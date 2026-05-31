# ruff: noqa: E402
from itertools import product
import time

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

import argparse
import json
import logging
import os
import signal
import sys
import threading
from functools import partial
from types import FrameType
from typing import Optional

import requests

from agents import AVAILABLE_AGENTS, Swarm
from agents.tracing import initialize as init_agentops

logger = logging.getLogger()

SCHEME = os.environ.get("SCHEME", "http")
HOST = os.environ.get("HOST", "localhost")
PORT = os.environ.get("PORT", 8001)

# Hide standard ports in URL
if (SCHEME == "http" and str(PORT) == "80") or (
    SCHEME == "https" and str(PORT) == "443"
):
    ROOT_URL = f"{SCHEME}://{HOST}"
else:
    ROOT_URL = f"{SCHEME}://{HOST}:{PORT}"
HEADERS = {
    "X-API-Key": os.getenv("ARC_API_KEY", ""),
    "Accept": "application/json",
}


def run_agent(swarm: Swarm) -> None:
    swarm.main()
    os.kill(os.getpid(), signal.SIGINT)


def cleanup(
    swarm: Swarm,
    signum: Optional[int],
    frame: Optional[FrameType],
) -> None:
    logger.info("Received SIGINT, exiting...")
    card_id = swarm.card_id
    if card_id:
        scorecard = swarm.close_scorecard(card_id)
        if scorecard:
            logger.info("--- EXISTING SCORECARD REPORT ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))
            swarm.cleanup(scorecard)
        
        # Provide web link to scorecard
        if card_id:
            scorecard_url = f"{ROOT_URL}/scorecards/{card_id}"
            logger.info(f"View your scorecard online: {scorecard_url}")

    sys.exit(0)


def main() -> None:
    log_level = logging.INFO
    if os.environ.get("DEBUG", "False") == "True":
        log_level = logging.DEBUG

    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler("logs.log", mode="w")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    # Specify which agents and (click-based) games to test run
    # ["base", "vanilla", "maml", "nn", "sac", "ppo"]
    agents = ["base"]
    # ["vc33", "tn36", "su15", "s5i5", "r11l", "lp85", "ft09"]
    _games = ["vc33"]
    # construct unique 'hash' for this eval run (based on date/time)
    date_str = time.strftime("%m/%d-%H:%M")  
    run_id = f"eval-{date_str}"
    print("Run ID:", run_id)
    print("Agents:", agents)
    print("Games:", _games)
    print('==================================')

    # Get the list of games from the API (name is slightly different)
    assert len(_games)
    full_games = []
    try:
        with requests.Session() as session:
            session.headers.update(HEADERS)
            r = session.get(f"{ROOT_URL}/api/games", timeout=10)

        if r.status_code == 200:
            try:
                full_games = [g["game_id"] for g in r.json()]
            except (ValueError, KeyError) as e:
                logger.error(f"Failed to parse games response: {e}")
                logger.error(f"Response content: {r.text[:200]}")
        else:
            logger.error(
                f"API request failed with status {r.status_code}: {r.text[:200]}"
            )
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to API server: {e}")
    games = [
        gid
        for gid in full_games
        if any(gid.startswith(prefix) for prefix in _games)
    ]

    logger.info(f"Game list: {games}")

    # Run all combinations of (agent, game)
    for a in agents:
        print(f"Running agent {a}\n==============================")
        # Start with base tag
        tags = [f"{run_id}-{a}"]  # e.g., "eval-09/01-12:00-base-vc33"

        # Initialize AgentOps client
        init_agentops(api_key=os.getenv("AGENTOPS_API_KEY"), log_level=log_level)

        swarm = Swarm(
            a,
            ROOT_URL,
            games,
            tags=tags,  # Pass tags as keyword argument
        )
        agent_thread = threading.Thread(target=partial(run_agent, swarm))
        agent_thread.daemon = True  # die when the main thread dies
        agent_thread.start()

        signal.signal(signal.SIGINT, partial(cleanup, swarm))  # handler for Ctrl+C

        try:
            # Wait for the agent thread to complete
            while agent_thread.is_alive():
                agent_thread.join(timeout=5)  # Check every 5 second
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received in main thread")
            cleanup(swarm, signal.SIGINT, None)
        except Exception as e:
            logger.error(f"Unexpected error in main thread: {e}")
            cleanup(swarm, None, None)

if __name__ == "__main__":
    os.environ["TESTING"] = "False"
    main()
