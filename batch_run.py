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

# Root logger for main-thread messages only
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

# Global flag so Ctrl+C aborts all running agents cleanly
_abort = threading.Event()


def make_agent_logger(agent_name: str, log_level: int) -> logging.Logger:
    """Each agent gets its own logger writing to logs-{agent_name}.log so
    concurrent output doesn't interleave in a single file."""
    agent_logger = logging.getLogger(agent_name)
    agent_logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    # Per-agent file handler
    fh = logging.FileHandler(f"logs-{agent_name}.log", mode="w")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    agent_logger.addHandler(fh)
    # Also mirror to stdout so all agents are visible in the console
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(log_level)
    sh.setFormatter(formatter)
    agent_logger.addHandler(sh)
    return agent_logger


def run_swarm(swarm: Swarm, done_event: threading.Event, time_limit: float | None) -> None:
    """Run one swarm to completion, then signal done. Does NOT kill the process."""
    try:
        swarm.main(time_limit=time_limit)
    finally:
        done_event.set()


def cleanup_swarm(
    swarm: Swarm,
    agent_logger: logging.Logger,
) -> None:
    """Close and log the scorecard for one swarm. Safe to call from any thread."""
    card_id = swarm.card_id
    if card_id:
        scorecard = swarm.close_scorecard(card_id)
        if scorecard:
            agent_logger.info("--- EXISTING SCORECARD REPORT ---")
            agent_logger.info(json.dumps(scorecard.model_dump(), indent=2))
            swarm.cleanup(scorecard)

        # Provide web link to scorecard
        scorecard_url = f"{ROOT_URL}/scorecards/{card_id}"
        agent_logger.info(f"View your scorecard online: {scorecard_url}")


def sigint_handler(signum: int, frame: Optional[FrameType]) -> None:
    # signal.signal is only registered from the main thread
    logger.info("Ctrl+C received — aborting all runs...")
    _abort.set()


def run_one(agent_name: str, games: list[str], tags: list[str], time_limit: float | None, log_level: int) -> None:
    """Run a single agent swarm, blocking until done. Designed to be called from a thread."""
    agent_logger = make_agent_logger(agent_name, log_level)

    init_agentops(api_key=os.getenv("AGENTOPS_API_KEY"), log_level=log_level)

    swarm = Swarm(agent_name, ROOT_URL, games, tags=tags)
    done_event = threading.Event()
    # daemon=False so scorecard writing is never cut off mid-run
    thread = threading.Thread(target=run_swarm, args=(swarm, done_event, time_limit), daemon=False)
    thread.start()

    # Poll so _abort (Ctrl+C) can interrupt the wait
    while not done_event.is_set():
        if _abort.is_set():
            agent_logger.info(f"Aborting run for agent={agent_name}")
            break
        done_event.wait(timeout=2)

    thread.join(timeout=10)
    cleanup_swarm(swarm, agent_logger)


def main() -> None:
    log_level = logging.INFO
    if os.environ.get("DEBUG", "False") == "True":
        log_level = logging.DEBUG

    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)

    # Main-thread log (agent runs write to their own logs-{agent}.log files)
    file_handler = logging.FileHandler("logs.log", mode="w")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    # Must be registered from the main thread
    signal.signal(signal.SIGINT, sigint_handler)

    # Specify which agents and (click-based) games to test run
    # ["base", "vanilla", "maml", "nn", "sac", "ppo"]
    agents = ["base", "vanilla", "maml", "nn", "sac", "ppo"]
    # ["vc33", "tn36", "su15", "s5i5", "r11l", "lp85", "ft09"]
    _games = ["vc33"]
    # time limit (s) per agent across all games(!)
    time_limit = 60
    # construct unique 'hash' for this eval run (based on date/time)
    date_str = time.strftime("%m/%d-%H:%M")  
    run_id = f"eval-{time_limit}-{date_str}"
    print("Run ID:", run_id)
    print("Agents:", agents)
    print("Games:", _games)
    print("Time limit (s):", time_limit)
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

    # Launch all agents in parallel, each in its own thread
    agent_threads = []
    for a in agents:
        print(f"Launching agent {a}\n==============================")
        # Start with base tag
        tags = [f"{run_id}-{a}"]  # e.g., "eval-09/01-12:00-base-vc33"

        t = threading.Thread(target=run_one, args=(a, games, tags, time_limit, log_level), daemon=False)
        t.start()
        agent_threads.append((a, t))

    # Wait for all agents to finish (or _abort to be set)
    for a, t in agent_threads:
        while t.is_alive():
            if _abort.is_set():
                logger.info("Eval aborted — waiting for running agents to clean up...")
                break
            t.join(timeout=2)
        t.join()  # final join after abort to let cleanup finish
        print(f"Agent {a} done.")

    print('==================================')
    print("All agents done." if not _abort.is_set() else "Eval aborted.")


if __name__ == "__main__":
    os.environ["TESTING"] = "False"
    main()