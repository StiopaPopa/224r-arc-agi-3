from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm
from .templates.langgraph import LangGraph
from .templates.langgraph_functional_agent import LangGraphFunc, LangGraphTextOnly
from .templates.langgraph_random_agent import LangGraphRandom
from .templates.llm_agents import LLM, FastLLM, GuidedLLM, ReasoningLLM
from .templates.random_agent import Random
from .templates.reasoning_agent import ReasoningAgent
from .templates.smolagents import SmolCodingAgent, SmolVisionAgent
from .heuristic_agent import HeuristicAgent
from .heuristic_rl_prioritiy_agent import (
    HeuristicRLAgent,
    HeuristicRLMAMLAgent,
    HeuristicRLNNAgent,
    HeuristicRLSACAgent,
    HeuristicRLVanillaAgent,
)

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

# update the agent dictionary to include subclasses of LLM class
AVAILABLE_AGENTS["reasoningagent"] = ReasoningAgent

# RL/MAML/NN priority-mode agents (subclasses of HeuristicAgent, not direct Agent subclasses)
AVAILABLE_AGENTS["heuristicrlagent"] = HeuristicRLAgent
AVAILABLE_AGENTS["heuristicrlvanillaagent"] = HeuristicRLVanillaAgent
AVAILABLE_AGENTS["heuristicrlmamlagent"] = HeuristicRLMAMLAgent
AVAILABLE_AGENTS["heuristicrlnnagent"] = HeuristicRLNNAgent
AVAILABLE_AGENTS["heuristicrlsacagent"] = HeuristicRLSACAgent

__all__ = [
    "Swarm",
    "Random",
    "LangGraph",
    "LangGraphFunc",
    "LangGraphTextOnly",
    "LangGraphRandom",
    "LLM",
    "FastLLM",
    "ReasoningLLM",
    "GuidedLLM",
    "ReasoningAgent",
    "SmolCodingAgent",
    "SmolVisionAgent",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
    "HeuristicRLAgent",
    "HeuristicRLVanillaAgent",
    "HeuristicRLMAMLAgent",
    "HeuristicRLNNAgent",
    "HeuristicRLSACAgent",
]
