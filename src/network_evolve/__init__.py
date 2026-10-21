"""NetworkEvolveLab: Co-evolution of LLM Cognition and Dynamic Social Topology."""

from .graph import DynamicSocialGraph
from .agent import EvolvingAgent
from .engine import CoEvolutionEngine
from .metrics import snapshot, init_incident_edges
from .datasets import (
    load_debate_benchmark,
    load_snap_higgs,
    load_twitter_election,
    load_reddit_convokit,
    load_birdwatch_community_notes,
)
from .baselines import (
    BaseOpinionModel,
    DeGrootModel,
    HegselmannKrauseModel,
    FriedkinJohnsenModel,
    GenerativeAgentsMemoryModel,
    GuEtAlPromptRewireModel,
    OursCoEvolutionModel,
)

__version__ = "0.1.0"

__all__ = [
    "DynamicSocialGraph",
    "EvolvingAgent",
    "CoEvolutionEngine",
    "snapshot",
    "init_incident_edges",
    "load_debate_benchmark",
    "load_snap_higgs",
    "load_twitter_election",
    "load_reddit_convokit",
    "load_birdwatch_community_notes",
    "BaseOpinionModel",
    "DeGrootModel",
    "HegselmannKrauseModel",
    "FriedkinJohnsenModel",
    "GenerativeAgentsMemoryModel",
    "GuEtAlPromptRewireModel",
    "OursCoEvolutionModel",
    "__version__",
]
