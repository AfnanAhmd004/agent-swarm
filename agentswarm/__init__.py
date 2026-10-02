"""agent-swarm: coordination protocols for teams of agents, measured on cost and accuracy."""
from .agents import DOMAINS, Agent, AgentFailure, Answer, CallableAgent, SimulatedAgent, Task, make_tasks
from .benchmark import HEADER, Score, evaluate
from .protocols import (
    Decision,
    Swarm,
    cascade,
    confidence_vote,
    debate,
    majority_vote,
    reliability_vote,
    route,
    single,
)
from .reliability import ReliabilityTracker

__all__ = [
    "DOMAINS", "Agent", "AgentFailure", "Answer", "CallableAgent", "SimulatedAgent", "Task", "make_tasks",
    "Decision", "Swarm", "single", "majority_vote", "confidence_vote", "reliability_vote", "route", "cascade",
    "debate", "ReliabilityTracker", "evaluate", "Score", "HEADER",
]
