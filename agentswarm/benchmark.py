"""Score protocols on a task set: accuracy, cost, and how they degrade under stress."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .agents import Task
from .protocols import Decision
from .reliability import ReliabilityTracker


@dataclass
class Score:
    name: str
    accuracy: float
    ci95: float
    cost: float
    abstain: float
    extra: dict

    def row(self) -> str:
        extra = " ".join(f"{k}={v:.2f}" for k, v in self.extra.items())
        return f"{self.name:<26}{self.accuracy:>8.1%} ±{self.ci95:>5.1%}{self.cost:>9.2f}{self.abstain:>9.1%}  {extra}"


HEADER = f"{'protocol':<26}{'accuracy':>8}{'':>7}{'cost':>9}{'abstain':>9}"


def evaluate(name: str, protocol: Callable[[Task], Decision], tasks: list[Task],
             tracker: ReliabilityTracker | None = None, warmup: int = 0) -> Score:
    """Run ``protocol`` on every task.

    If a tracker is given, every answer's correctness is fed back after the decision
    (online learning: the protocol never sees a task's label before deciding on it).
    The first ``warmup`` tasks are used only for that feedback and are not scored.
    """
    hits, costs, abstain, escal, rounds = [], [], 0, [], []
    for i, t in enumerate(tasks):
        d = protocol(t)
        if tracker is not None:
            for a in d.answers:
                tracker.update(a.agent, t.domain, a.choice == t.answer)
        if i < warmup:
            continue
        hits.append(d.choice == t.answer)
        costs.append(d.cost)
        abstain += d.choice is None
        if "escalated" in d.meta:
            escal.append(d.meta["escalated"])
        if "rounds" in d.meta:
            rounds.append(d.meta["rounds"])
    n = len(hits)
    acc = float(np.mean(hits))
    extra = {}
    if escal:
        extra["escalated"] = float(np.mean(escal))
    if rounds:
        extra["rounds"] = float(np.mean(rounds))
    return Score(name, acc, 1.96 * np.sqrt(acc * (1 - acc) / n), float(np.mean(costs)), abstain / n, extra)
