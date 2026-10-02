"""Online agent reliability from feedback, and the optimal vote weights it implies."""
from __future__ import annotations

import math
from collections import defaultdict


class ReliabilityTracker:
    """Beta–Bernoulli posterior on P(agent is correct) per (agent, domain).

    With independent voters of accuracy p_i over K choices, the decision rule that
    maximises the probability of a correct group answer weights each vote by
    ``log(p_i (K-1) / (1 - p_i))`` (Nitzan & Paroush, 1982). Agents at or below
    chance get weight 0: they are muted rather than inverted, because an adversary
    could otherwise steer the group by being predictably wrong.
    """

    def __init__(self, prior_correct: float = 2.0, prior_wrong: float = 1.0):
        self.a = defaultdict(lambda: prior_correct)
        self.b = defaultdict(lambda: prior_wrong)

    def update(self, agent: str, domain: str, correct: bool) -> None:
        key = (agent, domain)
        if correct:
            self.a[key] += 1
        else:
            self.b[key] += 1

    def accuracy(self, agent: str, domain: str) -> float:
        key = (agent, domain)
        return self.a[key] / (self.a[key] + self.b[key])

    def weight(self, agent: str, domain: str, n_choices: int) -> float:
        p = min(max(self.accuracy(agent, domain), 1e-3), 1 - 1e-3)
        return max(0.0, math.log(p * (n_choices - 1) / (1 - p)))
