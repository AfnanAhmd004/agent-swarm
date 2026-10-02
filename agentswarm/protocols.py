"""Coordination protocols. Each returns a :class:`Decision` with the answer and its cost.

All protocols tolerate agent failures (timeouts, exceptions): failed calls still
cost money, but simply do not vote. A protocol that ends up with no votes abstains
(``choice = None``), which is scored as wrong.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .agents import Agent, Answer, Task
from .reliability import ReliabilityTracker


@dataclass
class Decision:
    choice: int | None
    cost: float
    answers: list[Answer] = field(default_factory=list)
    failures: int = 0
    meta: dict = field(default_factory=dict)


class Swarm:
    """Fans calls out to agents concurrently with a per-call timeout."""

    def __init__(self, max_workers: int = 16, timeout_s: float = 30.0, concurrent: bool = True):
        self.pool = ThreadPoolExecutor(max_workers=max_workers) if concurrent else None
        self.timeout_s = timeout_s

    def ask(self, agents: Sequence[Agent], task: Task, peers: list[Answer] | None = None) -> tuple[list[Answer], int, float]:
        """Returns (answers, n_failures, cost). Failed calls are still billed."""
        answers, failures, cost = [], 0, 0.0
        if self.pool is None:
            for a in agents:
                cost += a.cost
                try:
                    answers.append(a.answer(task, peers))
                except Exception:
                    failures += 1
            return answers, failures, cost
        futs = [(a, self.pool.submit(a.answer, task, peers)) for a in agents]
        for a, f in futs:
            cost += a.cost
            try:
                answers.append(f.result(timeout=self.timeout_s))
            except (Exception, FutTimeout):
                failures += 1
        return answers, failures, cost

    def close(self) -> None:
        if self.pool:
            self.pool.shutdown()


def _vote(answers: list[Answer], n_choices: int, weight: Callable[[Answer], float]) -> int | None:
    if not answers:
        return None
    score = np.zeros(n_choices)
    for a in answers:
        score[a.choice] += weight(a)
    best = np.flatnonzero(score == score.max())
    # deterministic tie-break: the highest-confidence answer among the tied choices
    tied = [a for a in answers if a.choice in best]
    return max(tied, key=lambda a: a.confidence).choice if tied else int(best[0])


def single(swarm: Swarm, agent: Agent, task: Task) -> Decision:
    ans, fails, cost = swarm.ask([agent], task)
    return Decision(ans[0].choice if ans else None, cost, ans, fails)


def majority_vote(swarm: Swarm, agents: Sequence[Agent], task: Task) -> Decision:
    ans, fails, cost = swarm.ask(agents, task)
    return Decision(_vote(ans, task.n_choices, lambda a: 1.0), cost, ans, fails)


def confidence_vote(swarm: Swarm, agents: Sequence[Agent], task: Task) -> Decision:
    """Weights votes by self-reported confidence (log-odds). Trusts what agents say about themselves."""
    def w(a: Answer) -> float:
        c = min(max(a.confidence, 1e-3), 1 - 1e-3)
        return float(np.log(c * (task.n_choices - 1) / (1 - c)))
    ans, fails, cost = swarm.ask(agents, task)
    return Decision(_vote(ans, task.n_choices, w), cost, ans, fails)


def reliability_vote(swarm: Swarm, agents: Sequence[Agent], task: Task, tracker: ReliabilityTracker) -> Decision:
    """Weights votes by each agent's *measured* track record on this domain."""
    ans, fails, cost = swarm.ask(agents, task)
    choice = _vote(ans, task.n_choices, lambda a: tracker.weight(a.agent, task.domain, task.n_choices))
    return Decision(choice, cost, ans, fails)


def route(swarm: Swarm, router: Callable[[Task], str], specialists: dict[str, Agent], task: Task,
          router_cost: float = 0.5, fallback: Agent | None = None) -> Decision:
    """A supervisor classifies the task and sends it to one specialist."""
    domain = router(task)
    agent = specialists.get(domain, fallback)
    d = single(swarm, agent, task)
    if d.choice is None and fallback is not None and agent is not fallback:  # specialist failed
        f = single(swarm, fallback, task)
        d = Decision(f.choice, d.cost + f.cost, d.answers + f.answers, d.failures + f.failures)
    d.cost += router_cost
    d.meta["routed_to"] = domain
    return d


def cascade(swarm: Swarm, cheap: Agent, escalate: Callable[[Task], Decision], task: Task, threshold: float) -> Decision:
    """Ask a cheap agent first; escalate only when it is unsure (or fails)."""
    first = single(swarm, cheap, task)
    conf = first.answers[0].confidence if first.answers else 0.0
    if first.choice is not None and conf >= threshold:
        first.meta["escalated"] = False
        return first
    second = escalate(task)
    return Decision(second.choice, first.cost + second.cost, first.answers + second.answers,
                    first.failures + second.failures, {"escalated": True})


def debate(swarm: Swarm, agents: Sequence[Agent], task: Task, rounds: int = 2) -> Decision:
    """Agents answer, then see each other's answers and may revise; final majority vote."""
    ans, fails, cost = swarm.ask(agents, task)
    history = [ans]
    for _ in range(rounds):
        if not ans or len({a.choice for a in ans}) == 1:  # consensus reached: stop early
            break
        ans, f, c = swarm.ask(agents, task, peers=ans)
        fails, cost = fails + f, cost + c
        history.append(ans)
    return Decision(_vote(ans, task.n_choices, lambda a: 1.0), cost, ans, fails, {"rounds": len(history) - 1})
