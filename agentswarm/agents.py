"""Agents, tasks and a simulator with *controlled* error structure.

Real LLM agents plug in through :class:`CallableAgent`. For measuring protocols we
need ground truth and knobs we can turn, so :class:`SimulatedAgent` answers
multiple-choice tasks with:

* a per-domain skill (probability of being right),
* **correlated errors** inside a ``family`` (agents built on the same base model
  fail on the same tasks and fall for the same wrong answer), via a Gaussian copula,
* a confidence report that is informative (calibrated) or not (overconfident),
* a per-call cost, a failure rate, and optional adversarial (Byzantine) behaviour.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

DOMAINS = ("perception", "planning", "control", "quant")


@dataclass(frozen=True)
class Task:
    id: int
    domain: str
    n_choices: int
    answer: int
    trap: int  # the most tempting wrong answer
    seed: int  # per-task randomness shared by all agents


@dataclass
class Answer:
    agent: str
    choice: int
    confidence: float
    cost: float


class AgentFailure(RuntimeError):
    pass


class Agent(Protocol):
    name: str
    cost: float

    def answer(self, task: Task, peers: list[Answer] | None = None) -> Answer: ...


def make_tasks(n: int, n_choices: int = 4, seed: int = 0) -> list[Task]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        a = int(rng.integers(n_choices))
        trap = int(rng.choice([c for c in range(n_choices) if c != a]))
        out.append(Task(i, DOMAINS[int(rng.integers(len(DOMAINS)))], n_choices, a, trap, int(rng.integers(2**31))))
    return out


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


@dataclass
class SimulatedAgent:
    name: str
    skills: dict[str, float]
    family: str = "base"
    correlation: float = 0.0  # error correlation with agents of the same family
    trap_rate: float = 0.6  # when wrong, P(choose the shared trap answer)
    cost: float = 1.0
    calibrated: bool = True
    overconfidence: float = 0.0  # if not calibrated: reported confidence
    confidence_signal: float = 0.4  # how much reported confidence reveals correctness (logits)
    byzantine: bool = False  # always picks the trap, claims high confidence
    failure_rate: float = 0.0
    conformity: float = 0.3  # debate: tendency to adopt the peer plurality
    persuadability: float = 0.5  # debate: P(adopting a correct peer argument when wrong)
    calls: int = field(default=0, init=False)
    _rng: np.random.Generator = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(zlib.crc32(self.name.encode()))

    def _first_answer(self, task: Task) -> tuple[int, bool]:
        # Shared latent per (task, family) + private latent per (task, agent):
        # deterministic per task, so repeated calls reproduce the same "belief".
        fam = np.random.default_rng([task.seed, zlib.crc32(self.family.encode())])
        own = np.random.default_rng([task.seed, zlib.crc32(self.name.encode())])
        z = math.sqrt(self.correlation) * fam.standard_normal() + math.sqrt(1 - self.correlation) * own.standard_normal()
        correct = _phi(z) < self.skills.get(task.domain, 0.5)
        if correct:
            return task.answer, True
        wrong = [c for c in range(task.n_choices) if c != task.answer]
        if fam.random() < self.trap_rate:
            return task.trap, False
        return int(own.choice(wrong)), False

    def _confidence(self, task: Task, correct: bool) -> float:
        if not self.calibrated:
            return self.overconfidence
        p = self.skills.get(task.domain, 0.5)
        # weakly informative, like verbalised LLM confidence: a small shift plus noise
        z = _logit(p) + (self.confidence_signal if correct else -self.confidence_signal) + self._rng.standard_normal()
        return 1 / (1 + math.exp(-z))

    def answer(self, task: Task, peers: list[Answer] | None = None) -> Answer:
        self.calls += 1
        if self.failure_rate and self._rng.random() < self.failure_rate:
            raise AgentFailure(f"{self.name} timed out")
        if self.byzantine:
            return Answer(self.name, task.trap, 0.99, self.cost)
        choice, correct = self._first_answer(task)
        if peers:
            choice, correct = self._reconsider(task, choice, peers)
        return Answer(self.name, choice, self._confidence(task, correct), self.cost)

    def _reconsider(self, task: Task, choice: int, peers: list[Answer]) -> tuple[int, bool]:
        """One debate round: maybe get convinced by a correct argument, maybe conform."""
        others = [p for p in peers if p.agent != self.name]
        if not others:
            return choice, choice == task.answer
        if choice != task.answer and any(p.choice == task.answer for p in others):
            if self._rng.random() < self.persuadability * self.skills.get(task.domain, 0.5):
                return task.answer, True
        counts = np.bincount([p.choice for p in others], minlength=task.n_choices)
        plurality = int(counts.argmax())
        share = counts[plurality] / len(others)
        if plurality != choice and self._rng.random() < self.conformity * share:
            return plurality, plurality == task.answer
        return choice, choice == task.answer


@dataclass
class CallableAgent:
    """Wrap any function (e.g. an LLM call) as an agent.

    ``fn(task, peers) -> (choice, confidence)``.
    """

    name: str
    fn: Callable[[Task, list[Answer] | None], tuple[int, float]]
    cost: float = 1.0

    def answer(self, task: Task, peers: list[Answer] | None = None) -> Answer:
        choice, conf = self.fn(task, peers)
        return Answer(self.name, int(choice), float(conf), self.cost)
