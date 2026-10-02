"""Benchmark coordination protocols on cost and accuracy, under four conditions.

    python examples/benchmark.py            # ~20 s, writes docs/pareto.png
"""
from __future__ import annotations

import os

import numpy as np

from agentswarm import (
    DOMAINS,
    HEADER,
    ReliabilityTracker,
    SimulatedAgent,
    Swarm,
    cascade,
    confidence_vote,
    debate,
    evaluate,
    majority_vote,
    make_tasks,
    reliability_vote,
    route,
    single,
)

N_TASKS, WARMUP = 2400, 400
TASKS = make_tasks(N_TASKS, seed=1)


def flat(p):
    return {d: p for d in DOMAINS}


def diverse_team(**kw):
    """Five general models from five different vendors/base models (independent errors)."""
    specs = [("gpt-like", 0.72, 10), ("claude-like", 0.70, 10), ("open-70b", 0.68, 6), ("open-32b", 0.65, 4), ("open-14b", 0.62, 2)]
    return [SimulatedAgent(n, flat(p), family=n, cost=c, **kw) for n, p, c in specs]


def same_model_team(corr=0.7):
    """Five samples of one model: errors are correlated (same blind spots)."""
    return [SimulatedAgent(f"sample-{i}", flat(0.72), family="one-model", correlation=corr, cost=10) for i in range(5)]


def specialists():
    out = {}
    for i, d in enumerate(DOMAINS):
        skills = flat(0.55)
        skills[d] = 0.88
        out[d] = SimulatedAgent(f"{d}-specialist", skills, family=f"ft-{d}", cost=3)
    return out


def router(accuracy=0.9):
    def pick(task):
        rng = np.random.default_rng([task.seed, 7])
        return task.domain if rng.random() < accuracy else DOMAINS[int(rng.integers(len(DOMAINS)))]
    return pick


def run(title, rows):
    print(f"\n=== {title} ===\n{HEADER}")
    scores = []
    for name, fn, tracker in rows:
        s = evaluate(name, fn, TASKS, tracker=tracker, warmup=WARMUP)
        scores.append(s)
        print(s.row())
    return scores


def main() -> None:
    sw = Swarm(concurrent=False)  # simulated agents are instant; threads only add overhead here

    # 1. diverse team, independent errors -------------------------------------------------
    team = diverse_team()
    herd = diverse_team(persuadability=0.0, conformity=0.6)  # no truth advantage, just social pressure
    rt = ReliabilityTracker()
    rt_c = ReliabilityTracker()
    cheap = SimulatedAgent("open-8b", flat(0.58), family="open-8b", cost=1)
    spec = specialists()
    s1 = run("1. diverse team (independent errors)", [
        ("single best model", lambda t: single(sw, team[0], t), None),
        ("majority vote x5", lambda t: majority_vote(sw, team, t), None),
        ("confidence-weighted x5", lambda t: confidence_vote(sw, team, t), None),
        ("reliability-weighted x5", lambda t: reliability_vote(sw, team, t, rt), rt),
        ("debate x5, 2 rounds", lambda t: debate(sw, team, t, rounds=2), None),
        ("debate, conformity only", lambda t: debate(sw, herd, t, rounds=2), None),
        ("router -> specialist", lambda t: route(sw, router(0.9), spec, t), None),
        ("cascade 8b -> vote x5", lambda t: cascade(sw, cheap, lambda u: reliability_vote(sw, team, u, rt_c), t, 0.75), rt_c),
    ])

    # 2. one model sampled five times ---------------------------------------------------
    same = same_model_team(0.7)
    run("2. same model x5 (error correlation 0.7)", [
        ("single sample", lambda t: single(sw, same[0], t), None),
        ("majority vote x5", lambda t: majority_vote(sw, same, t), None),
    ])

    # 3. one adversarial, overconfident agent -------------------------------------------
    bad_team = diverse_team()[:4] + [SimulatedAgent("rogue", flat(0.7), family="rogue", byzantine=True, cost=2)]
    rt3 = ReliabilityTracker()
    run("3. one Byzantine agent (always wrong, 99% 'confident')", [
        ("majority vote x5", lambda t: majority_vote(sw, bad_team, t), None),
        ("confidence-weighted x5", lambda t: confidence_vote(sw, bad_team, t), None),
        ("reliability-weighted x5", lambda t: reliability_vote(sw, bad_team, t, rt3), rt3),
    ])

    # 4. flaky infrastructure -----------------------------------------------------------
    flaky = diverse_team(failure_rate=0.3)
    run("4. 30% of agent calls fail (timeouts)", [
        ("single best model", lambda t: single(sw, flaky[0], t), None),
        ("majority vote x5", lambda t: majority_vote(sw, flaky, t), None),
    ])

    # Pareto plot for scenario 1 ------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in s1:
        ax.errorbar(s.cost, s.accuracy, yerr=s.ci95, fmt="o", capsize=3)
        ax.annotate(s.name, (s.cost, s.accuracy), textcoords="offset points", xytext=(6, -3), fontsize=8)
    ax.set_xlabel("mean cost per task (relative units)")
    ax.set_ylabel("accuracy")
    ax.set_title("Coordination protocols: accuracy vs cost (diverse team)")
    ax.set_xscale("log")
    ax.set_xlim(2.5, 300)
    ax.grid(alpha=0.3)
    os.makedirs("docs", exist_ok=True)
    fig.tight_layout()
    fig.savefig("docs/pareto.png", dpi=130)
    print("\nsaved docs/pareto.png")


if __name__ == "__main__":
    main()
