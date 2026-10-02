import time

import numpy as np
import pytest

from agentswarm import (
    DOMAINS,
    Answer,
    CallableAgent,
    ReliabilityTracker,
    SimulatedAgent,
    Swarm,
    Task,
    cascade,
    debate,
    evaluate,
    majority_vote,
    make_tasks,
    reliability_vote,
    route,
    single,
)

TASKS = make_tasks(1500, seed=3)
FLAT = {d: 0.7 for d in DOMAINS}


@pytest.fixture
def sw():
    s = Swarm(concurrent=False)
    yield s
    s.close()


def fixed(name, choice, conf=0.8, cost=1.0):
    return CallableAgent(name, lambda t, p: (choice, conf), cost)


def test_simulated_accuracy_matches_skill():
    a = SimulatedAgent("a", FLAT)
    acc = np.mean([a.answer(t).choice == t.answer for t in TASKS])
    assert abs(acc - 0.7) < 0.04


def test_same_family_errors_are_correlated():
    def both_wrong(a, b):
        return np.mean([(a.answer(t).choice != t.answer) and (b.answer(t).choice != t.answer) for t in TASKS])

    indep = both_wrong(SimulatedAgent("x", FLAT, family="x"), SimulatedAgent("y", FLAT, family="y"))
    corr = both_wrong(SimulatedAgent("p", FLAT, family="m", correlation=0.8), SimulatedAgent("q", FLAT, family="m", correlation=0.8))
    assert indep == pytest.approx(0.09, abs=0.03) and corr > 2 * indep


def test_majority_vote_and_tie_break(sw):
    t = Task(0, "quant", 4, answer=1, trap=2, seed=0)
    assert majority_vote(sw, [fixed("a", 1), fixed("b", 1), fixed("c", 2)], t).choice == 1
    tie = majority_vote(sw, [fixed("a", 1, 0.9), fixed("b", 2, 0.6)], t)
    assert tie.choice == 1 and tie.cost == 2


def test_failures_are_billed_but_do_not_vote(sw):
    def boom(t, p):
        raise TimeoutError

    t = Task(0, "quant", 4, answer=0, trap=1, seed=0)
    d = majority_vote(sw, [fixed("a", 0), CallableAgent("b", boom, 5.0)], t)
    assert d.choice == 0 and d.failures == 1 and d.cost == 6.0
    assert single(sw, CallableAgent("b", boom), t).choice is None


def test_concurrent_swarm_respects_timeout():
    def slow(t, p):
        time.sleep(0.5)
        return 0, 0.9

    s = Swarm(timeout_s=0.05)
    t = Task(0, "quant", 4, answer=1, trap=0, seed=0)
    d = majority_vote(s, [fixed("fast", 1), CallableAgent("slow", slow)], t)
    s.close()
    assert d.choice == 1 and d.failures == 1


def test_reliability_weights_mute_bad_agents():
    tr = ReliabilityTracker()
    for _ in range(50):
        tr.update("good", "quant", True)
        tr.update("bad", "quant", False)
    assert tr.weight("good", "quant", 4) > 2 and tr.weight("bad", "quant", 4) == 0.0


def test_reliability_vote_neutralises_a_byzantine_agent(sw):
    good = [SimulatedAgent(f"g{i}", FLAT, family=f"g{i}") for i in range(3)]
    rogues = [SimulatedAgent(f"r{i}", FLAT, family=f"r{i}", byzantine=True) for i in range(2)]
    team = good + rogues
    tr = ReliabilityTracker()
    weighted = evaluate("w", lambda t: reliability_vote(sw, team, t, tr), TASKS, tracker=tr, warmup=200)
    plain = evaluate("m", lambda t: majority_vote(sw, team, t), TASKS, warmup=200)
    assert weighted.accuracy > plain.accuracy + 0.15


def test_router_falls_back_when_specialist_fails(sw):
    def boom(t, p):
        raise RuntimeError

    t = Task(0, "control", 4, answer=3, trap=0, seed=0)
    d = route(sw, lambda task: "control", {"control": CallableAgent("spec", boom, 2.0)}, t,
              router_cost=0.5, fallback=fixed("gen", 3, cost=4.0))
    assert d.choice == 3 and d.cost == 6.5 and d.meta["routed_to"] == "control"


def test_cascade_escalates_only_when_unsure(sw):
    t = Task(0, "quant", 4, answer=2, trap=0, seed=0)
    expensive = lambda u: single(sw, fixed("big", 2, cost=10), u)
    sure = cascade(sw, fixed("small", 0, conf=0.95), expensive, t, threshold=0.8)
    unsure = cascade(sw, fixed("small", 0, conf=0.4), expensive, t, threshold=0.8)
    assert (sure.choice, sure.cost, sure.meta["escalated"]) == (0, 1, False)
    assert (unsure.choice, unsure.cost, unsure.meta["escalated"]) == (2, 11, True)


def test_debate_stops_on_consensus(sw):
    t = Task(0, "quant", 4, answer=1, trap=0, seed=0)
    d = debate(sw, [fixed("a", 1), fixed("b", 1)], t, rounds=3)
    assert d.meta["rounds"] == 0 and d.cost == 2


def test_debate_with_truth_advantage_beats_first_round(sw):
    team = [SimulatedAgent(f"d{i}", FLAT, family=f"d{i}", persuadability=0.8, conformity=0.1) for i in range(5)]
    first = evaluate("v", lambda t: majority_vote(sw, team, t), TASKS)
    deb = evaluate("d", lambda t: debate(sw, team, t, rounds=2), TASKS)
    assert deb.accuracy > first.accuracy
