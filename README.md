# agent-swarm

Coordination protocols for **teams of AI agents** (voting, reliability weighting, supervisor routing, cascades, debate), plus a benchmark that measures what each one buys you in **accuracy, cost and robustness**.

Multi-agent systems are easy to build and hard to justify. This repo asks the questions a CTO would ask before paying for five model calls instead of one:

- When does an ensemble actually help?
- What does it cost?
- What happens when one agent is adversarial or the infrastructure is flaky?

![accuracy vs cost](docs/pareto.png)

## Protocols

| Protocol | Idea |
|---|---|
| `single` | one agent, the baseline |
| `majority_vote` | ask N agents, plurality wins (ties go to the most confident) |
| `confidence_vote` | weight each vote by the agent's *self-reported* confidence |
| `reliability_vote` | weight each vote by the agent's *measured* track record per domain: a Beta–Bernoulli posterior turned into the optimal log-odds weights of Nitzan & Paroush; sub-chance agents are muted |
| `route` | a supervisor classifies the task and sends it to one specialist, with a fallback if the specialist fails |
| `cascade` | a cheap agent answers first; escalate to the ensemble only when it is unsure |
| `debate` | agents see each other's answers and may revise over several rounds; stops early on consensus |

`Swarm` runs agent calls concurrently with per-call timeouts. Failed calls are billed but do not vote, so every protocol degrades gracefully instead of crashing. Real models plug in with `CallableAgent(name, fn)`.

## Why simulated agents

To measure protocols you need ground truth and the ability to turn individual knobs. `SimulatedAgent` answers multiple-choice tasks with:

- per-domain skill
- **correlated errors** within a model family (a Gaussian copula: same blind spots, same tempting wrong answer)
- weakly informative confidence, like verbalised LLM confidence
- cost per call and a failure rate
- optional adversarial (Byzantine) behaviour

The numbers below therefore show **mechanisms under stated assumptions**, not benchmark scores of real models.

## Results

```bash
pip install -e ".[dev]"
python examples/benchmark.py   # ~5 s; 2,000 scored tasks per row, 95% CIs
pytest
```

```
=== 1. diverse team (independent errors) ===
protocol                  accuracy            cost
single best model            72.2% ± 2.0%    10.00
majority vote x5             87.7% ± 1.4%    32.00
confidence-weighted x5       90.8% ± 1.3%    32.00
reliability-weighted x5      86.0% ± 1.5%    32.00
debate x5, 2 rounds          94.8% ± 1.0%    78.48   rounds=1.45
debate, conformity only      83.8% ± 1.6%    77.98   rounds=1.44
router -> specialist         86.3% ± 1.5%     3.50
cascade 8b -> vote x5        84.0% ± 1.6%    24.89   escalated=0.75

=== 2. same model x5 (error correlation 0.7) ===
single sample                71.2% ± 2.0%    10.00
majority vote x5             75.5% ± 1.9%    50.00

=== 3. one Byzantine agent (always wrong, 99% 'confident') ===
majority vote x5             64.8% ± 2.1%    32.00
confidence-weighted x5       44.2% ± 2.2%    32.00
reliability-weighted x5      84.6% ± 1.6%    32.00

=== 4. 30% of agent calls fail (timeouts) ===
single best model            50.0% ± 2.2%    10.00   (abstains 31%)
majority vote x5             83.0% ± 1.6%    32.00
```

### Takeaways

1. **Diversity is what you pay for.** Five *different* models lift accuracy from 72% to 88%. Five samples of *one* model with correlated blind spots only reach 76%, at 5× the cost.
2. **Routing beats brute force on cost.** A 90%-accurate router sending tasks to fine-tuned specialists gets 86% at **one-ninth** of the ensemble's cost. It is the clear Pareto winner.
3. **Debate is only as good as its "truth advantage".** When correct arguments persuade, debate reaches 95%. Under pure social pressure it *herds* to 84%, worse than a plain vote at 2.4× the cost.
4. **Never trust self-reported confidence in an open system.** One overconfident adversary drags confidence-weighting to 44%. Weighting by *measured* reliability holds 85%. With honest agents of similar skill, reliability weights add nothing over a plain vote (86% vs 88%, within noise); their value is robustness.
5. **Cascades need a cheap model whose confidence means something.** With weakly informative confidence, 75% of tasks escalate and the cascade is dominated by routing.
6. **Redundancy is fault tolerance.** With 30% of calls failing, a single agent abstains on 31% of tasks, while a 5-way vote still answers 83% correctly.

## Layout

```
agentswarm/
  agents.py       Task, Answer, SimulatedAgent (copula-correlated errors), CallableAgent
  protocols.py    Swarm (concurrent calls, timeouts) and the seven protocols
  reliability.py  Beta–Bernoulli reliability tracker and optimal vote weights
  benchmark.py    online evaluation with feedback, normal-approximation 95% CIs
examples/benchmark.py
tests/            11 tests incl. timeouts, fallbacks, Byzantine muting, correlated errors
```

## License

MIT
