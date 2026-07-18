# External research and independent challenge

External research expands the candidate set; it never replaces local evidence.
The optimization loop must remain usable with no network access.

## Modes

| Mode | Behavior |
|---|---|
| `off` | Use local evidence and bundled knowledge only |
| `search` | Check current primary documentation |
| `challenge` | Add independent external model critiques |
| `auto` | Use search or challenge only at the triggers below |

Use `auto` by default when network and policy permit it. Trigger research for:

- a version-specific or architecture-specific uncertainty;
- initial selection among materially different optimization directions;
- two failed candidates based on the same mechanism;
- a plateau where the current profiler evidence does not explain the limit;
- pre-release review of a major new method or compatibility claim.

## Search protocol

Search official vendor documentation, source repositories, specifications, and
research papers first. Record the query, URL, document version, access date,
claim, and the local observation it is meant to explain. Do not copy large
manuals into the repository.

## Independent challenge protocol

1. Build a small evidence packet: objective, constraints, environment identity,
   profiler summary, attempted mechanisms, and unresolved questions.
2. Remove proprietary source, inputs, credentials, hostnames, and raw logs
   unless the user explicitly approves sharing them.
3. Ask heterogeneous models or roles for independent proposals before exposing
   other answers.
4. Require each critique to identify assumptions, contradicting evidence, and a
   test that could falsify its preferred explanation.
5. Preserve disagreements. Do not force consensus or use model votes as a
   promotion rule.

External models are advisory. They do not execute the target, mutate the
repository, change the host, or decide which candidate wins. Local correctness,
paired measurements, constraints, and evidence integrity remain authoritative.

Record unavailable providers and continue locally. Never turn network failure
into a blocked optimization when the required local foundation exists.

## Design basis

- [Multiagent Debate](https://arxiv.org/abs/2305.14325) motivates independent
  proposals and critique.
- [Critical evaluation of multi-agent debate](https://arxiv.org/abs/2502.08788)
  shows that debate can lose to strong single-agent baselines and that model
  heterogeneity and evaluation design matter.
- [Self-Knowledge Guided Retrieval](https://arxiv.org/abs/2310.05002) supports
  retrieving external material when uncertainty warrants it rather than on
  every step.
