# 3.0 skill evaluation

This directory defines one preregistered five-arm experiment: no skill, 2.9,
the 3.0 control framework with a random planner, 3.0 with shuffled registry
metadata, and full 3.0. It is not a collection of claimed speedups.

An executor runs every arm and replicate with paired seeds and fixed model,
prompt, target, workload contract, environment, and budget. Every result binds
those identities plus the exact skill content. `tools/run_skill_eval.py` checks
coverage and reports correctness, policy, direction, recovery, time, GPU cost,
and candidate efficiency without a model-generated quality score.

Scenario events must name their ledger sequence and source artifact SHA-256.
The scenario oracle derives them from retained run artifacts; a human or model
may not add an event merely to make a score pass. Physical-GPU results are stored
outside the repository until private paths and inputs have been sanitized.

Example scoring command:

```bash
python3 tools/run_skill_eval.py \
  --suite tests/evals/v3/scenarios.json \
  --results /path/to/v2.9-results.json \
  --mode v2.9 \
  --out /path/to/v2.9-score.json
```
