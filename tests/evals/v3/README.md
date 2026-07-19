# 3.0 skill evaluation

This directory defines comparable behavior scenarios for 2.9, the 3.0 control
framework with a random planner, and full 3.0. It is not a collection of claimed
speedups.

An executor runs every mode with the same model, target, workload contract, and
budget, then writes one strict result object per scenario. `tools/run_skill_eval.py`
checks coverage and reports correctness, policy, direction, recovery, time, GPU
cost, and candidate efficiency without a model-generated quality score.

Scenario events must come from retained run artifacts. A human or model may not
add an event merely to make a score pass. Physical-GPU results are stored outside
the repository until private paths and inputs have been sanitized.

Example scoring command:

```bash
python3 tools/run_skill_eval.py \
  --suite tests/evals/v3/scenarios.json \
  --results /path/to/v2.9-results.json \
  --mode v2.9 \
  --out /path/to/v2.9-score.json
```
