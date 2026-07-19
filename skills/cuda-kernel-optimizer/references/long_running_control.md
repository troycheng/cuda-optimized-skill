# Long-running optimization control

Read this reference when an optimization may span many candidates, survive an
interruption, or run long enough for the workload or environment to drift.

## Ownership

- The Planner proposes one bounded candidate from verified observations.
- The Controller owns time, budget, state transitions, adapter execution,
  candidate admission, and ledger writes.
- Capability cards provide applicable methods and checks. They never authorize
  execution or promotion.
- Local correctness, paired measurement, workload replay, and constraints decide
  whether a candidate can be retained.

Do not give the Planner a Controller seal key or a writable run ledger.

## Start a run

1. Use `scripts/readiness.py` to establish the claim ceiling.
2. Build a workload-contract draft. Let the selected `quick`, `balanced`, or
   `thorough` preset supply stability defaults unless the user has chosen an
   explicit policy.
3. Use `scripts/workload_contract.py` to freeze the contract. The frozen file
   binds artifacts, objective, budget, stability policy, mutation roots, and
   `recommend_only` host policy.
4. Initialize the run ledger and enter `CALIBRATING`.
5. Collect baseline-only paired blocks on the target path. Use
   `scripts/stability_calibration.py` to estimate noise and the minimum
   detectable effect (MDE).

Never pass MPE, confidence, power, bootstrap count, minimum pair count, seed, or
audit cadence as ad hoc calibration arguments. They come from the verified
contract.

## Environment states

- `green`: the calibrated noise and MDE are no larger than the contract's
  minimum practical effect. New candidates may be admitted.
- `yellow`: the effect cannot be measured reliably, too few pairs are valid, or
  a periodic baseline shifted. Pause new candidates and improve or replay the
  measurement setup.
- `red`: a hard guardrail failed. Stop the run.

Invalid pairs remain counted but do not contribute to the baseline, noise, or
MDE. Zero valid pairs produce no invented statistic.

## Candidate loop

1. Derive the observation summary from Controller-sealed diagnostic evidence.
2. Query `scripts/capability_query.py` with exact architecture, task, signals,
   available evidence, and a hard context budget. Load only returned playbooks.
3. Register the candidate through `scripts/planner_boundary.py`. Bind its
   summary digest, capability-query digest, hypothesis, cost, kill gate,
   capability versions, and mutation paths before execution.
4. Let `scripts/evidence_controller.py` run allowlisted adapters. Adapters return
   measurements; the Controller constructs and signs evidence.
5. Evaluate correctness before performance. Record `PASS`, `KILL`,
   `INCONCLUSIVE`, or `DEFERRED`; never rewrite an earlier result.

The workload contract's `audit_every_candidates` field sets the maximum number
of newly registered candidates between baseline audits. Both online writes and
ledger replay enforce it.

## Audit and recovery

Enter `AUDITING` when cadence is reached or an anomaly appears. A periodic audit
must use the same green calibration anchor, contract, source identity, and
environment identity. Only a Controller-attested audit can resume `EXPLORING`.

On restart, verify the frozen artifacts and the complete append-only hash chain,
then replay every state transition, calibration, audit, admission, and candidate
result. Reject a truncated chain, stale tail, duplicate observation, cadence
violation, identity drift, or a state snapshot that cannot be reproduced.

If the workload, source, target, objective, or environment identity changes,
stop the old run. Freeze a new contract and create a new ledger with an explicit
parent reference. Never mix candidates or budget across contracts.

## External research

External search and independent models may propose sources, alternatives, and
counterexamples after private material is removed. They do not receive the
Controller key, write the ledger, change the contract, or vote on promotion.
When network access is unavailable, use the bundled source manifest and
capability registry. Local evidence remains decisive in both modes.
