# CUDA Skill V2.5 Evidence Automation Implementation Plan

> **Worker note:** Execute inline with strict RED-GREEN-refactor cycles. Do not
> enable GPU tests, contact `/data/triton-handoff`, or mutate host/external
> services.

**Goal:** Add fail-closed, executable V2.5 evidence automation while preserving
V2.4.1 manifests and APIs.

**Architecture:** Extend the existing experiment-design seam, add one reusable
evidence protocol module plus a thin CLI, and bind formal decisions through an
immutable seal/audit/decision closure. Keep all runtime logic Python 3.10+
standard library and all local tests CPU/static.

**Technical stack:** Python 3.10+, `unittest`, strict JSON/JSONL, SHA-256,
atomic local artifacts.

---

## Files

- Modify `skills/cuda-kernel-optimizer/scripts/experiment_design.py`: frozen
  formal design validation.
- Modify `skills/cuda-kernel-optimizer/scripts/workload_evaluate.py`: optional
  frozen schedule and zero per-role retries in formal mode.
- Create `skills/cuda-kernel-optimizer/scripts/evidence_protocol.py`: guard,
  coverage, serving, identity, seal, audit, decision, and import validators.
- Create `skills/cuda-kernel-optimizer/scripts/evidence.py`: CLI facade.
- Create `skills/cuda-kernel-optimizer/scripts/self_check.py`: installed CPU
  validation.
- Create strict schemas under `skills/cuda-kernel-optimizer/templates/` for
  guard policy, experiment design, attempt, serving experiment, and profiler
  bundle.
- Create `skills/cuda-kernel-optimizer/references/evidence_automation.md` and
  `migration_v2_5.md`; update `SKILL.md`, agent metadata, and readmes.
- Create `tests/test_evidence_protocol.py` and `tests/test_evidence_cli.py`;
  extend experiment/workload/metadata/readme tests.

### Task 1: Frozen design and evaluator boundary

- [x] Add RED tests for closed keys, complete balanced schedules, statistical
  units, both guardrail forms, no-exclusion, and whole-pair-only retry policy.
- [x] Run the focused tests and confirm failures are missing V2.5 behavior.
- [x] Implement the smallest detached validator and schedule accessor.
- [x] Add RED tests proving formal evaluation rejects role retries and uses the
  exact frozen schedule.
- [x] Implement the optional formal evaluator seam; preserve legacy defaults.
- [x] Run experiment and workload evaluator tests to GREEN.

### Task 2: Shared-host environment guard

- [x] Add RED tests for one complete clean attempt and fail-closed cases:
  missing sample/metric/identity, excessive gap, missing watcher readiness,
  insufficient clean window, CPU/NUMA drift, foreign process/container,
  swap/memory pressure, clock/temp/power/thermal, and contamination markers.
- [x] Implement strict policy/sample/phase validation and a deterministic audit
  result with per-phase reasons.
- [x] Run the focused guard tests to GREEN.

### Task 3: Execution path, serving, artifact identity, and profiler bundle

- [x] Add RED tests for expected-case coverage and positive dispatch hits.
- [x] Add RED tests requiring diagnostic removal, rebuild/rehash, source/config
  binding, and a residue-free timed binary.
- [x] Add RED tests for c1/c2/c4/c8/c12 HTTP/gRPC strata, fresh process,
  request counts, required metrics, and per-stratum constraints.
- [x] Add RED tests for plugin/engine/backend/server/image identities, tactics,
  timing-cache versions, and non-promotional Nsys/NCU authority.
- [x] Implement the minimum pure validators and run focused tests to GREEN.

### Task 4: Immutable attempt and evidence closure

- [x] Add RED tests for all terminal states, create-once seals, required artifact
  kinds, safe regular paths, raw rows matching the frozen schedule, and digest
  drift.
- [x] Add RED tests separating performance verdict from evidence integrity.
- [x] Add RED tests for seal -> audit -> decision -> closure references and
  fail-closed promotion.
- [x] Implement atomic immutable outputs and independent rehash audit.
- [x] Add a read-only imported serving audit test that writes only to a separate
  output directory.
- [x] Run lifecycle and CLI tests to GREEN.

### Task 5: Schemas, docs, migration, and self-check

- [x] Add RED static tests requiring every V2.5 schema/reference/CLI entry and
  V2.4.1 compatibility language.
- [x] Add schemas and concise on-demand references; update `SKILL.md`, agent
  prompt, and user readmes without claiming GPU validation.
- [x] Add RED self-check tests for success and missing/corrupt installed files.
- [x] Implement `self_check.py` and run focused tests to GREEN.

### Task 6: Verification and delivery

- [x] Run the complete CPU/static suite without `CUDA_SM120_E2E`.
- [x] Rerun any failure separately and report confirmed flaky behavior.
- [x] Run `quick_validate.py`, `self_check.py`, `compileall`, and
  `git diff --check`.
- [x] Review the diff for fail-open defaults, evidence fabrication, unsafe path
  handling, compatibility regressions, and promotional profiler language.
- [x] Commit on `codex/v2.5-evidence-automation` and push that branch only to
  GitHub and GitLab; do not create a PR or update either `main`.
