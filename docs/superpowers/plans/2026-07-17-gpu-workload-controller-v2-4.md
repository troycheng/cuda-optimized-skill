# GPU Workload Controller v2.4 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a resumable workload-level GPU optimization controller that combines deterministic evidence, Codex-authored bounded changes, optional local third-party review, and the existing paired workload evaluator without permitting host mutation.

**Architecture:** Keep the existing kernel optimizer intact and add a thin linear controller around `workload_adapter` and `workload_evaluate`. Strict JSON contracts define control input, normalized probes, diagnoses, ChangeSets, and reviewer output. Pure validation and diagnosis functions stay separate from process execution so CPU tests cover policy and safety boundaries; the RTX 5090 lane verifies one real workload round.

**Tech Stack:** Python 3 standard library, JSON Schema documents, `unittest`, existing workload adapter/evaluator, existing artifact store, GitHub and GitLab dual-publish helper.

---

## Task 1: Add strict control and ChangeSet contracts

**Files:**
- Create: `skills/cuda-kernel-optimizer/templates/workload_control.schema.json`
- Create: `skills/cuda-kernel-optimizer/templates/change_set.schema.json`
- Create: `skills/cuda-kernel-optimizer/scripts/workload_controller.py`
- Create: `tests/test_workload_controller.py`

**Step 1: Write failing contract tests**

Add tests that import `workload_controller`, accept the smallest valid control manifest and ChangeSet, and reject:

- unknown keys;
- shell-string argv values;
- relative or escaping project roots;
- mutation roots outside `project_root`;
- `host` scope;
- ChangeSet paths outside the declared mutation roots;
- a missing candidate descriptor;
- invalid budgets, timeouts, and schema versions.

The valid fixture must use an absolute temporary project root and a real workload manifest path.

**Step 2: Verify RED**

Run: `python3 -m unittest -v tests.test_workload_controller`

Expected: FAIL because `workload_controller.py` does not exist.

**Step 3: Implement the minimum validators**

Implement:

- `ValidationError`;
- `load_json_object(path)` with duplicate-key rejection;
- `validate_control_manifest(value, source_path)`;
- `validate_change_set(value, control)`;
- canonical path containment checks that do not require candidate files to exist;
- CLI `validate --control <path> [--change-set <path>]` returning non-zero with a concise stderr error.

The Python validator is authoritative at runtime. The two JSON Schema files document the same closed contracts and use `additionalProperties: false` at every object layer.

**Step 4: Verify GREEN and contract synchronization**

Run: `python3 -m unittest -v tests.test_workload_controller`

Expected: PASS.

Run: `python3 -m unittest -v tests.test_skill_metadata tests.test_readme_sync`

Expected: existing metadata tests remain green.

**Step 5: Commit**

```bash
git add skills/cuda-kernel-optimizer/templates/workload_control.schema.json skills/cuda-kernel-optimizer/templates/change_set.schema.json skills/cuda-kernel-optimizer/scripts/workload_controller.py tests/test_workload_controller.py
git commit -m "feat: add workload controller contracts"
```

## Task 2: Add normalized probes and deterministic diagnosis

**Files:**
- Create: `skills/cuda-kernel-optimizer/references/workload_diagnosis_policy.json`
- Create: `skills/cuda-kernel-optimizer/scripts/workload_diagnosis.py`
- Create: `tests/test_workload_diagnosis.py`
- Modify: `skills/cuda-kernel-optimizer/scripts/workload_controller.py`
- Modify: `tests/test_workload_controller.py`

**Step 1: Write failing diagnosis tests**

Cover strict normalized probe validation and classifications for:

- kernel-heavy GPU work;
- framework launch gaps;
- CPU/data wait;
- host-to-device transfer;
- communication;
- storage I/O;
- environment/toolchain failure;
- close competing scores returning `mixed`;
- insufficient evidence returning `inconclusive`;
- rule provenance and policy digest in every diagnosis.

Add controller tests for probe command timeout, non-zero exit, oversized stdout, malformed JSON, secret redaction, and successful artifact capture. Use injected process runners where process behavior is the subject of the test.

**Step 2: Verify RED**

Run: `python3 -m unittest -v tests.test_workload_diagnosis tests.test_workload_controller`

Expected: FAIL because diagnosis APIs and probe execution are absent.

**Step 3: Implement the minimum evidence path**

Implement:

- versioned policy loading with SHA-256 digest;
- closed metric names and finite `[0, 100]` percentage validation;
- a pure `diagnose(probes, policy)` classifier with explicit matched rules and coverage;
- `run_probe` using argv only, process-group timeout, bounded output, minimal environment, JSON stdout, and redacted error artifacts;
- built-in `environment` probe based on the existing `check_env` output when requested;
- controller CLI `probe --control <path> --run-dir <path>` and `diagnose --run-dir <path>`.

External tools such as Nsight Systems remain user-supplied probe argv wrappers in v2.4; the controller only consumes their normalized JSON.

**Step 4: Verify GREEN**

Run: `python3 -m unittest -v tests.test_workload_diagnosis tests.test_workload_controller`

Expected: PASS.

**Step 5: Commit**

```bash
git add skills/cuda-kernel-optimizer/references/workload_diagnosis_policy.json skills/cuda-kernel-optimizer/scripts/workload_diagnosis.py skills/cuda-kernel-optimizer/scripts/workload_controller.py tests/test_workload_diagnosis.py tests/test_workload_controller.py
git commit -m "feat: diagnose workload bottlenecks from normalized probes"
```

## Task 3: Add the advisory local reviewer protocol

**Files:**
- Create: `skills/cuda-kernel-optimizer/scripts/workload_reviewer.py`
- Create: `tests/test_workload_reviewer.py`
- Modify: `skills/cuda-kernel-optimizer/scripts/workload_controller.py`
- Modify: `tests/test_workload_controller.py`

**Step 1: Write failing reviewer tests**

Test that the adapter:

- sends canonical JSON on stdin and binds a SHA-256 request digest;
- accepts only `support`, `challenge`, or `insufficient` with the same digest;
- rejects unknown keys, command callbacks, mismatched digests, malformed JSON, duplicate keys, and oversized output;
- uses argv only, an empty temporary cwd, a minimal environment, and a deadline;
- redacts secrets from stored stderr;
- returns a degraded `unavailable` artifact on timeout or missing CLI without blocking evaluation.

**Step 2: Verify RED**

Run: `python3 -m unittest -v tests.test_workload_reviewer tests.test_workload_controller`

Expected: FAIL because reviewer APIs do not exist.

**Step 3: Implement the minimum reviewer adapter**

Implement canonical request creation, digest verification, strict response validation, bounded subprocess execution, redaction, and an explicit advisory result type. Do not expose filesystem paths, execution callbacks, promotion handles, or network/model SDK integration to the reviewer protocol.

Wire `review --control <path> --run-dir <path> --change-set <path>` into the controller. A missing reviewer configuration writes a `skipped` artifact and exits successfully.

**Step 4: Verify GREEN**

Run: `python3 -m unittest -v tests.test_workload_reviewer tests.test_workload_controller`

Expected: PASS.

**Step 5: Commit**

```bash
git add skills/cuda-kernel-optimizer/scripts/workload_reviewer.py skills/cuda-kernel-optimizer/scripts/workload_controller.py tests/test_workload_reviewer.py tests/test_workload_controller.py
git commit -m "feat: add advisory local model review"
```

## Task 4: Build the resumable optimization round

**Files:**
- Modify: `skills/cuda-kernel-optimizer/scripts/workload_controller.py`
- Modify: `tests/test_workload_controller.py`

**Step 1: Write failing round and safety tests**

Cover:

- run directory initialization with canonical manifest copies and digests;
- stage transitions `baseline -> probes -> diagnosis -> change -> review -> evaluation -> decision`;
- checkpoint resume without repeating completed workload pairs;
- baseline and candidate execution through existing `normalize_workload` and `evaluate_pairs`;
- correctness failure, constraint rejection, practical-effect loss, and successful promotion;
- identity drift rejection for manifest, candidate, and project files;
- actual Git diff outside ChangeSet paths;
- rollback restoration after rejection;
- hard deadline and round budget enforcement;
- host recommendations being recorded but never executed;
- JSON summary output suitable for a calling Codex session.

Use a tiny temporary Python workload fixture so integration tests execute real subprocesses and the real paired evaluator.

**Step 2: Verify RED**

Run: `python3 -m unittest -v tests.test_workload_controller`

Expected: FAIL on missing round orchestration behavior.

**Step 3: Implement the minimum controller state machine**

Add:

- `init_run`, atomic state/checkpoint writes, and artifact hashes;
- frozen project identity for declared mutation roots;
- baseline capture via existing workload APIs;
- validated ChangeSet registration before Codex edits;
- before/after diff verification limited to declared paths;
- correctness commands using argv only;
- paired evaluation and deterministic promotion decision;
- rollback from a frozen snapshot for rejected changes;
- idempotent `run`, `status`, `register-change`, `evaluate`, and `resume` CLI subcommands.

The controller never authors or applies a patch. Codex edits only the validated roots between `register-change` and `evaluate`, then the controller verifies the resulting diff.

**Step 4: Verify GREEN and regression safety**

Run: `python3 -m unittest -v tests.test_workload_controller tests.test_workload_adapter tests.test_workload_evaluate tests.test_artifact_store`

Expected: PASS.

Run: `git diff --check`

Expected: no output.

**Step 5: Commit**

```bash
git add skills/cuda-kernel-optimizer/scripts/workload_controller.py tests/test_workload_controller.py
git commit -m "feat: orchestrate resumable workload optimization rounds"
```

## Task 5: Update the skill, examples, and bilingual project documentation

**Files:**
- Modify: `skills/cuda-kernel-optimizer/SKILL.md`
- Create: `skills/cuda-kernel-optimizer/examples/workload-controller.md`
- Modify: `README.md`
- Modify: `README.zh-CN.md`
- Modify: `skills/cuda-kernel-optimizer/agents/openai.yaml`
- Modify: `tests/test_skill_metadata.py`
- Modify: `tests/test_readme_sync.py`

**Step 1: Write failing documentation contract tests**

Extend tests to require both READMEs and skill metadata to expose the same v2.4 capabilities and safety boundaries:

- workload-level diagnosis beyond kernels;
- user-provided runnable workload and optional normalized probes;
- Codex as the primary optimizer;
- optional local third-party JSON reviewer;
- project/container/isolation mutations allowed;
- host tuning is advice-only;
- reviewer is advisory and not an OS sandbox;
- fixed v2.4 controller stages and resume behavior.

Require every documented command and referenced file to exist. Preserve the existing bilingual semantic-sync checks without requiring literal translation.

**Step 2: Verify RED**

Run: `python3 -m unittest -v tests.test_skill_metadata tests.test_readme_sync`

Expected: FAIL because v2.4 documentation is not present.

**Step 3: Write and tighten the skill documentation**

Update the skill workflow only after the controller behavior is real. Keep `SKILL.md` operational and concise; move the complete runnable example and JSON samples to `examples/workload-controller.md`.

Rewrite the relevant README sections in natural English and Chinese rather than line-by-line translation. Use one Mermaid diagram for the controller and one compact boundary table. State limitations directly, including external probe normalization and reviewer process permissions.

Update `skills/cuda-kernel-optimizer/agents/openai.yaml` to route workload-level GPU optimization requests to the new controller while preserving kernel-only triggers.

**Step 4: Verify GREEN**

Run: `python3 -m unittest -v tests.test_skill_metadata tests.test_readme_sync`

Expected: PASS.

Run: `python3 skills/cuda-kernel-optimizer/scripts/workload_controller.py --help`

Expected: lists the documented subcommands and exits zero.

**Step 5: Commit**

```bash
git add skills/cuda-kernel-optimizer/SKILL.md skills/cuda-kernel-optimizer/examples/workload-controller.md README.md README.zh-CN.md skills/cuda-kernel-optimizer/agents/openai.yaml tests/test_skill_metadata.py tests/test_readme_sync.py
git commit -m "docs: explain the v2.4 workload optimization workflow"
```

## Task 6: Add and run RTX 5090 workload acceptance

**Files:**
- Modify: `tests/gpu/sm120/fixtures/workload_smoke.py`
- Modify: `tests/gpu/sm120/test_sm120_acceptance.py`
- Modify: `tests/gpu/sm120/remote/run_lane.sh`
- Modify: `tests/gpu/sm120/README.md`

**Step 1: Write the acceptance expectation before changing the fixture**

Add a test that requires the remote lane to produce and validate:

- a real baseline workload observation;
- at least one normalized probe and deterministic diagnosis;
- one bounded ChangeSet candidate;
- paired candidate evaluation;
- reviewer `skipped` or valid advisory output;
- a deterministic decision and resumable final state;
- no host mutation command.

**Step 2: Verify the local RED signal**

Run: `python3 -m unittest -v tests.gpu.sm120.test_sm120_acceptance`

Expected: local environment skips GPU execution, while fixture/static assertions fail until the lane invokes the controller.

**Step 3: Implement the minimum remote fixture**

Extend the existing smoke workload and remote lane with a deterministic project-scoped candidate. Keep the runtime short and use the target's existing CUDA/Triton environment. Do not install or change driver, persistence mode, clocks, power limits, profiling permissions, or host packages; record any host-level opportunity as a recommendation.

**Step 4: Run RTX 5090 validation**

Use the repository's established SSH alias and remote lane command from `tests/gpu/sm120/README.md`. Transfer only the committed worktree content or a clean archive, run the lane, and retrieve the artifact summary.

Expected: all acceptance checks pass. `ERR_NVGPUCTRPERM`, if still present, is recorded as reduced profiler coverage and does not trigger a privileged change.

**Step 5: Commit**

```bash
git add tests/gpu/sm120/fixtures/workload_smoke.py tests/gpu/sm120/test_sm120_acceptance.py tests/gpu/sm120/remote/run_lane.sh tests/gpu/sm120/README.md
git commit -m "test: cover workload controller on RTX 5090"
```

## Task 7: Full verification, branch integration, and dual publish

**Files:**
- Modify only files required by failures found during verification.

**Step 1: Run the complete CPU suite**

Run: `python3 -m unittest discover -v`

Expected: every non-GPU test passes; only explicitly environment-gated GPU tests skip.

**Step 2: Run repository checks**

Run: `git diff --check`

Expected: no output.

Run: `git status --short`

Expected: clean after committing any verification fixes.

Run the dual-publish helper in dry-run mode against `main` and both configured remotes. Expected: it refuses upstream and reports matching intended GitHub/GitLab targets.

**Step 3: Review the implementation against the design**

Use `superpowers:requesting-code-review` for a focused local review of contracts, subprocess boundaries, rollback, resume, documentation truthfulness, and compatibility. Fix validated findings with failing regression tests first.

**Step 4: Integrate the branch**

Use `superpowers:finishing-a-development-branch`. Re-run the full suite against the latest local `main`, merge `agent/v2-4-workload-controller` into `main`, and re-run the complete CPU suite after merge.

**Step 5: Publish only the user's two repositories**

Create annotated tag `v2.4.0` only after merged-main verification. Run the controlled dual-publish helper with `--execute` so GitHub `origin` and internal GitLab receive the same `main` commit and tag. Confirm by fetching both remotes and comparing object IDs.

Never push to the original upstream repository.
