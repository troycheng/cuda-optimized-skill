# Task 8 Quality Review Implementation Plan

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** Make setup, checkpoint resume, multi-round execution, hard deadlines, workload secrecy, and artifact durability evidence-driven and restart-safe.

**架构：** Keep `orchestrate.py` as the lifecycle state machine, but make baseline and expensive subprocesses go through a process-group runner backed by a monotonic execution budget. Persist every resumable contract as a strict run-root artifact; extend state with an idempotent decision recorder and checkpoints with an iteration number. Keep legacy CLI aliases explicit and isolated at parser normalization.

**技术栈：** Python 3 standard library, `unittest`, atomic JSON artifacts, POSIX process groups with portable fallbacks.

---

### 任务 1：Baseline producer truth

**文件：**
- 修改：`skills/cuda-kernel-optimizer/scripts/orchestrate.py`
- 测试：`tests/test_orchestrate.py`

- [ ] **步骤 1：编写失败测试**：setup mock chain must observe `run_iteration.py seed-baseline`; reject missing/symlink/malformed/incorrect/non-positive benchmark output and mismatched state metric; assert failed setup removes the run directory and never leaves a passed checkpoint.
- [ ] **步骤 2：运行 RED**：`python3 -m unittest tests.test_orchestrate.BudgetedParserTests -v`; expected failures show setup currently writes a passed baseline without producer evidence.
- [ ] **步骤 3：最小实现**：run the seed command through the budgeted process runner, validate `iterv0/bench.json` as a current-run regular file, require literal `correctness.passed is True`, finite positive `kernel.average_ms`, and matching `state.best_metric_ms`; bind path, sha256, metric, and correctness into baseline checkpoint evidence.
- [ ] **步骤 4：运行 GREEN**：repeat the RED command and require zero failures.

### 任务 2：Round-aware checkpoints and no-win state history

**文件：**
- 修改：`skills/cuda-kernel-optimizer/scripts/orchestrate.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/state.py`
- 测试：`tests/test_orchestrate.py`
- 测试：`tests/test_state_schema.py`

- [ ] **步骤 1：编写失败测试**：require checkpoint `iteration`; baseline uses zero; decision of round one resumes at candidate correctness round two; `close-iter --iter` mismatch fails; decision-to-next-round is the only wrap; no-win record is idempotently appended to state history; max round remains at decision until finalize.
- [ ] **步骤 2：运行 RED**：run the lifecycle and state test classes; expected failures identify missing iteration and terminal decision lock-in.
- [ ] **步骤 3：最小实现**：validate and persist iteration; return `next_iteration`; permit only decision(N) to candidate_correctness(N+1); add `state.py record-decision` with an atomic idempotency key; make close advance after decision only when another configured round exists.
- [ ] **步骤 4：运行 GREEN**：run the same lifecycle/state tests and require zero failures.

### 任务 3：Monotonic hard-deadline process groups

**文件：**
- 修改：`skills/cuda-kernel-optimizer/scripts/budget.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/orchestrate.py`
- 测试：`tests/test_budget.py`
- 测试：`tests/test_orchestrate.py`

- [ ] **步骤 1：编写失败测试**：wall-clock rollback cannot increase execution time; restored elapsed time reduces a new monotonic budget; TERM-ignoring descendants are killed and reaped; baseline and branch timeout persist structured inconclusive checkpoints and do not advance.
- [ ] **步骤 2：运行 RED**：run budget/orchestrate deadline tests; expected failures show wall-time anchoring and `subprocess.run` without group cleanup.
- [ ] **步骤 3：最小实现**：make `BudgetClock` monotonic/elapsed based; persist elapsed; add a `Popen(start_new_session=True)` runner with reserve-aware timeout, TERM, bounded grace, KILL, wait, and stream close; route seed, branch, profile, ablation, SASS, and state producer commands through it.
- [ ] **步骤 4：运行 GREEN**：repeat deadline tests and check descendant PID no longer exists.

### 任务 4：Full-mode exhaustion semantics and secret-safe workload handoff

**文件：**
- 修改：`skills/cuda-kernel-optimizer/scripts/orchestrate.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/state.py`
- 测试：`tests/test_orchestrate.py`

- [ ] **步骤 1：编写失败测试**：exhausted full workload produces authoritative inconclusive without invoking the decision engine; kernel evidence remains separate; workload snapshots containing `API_TOKEN` never appear in argv or logs; state init accepts only a safe run-root `--workload-file`.
- [ ] **步骤 2：运行 RED**：run outer-loop/setup tests; expected failures show `kernel_only_win` and inline JSON argv.
- [ ] **步骤 3：最小实现**：short-circuit full exhaustion to an inconclusive terminal payload; atomically write `workload/spec.json` mode 0600; pass only its path; redact sensitive option values and secret-pattern arguments from execution logs.
- [ ] **步骤 4：运行 GREEN**：repeat outer-loop/setup tests and require no secret bytes in captured command/log text.

### 任务 5：Durability, compatibility, and workload drift verification

**文件：**
- 修改：`skills/cuda-kernel-optimizer/scripts/artifact_store.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/workload_adapter.py`
- 修改：`skills/cuda-kernel-optimizer/scripts/orchestrate.py`
- 测试：`tests/test_artifact_store.py`
- 测试：`tests/test_workload_adapter.py`
- 测试：`tests/test_orchestrate.py`

- [ ] **步骤 1：编写失败测试**：atomic replace is followed by parent-directory fsync with portable fallback; setup accepts `--iterations` as `max_rounds`, rejects conflicts, and emits migration errors for `--env-out` and noise; resume detects workload source/hash drift without benchmarking.
- [ ] **步骤 2：运行 RED**：run artifact/workload/parser tests; expected failures identify missing dir fsync, aliases, and source verification.
- [ ] **步骤 3：最小实现**：fsync parent dir after replace; expose a verify-only workload helper; reconstruct frozen spec on resume and verify bytes/hash; normalize parser aliases and removed options.
- [ ] **步骤 4：运行 GREEN**：repeat the focused tests and require zero failures.

### 任务 6：Final verification and commit

**文件：**
- 验证：all files above; README changes remain reserved for Task 12.

- [ ] **步骤 1：指定验证**：run Task 8 plus budget, artifact, state, and workload tests.
- [ ] **步骤 2：全量验证**：run `python3 -m unittest discover -s tests -q`.
- [ ] **步骤 3：CLI/静态验证**：run seven help surfaces under normal Python and `python3 -S`, `py_compile`, and `git diff --check`.
- [ ] **步骤 4：提交**：create a new commit, do not amend and do not push; report RED/GREEN counts and SHA.
