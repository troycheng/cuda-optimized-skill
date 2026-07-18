# CUDA Skill V2.6 Performance-First Iteration Implementation Plan

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** Add a deterministic, cross-workflow iteration gate that keeps AI optimization rounds centered on real candidates and measured results.

**架构：** A standard-library validator reads one strict round record and a frozen prevalidated measurement-path registry, mechanically derives the work class and next action, and writes one create-once decision. Existing V2.5 correctness, benchmark, evidence, and promotion components remain unchanged.

**技术栈：** Python 3 standard library, JSON Schema documents, `unittest`, Markdown.

---

### 任务 1：锁定行为契约

**文件：**
- 创建：`tests/test_iteration_guard.py`
- 创建：`skills/cuda-kernel-optimizer/templates/performance_iteration.schema.json`
- 创建：`skills/cuda-kernel-optimizer/templates/measurement_path_registry.schema.json`

- [ ] **步骤 1：编写失败的核心分类测试。** Build strict fixtures for a frozen registry and round record. Assert a correctness failure and a correctness-pass plus timing result both derive `candidate_evaluated`, while only a bound `confirmed_win` derives `performance_gain`.
- [ ] **步骤 2：运行 RED。** Run `python3 -m unittest tests.test_iteration_guard -v`; expect import failure because `iteration_guard.py` does not exist.
- [ ] **步骤 3：添加严格 schema。** Close every object, require SHA-256 identities, finite positive budgets and minimum effect, safe relative mutation paths, and explicit nullable candidate/evidence fields.

### 任务 2：实现最小分类器和 CLI

**文件：**
- 创建：`skills/cuda-kernel-optimizer/scripts/iteration_guard.py`
- 修改：`tests/test_iteration_guard.py`

- [ ] **步骤 1：实现严格 JSON、registry 和 round validation。** Reject duplicate keys, unknown keys, booleans used as numbers, non-finite values, unsafe paths, missing registry entries, and registry digest drift.
- [ ] **步骤 2：实现身份绑定和分类。** Require candidate/baseline inequality and bind correctness/performance to the same baseline, candidate, environment, metric, direction, and measurement path.
- [ ] **步骤 3：实现预算与历史停止规则。** Calculate `min(1200, floor(round_seconds * 0.15))`, permit one repair, and treat two consecutive non-candidate rounds as a fallback-or-stop condition.
- [ ] **步骤 4：实现只读 CLI。** `check --record --registry [--history] --out` reads inputs, performs no subprocess execution, and create-once publishes strict JSON.
- [ ] **步骤 5：运行 GREEN。** Run `python3 -m unittest tests.test_iteration_guard -v`; expect all focused tests to pass.

### 任务 3：接入 skill 和安装自检

**文件：**
- 修改：`skills/cuda-kernel-optimizer/scripts/self_check.py`
- 修改：`tests/test_evidence_cli.py`
- 修改：`tests/test_skill_metadata.py`
- 修改：`skills/cuda-kernel-optimizer/SKILL.md`
- 创建：`skills/cuda-kernel-optimizer/references/performance_iteration.md`
- 修改：`skills/cuda-kernel-optimizer/templates/iteration_report.md`

- [ ] **步骤 1：先写 RED 元数据与 self-check 测试。** Require V2.6 title/routing, the guard command, both schemas, the reference, derived classes, default budget, forced stop, and a static installation check.
- [ ] **步骤 2：运行 RED。** Run the focused metadata and CLI tests; expect failures for missing V2.6 routing and install checks.
- [ ] **步骤 3：写最小协议文档。** Put the full record, registry, state rules, fallback boundary, examples, and reporting contract in the on-demand reference; keep `SKILL.md` concise.
- [ ] **步骤 4：更新 iteration report。** Lead with hypothesis, candidate delta, measured result, decision, and next performance action; infrastructure details appear only when they block measurement.
- [ ] **步骤 5：运行 GREEN。** Run the focused metadata, CLI, and guard tests.

### 任务 4：更新公共文档并验证发布面

**文件：**
- 修改：`README.md`
- 修改：`README.zh-CN.md`
- 修改：`docs/workflows.md`
- 修改：`tests/test_readme_sync.py`
- 修改：`tests/test_public_docs.py`

- [ ] **步骤 1：先写 RED 文档测试。** Require both README files to explain the performance-first loop in user-facing language and route technical details to the new reference.
- [ ] **步骤 2：更新英文和中文说明。** Explain that each round starts from a testable performance idea and ends with a real candidate result; tool repair has a hard budget and is never reported as a speedup.
- [ ] **步骤 3：运行 focused tests。** Run guard, metadata, self-check, README, and public-doc tests.

### 任务 5：审查、回归和发布

**文件：**
- 修改：仅限审查发现的 V2.6 文件

- [ ] **步骤 1：运行静态与完整回归。** Run `python3 skills/cuda-kernel-optimizer/scripts/self_check.py --skill-dir skills/cuda-kernel-optimizer`, `python3 -m unittest discover -s tests -v`, and `git diff --check`.
- [ ] **步骤 2：做独立代码审查和外部 AI 二次评审。** Review the public diff for bypasses, false progress, baseline drift, and accidental runner growth; fix every critical or important issue and rerun tests.
- [ ] **步骤 3：提交并合并。** Commit on `feat/v2-6-performance-first`, fast-forward or merge into `main`, then rerun the full suite on merged main.
- [ ] **步骤 4：双端发布并验证。** Push the same main SHA to GitHub `origin` and internal GitLab `internal`; never push `upstream`.
- [ ] **步骤 5：更新本地 skill。** Replace the installed skill from merged main, run installed `self_check`, and compare source and installed hashes.

