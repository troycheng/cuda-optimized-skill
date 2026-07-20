# CUDA Kernel Optimizer 3.1 Phase 0 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 `superpowers-zh:executing-plans`
> 逐任务实现本计划。每项行为变更先写失败测试，步骤使用复选框跟踪进度。

**目标：** 在真实 workload 初始化和正式 profile 之前，用独立、受预算约束的 capability
协议证明任务所需工具真实可用；必要能力缺失时自动修复已授权的隔离 Python 环境，或给出
宿主机操作建议，并由 Controller 阻止不具备证据条件的诊断运行。

**架构：** 新增封闭的 readiness contract、probe evidence 和 readiness report，和现有
workload diagnosis probe 分开。Controller v2 冻结 readiness contract，在 baseline 前按
`foundation -> workload` 顺序执行，只有 required 项通过才进入原有基线与诊断流程。
第一版自动修复只支持带哈希 requirements 的隔离环境 pip 安装；宿主机仍是
`recommend_only`。

**技术栈：** Python 3 标准库、严格 JSON、JSON Schema 2020-12、`unittest`、现有
`artifact_store.py`、Nsys/NCU/Compute Sanitizer/CUDA CLI、RTX 5090 `sm_120`。

---

## 0. 边界与起点

工作区：
`/Users/tcheng/Documents/Codex/2026-07-15-triton-skill/cuda-optimized-skill/.worktrees/codex-v3.1`

分支：`codex/v3.1-active-diagnosis`

基线：`557efad`，本地 `935` 项测试通过，`6` 项物理 GPU 测试跳过。

本计划只交付 Phase 0。执行路径、竞争假设、知识卡和行动规划器在 Phase 0 通过 5090 验证
后分别编写后续计划，避免一个提交同时改四个独立子系统。

## 1. 文件结构

### 新建

- `skills/cuda-kernel-optimizer/templates/readiness_contract.schema.json`：授权、必要性、控制范围、
  阶段、probe 和修复方式。
- `skills/cuda-kernel-optimizer/templates/readiness_probe.schema.json`：单项 capability 原始输出。
- `skills/cuda-kernel-optimizer/templates/readiness_report.schema.json`：Controller 准入报告。
- `skills/cuda-kernel-optimizer/scripts/readiness_contract.py`：严格读取、验证和摘要。
- `skills/cuda-kernel-optimizer/scripts/readiness_probe.py`：限时、限日志的独立 runner。
- `skills/cuda-kernel-optimizer/scripts/readiness_gate.py`：阶段、状态、预算和 claim ceiling。
- `skills/cuda-kernel-optimizer/scripts/readiness_install.py`：合同授权的隔离 pip 修复。
- `tests/test_readiness_contract.py`、`tests/test_readiness_probe.py`、
  `tests/test_readiness_gate.py`：CPU/static 单元测试。
- `tests/gpu/sm120/fixtures/readiness_smoke.py`：5090 最小真实 probe。

### 修改

- `skills/cuda-kernel-optimizer/scripts/check_env.py`：增加只读库存，不拥有准入权。
- `skills/cuda-kernel-optimizer/scripts/workload_controller.py`：control v2 和 baseline 前 readiness。
- `skills/cuda-kernel-optimizer/templates/workload_control.schema.json`：v2 字段。
- `skills/cuda-kernel-optimizer/scripts/self_check.py`：验证新协议的静态结构。
- 对应的 Controller、metadata、文档和 SM120 测试。

## 2. 固定协议

### 2.1 Readiness contract

```json
{
  "schema_version": "cuda-workload-optimizer/readiness-contract-v1",
  "requested_claim": "workload",
  "budget": {"max_seconds": 300, "max_repairs": 1},
  "requirements": [
    {
      "id": "gpu-execute",
      "necessity": "required",
      "control_scope": "host",
      "phase": "foundation",
      "kind": "gpu_execute",
      "max_age_seconds": 300,
      "probe": {
        "argv": ["python3", "tools/gpu_smoke.py"],
        "timeout_seconds": 30
      },
      "remediation": {
        "mode": "user_action",
        "message": "Make one target GPU visible to the isolated environment."
      }
    }
  ]
}
```

封闭枚举：

```python
NECESSITIES = {"required", "diagnostic", "optional"}
CONTROL_SCOPES = {"project", "isolated_environment", "host"}
PHASES = {"foundation", "workload"}
KINDS = {
    "target_compile", "gpu_execute", "nsys_trace", "ncu_counters",
    "sanitizer", "sass", "benchmark_noise", "workload_smoke", "rollback",
}
REMEDIATION_MODES = {"none", "user_action", "isolated_pip"}
```

`isolated_pip` 固定格式：

```json
{
  "mode": "isolated_pip",
  "authorization_id": "user-approved-env-20260720",
  "python": "/abs/env/bin/python",
  "requirements_file": "/abs/project/requirements-gpu.lock",
  "requirements_sha256": "64-lowercase-hex",
  "timeout_seconds": 180
}
```

`python` 必须位于 `environment_root`，requirements 必须位于 `project_root`。唯一允许构造的
安装命令是：

```text
<python> -I -m pip install --require-hashes -r <requirements_file>
```

`authorization_id` 是冻结合同中的审计标签，不是独立授权令牌；同一合同内不得重复。实际授权
边界由合同摘要、精确 requirements 摘要、隔离环境根和控制范围共同组成，修改任一项都产生
新合同身份。

### 2.2 Probe evidence

probe 通过 `CUDA_OPTIMIZER_READINESS_OUTPUT` 写入：

```json
{
  "schema_version": "cuda-workload-optimizer/readiness-probe-v1",
  "requirement_id": "gpu-execute",
  "status": "ready",
  "observations": {"sm_arch": "sm_120", "device_count": 1},
  "artifacts": []
}
```

`status` 只允许 `ready`、`degraded`、`unavailable`、`failed`。Runner 另存 execution
artifact，记录 argv digest、实际工具路径与版本、return code、timeout、duration、截断日志、
执行 uid、容器或隔离环境 identity、GPU identity、可见设备、权限状态和输出摘要。

### 2.3 Readiness report 与裁决

报告必须包含：schema、requested claim、总状态、`can_start_diagnosis`、claim ceiling、合同与
环境摘要、开始/结束时间、预算、逐项结果、状态计数和 next actions。逐项结果必须保存
`valid_until` 和 identity digest；`max_age_seconds` 由合同按能力冻结，不设置跨任务统一 TTL。
在依赖该能力的高成本动作前，证据过期时只重跑对应 probe；任一环境 identity 改变时，v1
保守地让全部 readiness 证据失效，不推断未显式声明的跨 requirement 依赖。

```python
def admission_status(necessity, probe_status, remediation_mode, repairs_left):
    if probe_status == "ready":
        return "ready"
    if remediation_mode == "isolated_pip" and repairs_left > 0:
        return "auto_fixable"
    if remediation_mode == "user_action":
        return "user_action_required"
    if necessity in {"diagnostic", "optional"}:
        return "degraded"
    return "blocked"
```

只有全部 `required` 最终为 `ready`，`can_start_diagnosis` 才为 `true`。diagnostic 或 optional
缺失可以进入降级路径，但报告必须列出无法支持的分析能力。

## 3. 实现任务

### 任务 1：冻结 readiness contract 和 schema

**文件：**

- 创建：`skills/cuda-kernel-optimizer/templates/readiness_contract.schema.json`
- 创建：`skills/cuda-kernel-optimizer/templates/readiness_probe.schema.json`
- 创建：`skills/cuda-kernel-optimizer/templates/readiness_report.schema.json`
- 创建：`skills/cuda-kernel-optimizer/scripts/readiness_contract.py`
- 测试：`tests/test_readiness_contract.py`

- [x] **步骤 1：编写失败测试**

覆盖有效合同 detached；重复键、未知字段、重复 id、非法枚举、非法 `max_age_seconds`、非有限
预算、空 argv、相对路径、host 使用 `isolated_pip`、越界 Python/requirements、错误摘要和
symlink 全部拒绝。

```python
def test_host_requirement_cannot_auto_install(self):
    value = valid_contract()
    value["requirements"][0]["control_scope"] = "host"
    value["requirements"][0]["remediation"] = valid_isolated_pip()
    with self.assertRaisesRegex(self.module.ValidationError, "host.*user_action"):
        self.module.validate_contract(value, project_root=PROJECT,
                                      environment_root=ENV)
```

- [x] **步骤 2：运行并确认失败**

```bash
python3 -m unittest tests.test_readiness_contract -v
```

预期：导入失败，因为 `readiness_contract.py` 尚不存在。

- [x] **步骤 3：实现固定接口**

```python
class ValidationError(ValueError):
    pass

def load_contract(path: str | os.PathLike) -> dict:
    """No-follow 读取严格 JSON object。"""

def validate_contract(value: Mapping[str, Any], *, project_root: Path,
                      environment_root: Path) -> dict:
    """返回 detached normalized contract；不联网、不执行 probe。"""

def contract_digest(value: Mapping[str, Any]) -> str:
    """对 validated canonical JSON 计算 SHA-256。"""
```

复用 `artifact_store.read_regular_bytes()`，不复制软链接和原子文件实现。

- [x] **步骤 4：验证并提交**

```bash
python3 -m unittest tests.test_readiness_contract -v
python3 skills/cuda-kernel-optimizer/scripts/self_check.py
git add skills/cuda-kernel-optimizer/templates/readiness_*.schema.json \
  skills/cuda-kernel-optimizer/scripts/readiness_contract.py \
  tests/test_readiness_contract.py
git commit -m "feat(v3.1): define readiness capability contracts"
```

### 任务 2：实现独立 capability probe runner

**文件：**

- 创建：`skills/cuda-kernel-optimizer/scripts/readiness_probe.py`
- 测试：`tests/test_readiness_probe.py`

- [x] **步骤 1：编写失败测试**

覆盖安全环境、create-once 输出、命令不存在、非零退出、超时、输出缺失、重复键、超过 1 MiB、
id 不符、symlink/hardlink、父目录替换、子进程残留、日志头尾截断、secret redact、执行文件和
argv 输入漂移，以及工具版本、uid、容器、GPU identity、可见设备和权限摘要完整记录。

```python
def test_timeout_kills_descendants_and_returns_unavailable(self):
    result = self.module.run_requirement(
        requirement("slow", [sys.executable, str(SPAWN_CHILD)]),
        run_dir=self.root, project_root=self.project,
        environment_identity_digest="a" * 64,
        deadline_epoch=time.time() + 0.2,
    )
    self.assertEqual(result["status"], "unavailable")
    self.assertFalse(pid_exists(read_child_pid(self.root)))
```

- [x] **步骤 2：运行并确认失败**

```bash
python3 -m unittest tests.test_readiness_probe -v
```

- [x] **步骤 3：实现固定接口**

```python
def validate_probe(value: Mapping[str, Any], expected_requirement_id: str) -> dict:
    """验证并 detached 返回 readiness-probe-v1。"""

def run_requirement(requirement: Mapping[str, Any], *, run_dir: Path,
                    project_root: Path, environment_identity_digest: str,
                    deadline_epoch: float) -> dict:
    """运行一个 capability probe，并发布 probe/execution artifact。"""
```

输出位于 `readiness/probes/<id>.json` 和 `<id>.execution.json`。completion marker 最后发布；
任何失败先删除旧 marker，不能复用上一次成功。

- [x] **步骤 4：验证并提交**

```bash
python3 -m unittest tests.test_readiness_probe -v
python3 -m unittest tests.test_workload_controller.ProbeRunnerTests -v
git add skills/cuda-kernel-optimizer/scripts/readiness_probe.py \
  tests/test_readiness_probe.py
git commit -m "feat(v3.1): run bounded capability probes"
```

### 任务 3：实现聚合门禁和隔离 pip 修复

**文件：**

- 创建：`skills/cuda-kernel-optimizer/scripts/readiness_gate.py`
- 创建：`skills/cuda-kernel-optimizer/scripts/readiness_install.py`
- 测试：`tests/test_readiness_gate.py`

- [x] **步骤 1：编写失败测试**

覆盖 foundation 先于 workload；required foundation 失败后 workload 不执行；diagnostic 可降级；
host 只输出 user action；预算单调且崩溃不重置；requirements hash/Python identity 漂移、安装失败、
安装后 identity 未刷新或 probe 仍失败均 blocked；有效且 identity 未变时 resume 不重复 probe 或
安装；证据过期时只重跑对应 probe；环境 identity 变化或修复成功时从 foundation 重跑全部所需
probe；report/marker 篡改拒绝；每个结果保留不覆盖旧证据的相对 `evidence_path`。

```python
def test_required_foundation_failure_skips_workload_phase(self):
    report = self.module.run_gate(
        contract=contract_with_foundation_and_workload(), control=control(),
        run_dir=self.run_dir,
        probe_runner=fake_probe({"gpu": "failed", "workload": "ready"}),
        installer=failing_if_called,
    )
    self.assertFalse(report["can_start_diagnosis"])
    self.assertEqual(called_probe_ids(), ["gpu"])
```

- [x] **步骤 2：运行并确认失败**

```bash
python3 -m unittest tests.test_readiness_gate -v
```

- [x] **步骤 3：实现 gate**

```python
def evaluate_result(requirement: Mapping[str, Any], probe: Mapping[str, Any],
                    repairs_left: int) -> dict:
    """按 2.3 的固定映射生成单项结果。"""

def run_gate(*, contract: Mapping[str, Any], control: Mapping[str, Any],
             run_dir: Path, probe_runner=run_requirement,
             installer=install_isolated_pip, identity_provider=None,
             now=time.time) -> dict:
    """按阶段执行、修复、重试并原子发布 readiness/report.json。"""
```

不使用模型评分或概率。排序固定为 phase 在前、原合同数组顺序在后。required foundation 未通过
立即停止。gate 在执行动作前持久化绝对开始时间，并在 installer 前先扣减和持久化修复次数。

- [x] **步骤 4：实现唯一自动修复**

```python
def install_isolated_pip(remediation: Mapping[str, Any], *, project_root: Path,
                         environment_root: Path, run_dir: Path,
                         deadline_epoch: float) -> dict:
    command = [remediation["python"], "-I", "-m", "pip", "install",
               "--require-hashes", "-r", remediation["requirements_file"]]
    # 先复核 containment 和 requirements SHA-256；使用安全环境、进程组超时和限长日志。
```

installer 不接受 shell、任意 argv 或 sudo。修复成功后必须通过 `identity_provider` 刷新完整环境
身份；身份未变化即 blocked，变化后废弃本轮已有结果并从 foundation 开头重跑。修复预算耗尽后
不再次安装。

- [x] **步骤 5：验证并提交**

```bash
python3 -m unittest tests.test_readiness_gate tests.test_readiness_probe -v
git add skills/cuda-kernel-optimizer/scripts/readiness_gate.py \
  skills/cuda-kernel-optimizer/scripts/readiness_install.py \
  tests/test_readiness_gate.py
git commit -m "feat(v3.1): gate diagnosis on verified readiness"
```

### 任务 4：接入 Controller baseline 之前

**文件：**

- 修改：`skills/cuda-kernel-optimizer/templates/workload_control.schema.json`
- 修改：`skills/cuda-kernel-optimizer/scripts/workload_controller.py`
- 测试：`tests/test_workload_controller.py`

- [x] **步骤 1：编写 v2 和状态迁移失败测试**

v2 固定增加：

```json
{
  "schema_version": "cuda-workload-optimizer/control-v2",
  "readiness_contract": "/abs/project/readiness.json"
}
```

覆盖 v2 缺 readiness、合同越界、初始化中 project/workload 漂移、required blocked 仍测 baseline、
resume 重复 readiness、报告篡改、marker 篡改，以及 baseline 或后续高成本 profiler 前错误接受
过期证据。blocked resume 保持 `readiness_action`，不在变化后的环境中原地重开。

```python
def test_blocked_readiness_never_measures_baseline(self):
    with mock.patch.object(self.controller, "_run_readiness_gate",
                           return_value=blocked_report()), \
         mock.patch.object(self.controller, "_load_evaluate_module") as evaluate:
        state = self.controller.start_run(v2_control(self.root), self.run_dir)
    self.assertEqual(state["next_action"], "readiness_action")
    evaluate.assert_not_called()
```

- [x] **步骤 2：运行并确认失败**

```bash
python3 -m unittest \
  tests.test_workload_controller.WorkloadControllerContractTests \
  tests.test_workload_controller.WorkloadRoundTests -v
```

- [x] **步骤 3：实现兼容边界**

```python
CONTROL_SCHEMA_V1 = "cuda-workload-optimizer/control-v1"
CONTROL_SCHEMA_V2 = "cuda-workload-optimizer/control-v2"
```

v1 保留 validate、内部兼容和历史 resume/replay；面向用户的 CLI 新建 3.1 run 只接受 v2。
不得原地改变 v1 字段含义。

- [x] **步骤 4：增加初始状态和迁移**

```python
{
    "status": "active", "stage": "readiness",
    "completed_stages": [], "next_action": "readiness",
    "readiness_contract_digest": digest,
    "readiness_report_digest": None,
}
```

允许 `readiness -> baseline` 或 `readiness -> readiness_action`。用户处理宿主机后创建 child run，
旧 run 不在环境变化后原地重开。

- [x] **步骤 5：验证并提交**

```bash
python3 -m unittest tests.test_workload_controller tests.test_state_schema -v
git add skills/cuda-kernel-optimizer/templates/workload_control.schema.json \
  skills/cuda-kernel-optimizer/scripts/workload_controller.py \
  tests/test_workload_controller.py
git commit -m "feat(v3.1): require readiness before workload baseline"
```

### 任务 5：扩展只读环境库存

**文件：**

- 修改：`skills/cuda-kernel-optimizer/scripts/check_env.py`
- 创建：`skills/cuda-kernel-optimizer/scripts/readiness_identity.py`
- 测试：`tests/test_check_env.py`

- [x] **步骤 1：编写失败测试**

覆盖 `nsys`、`compute-sanitizer`、`ptxas`、`cuobjdump`、`nvdisasm`、`cmake`、`ninja`、C/C++
编译器。工具存在但版本失败时仍 `available=true`、`usable=null`，不能标记 ready。环境 identity
必须绑定实际工具路径/realpath/版本/摘要、driver、隔离 Python 与已安装 distribution/RECORD
清单、uid、容器、GPU、可见设备和权限状态；隔离 pip 后重新生成 identity 时必须发生可解释的
摘要变化。标准 venv 的 Python leaf symlink 可以使用，但父目录、解析目标和执行前后身份必须
复核；requirements 仍拒绝 symlink。

```python
def test_inventory_never_claims_tool_capability(self):
    result = self.module._detect_tool("nsys", ["--version"])
    self.assertIn("available", result)
    self.assertNotIn("ready", result)
    self.assertNotIn("can_profile", result)
```

- [x] **步骤 2：运行并确认失败**

```bash
python3 -m unittest tests.test_check_env -v
```

- [x] **步骤 3：实现统一 detector**

```python
def _detect_tool(name: str, version_args: list[str], timeout: int = 10) -> dict:
    path = shutil.which(name)
    if path is None:
        return {"available": False, "path": None, "version": None,
                "version_query_returncode": None}
    rc, out, err = _run([path, *version_args], timeout=timeout)
    return {"available": True, "path": path,
            "version": _first_version_line(out, err),
            "version_query_returncode": rc}
```

NCU 的 `can_read_counters` 保持 `None`；真实状态只来自 readiness 或 `profile_ncu.py`。
`readiness_identity.py` 提供 gate 的 `identity_provider`；它只读库存，不运行 profile，不把 ECC
计数、瞬时 clock throttling 等波动指标混入稳定身份。

- [x] **步骤 4：验证并提交**

```bash
python3 -m unittest tests.test_check_env tests.test_profile_ncu -v
git add skills/cuda-kernel-optimizer/scripts/check_env.py \
  skills/cuda-kernel-optimizer/scripts/readiness_identity.py tests/test_check_env.py
git commit -m "feat(v3.1): inventory GPU diagnostic tools"
```

### 任务 6：接入 self-check 和本地 vertical slice

**文件：**

- 修改：`skills/cuda-kernel-optimizer/scripts/self_check.py`
- 修改：`tests/test_skill_metadata.py`
- 创建：`tests/test_readiness_vertical_slice.py`
- 创建：`tests/fixtures/readiness/emit_probe.py`
- 创建：`tests/fixtures/readiness/readiness-contract.json.in`
- 创建：`tests/fixtures/readiness/control-v2.json.in`

- [x] **步骤 1：编写失败测试**

验证 schema 可解析、CLI help 可运行、示例 contract 可验证、blocked fixture 不运行 baseline、
degraded fixture 可进入 baseline mock。

- [x] **步骤 2：接入 self-check**

```json
{
  "readiness_contract": "passed",
  "readiness_probe_schema": "passed",
  "readiness_report_schema": "passed",
  "gpu_environment_validated": false
}
```

self-check 保持 CPU/static，不探测本机 GPU。

- [x] **步骤 3：验证并提交**

```bash
python3 -m unittest \
  tests.test_readiness_contract tests.test_readiness_probe \
  tests.test_readiness_gate tests.test_workload_controller \
  tests.test_skill_metadata -v
python3 skills/cuda-kernel-optimizer/scripts/self_check.py
git add skills/cuda-kernel-optimizer/scripts/self_check.py \
  tests/test_skill_metadata.py tests/fixtures/readiness
git commit -m "test(v3.1): exercise readiness admission locally"
```

### 任务 7：RTX 5090 真实能力验证和反证

**文件：**

- 创建：`tests/gpu/sm120/fixtures/readiness_smoke.py`
- 修改：`tests/gpu/sm120/test_sm120_acceptance.py`
- 修改：`tests/gpu/sm120/remote/run_lane.sh`

- [x] **步骤 1：先写 GPU acceptance**

固定五条路径：target compile/GPU execute/SASS；Nsys report + stats；NCU 真实 target range；
Compute Sanitizer memcheck；foundation 之后的 workload correctness/KPI smoke。

```python
@unittest.skipUnless(os.environ.get("CUDA_SM120_E2E") == "1", "requires sm_120")
def test_readiness_gate_precedes_real_workload(self):
    report = run_remote_readiness_fixture()
    self.assertEqual(report["environment"]["sm_arch"], "sm_120")
    self.assertLess(report["events"].index("foundation-complete"),
                    report["events"].index("workload-smoke-start"))
```

- [x] **步骤 2：本机确认只跳过物理 GPU**

```bash
python3 -m unittest tests.gpu.sm120.test_sm120_acceptance -v
```

- [x] **步骤 3：5090 新鲜临时目录运行**

```bash
CUDA_SM120_E2E=1 bash tests/gpu/sm120/remote/run_lane.sh
```

不修改驱动、counter 权限、频率、功耗和服务。NCU 成功时记录 counter；无权限时精确记录
`ERR_NVGPUCTRPERM` 和 user action，不尝试 sudo。`--query-metrics` 不算通过。

- [x] **步骤 4：故障注入**

隐藏 `nsys`、让 NCU 返回权限错误、破坏 requirements hash、让 probe 超时、让 workload smoke
错误。每种情况必须得到预期 degraded/blocked，且 baseline 不被错误启动。

- [x] **步骤 5：冻结 3.0 对照**

同机器、workload、容器记录环境准备耗时、首次 baseline 时间、首个方向前 profile 轮次、工具
修复数和重复 probe 数。先保存分布，再冻结 3.1 门槛，不预设改善百分比。

- [x] **步骤 6：提交**

```bash
git add tests/gpu/sm120/fixtures/readiness_smoke.py \
  tests/gpu/sm120/test_sm120_acceptance.py tests/gpu/sm120/remote/run_lane.sh
git commit -m "test(v3.1): validate readiness on sm120"
```

### 任务 8：更新 AI 使用文档与 Release Notes 草案

**文件：**

- 修改：`skills/cuda-kernel-optimizer/references/environment_readiness.md`
- 修改：`docs/environment-readiness.md`
- 修改：`skills/cuda-kernel-optimizer/SKILL.md`
- 修改：`README.md`、`README.zh-CN.md`
- 测试：`tests/test_public_docs.py`、`tests/test_readme_sync.py`

- [ ] **步骤 1：先写文档行为测试**

中英文都必须说明：AI 自动执行 readiness；用户提供 workload 和授权；required 未通过不开始；
隔离 pip 是唯一自动修复；宿主机只建议；self-check 不代表 GPU ready；3.1 未发布时 Release Notes
标为 development。

- [ ] **步骤 2：运行并确认失败**

```bash
python3 -m unittest tests.test_public_docs tests.test_readme_sync -v
```

- [ ] **步骤 3：更新文档**

README 讲用户能获得什么和 AI 如何运行，不要求用户手工复制 readiness CLI。CLI、schema 和
状态细节放 `docs/environment-readiness.md`，skill reference 只保留代理决策规则。

- [ ] **步骤 4：全量验证并提交**

```bash
python3 -m unittest tests.test_public_docs tests.test_readme_sync -v
python3 -m unittest discover -s tests -v
python3 skills/cuda-kernel-optimizer/scripts/self_check.py
git diff --check
git add README.md README.zh-CN.md docs/environment-readiness.md \
  skills/cuda-kernel-optimizer/SKILL.md \
  skills/cuda-kernel-optimizer/references/environment_readiness.md \
  tests/test_public_docs.py tests/test_readme_sync.py
git commit -m "docs(v3.1): explain environment readiness workflow"
```

## 4. Phase 0 拒绝条件

出现任一项，不进入分析引擎开发：

- required 未通过时 Controller 仍启动 baseline 或正式 profile；
- `--query-metrics` 被记录为 NCU counter ready；
- readiness 复用 workload diagnosis artifact 或预算；
- 宿主机问题触发 sudo、驱动、权限、频率、功耗或服务修改；
- 安装命令不是合同固定的 hash-locked isolated pip 形式；
- 安装或 probe 超时后残留子进程，或恢复时重复消耗预算；
- 环境 identity 改变后仍使用旧 readiness；
- readiness 已超过合同有效期，或工具路径/版本、uid、容器、GPU identity、可见设备、权限状态
  已变化，却仍被 baseline 或依赖它的高成本 profiler 接受；
- Nsys/NCU/sanitizer 插桩耗时被当成正式 workload KPI；
- Phase 0 相比 3.0 没减少工具修复或重复采集，并增加首次有效方向总耗时；
- 5090 故障注入不能稳定区分 ready、degraded、user action 和 blocked。

## 5. 后续入口

Phase 0 通过后，下一份计划只实现结构化 execution path、带关系和 epoch 的竞争假设，以及
不使用虚假概率的下一证据选择。知识卡和方向实验 planner 作为第三个 vertical slice，分别
做 analysis-only、experiment-only 和完整闭环消融。
