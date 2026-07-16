# cuda-kernel-optimizer V2.2

[English](README.md) | **简体中文**

一个面向 Codex 的可信 CUDA、CUTLASS 与 Triton 优化 skill。V2.2 使用双环
工作流：内环通过成对测量证明 kernel 正确且更快；可选外环再用用户真实负载
证明收益能体现在业务 KPI 上。

这是 skill package，不是独立优化器。Agent 读取 `SKILL.md`；确定性脚本负责
冻结输入、约束预算、采集证据、执行晋级判定并保存可恢复产物。

## 在 Codex 中安装

从维护中的 fork `main` 安装：

```bash
python3 "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-installer/scripts/install-skill-from-github.py" \
  --repo troycheng/cuda-optimized-skill \
  --ref main \
  --path skills/cuda-kernel-optimizer
```

安装器不会覆盖已有 skill。升级时先备份或移走旧目录，重新安装后重启 Codex
会话。

```bash
cd skills/cuda-kernel-optimizer
```

## V2.2 的变化

- **双环证据**：kernel 微基准证据和真实负载 KPI 证据分别展示。
- **用户自有负载**：skill 不发现、不下载、也不编造“代表性负载”。端到端结论
  必须来自三种显式 workload 输入之一。
- **预算预设**：默认 `balanced`；setup 时冻结墙钟、分支、轮数、样本对、候选、
  case 和 sanitizer 上限。
- **成对判定**：随机 AB/BA block、telemetry gate、置信区间和最小实际收益取代
  “最快单样本晋级”。
- **唯一晋级权威**：只有 `decision.json` 能推进最佳候选；`inconclusive` 永不晋级。
- **持久证据**：冻结 manifest、checkpoint、编译 provenance、原始
  `paired_samples.jsonl` 和双层 summary 让结论可审计、可复算。

## 预算预设

用户没有选择时默认使用 `balanced`。

| 预设 | 最长秒数 | 分支数 | 最大轮数 | 最少 pairs | 最多 pairs | 外环候选 | 最多 cases | Sanitizer |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `quick` | 2700 | 4 | 2 | 20 | 50 | 1 | 3 | targeted |
| `balanced`（默认） | 10800 | 8 | 4 | 20 | 100 | 2 | 10 | targeted |
| `thorough` | 36000 | 16 | 8 | 30 | 200 | 3 | unlimited | full |

只有提供全部必填限制时才使用 `--budget custom`。调度器会预留收尾时间，在
deadline 前停止接纳新阶段并写入可恢复 checkpoint。

## 输入

始终需要：

1. baseline `.cu` 或 Triton `.py` kernel；
2. 暴露 `reference(**kwargs)` 的 Python reference；
3. JSON 格式的签名维度。

可以额外提供且只能选择一种真实负载形式：

- `--workload ./workload.py`：Python adapter，从
  `skills/cuda-kernel-optimizer/templates/workload.py` 开始填写；
- `--workload-cmd 'command ...' --objective ./objective.json`：不经 shell
  解析的命令，加一个显式 objective；
- `--workload-manifest ./workload.json`：包含 source、objective 和 cases 的
  严格 manifest。

Objective schema 位于
[`templates/objective.schema.json`](skills/cuda-kernel-optimizer/templates/objective.schema.json)，
用于定义一个主指标、方向、最小实际收益，以及每项约束允许的最大回退。

未提供 workload 时运行 kernel-only 模式。确认的 kernel 收益可以得到
`kernel_only_win`，但不能声称应用吞吐或时延改善。提供 workload 后，只有
kernel 与主 KPI 都是确认收益且所有约束通过，才能得到 `end_to_end_win`。

## 快速开始

Kernel-only：

```bash
python3 skills/cuda-kernel-optimizer/scripts/orchestrate.py setup \
  --baseline ./gemm.cu \
  --ref ./ref.py \
  --dims '{"M":4096,"N":4096,"K":4096}' \
  --budget balanced
```

使用 Python workload 的 full 模式：

```bash
python3 skills/cuda-kernel-optimizer/scripts/orchestrate.py setup \
  --baseline ./gemm.cu \
  --ref ./ref.py \
  --dims '{"M":4096,"N":4096,"K":4096}' \
  --budget balanced \
  --workload ./workload.py
```

Agent 生成本轮候选后关闭该轮：

```bash
python3 skills/cuda-kernel-optimizer/scripts/orchestrate.py close-iter \
  --run-dir ./run_YYYYMMDD_HHMMSS --iter 1
```

中断后校验冻结输入并查看下一个未完成阶段，不重放已完成工作：

```bash
python3 skills/cuda-kernel-optimizer/scripts/orchestrate.py resume --run-dir \
  ./run_YYYYMMDD_HHMMSS
```

Decision 阶段完成后生成最终总结：

```bash
python3 skills/cuda-kernel-optimizer/scripts/orchestrate.py finalize \
  --run-dir ./run_YYYYMMDD_HHMMSS
```

## 晋级规则

内环依次经过 reference 正确性、配置的 sanitizer、compiler/SASS 证据以及
随机成对 baseline/candidate 测量。只有有限置信区间越过最小实际收益阈值的
`confirmed_win` 才进入 shortlist。

外环只对用户提供的 workload 运行。它在冻结 cases 上采集成对
baseline/candidate observation，检查主指标和所有约束，然后输出终局判定。
loss、timeout、畸形证据、约束失败和 `inconclusive` 都保留当前 best。

## 产物目录

精确文件取决于 backend 和运行结果；缺失的可选证据不会伪装为成功。

```text
run_YYYYMMDD_HHMMSS/
├── manifest.json                   # 冻结输入、策略与 input_hash
├── state.json                      # 候选注册表与历史
├── checkpoint.json                 # 持久恢复边界
├── env.json                        # GPU 与工具链快照
├── workload/spec.json              # 冻结 workload 快照或 null
├── baseline/
│   ├── <baseline>
│   └── bench.json
├── iterv1/
│   ├── analysis.md
│   ├── methods.json
│   ├── branches/
│   │   └── <candidate>/
│   │       ├── kernel.{cu,py}
│   │       ├── bench.json
│   │       ├── compiler_evidence/manifest.json
│   │       └── paired_samples.jsonl
│   ├── sanitizer.json
│   ├── sanitizer/*.json
│   ├── sass_check.json
│   ├── workload/<candidate-hash>/paired_samples.jsonl
│   ├── decision.json               # 唯一晋级判定
│   └── *.ncu.log                   # 成功或降级的 profiler 日志
├── iterv2/ ...
└── summary.md                      # 分开的 kernel/workload 结论
```

原始 pair 文件包含冻结候选身份和 classifier 配置，因此可以重新计算置信结果。
`summary.md` 会链接证据，并明确 profiler、sanitizer、compiler 或 workload
覆盖是否降级。

## RTX 5090 验证与 NCU 权限

2026-07-16 已在物理 RTX 5090 上完成可选 SM120 矩阵；Triton、原生 CUDA 和
CUTLASS 的正确性与耗时产物均通过。

| Lane | CUDA 编译器 | Triton | CUTLASS | Nsight Compute | 结果 |
|---|---:|---:|---:|---:|---|
| 兼容环境 | 13.0.1 | 3.6.0 | 4.6.1 | 2025.3.1 | 3/3 后端通过 |
| 当前环境 | 13.3.73 | 3.7.1 | 4.6.1 | 2026.2.1 | 3/3 后端通过 |

两个环境中宿主机都对硬件 counter 返回 `ERR_NVGPUCTRPERM`。Skill 保留命令、
返回码和日志，记录 counter coverage 不可用，并继续使用其它证据；测试没有增加
特权、容器 capability 或修改驱动策略。counter 可用时，NCU 证据会补充判定，
但正确性与成对计时不依赖它。`ncu --query-metrics` 本身不能证明 counter 权限。

可选 GPU 测试见 [`tests/gpu/sm120/README.md`](tests/gpu/sm120/README.md)，
已验证版本和架构路由见
[`references/compatibility.md`](skills/cuda-kernel-optimizer/references/compatibility.md)。

## 运行环境

- 驱动正常且 `nvidia-smi` 可用的 CUDA GPU；
- Python 3.10+ 与 CUDA 版 `torch`，Triton kernel 另装 `triton`；
- CUDA/CUTLASS 使用 `nvcc`，SASS 证据使用 `cuobjdump`；
- 使用 CUTLASS 时，通过 `$CUTLASS_PATH` 或 `$CUTLASS_INCLUDE_DIR` 提供头文件；
- `ncu` 可选；缺少 counter 权限属于明确记录的降级模式。

通用 benchmark driver 已内置。Skill 不重新分发 CUDA、CUTLASS、Triton 或
Nsight Compute。

## 参考资料

- [正式流程](skills/cuda-kernel-optimizer/SKILL.md)
- [完整示例](skills/cuda-kernel-optimizer/examples/walkthrough.md)
- [兼容性](skills/cuda-kernel-optimizer/references/compatibility.md)
- [优化目录](skills/cuda-kernel-optimizer/references/optimization_catalog.md)
- [NCU 指标指南](skills/cuda-kernel-optimizer/references/ncu_metrics_guide.md)
- [Sanitizer 策略](skills/cuda-kernel-optimizer/references/sanitizer_policy.json)

## 许可证 / 说明

这个 skill 独立于 CUTLASS、Triton 和 Nsight Compute，也不重新分发它们；依赖
需要单独安装。
