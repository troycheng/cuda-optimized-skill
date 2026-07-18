<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="asset/logo-wordmark-dark.svg">
    <img src="asset/logo-wordmark.svg" width="640" alt="CUDA Kernel Optimizer">
  </picture>
</p>

<p align="center"><strong>以证据驱动 Codex 优化 CUDA、CUTLASS 与 Triton</strong></p>

<p align="center">
  <a href="docs/getting-started.md">快速开始</a> ·
  <a href="docs/workflows.md">工作流</a> ·
  <a href="docs/evidence-and-safety.md">证据与安全</a> ·
  <a href="skills/cuda-kernel-optimizer/examples/walkthrough.md">示例</a> ·
  <a href="README.md">English</a>
</p>

## 项目简介

`cuda-kernel-optimizer` 是一个供 Codex 使用的可复用性能优化 skill。它可以优化单个
CUDA、CUTLASS 或 Triton kernel，诊断完整 GPU workload，验证 kernel 改动能否改善
serving 目标，也可以在不启动原程序的情况下分析已有 Nsight Compute report。

它把环境检查、profiling、限定范围的代码修改、正确性验证和成对性能测量组织成一个
可恢复的工作流。提供完整 workload 后，诊断范围不局限于 GPU kernel，还包括框架调度、
CPU 处理、数据传输、多卡通信、I/O 和运行环境。

Skill 只会修改已声明的项目路径和隔离项目环境，不会自动修改宿主机配置。驱动、权限、
频率、功耗限制和系统设置只提供建议。

## 快速开始

安装由 Codex 完成。让 Codex 从
[troycheng/cuda-optimized-skill](https://github.com/troycheng/cuda-optimized-skill)
仓库的 `skills/cuda-kernel-optimizer` 路径安装或更新 skill，然后开启新会话以重新加载
指令。

请提供可运行目标、正确性 reference、测试环境、性能目标、业务约束和允许修改的范围。
真实 workload 必须由用户提供；skill 不会自行下载或编造。

`quick` 最长 45 分钟；`balanced` 是默认的 3 小时预算；`thorough` 最长 10 小时，
用于更广的搜索和验证。

> 使用 cuda-kernel-optimizer 优化当前目录中的 Triton kernel。先确认 reference 和输入，保持宿主机设置不变，只有正确性与成对性能证据都通过时才保留改动。

输入清单和第一次任务边界见[快速开始](docs/getting-started.md)。

## 选择工作流

| 工作流 | 适用场景 | 结论边界 |
|---|---|---|
| **Kernel 优化** | 已有 CUDA、CUTLASS 或 Triton 实现及可比较 reference | 产出 kernel 级结论，包括正确性、编译器/profiler 证据、成对样本和置信度结果 |
| **完整 workload** | 延迟、吞吐或成本未达目标，但瓶颈未知 | 覆盖 kernel、框架、CPU、传输、通信、I/O 和环境的诊断，并进行限定范围的端到端评测 |
| **Serving 验证** | Kernel benchmark 已提升，需要验证产品 KPI | 冻结 c1/c2/c4/c8/c12 分层、serving-stack identity、逐层约束，并分别判定性能和证据完整性 |
| **已有 NCU report** | 已有 `.ncu-rep`，且不能重新运行被 profile 的 workload | 只读分析 report；导入结果不能证明当前 counter 权限或当前目标 identity |

[工作流说明](docs/workflows.md)列出每条路径需要的输入、允许修改的范围以及能够支持的结论。

## 工作方式

```mermaid
flowchart LR
    goal["目标、代码和约束"] --> environment["检查测试环境"]
    environment --> baseline["建立可复现 baseline"]
    baseline --> profiling["Profiling 并定位瓶颈"]
    profiling --> change["创建限定范围的修改"]
    change --> evaluation["检查正确性和成对性能"]
    evaluation --> keep["证据充分：保留修改"]
    evaluation --> restore["证据不足：恢复原实现"]
```

每轮优化先提出一个能被实测推翻的性能假设，再用真实候选给出正确性结果；正确性通过后，
还要给出可比较的性能结果。测量工具的修复有明确的时间和次数上限；超限后只切换到预先
验证的测量路径，没有可用路径就停止该方向。修工具不等于性能提升，也不会作为优化成果
汇报。具体规则见[性能优先的迭代约束](skills/cuda-kernel-optimizer/references/performance_iteration.md)。

工作流在正式计时前冻结目标和授权范围。每个候选方案都绑定 source、binary、输入、
schedule、raw rows 和运行时 identity。被拒绝或中断的尝试会留下记录，但不会覆盖之前
有效的结果。

## 以证据为准，而不是选择最快样本

性能结论只有在证据闭合后才能成立：

- 正确性和所有声明约束通过；
- 成对 A/B 样本使用冻结的 schedule 和 aggregation 规则；
- 默认 95% 置信区间支持相对与绝对提升门槛，并且有效 pair 数量足够；
- continuous shared-host guard 完整覆盖正式计时阶段，不存在 unknown、缺采样、过期或
  污染样本；
- 正式 serving run 覆盖全部 c1/c2/c4/c8/c12 分层，并把 timed binary 绑定到已证明的
  execution path。

正式证据出现不确定、必需字段缺失、identity 漂移或环境污染时必须 fail closed。冻结的
实验开始后，不能排除不利样本，也不能只重试一侧 role 来补救结果。

`performance_verdict` 与 `evidence_integrity` 分开判定：更快的数字不能补偿无效 attempt。
安装后的 `self_check` 只执行 CPU/static 检查，不验证 GPU 环境。Claim ladder 和宿主机
边界见[证据与安全](docs/evidence-and-safety.md)。

## 已测试范围

以下数字是历史验收证据，不代表任意项目都能获得相同提升。本次文档修改不会从
CPU/static 检查推断出新的 GPU 结果。

| 验证路径 | 已记录结果 | 含义 |
|---|---|---|
| CPU/static 验收 | 746 项测试：741 通过，5 项 RTX 5090 opt-in 测试跳过，0 失败 | 覆盖状态恢复、证据绑定、shared-host guard、超时、恢复和输入验证 |
| 物理 RTX 5090 路径 | 13/13 项检查耗时 34.302 秒；目标侧 NCU 返回 `ERR_NVGPUCTRPERM` | GPU 工作流已运行，且未修改权限或驱动策略 |
| 可复现 workload fixture | 端到端延迟提升 60.4616%，约束通过 | 只证明该 fixture 上的完整工作流 |
| 用户提供的 vLLM workload | Kernel 指标提升 26.3287%，真实 workload 变化 -0.0097% | 更快的 kernel 没有改善产品 workload，因此保留原实现 |
| 导入 NCU report | 未启动原程序并解析 140 项指标 | 不代表当前 counter 权限或当前 runtime identity 有效 |

详细版本和 opt-in 条件见[兼容性](docs/compatibility.md)。历史性能数字不是通用性能承诺。

## 文档

- [快速开始](docs/getting-started.md)
- [工作流选择](docs/workflows.md)
- [证据与安全](docs/evidence-and-safety.md)
- [兼容性](docs/compatibility.md)
- [AI 执行协议](skills/cuda-kernel-optimizer/SKILL.md)
- [Kernel 与 workload walkthrough](skills/cuda-kernel-optimizer/examples/walkthrough.md)
- [性能优先的迭代约束](skills/cuda-kernel-optimizer/references/performance_iteration.md)
- [V2.5 正式证据参考](skills/cuda-kernel-optimizer/references/evidence_automation.md)
- [Canonical 兼容性参考](skills/cuda-kernel-optimizer/references/compatibility.md)
- [RTX 5090 opt-in 测试说明](tests/gpu/sm120/README.md)
- [MIT License](LICENSE)

本项目独立于 CUDA、CUTLASS、Triton 和 Nsight Compute。请按照各依赖自身的许可证使用。
