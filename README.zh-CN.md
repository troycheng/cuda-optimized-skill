# cuda-kernel-optimizer v2.1

[English](README.md) | **简体中文**

一个兼容 Codex 的 skill，用于围绕 Python reference 对 CUDA / CUTLASS / Triton kernel 进行迭代优化。它组合正确性检查、稳健的耗时分布、可选的 `nsight-compute`（`ncu`）profiling、分支选择、消融与 SASS 验证。

这是一个 **skill package**，不是独立工具。Agent 会读取 `SKILL.md` 并驱动整个循环。`scripts/` 下的脚本负责确定性的部分（环境检测、profiling、benchmarking、state 管理）。

---

![alt text](asset/v2_ch_arch.png)

## 使用方法

```text
在 agent 中使用下面的 prompt：
@cuda-kernel-optimizer 使用这个 skill 对“你想优化的算子”进行优化，迭代次数为 N 次。
```

## 在 Codex 中安装

首次安装时，从维护中的 fork `main` 获取：

```bash
python "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-installer/scripts/install-skill-from-github.py" \
  --repo troycheng/cuda-optimized-skill \
  --ref main \
  --path skills/cuda-kernel-optimizer
```

安装器不会覆盖已有 skill。升级时应先备份或移走旧的
`cuda-kernel-optimizer`，安装后重新启动 Codex 会话以加载新版。

## V2.1 新增内容

V2.1 把循环从“试错–记录”升级为“试错–归因–验证–学习”。在 V1
基础上新增了五个机制，下文所有描述均反映 V2.1 行为：

- **Roofline 驱动的轴预算分配** — 取代 V1 固定的“每个 axis 选 1 个方法”，V2.1 每轮先计算 compute / memory / latency 三个间隙（Δc、Δm、Δl），然后按比例把总预算 3 个方法名额切给三个 axis（单轴上限 2）。只有三个 Δ 都有证据且全部小于 0.15 时，才输出 `near_peak: true` 并提前终止。
- **Branch-and-Select 分支探索** — 每次迭代基于同一组方法生成 K 个分支候选（默认 K=4），它们共享方法但在 tile size / pipeline stages / warp 数 / 实现变体上各不相同。最快且正确的分支被选为 champion，其余归入 `frontier` 归档。
- **Ablation 消融归因** — champion 选出后对每个方法做逐一消融（leave-one-out）。`attribution(m) = ms_without_m − ms_champion` 给出单个方法的因果贡献值，取代 V1 那种“三个打包一起判定”的粗粒度判断。
- **SASS 指令级验证** — 对 champion 调用 `cuobjdump --dump-sass`，并按签名表（`sass_signatures.json`）去 grep，确认每个声称的优化方法是否真的出现在编译后的机器码中。
- **噪声感知测量** — benchmark JSON 保留独立样本、median、nearest-rank p95、总体标准差和按 median 归一化的 CV。分支按 median 排名，并明确标注差异是否落在配置的噪声带内。

这四项共同把方法分类从“effective / ineffective”两个桶升级为三个桶：`effective_methods`（SASS ✓ 且归因超过噪声阈值）、`ineffective_methods`（SASS ✓ 但归因未超过阈值）、`implementation_failed_methods`（SASS ✗）。

## RTX 5090 验证

2026-07-16 已在物理 RTX 5090 上完成可选 SM120 矩阵，Triton、原生 CUDA
和 CUTLASS 的正确性与耗时产物均通过。

| Lane | CUDA 编译器 | Triton | CUTLASS | Nsight Compute | 结果 |
|---|---:|---:|---:|---:|---|
| 兼容环境 | 13.0.1 | 3.6.0 | 4.6.1 | 2025.3.1 | 3/3 后端通过 |
| 当前环境 | 13.3.73 | 3.7.1 | 4.6.1 | 2026.2.1 | 3/3 后端通过 |

宿主机在两个环境中都以 `ERR_NVGPUCTRPERM` 拒绝硬件 counter 访问。Skill
会正确记录 `can_read_counters: false` 并保留失败日志；测试没有增加特权
capability，也没有修改驱动策略。可选运行命令见
[`tests/gpu/sm120/README.md`](tests/gpu/sm120/README.md)。
版本目标与精确架构路由维护在
[`skills/cuda-kernel-optimizer/references/compatibility.md`](skills/cuda-kernel-optimizer/references/compatibility.md)。

另外还对 vLLM SM120 blockwise-FP8 `down_proj`
（`m=1,n=8704,k=5120`）做了隔离二进制 A/B：每个候选使用 5 个全新进程，
每个进程计时 200 次。两个候选都通过正确性，median 分别为 20.482 us 和
20.483 us，因此在 2% 噪声带内停止。采集到的源码头文件字节相同，但扩展
哈希不同，所以这里只记录为二进制证据，不声称重新验证了源码补丁。

## 你需要准备什么

在 Agent 运行的宿主机上：

- 一块可用的 CUDA GPU，并且驱动正常（`nvidia-smi` 可运行）
- `$PATH` 中有 `nvcc`（用于 CUDA / CUTLASS backend）
- 如果需要 profiler 指标，`$PATH` 中应有 `ncu`。没有 counter 权限时，skill 会记录具体失败，并继续使用正确性、耗时、源码与 SASS 证据。
- `$PATH` 中有 `cuobjdump`（CUDA toolkit 自带）— V2.1 的 SASS 验证步骤需要它
- Python 3.10+，安装了 `torch`（CUDA 版本）；如果要用 Triton backend，还需要 `triton`
- 对于 CUTLASS kernel：`$CUTLASS_PATH` 或 `$CUTLASS_INCLUDE_DIR` 需要指向同时包含 `cutlass/` 和 `cute/` 头文件的目录树

`benchmark.py`（通用算子 benchmark driver）已经内置在 `scripts/benchmark.py` 中，不需要单独安装。

### `ncu` 权限常见问题

在很多云环境和容器环境中，profiling counter 访问默认是关闭的。你会在 `env.json` 中看到 `can_read_counters: false`。不得自动修改宿主机策略或增加容器 capability；只有在运维人员明确授权后，才考虑以下方式：

- 以 root 身份运行宿主机，或者
- 在 `/etc/modprobe.d/nvidia.conf` 中加入 `options nvidia NVreg_RestrictProfilingToAdminUsers=0` 并重启，或者
- 对于 docker：使用 `--cap-add=SYS_ADMIN`（Nsight 文档推荐）

## 你需要提供的内容

1. **Baseline kernel 文件**：`gemm.cu`（CUDA/CUTLASS）或 `gemm.py`（Triton）
2. **Reference 文件**：`ref.py`，需要暴露 `reference(**kwargs)`，并可选提供 `atol` / `rtol`
3. **Dims**：该签名所需的标量参数（例如 `M=4096 N=4096 K=4096`）
4. **`benchmark.py` 路径**：已经内置在 `scripts/benchmark.py` 下；`orchestrate.py` 默认使用它。只有在你有自定义版本时才需要传 `--benchmark <path>`
5. 可选：迭代次数 `N`（默认 3）、每个 axis 的 `ncu_num` top-K（默认 5）、噪声阈值（默认 2%）、**每轮分支数 `K`（默认 4，通过 `--branches` 传入）**

## 你会得到什么

在 baseline 同级目录下，会生成一个 `run_YYYYMMDD_HHMMSS/` 目录，内容如下：

```text
run_YYYYMMDD_HHMMSS/
├── state.json                   # 全局状态，可跨会话重新读取
│                                #   V2.1 新增：branches、implementation_failed_methods、
│                                #           roofline_history、frontier
├── env.json                     # GPU / nvcc / ncu / CUTLASS 环境快照
├── baseline/
│   ├── <baseline>               # 原样复制的 baseline
│   └── bench.json               # 初始时延与正确性结果
├── iterv1/
│   ├── roofline.json            # Δc / Δm / Δl 以及每个 axis 的预算分配
│   ├── methods.json             # 预算下选出的方法（含 trigger_strength）
│   ├── analysis.md              # 证据、决策、验证方式与风险
│   ├── best_input.ncu-rep       # 目标 profiling 成功时存在
│   ├── branches/                # K 个分支候选（方法相同，超参不同）
│   │   ├── b0/kernel.{cu,py} + bench.json
│   │   ├── b1/…
│   │   └── …
│   ├── kernel.{cu,py}           # champion kernel（最快且正确的分支）
│   ├── kernel.ncu-rep           # champion profiling 成功时存在
│   ├── ncu_top.json             # 可用的每个 axis top-K 指标
│   ├── *.ncu.log                # 保留 profiling 成功或失败日志
│   ├── sass_check.json          # 每个方法的 SASS 签名验证结果
│   ├── ablations/               # leave-one-out 消融实验的产物
│   │   ├── no_<method_a>/kernel.{cu,py} + bench.json
│   │   └── …
│   ├── attribution.json         # 每个方法的因果贡献（ms）
│   └── bench.json
├── iterv2/ …
├── iterv3/ …
└── summary.md                   # 总体加速、时间线、瓶颈漂移与回顾总结
```

## 手动调用

你不需要手动驱动整个循环；如果想调试这个 skill 本身，可以使用这些命令：

```bash
cd skills/cuda-kernel-optimizer

# 0 + 0b + 1 + 2 + 3a-for-iter1
python scripts/orchestrate.py setup \
  --baseline   ./gemm.cu \
  --ref        ./ref.py \
  --iterations 3 \
  --ncu-num    5 \
  --branches   4 \
  --dims       '{"M":4096,"N":4096,"K":4096}'
  # --benchmark 默认使用 scripts/benchmark.py（已内置）

# --- （Agent 会写入 iterv1/kernel.cu + iterv1/methods.json + iterv1/analysis.md
#       以及 iterv1/branches/ 下的 K 个分支候选） ---

# 3d + 3f + 3a-for-iter2 for iter 1
# close-iter 内部会依次执行：分支选拔 → SASS 验证 → 消融归因 → state 更新
python scripts/orchestrate.py close-iter \
  --run-dir   run_20260418_143022 \
  --iter      1
  # --benchmark 默认使用 scripts/benchmark.py（已内置）

# （对 iter 2 和 iter 3 重复代码生成 + close-iter）

# 4
python scripts/orchestrate.py finalize --run-dir run_20260418_143022
```

每个脚本都可以单独调用（对任意脚本执行 `--help` 即可）；`orchestrate.py` 只是一个便捷封装。

## Skill 结构

```text
cuda-optimized-skill/
├── README.md
├── README.zh-CN.md
└── skills/cuda-kernel-optimizer/
    ├── SKILL.md                     # skill 入口
    ├── scripts/
    │   ├── benchmark.py             # 内置 benchmark driver
    │   ├── check_env.py             # GPU/工具链环境检测
    │   ├── preflight.py             # baseline 与 reference 契约校验
    │   ├── state.py                 # state.json 写入者
    │   ├── validate_methods.py      # 优先级合规校验
    │   ├── run_iteration.py         # benchmark 执行与采集
    │   ├── profile_ncu.py           # 目标范围 ncu profiling
    │   ├── roofline.py              # 有证据的 gap 与方法预算
    │   ├── branch_explore.py        # median/噪声感知分支选择
    │   ├── ablate.py                # leave-one-out 方法归因
    │   ├── sass_check.py            # 每个方法的 SASS 验证
    │   ├── summarize.py             # summary 与瓶颈漂移
    │   └── orchestrate.py           # setup/close-iter/finalize CLI
    ├── references/
    │   ├── compatibility.md         # 版本与精确架构路由
    │   ├── ncu_metrics_guide.md     # 瓶颈到优化方法映射
    │   ├── optimization_catalog.md  # 按优先级排序的优化目录
    │   ├── method_registry.json     # 机器可读方法注册表
    │   └── sass_signatures.json     # 预期 SASS 签名
    ├── templates/
    │   ├── iteration_report.md      # analysis.md 骨架
    │   └── methods.schema.json      # methods.json schema
    └── examples/walkthrough.md      # 带注释的示例流程
```

## Agent 如何使用它

当用户说“优化 `gemm.cu`”时，Agent 会：

1. 读取 `SKILL.md`
2. 调用 `orchestrate.py setup`（env check → preflight → init → seed baseline → 目标范围 profile 尝试）
3. 读取当前最佳 kernel，以及可用的 profiler 证据或具体失败原因
4. counter 可用时，运行 `roofline.py` 计算有证据的 Δc / Δm / Δl 与每个 axis 的方法预算；指标缺失时不能得出 `near_peak` 结论
5. 查阅 `references/optimization_catalog.md` 与 `references/ncu_metrics_guide.md`
6. 根据现有证据预算选择方法，并把决策记录写入 `iterv1/methods.json` 与 `iterv1/analysis.md`；没有 counter 时，只基于正确性、耗时、源码和 SASS 证据做结论
7. 将 **K 个分支候选** 写入 `iterv1/branches/b{0..K-1}/kernel.<ext>` — 方法相同，但 tile / stages / warps / 实现变体各不相同
8. 调用 `orchestrate.py close-iter --iter 1`，它内部会：
   - 运行 `branch_explore.py` → 编译 + benchmark 所有分支，选出最快且正确的那个作为 champion（复制到 `iterv1/kernel.<ext>`），其余归入 `frontier`
   - counter 可用时用 `ncu` profile champion；否则保留完整命令、返回码和失败日志
   - 运行 `sass_check.py` → `iterv1/sass_check.json`
   - 运行 `ablate.py` → `iterv1/attribution.json`
   - 更新 state：每个方法按 `SASS ✓/✗ × 归因值是否超过噪声` 进入 `effective_methods` / `ineffective_methods` / `implementation_failed_methods` 之一
9. 如果正确性失败（所有 K 个分支都失败）：检查 `bench.json.correctness` 与 `bench.stderr.txt`，重写 kernel，并重试（最多 3 次）
10. 如果成功：若更快则推进 `best_file`；本轮 roofline 结果追加进 `roofline_history`
11. 回到第 3 步，进入下一轮迭代
12. 调用 `orchestrate.py finalize`，并将回顾总结写入 `summary.md` — 其中包含来自 `roofline_history` 的瓶颈漂移表

完整示例请见 `examples/walkthrough.md`，正式流程请见 `SKILL.md`。

## 限制与真实注意事项

- **上限**：如果你的 reference 已经是 cuBLAS / cuDNN / cuBLASLt，那么要获得显著提升通常需要算法级改动（如 split-K、stream-K、fused epilogues、mixed precision），3 轮预算内不一定能完成。baseline 是手写实现时，通常更容易得到大幅加速。
- **噪声**：当 kernel 运行时间低于约 `50 μs` 时，launch overhead 会占主导。skill 默认的 2% 噪声阈值有所帮助，但如果 dims 很小，建议提高 `--repeat` 或直接增大维度。消融归因也使用同一阈值 —— 低于噪声的贡献会被归为 `ineffective_methods`。
- **Triton + `@triton.autotune`**：在 `ncu` 下做 autotuning 会很慢，甚至超时。建议在 profiling 前先固定为单一 config，或者设置 `--launch-count 1` 并提高 warmup。
- **ncu CSV 列名**：较旧版本的 `ncu`（< 2022.1）会输出 `"Metric Value"`，其大小写和单位格式可能不同；`profile_ncu.py` 做了兼容处理，但如果你看到全 0，先检查迭代目录下的 `.ncu.log` 文件。
- **分支成本**：当 K=4 且开启消融时，每轮迭代最多要编译 K + (方法数) 个 kernel。在干净环境下首次构建会比较慢；如果更看重墙钟时间，可适当降低 `--branches`。
- **SASS 签名是启发式的**：`sass_signatures.json` 只是按指令模式做 grep，并不做完整的语义等价判定。一个方法可能通过了 grep 但实现依然次优 —— 这正是归因机制要兜住的部分。
- **重试是有上限的**：单轮迭代最多允许 3 次正确性失败。超过后，skill 会记录这次尝试失败并继续往下，而不是无限循环。一个 kernel 如果 3 次都无法修正，通常意味着存在需要人工审查的概念性问题。

## 示例结果

https://tensara.org/problems  以 Tensara 平台上的 Batch Normalization 题目为例，本项目展示了从基础实现到优化版本的显著性能提升。提交到 A100-80GB 环境后，程序 4/4 测试全部通过，平均运行时间由 82.94 ms 大幅降低到 439.13 μs，整体吞吐从 2.52 GFLOPS 提升至 476.20 GFLOPS。需要说明的是，日常开发与调优主要在本地 RTX 3060 环境中完成，因此本地结果无法完全体现 A100 的性能上限；最终性能数据以上述平台实测结果为准。

![alt text](asset/Tensara_baseline.png)

![alt text](asset/Tensara_best.png)

## 许可证 / 说明

这个 skill 独立于 CUTLASS、Triton 和 Nsight Compute，也不重新分发它们。你需要自行安装这些依赖。

## Star History

<a href="https://www.star-history.com/?repos=KernelFlow-ops%2Fcuda-optimized-skill&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=KernelFlow-ops/cuda-optimized-skill&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=KernelFlow-ops/cuda-optimized-skill&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=KernelFlow-ops/cuda-optimized-skill&type=date&legend=top-left" />
 </picture>
</a>
