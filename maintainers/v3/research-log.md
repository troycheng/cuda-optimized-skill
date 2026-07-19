# 3.0 研究与质证记录

日期：2026-07-19

这份记录只保存影响设计的结论，不保存冗长对话。外部意见用于发现盲区，不能替代本地实验和代码审查。

## kernel-skills 对照

检查对象：[tensormux/kernel-skills `7b7337a`](https://github.com/tensormux/kernel-skills/tree/7b7337a123f8711aa8e3d0452351d8fd30dde4b7)，MIT License。

本地复核结果：35 项 skill，结构校验和构建通过。它最值得采用的是细粒度任务拆分、统一章节结构、独立元数据、检索与 bundle、逐项版本和 proof 目录。不能直接照搬的部分是：bundle 仍以全文拼接为主；硬件元数据较粗；proof 多为 Markdown 和图片，没有形成可重放 benchmark、原始样本和统计裁决；示例主要证明单模型单任务表现。

3.0 因此采用“元数据先筛选、正文按需加载”的能力层，同时增加精确架构、反向信号、来源生命周期、本地 workload 裁决和消融评测。

## 一手资料

- [Agent Skills specification](https://agentskills.io/specification)：确认 skill、scripts、references、assets 和渐进式加载的边界。
- [Nsight Compute Python Report Interface](https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html)：确认 `.ncu-rep` 可离线结构化读取。
- [Nsight Compute documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)：确认报告、规则、合并和分析能力。
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：确认跨 CPU、GPU、通信和运行时的时间线证据。
- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)：确认框架事件和 execution trace 能力。
- [vLLM bench serve](https://docs.vllm.ai/en/stable/cli/bench/serve/) 与 [TensorRT-LLM Performance Tuning Guide](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/index.html)：确认端到端推理 workload 的指标和调优入口。

## 外部 AI 两轮质证

参与方：GLM、DeepSeek、Kimi。第一轮独立评审，不互相看到答案；第二轮去掉来源标签后交叉质疑。两轮结论均为 `REVISE`，没有把多数意见当成自动通过。

三方共识：

- 长期循环必须由确定性 Python 状态机控制；
- workload、源码、输入、环境和目标必须绑定身份；
- 候选在执行前登记假设、预计指标、成本和 kill gate；
- 证据账本必须追加写、可校验、不能由 Planner 覆盖；
- baseline/champion 要定期重放，预算要有硬停止；
- 先估计当前噪声和最小可测效应，不能使用通用硬编码阈值；
- Planner 可以看到结构化失败结果以便学习，但不能修改合同、账本、预算和晋级门禁；
- 外部模型只提供候选和反例，本地确定性验证决定结果。

主要分歧：

- GLM 倾向先做纯 kernel MVP；DeepSeek 主张由完整 workload 定位瓶颈、局部优化后回归；Kimi 主张 3.0 就纳入有界离线 workload。最终采用后两者的交集，因为项目目标明确包含非 kernel 瓶颈。
- 第一轮有人建议严格屏蔽 Planner 的失败细节；第二轮认为这会阻止学习。最终采用权限分离：Planner 能读验证后的摘要和谱系，不能改规则和原始证据。
- 有意见给出固定噪声、重放周期或宿主机阈值。最终拒绝通用数值，改由合同、校准和目标环境决定；宿主机继续只读检测、建议修改。

## 独立代码与评测审查

第一轮独立审查发现：早期 PASS 仅依赖布尔值、mutation 根可被软链接绕过、截止后仍可能晋级、校准黄灯状态不闭合、账本原地写在进程死亡时会留下毒尾、同一 run 可混入新合同。以上问题均转成失败测试后修复；证据适配器接通前，`PASS` 现在无条件 fail-closed。

评测审查还指出，只有事件名称的七个场景无法证明 3.0 的增量。开发计划已改为五臂对照，绑定模型、prompt、skill、合同、环境、种子和重复编号，并要求事件由已校验账本和带哈希产物派生。这个矩阵尚在实现，未完成前不能声称 3.0 优于 2.9。

## 当前裁决

保留：确定性控制、完整 workload 的有界离线闭环、按需能力库、可选外部质证、五臂消融和故障注入长跑。

推迟：线上非平稳自适应、跨架构自动迁移、外部模型自动投票和未经授权的宿主机修改。
