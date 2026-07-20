# V3.1 主动诊断本地验证

> 本页只记录合同和控制流程验证。这里没有用户 workload，也没有 GPU 性能收益结论。

## 验证范围

本地 fixture 覆盖五类执行形态：

| 场景 | 主要结构 | execution map 大小 | 结果 |
|---|---|---:|---|
| kernel hot path | 短 CPU launch + 长 GPU kernel | 2,069 B | 通过 |
| framework gap | framework 等待后进入 GPU kernel | 2,429 B | 通过 |
| transfer overlap | H2D copy 完全隐藏在 kernel 内 | 2,414 B | 通过 |
| unknown idle | 未解释 GPU idle + 未覆盖区间 | 2,352 B | 通过，强制 `unmodeled` |
| mixed | CPU、GPU 和 transfer 同时占据关键区间 | 2,432 B | 通过 |

大小按规范化紧凑 JSON 计算。64 KiB 是测试上限，不是建议把上下文填满。

## 采集与消融

- 一次全局扫描后即可进入 `propose_hypotheses`；恢复运行不会重复全局扫描。
- 正常路径在选择下一条证据前消耗 1 次全局扫描；选中的定向证据尚未在本地 fixture 上执行。
- 保留请求签名历史时，改名后的等价请求被拒绝，重复 profile 数为 0；移除历史后，同一请求会再次入选。
- 保留假设关系时，请求能明确区分 1 组排他假设；移除关系后，该值降为 0。
- 移除 execution map 时，假设登记直接失败，不会退化为只看 metric 的诊断。

## 时间重叠合同修正

最初的节点只有聚合耗时、lane 和调用边，无法区分完全隐藏、部分隐藏和串行 transfer。外部敌意评审确认了这个反例，但其建议的 `critical_path_contribution_us` 在多 lane 重叠下没有稳定的可加定义，评审推导本身也出现了守恒矛盾，因此没有采纳。

当前合同采用更窄的做法：

- `duration_us` 表示所有 occurrence 的活跃时间总和；
- 节点记录时间覆盖是否可用，以及最早开始和最晚结束；
- `overlaps` 边记录从原始 trace 得到的精确 `overlap_us`；
- 缺少时间覆盖时禁止构造 overlap 边，并把 map 降为 `inconclusive`；
- per-occurrence 时间戳仍留在原始 profiler artifact，不进入紧凑上下文。

这组字段用于判断时间是否被隐藏，不用于把各节点耗时相加成 workload wall time。

## 回归结果

- 主动诊断相关测试：100 项通过。
- 全量回归：1,044 项通过，8 项 GPU opt-in 测试跳过。
- 自检：PASS。

第一次全量回归中，旧的进程组超时测试因 0.2 秒启动窗口抖动失败；该测试隔离运行两次均通过，第二次全量回归通过。没有修改无关的 workload adapter 或其测试。
