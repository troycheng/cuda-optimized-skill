# CUDA Kernel Optimizer 3.1 研究与质证记录

状态：第一轮完成，2026-07-20

本文件只记录会影响 3.1 设计和实现的事实、外部反驳以及采纳决定。外部资料和模型
没有代码晋级权；本地测试、真实 workload 和 Controller 证据仍是最终依据。

## 1. 一手资料核对

### NVIDIA Nsight Compute 13.3

来源：

- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [Python Report Interface](https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html)
- [Nsight Compute Release Notes](https://docs.nvidia.com/nsight-compute/ReleaseNotes/)

结论：

- `--query-metrics` 只能证明 metric 元数据可查询，不能证明当前账号能读取硬件 counter；
- counter readiness 必须实际 profile 一个限定范围的目标 kernel；
- `.ncu-rep` 应优先通过 `ncu_report` 读取 range、action 和 metric，而不是重复运行 NCU；
- 2026.1 起 `ncu-report` 可以独立安装，但版本必须与报告格式一起记录，不能静默升级。

### NVIDIA Nsight Systems 2026.1+

来源：

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Nsight Systems 2026.1 Release Notes](https://docs.nvidia.com/nsight-systems/2026.1/ReleaseNotes/index.html)

结论：

- readiness 要验证 `.nsys-rep` 能生成并由 `nsys stats` 或 `nsys export` 解析；
- 新 exporter 的内部存储发生变化，不能依赖未公开的行顺序；
- Nsys timeline 与未插桩 workload 的 KPI 分开记录，不能用 trace 耗时替代正式基线。

### CUDA 13.3 编译与二进制工具

来源：

- [CUDA Toolkit 13.3 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [NVCC 13.3](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/)
- [CUDA Binary Utilities 13.3](https://docs.nvidia.com/cuda/archive/13.3.0/pdf/CUDA_Binary_Utilities.pdf)

结论：

- CUDA 组件独立版本化，不能用单个“CUDA 版本”代替 nvcc、runtime、driver 和工具版本；
- `sm_120` 是明确支持的 Blackwell 目标，target compile 必须生成对应 real architecture；
- `cuobjdump` 可处理 host binary 和 cubin，`nvdisasm` 只处理 cubin但提供更丰富的分析；
- SASS readiness 必须针对本次编译产物执行并校验非空输出。

### Compute Sanitizer 与 Triton

来源：

- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
- [Compute Sanitizer Release Notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html)
- [Triton debugging](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)

结论：

- memcheck、racecheck、initcheck、synccheck 解决不同问题，racecheck 和 synccheck 不能替代
  memcheck；
- sanitizer 可能显著减慢运行并改变资源使用，readiness 只能运行最小 kernel；
- Triton interpreter 有数据类型和间接寻址限制，不能作为 GPU 正确性或性能证据；
- sanitizer 不应在每个失败候选后无条件运行，仍由方法和风险选择 targeted tools。

### PyTorch Profiler 2.13

来源：[torch.profiler](https://docs.pytorch.org/docs/stable/profiler.html)

结论：

- Execution Trace Observer 可以生成图结构 workload 表示，适合作为执行路径输入；
- profiler schedule 的 wait、warmup、active 和 repeat 必须固定到合同；
- CUDA 可用但 CUPTI 不可用时可能退回 legacy 路径，因此“能看到 CUDA time”不等于完整
  trace 能力可用。

## 2. 外部 AI 匿名质证

### Gemini

提交内容只有公开设计摘要，没有代码、内部地址、账号、日志或真实 workload。主要反驳：

1. Phase 0 如果无条件运行全部工具，会先消耗大量优化预算；
2. 严格把所有问题都当成单变量会忽略耦合机制；
3. 知识卡可能带来维护债务和相近硬件误匹配；
4. 没有明确 host launch、Python 调度和 GPU kernel 的区分证据时仍会选错层级；
5. NCU 的高开销会改变队列、带宽和热状态，不能用插桩耗时证明端到端因果；
6. readiness 应使用独立的轻量 capability 协议，而不是复用真实 workload probe；
7. 机制修改后如果旧 profile 不失效，系统会在迁移后的瓶颈上循环。

采纳：1、4、5、6、7。设计增加任务裁剪、独立 readiness 协议、profiler 扰动记录、
假设 epoch 和瓶颈迁移失效规则。

部分采纳：2。单变量保留为因果问题边界，但允许预登记的原子耦合干预，随后必须做消融。

不直接采纳：质证建议使用未经校准的 Bayesian 分数和固定 `0.85/0.10` 阈值。3.1 第一版
不制造精确概率，先使用可重放的证据覆盖、反证能力、成本和风险排序；任何数值门槛都由
3.0 基线和校准数据冻结。

### Kimi

登录后使用 K2.6 标准模式提交同一份公开设计摘要，并明确禁止执行 Python、搜索、Agent 或
其他外部工具，只接收纯文本技术评审。有效反驳集中在：

1. readiness 证据如果不绑定工具链、容器、用户、GPU 可见性和权限状态，会在环境漂移后
   继续被误用；
2. 最小 capability probe 只能证明能力存在，不能代替真实 workload 的兼容性验证；
3. 知识卡不仅需要复核日期，还需要失效条件和冲突裁决；
4. “确认方向”不应要求排除所有想象中的原因，只需排除会实质改变行动或停止决定的竞争解释；
5. 诊断探针的代码不能自动晋级，但同一份 diff 可以重新登记为正式候选并重新通过全部门禁；
6. 固定标杆适合因果对比，还需要滚动留出案例检查对新 workload 的外部有效性。

采纳：1、2、3、4、5、6。设计增加 identity-bound readiness 失效规则、知识冲突裁决、
决策相关的反证边界、探针重新登记规则，以及固定标杆之外的滚动前向验证。

不直接采纳：未经本项目数据校准的 Bayesian 后验、固定小时数、固定百分比和统一置信阈值。
这些数字看似精确，但无法由当前证据支持；仍由 3.0 基线与校准数据冻结门槛。

对照现有实现后，Kimi 对“完全缺少漂移和知识时效机制”的判断只部分成立：3.0 已有
`conflicts`、`last_reviewed`、`max_review_age_days`、`DRIFTED` 和稳定性复核。3.1 的真实缺口是
readiness 证据自身的身份绑定与失效触发，以及冲突知识如何影响假设排序和结论上限。

### DeepSeek：Phase 0 合同实现质证

提交公开协议摘要，不含代码路径、账号、日志或 workload，并明确禁止搜索和工具调用。
有效反驳是：环境变化可能同时使多个 requirement 失效，v1 如果推断“只影响某一个 probe”
容易漏掉系统库、驱动和运行时的隐式依赖；`authorization_id` 如果被误解为独立授权令牌，
也会造成错误安全边界。

采纳：v1 在环境 identity 变化时让全部 readiness 证据失效；证据仅因时间过期时仍只重跑
对应 probe。`authorization_id` 明确为冻结合同内的唯一审计标签，真实授权绑定合同摘要、
requirements 摘要、隔离根和控制范围。

已由现有设计覆盖：canonical digest 包含 `schema_version`；空洞的 `/bin/true` 不会因为退出码
为零而 ready，Task 2 runner 还要求它发布匹配 requirement id 的严格 probe evidence。

留到 runner/gate：环境 identity 的精确输入、超时分类、valid-until 执行、修复后重试和预算
累计。未采纳把这些执行语义塞回静态合同的建议，以免合同变成脚本语言。

## 3. 第一轮开发决定

- 先交付一个独立的 readiness vertical slice，不同时实现知识卡和行动规划器；
- capability probe 与 workload diagnosis probe 使用不同 schema、目录和预算；
- Phase 0 先运行 foundation probe，再运行真实 workload smoke；
- 第一版自动修复只允许合同授权、带哈希 requirements 的隔离环境 pip 安装；
- 所有宿主机问题输出 `user_action_required`，不执行 sudo、驱动、权限、频率或服务修改；
- Controller 在 baseline 前执行 readiness gate，required 未通过时不启动真实 workload；
- vertical slice 通过本地故障注入后，再到 RTX 5090 验证真实 Nsys、NCU、sanitizer、
  SASS 和 workload smoke。
