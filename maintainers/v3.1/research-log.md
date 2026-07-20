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

### GLM：capability probe runner 敌意评审

提交内容只有 runner 的公开实现摘要，并明确禁止搜索、运行代码或调用工具。评审指出的有效
风险包括：版本查询与正式执行之间存在可执行文件替换窗口；只保留日志头部会丢失尾部错误；
普通文件校验仍需防范硬链接；进程组无法约束主动 `setsid` 逃逸的孙进程。

采纳并由故障注入测试复现：原实现确实会把自改的可执行文件、自改的 probe 脚本和硬链接
输出发布为 `ready`。runner 现已记录并复核执行前后的 executable 与 argv 文件身份，任何漂移
都降为 `failed`；staging 输出必须是当前 uid 创建的单链接普通文件；截断日志改为保留头尾并
显式记录丢弃字节数。版本查询沿用与正式 probe 相同的环境白名单，且不携带正式输出路径，
新增测试冻结这一约束。

不成立或已覆盖：评审担心版本查询继承完整父环境，但当前实现从一开始就使用白名单环境；
符号链接 staging 也已由 `follow_symlinks=False` 的元数据检查和 regular-leaf 读取拒绝。测试补充
的是此前未覆盖的硬链接，而不是放宽读取方式。

边界保留：进程组可以清理普通后代，不能充当恶意代码沙箱。需要执行不受信任的 probe 时，
由 Controller 要求容器/PID namespace 等外部隔离；runner 不自行创建 cgroup、不修改宿主机，
也不把 stdout/stderr 当成结构化 readiness 证据。并发上限和总体资源预算留到 Controller。

### Gemini：readiness gate 与隔离修复质证

提交的是公开 gate 摘要，不含代码、路径、账号、日志或 workload，并禁止搜索和工具调用。
评审最有价值的反例不是某个状态映射，而是修复与崩溃会改变证据的时间边界：隔离 pip 安装
改变环境后，如果仍用安装前的 identity 重跑，报告会把新能力绑定到旧工具链；如果只在最终
report 记录修复次数，安装过程中崩溃又会让 resume 重复消费授权和预算。

采纳并加入故障注入：安装成功后必须通过 identity provider 重新生成完整环境摘要；摘要未变
直接 blocked，摘要变化则让本轮全部结果失效并从 foundation 开头重跑。gate 在第一项 probe
前持久化绝对 `started_at`，并在调用 installer 前先持久化 `repairs_used`；中断后的 resume 用
墙上时间计算剩余预算，不能重置次数。每次 probe 使用独立 attempt，report 保存相对
`evidence_path`，旧证据不覆盖。report 与 completion marker 的篡改测试也已补充。

已有实现覆盖：`artifact_store` 对临时文件、目标文件和父目录执行 fsync 后再发布 marker；当前
预算本来就按 `now - started_at` 计算，而不是把各动作 duration 相加。

不直接采纳把 persistence mode、ECC 计数和 clock throttling 等波动状态全部塞进环境 identity。
它们是需要按时效复核的运行条件，不是稳定身份；无选择地纳入会让正常波动触发全量重跑。
同样不引入通用 probe DAG：capability probe 必须独立，不能依赖前一个进程留下的 CUDA context
或 IPC 状态；真实共享前置应封装进同一个原子 probe。也不对整个 site-packages 做 Merkle 扫描，
锁文件摘要负责安装输入，刷新后的工具链摘要与安装后真实 capability probe 负责可用性；恶意
环境完整性验证不扩入当前本地同用户信任边界。

### DeepSeek：Controller v2 准入与 resume 质证

提交公开的 v1/v2 兼容、readiness 状态迁移、TTL 复核与 marker 绑定摘要，不含代码、路径、
账号或 workload。有效反例集中在 Controller 与 gate 的交界：identity 输入若漏掉系统工具链，
resume 会错误复用旧证据；readiness 期间只检查声明的 mutation roots 还不够，workload adapter
和 manifest 也可能被安装脚本或 probe 改写；blocked readiness 的 resume 语义必须明确，不能
在原 run 上把用户操作后的环境当成同一证据 epoch。

采纳：Task 5 的 identity provider 明确覆盖实际工具路径、版本、摘要与隔离环境 distribution，
而不是只哈希 venv 目录；Controller 在 readiness 结束后重新计算声明 mutation roots 和 workload
source hash，任一漂移都在 baseline 前阻断。测试已覆盖 project 文件与 workload adapter 被
probe 改写、fresh resume 不新增 attempt、baseline 超过 TTL 后在 profiler 前刷新，以及 blocked
resume 原样返回 `readiness_action`、不在旧 run 重试。用户处理 host 后必须创建 child run。

不直接采纳：不监控整个 project_root 的 `.cache` 等非合同输出；其是否污染正式测量由 workload
合同和稳定性门禁处理，readiness 只冻结声明 mutation roots、workload source 和 contract。
report/marker 损坏继续 fail closed，不猜测是位翻转还是人为修改；恢复方式是保留损坏 run 供
审计并创建 child run，不在同一证据链静默重建。v1 不继承 v2 的 TTL 语义：CLI 只允许 v2 新建，
v1 仅保留 validate、内部兼容和历史 resume/replay。

## 3. 第一轮开发决定

- 先交付一个独立的 readiness vertical slice，不同时实现知识卡和行动规划器；
- capability probe 与 workload diagnosis probe 使用不同 schema、目录和预算；
- Phase 0 先运行 foundation probe，再运行真实 workload smoke；
- 第一版自动修复只允许合同授权、带哈希 requirements 的隔离环境 pip 安装；
- 所有宿主机问题输出 `user_action_required`，不执行 sudo、驱动、权限、频率或服务修改；
- Controller 在 baseline 前执行 readiness gate，required 未通过时不启动真实 workload；
- vertical slice 通过本地故障注入后，再到 RTX 5090 验证真实 Nsys、NCU、sanitizer、
  SASS 和 workload smoke。

本机用 Python 标准库创建 venv 后确认，`bin/python` 与 `bin/python3` 都是指向基础解释器的
leaf symlink。若一律拒绝会让合同声称支持 venv、实际只能使用复制解释器或 Conda。实现因此只
放开这一种受限形态：environment root 和父目录不跟随 symlink，记录 leaf target、realpath 与
目标 SHA-256，安装前后必须一致。requirements、合同、probe 输出和 report 继续拒绝 symlink。

本地发布自检现会在不访问 GPU 和网络的前提下，对照 runtime 检查 readiness contract、probe、
report 三份 schema 的版本、字段和枚举。确定性 fixture 还冻结了两条 Controller 边界：required
readiness 失败时绝不能加载 baseline evaluator；只有 diagnostic 能力降级时，允许进入 mock
baseline，并把 claim 保持在较低但有效的证据层。这能在远端 GPU 验证之前发现打包和准入顺序
回归。
