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

## 能力层首轮技能 TDD 与外部质证

首个场景是 RTX 5090 / `sm_120` 上的 Triton decode attention / GQA。无
playbook 基线能守住“当前不能晋级”，但一次加载多份大文档，并在尚未证明 K/V
重复读取时直接提出四 Q head 合并候选；它还自行分配了合同中不存在的基础设施
时间比例。首张 playbook 因此先要求修复尾块正确性、确认 dispatch 与生成代码，
再把短上下文 launch 和长上下文 KV 机制分开。

同一设计交给 GLM、DeepSeek、Kimi 独立攻击，三方均给出 `REVISE`。采纳：

- 把宽松的单信号 OR 改成“任一完整信号组”；
- 删除检索输出中的 `candidate_admissible`，明确检索没有执行权；
- 将门禁拆成 `pre_execution` 和 `promotion`：前者包含 correctness reference、
  dispatch identity、target compile probe，后者包含 candidate correctness、paired
  measurement 和 workload replay；
- 对 playbook 正文复算精确 UTF-8 字节成本，声明值不一致时安装自检失败；具体
  模型的 token 消耗留给评测记录，不再用启发式估算充当硬预算。

随后独立代码审查发现清单校验后重复读取、符号链接逃逸、门禁词表未封闭、运行时
校验弱于 schema，以及 `--validate` 仍依赖查询参数。实现改为单次快照校验和路由，
清单、来源与 playbook 拒绝符号链接，查询同时返回 registry/source 哈希；阶段门禁
采用封闭且完整的最小集合，版本、风险、playbook 后缀和版本区间由运行时同步校验。
复审进一步复现了上层 `references` 软链和未来复核日期问题。最终读取改为从可信
skill 根开始，用目录文件描述符逐层 `O_NOFOLLOW` 打开；安装自检执行的能力查询
脚本也来自同一安全快照。历史 `as_of` 早于能力卡或任一来源复核日期时，知识状态
降级为 `unverified_future`，避免未来知识污染旧实验。

留到阶段 3：证据类别必须解析成带合同摘要、产物哈希、环境范围和新鲜度的真实
引用；compile probe、autotune 配置和 dispatch identity 属于本次运行的证据，不
固化在知识卡中。

未采用：随机探索槽、固定 20% 探索概率、按命中次数计算固定冷却轮数、预写短长
上下文阈值、知识卡级二进制哈希。这些建议要么破坏确定性和可归因性，要么把目标
环境的运行事实错误地变成通用知识。短长分界必须由冻结 workload 和校准结果决定。

## 门禁证据信任边界复审

阶段 3 的首版摘要先把六类门禁限制为封闭格式，并通过 reserved ledger event、产物
哈希和 Controller HMAC 阻断通用账本写入与私造摘要。独立复审随后指出：HMAC 只能
证明持钥者批准了某份 JSON；如果 JSON 中的 producer、PASS 和结果仍由 artifact
自报，它不能证明对应 adapter 真正运行过。这个问题不能靠增加 schema 字段解决。

已关闭的问题：摘要会在 Controller 当前时间重算新鲜度；门禁匹配绑定合同、环境、
账本尾、reference、target、workload、精确架构、candidate ID 与 candidate SHA；
错误密钥、篡改产物和 reserved event 注入均 fail closed。

随后实现改为由 Controller 捕获并运行 allowlist 中的自包含 adapter 哈希快照。实现
身份绑定入口、Python runtime 和固定隔离模式；快照在临时空目录以 `-I -S` 执行，
不能从可写 sealed 目录加载未声明 helper。adapter 输出不再包含 kind、producer、
status 和 recorded_at；Controller 校验所有 check、重算 PASS 并构造规范化证据。
密钥不进入 adapter 的 stdin、环境和 argv，输出与请求都有硬字节上限，执行期间入口
或 runtime 身份变化时不落账，退出后清理整个进程组。证明同时绑定 run、账本和
adapter 实现摘要。

复审还复现了两条落盘边界：ledger 已提交但返回异常时，旧清理会删掉被账本引用的
artifact；超大 request 会在进入 finally 前遗留隐藏快照。现在 append 异常会先重验
账本，精确记录已提交时恢复成功；所有 append 异常都保留不可变 final artifact，孤儿
留给后续按账本可达性安全回收，不能在并发路径同步删除；request 硬上限在创建快照
前检查，执行目录由统一生命周期清理。

第二轮复审继续复现了 acknowledgement loss：同一 `observation_id` 在提交后重试会
追加第二条事件，摘要随后因重复 ID 永久失败。修复后，Controller-owned 规范化请求
摘要进入 payload 和 HMAC；相同 ID、请求、实现、run 与账本身份直接复用旧记录，
任一项冲突都拒绝。账本在排他锁内再次检查 ID 唯一，封住并发竞态；同名字节 artifact
可用于恢复提交前崩溃留下的孤儿，不同字节则 fail closed。

一手资料复核采用当前 SLSA 1.2 的 provenance 与 build platform 文档。采用的原则是：
证明字段应由可信控制面生成或验证，用户步骤不能读取签名材料，消费者要同时核对
产物摘要和预期 builder 身份。这里借用的是控制面设计原则，不把 GPU 实验系统包装成
SLSA 认证。默认本地模式防止 Planner 越权与意外伪造；抵抗同账号主机攻击需要独立
账号、容器或密钥服务，作为高保证部署配置继续实现。

- [SLSA 1.2 Build Provenance](https://slsa.dev/spec/v1.2/build-provenance)
- [SLSA Build Levels](https://slsa.dev/spec/v1.2/levels)
- [Sigstore signing overview](https://docs.sigstore.dev/cosign/signing/overview/)

## 诊断观察与 Planner 登记边界

外部质证提出的“完整信号组、阶段门禁、确定性上下文预算、产物哈希与环境/时间绑定”
已经落到 Planner 运行时，而不是停在能力卡字段。Nsys 与 PyTorch Profiler 的 adapter
只能输出封闭 measurement；kind、producer、请求身份、环境和时间由 Controller 构造。
观察摘要只把当前目标、当前时间窗内的诊断证据交给能力查询，Planner 不能自行声明
`observed_signals` 或 `available_evidence`。

首版实现仍留下三个可复现绕过：旧 `run_control` 接口可凭两个形状正确的哈希直接
登记；登记 `now` 未绑定摘要 `as_of`，旧快照可在未来复用；诊断 schema 只表达信号和
producer 并集，安装自检看不出逐 kind 漂移。独立审查把三项都判为阻塞问题。

修复后，admission 绑定候选、摘要、能力查询、三项执行前门禁、证据年龄和登记时间，
并由 Controller HMAC。内存登记、持久化 run ledger 和 replay 都重新验证 admission；
parent run 含候选时，创建 child contract 也必须由 Controller key 验证旧账本。诊断
schema 使用 kind 条件约束，安装自检同时核对顶层词表、逐 kind producer/signal、
schema version 和 check 唯一性。默认本地模式的控制面闭环由此成立；同账号恶意进程
隔离仍属于高保证部署配置。

## 稳定性闭环与 RTX 5090 复核

稳定性校准首版有两个实质问题：审计周期只在在线状态中计数，重放账本不能恢复同一
约束；无有效配对或包含无效配对时，噪声基线仍可能被污染。独立审查将两项都复现为
阻塞问题。修复后，在线迁移与账本重放共同执行 `audit_every_candidates`，审计必须
绑定同一 green anchor、合同、源码和环境；无效 pair 完全排除，零有效 pair 的基线、
噪声和 MDE 均为空，并保持 yellow/red。第二次独立复审结果为 Critical 0、Important 0。

2026-07-19 在物理 RTX 5090 / `sm_120` 上使用不可变容器镜像
`sha256:a2d9d89bc4394eab3fadc62c6b5b3f739b6494c1f64c56f5ba5e6c008252a0e5`
执行新鲜 artifact lane，15/15 检查通过，用时 34.307 秒。V3 校准使用八组真实同
kernel 配对，得到 34.153% 噪声中位数、36.712% 置信上界和 40.193% 最小可测效应，
高于合同中的 0.5% 实用效应。Controller 正确停在 `CALIBRATING`，没有把不稳定环境
写成候选结果。NCU 返回 `ERR_NVGPUCTRPERM`；没有提升权限或修改驱动策略。

这个结果证明的是反偏航和降级路径在该环境按设计工作，不证明 3.0 已带来 workload
加速。后续效果评测仍需用户真实 workload，并按五臂矩阵比较 2.9、随机 Planner、
乱序能力库和完整 3.0。

## Triton 升级实践回灌

发布前同步本机 skill 时发现，另一条真实 Triton 升级任务留下了四项未进入仓库的
规则：软件栈升级必须做单变量对照；无效证据需要永久隔离；组合两个候选不能继承或
相加父候选收益；测量 runner 达到可封存、可复现后必须停止无关维护。这些规则与 3.0
的合同和证据模型一致，因此纳入 3.0.1。

原本的 `version_audit.py` 只做宽松 JSON 校验和普通文件写入，未直接复制。发布实现
改为严格字段、重复键和非有限数拒绝、no-follow 输入、原子 no-follow 输出，并要求
两边独立重建、自重复稳定和正确性先于计时。八项新测试覆盖混杂变量、跨栈 engine
复用、无效证据回流、非法 JSON、符号链接输出和计时顺序。
