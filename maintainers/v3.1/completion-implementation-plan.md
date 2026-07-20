# V3.1 完成实施计划

状态：代码完成，等待最终发布验证
基线：`1e4ce2d`
目标：把主动诊断从“合同原型”补成可执行、可恢复、可验证的闭环

## 完成标准

V3.1 只有同时满足以下条件才算代码完成：

1. Controller 能执行已选证据动作，登记结果并回到下一轮假设更新；
2. 证据动作只能使用冻结合同中的用户自有 adapter，能力必须来自当前 readiness 报告；
3. 已执行动作和等价请求会持久化，`resume` 不会重复消耗 profile 预算；
4. 互斥假设不能同时被判为方向成立，有缺口或冲突时不能进入候选修改；
5. public schema、CLI、SKILL、README 与真实实现一致；
6. 单元、集成、重放与 GPU smoke 验证都有可审计产物。

真实 workload 的收益仍由用户提供的 workload 决定。合成或示例 workload 只验证机制，
不能替代真实业务结论。

## 实施顺序

### 1. 先固定失败用例

- [x] `collect_evidence` 能从 selection 执行到新一轮 `propose_hypotheses`；
- [x] 中断后 `resume` 不重复执行已完成动作；
- [x] readiness 未证明的 capability 不能被分析合同伪造为可用；
- [x] 两个互斥假设不能同时 `direction_supported`；
- [x] 有反对证据或仍缺关键证据时不能 `direction_supported`；
- [x] 等价请求历史跨轮次生效，全局扫描不能被重复采集。

### 2. 实现证据执行闭环

- [x] 在分析合同中冻结 action adapter、argv、摘要、超时和输出路径；
- [x] Controller 校验 action 与 readiness capability 后执行；
- [x] 结果绑定 epoch、request signature、adapter、artifact digest 和 outcome；
- [x] 原子写入 evidence catalog、request history、预算与 hash-chain ledger；
- [x] 刷新 diagnosis context，进入下一轮提案。

### 3. 收紧方向判定

- [x] 先校验证据冲突、缺口和关系，再计算 `sufficient`；
- [x] 互斥连通分量最多允许一个成立方向；
- [x] `sufficient` 只表示主要竞争解释已排除，不等于优化成功。

### 4. 对齐公开接口和文档

- [x] 更新 `workload_control.schema.json` 与分析合同模板；
- [x] 增加 `collect-evidence` CLI；
- [x] 统一预算名称，默认保持 `balanced`；
- [x] 更新 README、SKILL、验证说明和 V3.1 release note；
- [x] 明确离线知识、外部检索与多模型质证均只生成线索，不拥有晋级权。

### 5. 验证和发布

- [x] 运行定向测试和完整测试；
- [x] 运行已知根因的重放/合成 workload，记录方向轮次与耗时；
- [ ] 在 5090 目标机运行 GPU smoke；宿主机配置只给建议；
- [ ] 复审 diff、schema、文档与发布边界；
- [ ] 合并主干，同步个人 GitHub、内网 GitLab 和本地已安装 skill；
- [ ] 不向原始上游仓库推送。
