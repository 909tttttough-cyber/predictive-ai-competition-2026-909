# V37 固定实验协议

冻结于读取 response outcomes 之前。本协议只使用已确认 inventory-listed 的 AlgoTune、Multi-SWE-bench，以及 Measurement DB 固定 revision `54074453da7b9911be0f48e4c560be547bc84d21`。Multi-SWE-bench 当前 revision 的 metadata/subjects 已由 V35 获取；items/responses/benchmarks 仅下载本实验需要的格式化表。不得读取 AppWorld、MathArena、ResearchCodeBench、CMMLU outcomes。

## 特征定义（固定，不按 outcome 选词）

只检查正式预测时可见的 `item_content` 字符串，转为 Unicode casefold 后，以以下固定、大小写不敏感、词边界正则提取 5 个可重叠类别：

- `testing`: `test`, `tests`, `testing`, `pytest`, `unittest`, `junit`, `assert`。
- `dependency_installation`: `dependency`, `dependencies`, `installation`, `install`, `pip`, `npm`, `yarn`, `pnpm`, `gradle`, `maven`, `pom.xml`, `cargo`, `go.mod`, `requirements.txt`。
- `api_interface`: `api`, `interface`, `endpoint`, `signature`, `protocol`, `request`, `response`。
- `bug_exception`: `bug`, `exception`, `error`, `crash`, `traceback`, `panic`, `nullpointer`, `typeerror`, `valueerror`, `failure`, `failed`。
- `documentation_configuration`: `documentation`, `docs`, `readme`, `configuration`, `config`, `yaml`, `toml`, `ini`, `environment variable`, `settings`。

空/非字符串文本不给类别。一题可有多个类别或零类别。任务类别不读取 item ID、repo/issue 路径、grading 字段、任何 outcome 或模型输出。AlgoTune 与 Multi-SWE-bench 共用同一提取器，不做 benchmark-specific 词表。

## 固定预测规则

对当前完整 subject 字典及 benchmark 完全匹配的合法 revealed response 行，V12 基线为 `p12=(k+1)/(n+2)`；无行时为 0.5。V37 先检查 `n>=7`。对局部证据，按完整可见 item payload `(benchmark_id,item_content,item_features,interactors)` 保守去重；相同 payload 的标签冲突时排除该 payload。局部集合是与目标共享至少一个冻结类别的不同 revealed payload。需至少 3 个局部 payload 才计算 `p_local=(k_local+1)/(n_local+2)`，否则逐值回退 V12。满足时唯一公式为 `0.75*p12+0.25*p_local`，数值裁剪到 `[0,1]`。混合权重、阈值及类别词表不根据结果调整。

## 数据划分与计分

- AlgoTune：逐字节复用 V27-B 固定 outcome-blind content-cluster 划分、嵌套 acquisition 顺序及既有 12/6 subject 划分；主评分只包括 V27-B 原定 6 个 held-out subjects（seed `20261015`、六预算 `0,1,3,7,15,31`）。不重新切分、不把训练 subjects 混入 held-out 汇总。
- Multi-SWE-bench：对该固定 revision 的所有 item `content` 做与 V27 相同的 `casefold + whitespace collapse` 规范化，完全相同文本的 item ID 作为一个 cluster；按 `SHA256("item\x1f20261015\x1f" + normalized_content)` 稳定排序，按 V27-B 方法取约一半 acquisition clusters，其余 evaluation。每个 subject 的 acquisition 前缀在其有 response 的 acquisition items 上按 `SHA256("acq\x1f20261015\x1f" + subject_id + "\x1f" + item_id)` 排序，取嵌套六前缀。Outcome 不参与划分或排序。
- 每个 subject-benchmark 先要求 distinct item 数≥80，随后只有 acquisition ≥31 且 evaluation 非空的组进入完整六预算主评分。 acquisition 每题只使用最早 `(trial,response_id)` 的一个合法二元 label；评估保留所有合法二元 response 行，同一 item 的重复 response 共用该题一个预测，不跨池。
- 每个组先计算各预算 response-row 平均 Brier，再用 `[0.1,0.2,0.2,0.2,0.2,0.1]` 算组 ALC；总分为合格 subject-benchmark 组等权宏平均。两模型共用完全相同前缀、评估题与真实 outcomes。

## 解释边界

所有真实数据之前已用于项目其他研究，因此本实验是预先冻结规则后的回顾性探索，不是新样本独立验证或 Codabench hidden-test 结果。不同 collection 标签语义不同；AlgoTune 的标签按已发布 speedup threshold 生成，Multi-SWE 标签为 issue 是否解决。任何改善只支持该 collection 内的局部诊断，不能保证隐藏 benchmark 泛化。PAIEC public training pool 的逐项分配未由本地 metadata 单独列出，数据在此依组织者说明与 inventory-listed 来源要求进行开发实验，并保留这一限制。
