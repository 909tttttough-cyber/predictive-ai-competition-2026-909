# V27-B 实验协议（冻结于读取 outcomes 之前）

## 数据与边界

- 数据仓库：`aims-foundations/measurement-db`
- 固定 revision：`54074453da7b9911be0f48e4c560be547bc84d21`
- 仅使用 `algotune` 当前格式化的 `metadata.yaml`、`characterization.yaml`、`curation_record.md` 与 `formatted_tables/{benchmarks,items,responses,subjects}.parquet`；不读取 raw、traces 或其他 collection。
- 资格口径：依据 PAIEC 官网将 verified training data 指向 Measurement DB，以及该固定 revision 中 `algotune` 的 `release: public`。具体 benchmark 的赛事 pool 成员身份没有单独清单，本实验会如实标为“依据当前官方公开训练发布渠道进行研究，逐项归属尚未单独确认”。
- 当前 metadata 将 AlgoTune 标为 MIT、response_type=binary；curation 规则将每个已发布 task/model 最终 speedup `>=1.0` 映射为 1，否则包括显式 N/A 映射为 0。不得解释为独立重跑的 correctness。

## 冻结划分

- 单位：固定 revision 中的 distinct `item_id`；若不同 item ID 的规范化完整题面相同，则视为同一内容簇，整簇进入同一侧，避免同题跨 acquisition/evaluation。
- 全局 item 划分 seed：`20261015`。按 SHA-256 稳定排序，确定性 50/50 分配至 acquisition 与 evaluation；分配不读取 response outcome。
- subject 划分 seed：`20261015`。按 SHA-256 稳定排序，18 个匿名 subject 中 12 个训练、6 个 held-out；subject 划分不读取 outcome。匿名 ID 仅作不透明分组键。
- acquisition 前缀顺序：使用独立域分隔 SHA-256 对 acquisition item 排序；每个预算使用同一嵌套前缀 `[0,1,3,7,15,31]`。
- evaluation 只含全局 evaluation item；任何 item ID/规范化题面均不得同时存在于 acquisition/evaluation。每个已发布 response 行均保留为 evaluation outcome；本 collection `max_trial=1`，不需要重复试次抽样。

## 预测器及训练

- V12：`p=(k+1)/(n+2)`。
- 唯一候选：Beta-binomial 层级收缩 `p=(k+alpha)/(n+alpha+beta)`。仅用 12 个训练 subjects 在 acquisition item 上的二元标签，对各组 `(k_g,n_g)` 最大化 Beta-binomial 边际似然；用固定初值 `alpha=beta=1` 的二维 Nelder–Mead，log 参数边界 `[log(0.05), log(100)]`，收敛容差及迭代上限写入脚本。没有在 held-out outcomes 或形成性分数上调参。
- 两个预测器对 held-out subjects 使用完全相同的 acquisition 前缀和 evaluation response rows。每个 held-out subject-benchmark pair 内先对所有 evaluation response 行求 Brier 均值，再按六预算权重 `[0.1,0.2,0.2,0.2,0.2,0.1]` 聚合；总体对 pair 等权宏平均。
- 只有一个官方 benchmark ID，因此这是**同一 benchmark 内的匿名 subject-held-out + item-disjoint**验证，不是 benchmark-held-out 泛化。held-out 组数有限时的 bootstrap 仅作描述。

## 不可解释为

本实验不是 Codabench hidden-test 分数，不证明对 PAIEC 私有 benchmark 可迁移，也不构成组织者对每个 benchmark pool assignment 的书面确认。模型标识、来源试验设计和 AlgoTune 阈值标签的局限须随结果报告。
