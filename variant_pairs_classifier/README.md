# Variant pairs：11 模态 → GTEx 分类器 → FDR

本目录是新的独立流程，不删除、移动或覆盖旧 variant-pair 代码与结果。

## 统计单位与 AG 打分

输入是 `same_tissue_pairs.parquet`，一行代表一个 `pair_id × target_tissue`。
每个物理 pair 仍用 16,384 bp 序列构造 REFREF、ALTREF、REFALT、ALTALT 四种状态。
对 11 种官方模态分别计算：

`interaction = score(ALTALT vs REFREF) - score(ALTREF vs REFREF) - score(REFALT vs REFREF)`

所有 CenterMask scorer（ATAC、DNase、ChIP-TF、ChIP-histone、CAGE、PRO-cap）
统一使用以 pair 中点为中心的 10,001 bp mask。其他模态使用官方 GeneMask、
ContactMap 或 Splice scorer。

归约顺序是：先匹配 pair 对应 gene（有 gene 时），再匹配 `target_tissue`，最后仅在
同一 `pair_id × tissue × modality` 的候选 scorer rows 中保留最大绝对值的 signed
interaction。这一规则复用 GTEx 二分类训练分数的特征定义；不跨 tissue、也不跨
modality 取最大值。所有匹配 scorer rows 都另存于 track-detail 文件。

## 分类器

使用正式实验 `results/GTEX_all_11` 在 Valid 上锁定的主模型 ExtraTrees。严格复用
Train 冻结的 82 维 schema：

- 11 个 signed interaction score；
- 11 个 absolute score；
- 11 个 missing indicator；
- 49 个 Train-frozen GTEx tissue one-hot。

模型输出的正类概率解释为“该 pair 的 11 模态非加性交互谱有多像 GTEx 因果正样本”。
它不是未经校准的生物学 PIP；显著性由经验 null 和 FDR 另行确定。

## Null 与 FDR

延续旧流程的三套 null：`PIP_lt_0.01`、`control`、以及二者合并。主报告建议使用
合并 null，另外两套用于敏感性分析。对 PIP > 0.9 pair 做右尾经验检验：

`p = (1 + count(null_probability >= observed_probability)) / (N_null + 1)`

每个 null definition 独立进行 Benjamini–Hochberg 校正，`FDR <= 0.05` 定义为预测的
非加性 variant pair。

## 运行与续跑

正式流程使用两张 A100 80 GB，每卡 batch size 8：

```bash
bash run_full_dual_gpu_pipeline.sh
```

worker 按 `pair_key` 固定分区并原子写 shard；重复运行会通过 `--resume` 跳过已经完整
写出 11 模态的 physical pair。两卡完成后，脚本自动合并、载入锁定模型、生成概率、
计算三套经验 P 值与 BH-FDR。

## 结果位置

全部新结果保存在：

`GTEx_self/results/variant_pairs_classifier_11modal_v1/`

关键文件：

- `pair_tissue_modality_scores_long.parquet`：每个 pair×tissue×modality 的 AG 值；
- `pair_tissue_modality_track_scores.parquet`：全部 gene/tissue 匹配后的 scorer rows；
- `same_tissue_pairs_with_tissue_matched_11modal_scores.parquet`：原始 pair 主表加 11 模态宽表；
- `score_coverage_by_tissue_modality.csv`：逐 tissue、逐模态覆盖率与回退规则；
- `classifier_fdr/pair_classifier_predictions.parquet`：11 模态值、缺失状态和模型概率；
- `classifier_fdr/classifier_features_82d.parquet`：实际进入模型的冻结 82 维输入；
- `classifier_fdr/all_high_pip_fdr_tests.parquet`：所有高 PIP 检验；
- `classifier_fdr/significant_variant_pairs.csv`：FDR <= 0.05 的 pair；
- `classifier_fdr/significance_summary.csv`：三套 null 的显著数量；
- `classifier_fdr/density_plots/`：概率密度图和 FDR 分界线；
- `logs/`：双 GPU、merge、分类和 FDR 日志。

