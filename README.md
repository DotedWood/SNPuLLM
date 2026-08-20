# GTEx tissue-aware variant classification with AlphaGenome

本仓库整理了两个相互衔接的分析任务：

1. 使用 AlphaGenome 的 11 种模态分数，对 GTEx 变异进行二分类；
2. 为同一 tissue 中的 variant pairs 计算非加性交互分数，并使用训练好的 GTEx 分类器进行预测和 FDR 检验。

仓库主要保存代码和 Notebook。输入数据、AlphaGenome 模型权重、参考基因组以及正式运行结果需要单独准备。

## Repository structure

```text
.
├── notebooks/
│   ├── 01_data_audit_and_selection.ipynb
│   └── 02_train_classifiers.ipynb
├── ag_scoring/
│   ├── README.md
│   ├── score_gtex_11modal.py
│   ├── score_cell_lines.py
│   ├── tissue_track_matching.py
│   ├── merge_gtex_11modal_scores.py
│   └── run_dual_gpu.sh
└── variant_pairs_classifier/
    ├── README.md
    ├── score_pair_tissue_11modal.py
    ├── merge_classify_fdr.py
    └── run_full_dual_gpu_pipeline.sh
```

## `notebooks/`

该目录包含 GTEx 二分类任务的数据整理、模型训练和独立测试流程。两个 Notebook 均为自包含文件，主要代码直接写在代码单元中。

### `01_data_audit_and_selection.ipynb`

用于数据审计、标准化和样本选择，主要完成：

- 读取 GTEx 高 PIP 正样本、低 PIP 负样本和 matched control；
- 统一 variant、target gene、target tissue 和标签字段；
- 检查重复 `sample_id`、冲突 `variant_key` 和跨 split 泄漏；
- 按固定染色体划分 Train、Valid 和 Test；
- Train 和 Test 保留原始样本组成；
- Valid 在每条染色体内部进行正负样本平衡；
- 统计每个 split、染色体、tissue 和模态的样本数与覆盖率；
- 保存后续模型训练使用的统一样本表和审计表。

固定染色体划分为：

- Train：chr1、chr4、chr7、chr8、chr10、chr13、chr15；
- Valid：chr2、chr5、chr11、chr14、chr17、chr20、chr22、chrX；
- Test：chr3、chr6、chr9、chr12、chr16、chr18、chr19、chr21。

### `02_train_classifiers.ipynb`

用于构建特征、训练分类器和评估独立 Test，主要完成：

- 构建 `score11`、`score33` 和 `score33_plus_tissue` 三种特征；
- 仅使用 Train 拟合缺失值插补、标准化和 tissue one-hot 类别；
- 训练 Logistic Regression、Random Forest、Extra Trees、XGBoost、LightGBM 和 DeepMLP；
- 使用 Valid 选择主模型并锁定分类阈值；
- 在模型和阈值锁定后评估完整 Test；
- 构造穷尽式 1:1 Test 评估轮次；
- 按 `ALL`、`0-3kb`、`3-12kb`、`12-35kb` 和 `>35kb` 进行 TSS 距离分层；
- 保存模型、预测概率、评价指标、混淆矩阵、ROC 和 PR 图。

Test 不参与模型选择、调参或阈值确定。

## `ag_scoring/`

该目录用于从单个 variant 的 REF/ALT 序列重新计算 tissue-aware AlphaGenome 11 模态分数。

输入序列长度为 16,384 bp。计算的模态包括：

```text
ATAC
DNASE
CHIP_TF
CHIP_HISTONE
CAGE
PROCAP
RNA_SEQ
CONTACT_MAPS
SPLICE_SITES
SPLICE_SITE_USAGE
SPLICE_JUNCTIONS
```

### `score_gtex_11modal.py`

单变异 11 模态评分的主程序，负责：

- 按物理 `variant_key` 去重；
- 构建 16,384 bp REF/ALT 输入；
- 一次前向预测多个 AlphaGenome 输出；
- 按 target gene 和 target tissue 规约官方 scorer 输出；
- 保存每个 context 的 signed score；
- 保存匹配规则、winning track、winning gene 和候选数量；
- 使用稳定分区支持多 worker 并行；
- 使用原子 Parquet shard 支持断点续跑；
- 在显存不足时自动降低 batch size。

一个物理 variant 只进行一次模型前向计算，但同一 variant 对应的不同 gene/tissue context 会分别保存分数。

### `score_cell_lines.py`

底层模型和批处理工具，提供：

- variant 坐标解析与 REF/ALT 序列构建；
- AlphaGenome checkpoint、hg38 FASTA 和参考注释加载；
- 单 variant 与批量 variant 前向预测；
- track 分组和分数规约；
- shard 写入与 resume 辅助功能。

`score_gtex_11modal.py` 会复用该文件中的模型加载和批量输入函数。该文件也保留了独立的 cell-line 评分入口。

### `tissue_track_matching.py`

负责把 GTEx tissue 与 AlphaGenome 输出 track 对齐。

RNA-seq 要求精确匹配 GTEx tissue。其他组织相关模态按照以下顺序逐级匹配：

1. 精确 GTEx tissue；
2. 相同器官系统；
3. 相同细胞谱系；
4. 通用细胞模型；
5. 全局非 padding track。

每次匹配都会返回明确的 `match_rule`，便于在下游分析中区分精确匹配和模糊回退。

### `merge_gtex_11modal_scores.py`

负责合并评分 shard 并进行质量检查，主要包括：

- 检查输入样本是否完整覆盖；
- 检查重复或额外 `sample_id`；
- 比较输入与评分结果的 variant、label、tissue、gene 和 split；
- 合并 11 模态分数；
- 统计逐模态和逐样本来源覆盖率；
- 统计 tissue-track 匹配规则；
- 导出仍未匹配的 context 和 scorer 错误；
- 生成最终评分主表和 JSON 汇总。

### `run_dual_gpu.sh`

多 GPU 运行入口，负责：

- 检查输入文件和运行环境；
- 在正式评分前执行 `validate-only`；
- 将物理 variant 稳定分配给两个 worker；
- 分别保存每个 worker 的日志和 PID；
- 等待所有 worker 完成；
- 自动调用合并和覆盖率检查程序。

硬件编号、batch size、输入目录和输出目录应根据实际运行环境配置。

## `variant_pairs_classifier/`

该目录用于计算同一 tissue 下 variant pairs 的 11 模态非加性交互值，并使用冻结的 GTEx 分类器进行预测和显著性检验。

每个 pair 构建四种序列状态：

```text
REFREF
ALTREF
REFALT
ALTALT
```

非加性交互值定义为：

```text
interaction = score(ALTALT vs REFREF)
            - score(ALTREF vs REFREF)
            - score(REFALT vs REFREF)
```

### `score_pair_tissue_11modal.py`

variant-pairs AlphaGenome 评分主程序，负责：

- 为每个物理 pair 构建四种单倍型组合；
- 对 11 种模态分别调用官方 scorer；
- 对 CenterMask 模态使用以 pair 中点为中心的扩展 mask；
- 在存在 gene 信息时先进行 pair gene 匹配；
- 再根据 `target_tissue` 进行 tissue-track 匹配；
- 仅在同一个 `pair × tissue × modality` 内选择最大绝对值的 signed interaction；
- 不跨 tissue 或模态进行 `idxmax`；
- 同时保存聚合分数和全部匹配 track 明细；
- 支持稳定分区、Parquet shard 和断点续跑。

### `merge_classify_fdr.py`

负责将 pair 分数转换成冻结 GTEx 分类器的输入，并进行概率预测和 FDR 分析：

- 合并所有 pair 聚合 shard 和 track-detail shard；
- 检查每个 pair 是否完整包含 11 种模态；
- 构建 11 个 signed、11 个 absolute、11 个 missing indicator；
- 按 Train 冻结顺序构建 tissue one-hot；
- 加载 Valid 阶段锁定的分类器和分类阈值；
- 保存每个 `pair × tissue` 的正类概率；
- 分别使用低 PIP、control 以及二者合并作为 null distribution；
- 计算右尾经验 p-value；
- 使用 Benjamini–Hochberg 方法进行 multiple-testing correction；
- 保存 FDR 显著的 variant pairs、统计表和密度图。
