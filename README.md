# GTEx tissue-aware variant classification with AlphaGenome

本项目使用 AlphaGenome 的组织相关多模态变异效应分数，对 GTEx 精细定位变异进行二分类。项目包含数据审计、固定染色体划分、特征构建、六类模型训练、独立测试、TSS 距离分层评估，以及从 REF/ALT 序列重新计算 11 种 AlphaGenome 分数的代码。

分类单位是 `variant–target gene–target tissue/context`，而不是去重后的物理 variant。

## 1. 样本定义

- 正样本：GTEx association，`PIP > 0.9`；
- GTEx 负样本：GTEx association，`PIP < 0.01`；
- Control 负样本：上游数据中冻结的 annotation/location-matched control。

Control 没有真实 target gene 和实测 tissue。其 `target_tissue` 只代表 AlphaGenome 评分上下文，不能解释成实验测量组织。样本来源字段仅用于审计，禁止作为分类特征。

## 2. 数据规模

| 文件 | Context 数 | 唯一物理 variant | 含义 |
|---|---:|---:|---|
| `data/positive_gtex_pip_gt_0p9_scored_11modal.parquet` | 42,719 | 14,533 | GTEx `PIP > 0.9` |
| `data/negative_gtex_pip_lt_0p01_plus_control_scored_11modal.parquet` | 54,518 | 50,722 | GTEx `PIP < 0.01` 与 matched control |
| `data/unified_binary_dataset.parquet` | 97,237 | 65,084 | 标准化统一主表 |
| `data/selected_samples.parquet` | 92,895 | 61,574 | 正式训练和评估样本 |

负样本包括 16,430 条 GTEx 低 PIP context 和 38,088 条 matched control context。统一数据覆盖 49 个标准化 GTEx tissue。

审计结果显示：正负样本之间没有重复 `sample_id`，没有物理 variant 跨越数据 split，坐标重建与冻结的 `variant_key` 一致。171 个物理 variant 在不同 gene/tissue context 中具有不同标签，因此保留为 context-specific association。

## 3. AlphaGenome 评分

每个物理 variant 使用以变异为中心的 16,384 bp REF 和 ALT 序列进行预测。一个物理 variant 只进行一次前向计算，再按照 target tissue 和 target gene 把官方 scorer 输出归约到各 association context。

11 种模态为：

```text
ATAC, DNASE, CHIP_TF, CHIP_HISTONE, CAGE, PROCAP,
RNA_SEQ, CONTACT_MAPS, SPLICE_SITES,
SPLICE_SITE_USAGE, SPLICE_JUNCTIONS
```

ATAC、DNASE、CHIP_TF、CAGE 和 PROCAP 使用官方 501 bp CenterMask；CHIP_HISTONE 使用 2,001 bp CenterMask；RNA_SEQ 使用 GeneMask LFC；contact 和 splice 模态保留官方 scorer。

RNA_SEQ 要求精确匹配 GTEx tissue，并在存在准确 target gene 时保留该 gene。其他组织相关模态按“精确 tissue、相同器官系统、相同细胞谱系、通用 track、全局回退”的顺序匹配。匹配规则、winning track、winning gene 和候选数量只用于审计。

## 4. 固定染色体划分

项目不使用随机行划分。

- Train：chr1、chr4、chr7、chr8、chr10、chr13、chr15；
- Valid：chr2、chr5、chr11、chr14、chr17、chr20、chr22、chrX；
- Test：chr3、chr6、chr9、chr12、chr16、chr18、chr19、chr21。

| Split | Positive | Negative | Total | Negative/Positive |
|---|---:|---:|---:|---:|
| Train | 13,086 | 19,433 | 32,519 | 1.4850 |
| Valid | 13,520 | 13,520 | 27,040 | 1.0000 |
| Test | 15,551 | 17,785 | 33,336 | 1.1437 |

Train 和 Test 保留冻结染色体中的全部样本。Valid 在每条染色体内部保持 1:1。模型对全量 Test 各预测一次；在模型和阈值锁定后，代码还会构造穷尽式 1:1 Test 评估轮次，不重新训练或选择阈值。

## 5. 特征配置

- `score11`：11 个 signed score；
- `score33`：11 个 signed score、11 个 absolute score、11 个 missing indicator；
- `score33_plus_tissue`：`score33` 加 Train-fitted GTEx tissue one-hot。

主分析使用 `score33_plus_tissue`。缺失值插补、标准化、tissue 类别集合和 one-hot 列顺序只在 Train 上拟合。Valid/Test 中未在 Train 出现的 tissue 编码为全 0。

禁止作为特征的字段包括 label、样本来源、PIP、Beta、SE、credible-set 统计量、variant 坐标、chromosome、split、`variant_key`、`sample_id`、target gene、匹配规则、track 名称及负样本构建信息。

## 6. 模型与评估

项目实现 Logistic Regression、Random Forest、Extra Trees、XGBoost、LightGBM 和 PyTorch DeepMLP。

所有模型只在 Train 上训练。Valid 用于选择主模型并通过 Youden J 锁定阈值。只有 `selection_lock_before_test.json` 保存后才运行 Test。Test 不参与模型选择、调参和阈值选择。

评价指标包括 AUROC、AUPRC、Balanced Accuracy、F1、Sensitivity、Specificity、混淆矩阵、ROC/PR 曲线，以及按 `variant_key` 聚类的 AUROC bootstrap 95% 置信区间。

## 7. 正式结果

主实验使用 `score33_plus_tissue`。Valid 结果为：

| Model | AUROC | AUPRC | Balanced Accuracy |
|---|---:|---:|---:|
| Extra Trees | 0.8931 | 0.8941 | 0.8255 |
| DeepMLP | 0.8914 | 0.8923 | 0.8224 |
| XGBoost | 0.8893 | 0.8922 | 0.8229 |
| LightGBM | 0.8864 | 0.8888 | 0.8192 |
| Logistic Regression | 0.8843 | 0.8850 | 0.8200 |
| Random Forest | 0.8522 | 0.8603 | 0.8024 |

Valid 选择 Extra Trees，并锁定阈值 0.4467。在全部 33,336 条 Test context 上：

| AUROC | AUROC 95% CI | AUPRC | Balanced Accuracy | F1 | Sensitivity | Specificity |
|---:|---:|---:|---:|---:|---:|---:|
| 0.8909 | 0.8837–0.8979 | 0.8758 | 0.8235 | 0.8135 | 0.8264 | 0.8205 |

置信区间来自 2,000 次 `variant_key` cluster bootstrap。Test 结果只报告 Valid 预先选定的主模型。

## 8. TSS 分层

目标基因 TSS 使用 hg38 GENCODE v46 和标准化 Ensembl gene ID 精确映射，距离为 `abs(variant_position - target_gene_tss)`。TSS 仅用于评估，不进入模型。

Test 结果分为 `ALL`、`0-3kb`、`3-12kb`、`12-35kb` 和 `>35kb`。Control 没有真实 target gene，因此只进入 `ALL`。

## 9. 项目结构

```text
GTEx_self/
├── data/
├── notebooks/
│   ├── 01_data_audit_and_selection.ipynb
│   └── 02_train_classifiers.ipynb
├── ag_scoring/
│   ├── score_gtex_11modal.py
│   ├── tissue_track_matching.py
│   ├── merge_gtex_11modal_scores.py
│   └── run_dual_gpu.sh
├── variant_pairs_classifier/
└── results/
```

大型数据、模型 checkpoint 和参考基因组文件是否随仓库发布，应根据数据使用协议和文件大小单独决定。

## 10. 运行

分类部分需要 pandas、NumPy、PyArrow、SciPy、scikit-learn、XGBoost、LightGBM、PyTorch、Matplotlib、Seaborn 和 Jupyter。

1. 打开 `notebooks/01_data_audit_and_selection.ipynb`，在首个配置单元设置项目和输入数据的相对或本地路径，完成审计与选样。
2. 打开 `notebooks/02_train_classifiers.ipynb`，先用 `FAST_MODE=True` 检查流程，再设置唯一 `RUN_TAG` 并用 `FAST_MODE=False` 运行正式实验。
3. 每次运行的模型、预测、指标和图像保存在 `results/<RUN_TAG>/`。

AlphaGenome 重新评分还需要官方 AlphaGenome/JAX 依赖、hg38 FASTA、GENCODE 注释、模型 checkpoint 和官方参考资源。准备 `data/ag_scoring_input_16kb.parquet` 后，可先检查输入：

```bash
python ag_scoring/score_gtex_11modal.py --dataset data/ag_scoring_input_16kb.parquet --output-dir results/ag_scoring/example_run --validate-only
```

正式评分前，请在评分配置中设置模型和参考资源路径。评分程序支持稳定分区、批处理、原子 Parquet shard、自动断点续跑和显存不足时自动降低 batch size。

## 11. 主要输出

- `validation_metrics.csv`；
- `selection_lock_before_test.json`；
- `feature_schema.json`；
- `test_all_metrics.csv`；
- `test_predictions.parquet`；
- `test_round_metrics.csv`；
- `test_tss_bin_metrics.csv`；
- `test_confusion_matrix.csv`；
- `test_roc.png` 和 `test_pr.png`；
- `models/`；
- `experiment_summary.json`。

## 12. 数据与参考资源

- GTEx Portal：https://gtexportal.org/
- GENCODE human release 46：https://www.gencodegenes.org/human/release_46.html
- AlphaGenome：请遵守模型官方仓库、论文和模型资源页面中的许可要求。

使用本项目时，请同时引用 GTEx、GENCODE、AlphaGenome 及所采用机器学习库的原始论文或官方资源。
