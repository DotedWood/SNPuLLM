# GTEx_self：AlphaGenome 16 kb、11模态重新评分

这个目录保存GTEx单变异AG评分的完整项目代码。它与原有评分代码相互独立，
不会覆盖旧结果。分类Notebook可以继续使用已经整理好的分数；只有需要从REF/ALT
序列重新生成11模态分数时，才运行这里的代码。

## 输入数据

默认输入是 `../data/ag_scoring_input_16kb.parquet`。它是不含AG分数的评分主表，
共97,237个variant–gene–tissue context、65,084个唯一物理variant、49种GTEx
tissue。每个唯一variant只做一次REF/ALT前向预测，同一variant关联的不同
gene/tissue context在官方scorer输出阶段分别规约。

## 评分内容

输入序列长度固定为16,384 bp，共计算ATAC、DNASE、CHIP_TF、CHIP_HISTONE、
CAGE、PROCAP、RNA_SEQ、CONTACT_MAPS、SPLICE_SITES、SPLICE_SITE_USAGE和
SPLICE_JUNCTIONS。

ATAC、DNASE、CHIP_TF、CAGE和PROCAP使用官方501 bp CenterMask；
CHIP_HISTONE使用2,001 bp CenterMask；RNA_SEQ使用GeneMask LFC；contact和
splice类保留官方scorer。

RNA_SEQ要求精确GTEx tissue，并在有精确target gene时只保留该gene。其他模态
按“精确GTEx tissue→相同器官系统→相同细胞谱系→通用模型→全局回退”逐级匹配。
每条结果同时保存match rule、winning track、winning gene和候选数。

## 文件

- `score_gtex_11modal.py`：主评分程序，支持稳定分区、batch、OOM自动减半、原子
  Parquet shard和自动resume。
- `score_cell_lines.py`：本地REF/ALT批量构造和模型加载工具。
- `tissue_track_matching.py`：49种GTEx tissue到官方track的分级匹配。
- `merge_gtex_11modal_scores.py`：合并shard并检查sample覆盖、重复、元数据一致性
  和11模态覆盖率。
- `run_dual_gpu.sh`：两张A100的正式启动器，完成后自动合并。

## 0. 进入项目目录

```bash
cd /vepfs-mlp2/xts001/400107
```


## 1. 只检查输入，不跑GPU

```bash
PY=/vepfs-mlp2/xts001/400107/miniconda3/envs/ft_alphagenome/bin/python

"$PY" code/AG_classification/GTEx_self/ag_scoring/score_gtex_11modal.py \
  --validate-only
```

应显示97,237个samples、65,084个variants、49个tissues和16,384 bp。

## 2. 小规模GPU测试

使用单卡、4个variant和独立输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 \
/vepfs-mlp2/xts001/400107/miniconda3/envs/ft_alphagenome/bin/python -u \
  code/AG_classification/GTEx_self/ag_scoring/score_gtex_11modal.py \
  --output-dir code/AG_classification/GTEx_self/results/ag_scoring/smoke_4variants \
  --num-parts 1 \
  --part-index 0 \
  --batch-size 2 \
  --flush-batches 1 \
  --limit-variants 4
```

smoke只覆盖4个variant，不要对这个目录运行全量merge检查。

## 3. 两张A100正式全量评分

默认每张A100使用batch size 8：

```bash
RUN_TAG=gtex_11modal_16kb_v1 \
BATCH_SIZE=8 \
bash code/AG_classification/GTEx_self/ag_scoring/run_dual_gpu.sh
```

需要退出终端后继续运行：

```bash
nohup env RUN_TAG=gtex_11modal_16kb_v1 BATCH_SIZE=8 \
  bash code/AG_classification/GTEx_self/ag_scoring/run_dual_gpu.sh \
  > code/AG_classification/GTEx_self/ag_scoring/launcher_gtex_11modal_16kb_v1.log 2>&1 &
```

查看两张卡的进度：

```bash
tail -f code/AG_classification/GTEx_self/results/ag_scoring/gtex_11modal_16kb_v1/gpu0.log
tail -f code/AG_classification/GTEx_self/results/ag_scoring/gtex_11modal_16kb_v1/gpu1.log
```

最终主文件：

```text
GTEx_self/results/ag_scoring/<RUN_TAG>/tissue_aligned_scores_11scorer.parquet
```

同时生成11模态覆盖率、来源覆盖率、匹配规则统计、错误表和
`final_score_summary.json`。

## Resume

不需要添加`--resume`。每次启动都会扫描两个partition中的
`scores.shard*.parquet`，已经完整落盘的variant自动跳过。shard先写临时文件，
完成后原子改名；中途终止不会把半写文件识别成完成结果。

重新执行相同RUN_TAG即可续跑。同一RUN_TAG不能更改dataset、分区数或scorer列表；
想运行新配置请使用新RUN_TAG。resume时可以降低batch size。

## 手动合并

如果评分完成但自动合并中断：

```bash
PY=/vepfs-mlp2/xts001/400107/miniconda3/envs/ft_alphagenome/bin/python
OUT=/vepfs-mlp2/xts001/400107/code/AG_classification/GTEx_self/results/ag_scoring/gtex_11modal_16kb_v1

"$PY" code/AG_classification/GTEx_self/ag_scoring/merge_gtex_11modal_scores.py \
  --results-dir "$OUT" \
  --dataset code/AG_classification/GTEx_self/data/ag_scoring_input_16kb.parquet
```
