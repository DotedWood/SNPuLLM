#!/usr/bin/env bash
set -euo pipefail

# 任一命令失败即停止；未定义变量和管道中间失败也视为错误。

# 项目、解释器、输入和输出均允许通过环境变量覆盖默认配置。
ROOT="/vepfs-mlp2/xts001/400107"
PROJECT="$ROOT/code/AG_classification/GTEx_self"
PYTHON="${PYTHON:-$ROOT/miniconda3/envs/ft_alphagenome/bin/python}"
SCRIPT="$PROJECT/ag_scoring/score_gtex_11modal.py"
MERGE="$PROJECT/ag_scoring/merge_gtex_11modal_scores.py"
DATASET="${DATASET:-$PROJECT/data/ag_scoring_input_16kb.parquet}"
RUN_TAG="${RUN_TAG:-gtex_11modal_16kb_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT/results/ag_scoring/$RUN_TAG}"
# BATCH_SIZE是每个worker的batch；OOM时Python评分器会自动减半。
BATCH_SIZE="${BATCH_SIZE:-8}"
FLUSH_BATCHES="${FLUSH_BATCHES:-10}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

# 显式固定11模态列表，避免不同worker使用不同scorer集合。
SCORERS=(
  ATAC DNASE CHIP_TF CHIP_HISTONE CAGE PROCAP RNA_SEQ
  CONTACT_MAPS SPLICE_SITES SPLICE_SITE_USAGE SPLICE_JUNCTIONS
)

# 先创建独立RUN_TAG目录，所有日志、shard和最终结果都写在其中。
mkdir -p "$OUTPUT_DIR"

# 在启动GPU任务前检查解释器和输入文件，避免两个worker同时失败。
if [ ! -x "$PYTHON" ]; then
  echo "[error] AlphaGenome Python不存在或不可执行: $PYTHON" >&2
  exit 1
fi
if [ ! -f "$DATASET" ]; then
  echo "[error] 输入数据不存在: $DATASET" >&2
  exit 1
fi

# validate-only检查字段、16kb窗口和全部tissue映射，不加载模型。
"$PYTHON" "$SCRIPT" --dataset "$DATASET" --output-dir "$OUTPUT_DIR" --validate-only

# 防止同一输出目录同时存在两套worker并发写相同partition。
if pgrep -f "$SCRIPT.*--output-dir $OUTPUT_DIR" >/dev/null; then
  echo "[error] 同一输出目录已有评分worker运行: $OUTPUT_DIR" >&2
  exit 1
fi

{
  echo "[launch] $(date -u +%FT%TZ)"
  echo "[config] dataset=$DATASET"
  echo "[config] output=$OUTPUT_DIR"
  echo "[config] gpu0=$GPU0 gpu1=$GPU1 batch_size_per_gpu=$BATCH_SIZE"
  echo "[config] resume=automatic_from_completed_parquet_shards"
} | tee -a "$OUTPUT_DIR/launcher_status.log"

# worker 0只处理稳定分区0，标准输出和错误均进入独立日志。
CUDA_VISIBLE_DEVICES="$GPU0" "$PYTHON" -u "$SCRIPT" \
  --dataset "$DATASET" --output-dir "$OUTPUT_DIR" \
  --num-parts 2 --part-index 0 --batch-size "$BATCH_SIZE" \
  --flush-batches "$FLUSH_BATCHES" --scorers "${SCORERS[@]}" \
  >> "$OUTPUT_DIR/gpu0.log" 2>&1 &
PID0=$!

# worker 1只处理稳定分区1；两个分区合起来覆盖全部物理variant。
CUDA_VISIBLE_DEVICES="$GPU1" "$PYTHON" -u "$SCRIPT" \
  --dataset "$DATASET" --output-dir "$OUTPUT_DIR" \
  --num-parts 2 --part-index 1 --batch-size "$BATCH_SIZE" \
  --flush-batches "$FLUSH_BATCHES" --scorers "${SCORERS[@]}" \
  >> "$OUTPUT_DIR/gpu1.log" 2>&1 &
PID1=$!

# 保存launcher和worker PID，便于监控或有控制地终止任务。
printf 'launcher_pid=%s\ngpu0_pid=%s\ngpu1_pid=%s\n' \
  "$$" "$PID0" "$PID1" > "$OUTPUT_DIR/worker_pids.txt"
echo "[workers] gpu0_pid=$PID0 gpu1_pid=$PID1" | tee -a "$OUTPUT_DIR/launcher_status.log"

# 分别收集两个worker退出码；先不让set -e在第一个失败时提前退出。
set +e
wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?
set -e

if [ "$STATUS0" -ne 0 ] || [ "$STATUS1" -ne 0 ]; then
  echo "[failed] gpu0_status=$STATUS0 gpu1_status=$STATUS1；已有shard安全保留，可用相同RUN_TAG续跑。" \
    | tee -a "$OUTPUT_DIR/launcher_status.log" >&2
  exit 1
fi

# 只有两分区均成功才执行一对一覆盖检查和最终Parquet合并。
echo "[merge] 两个worker完成，开始合并与覆盖率检查。" | tee -a "$OUTPUT_DIR/launcher_status.log"
"$PYTHON" "$MERGE" --dataset "$DATASET" --results-dir "$OUTPUT_DIR" \
  >> "$OUTPUT_DIR/merge.log" 2>&1

echo "[done] $OUTPUT_DIR/tissue_aligned_scores_11scorer.parquet" \
  | tee -a "$OUTPUT_DIR/launcher_status.log"
