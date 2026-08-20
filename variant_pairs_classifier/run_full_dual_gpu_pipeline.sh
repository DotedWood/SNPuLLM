#!/usr/bin/env bash
set -euo pipefail

# 评分、合并或FDR任一阶段失败都立即停止流水线。

# 集中定义项目、输入、冻结模型和输出目录，避免命令之间路径不一致。
ROOT="/vepfs-mlp2/xts001/400107"
PROJECT="${ROOT}/code/AG_classification/GTEx_self"
AG_PY="${ROOT}/miniconda3/envs/ft_alphagenome/bin/python"
ANALYSIS_PY="${ROOT}/miniconda3/bin/python"
SCORER="${PROJECT}/variant_pairs_classifier/score_pair_tissue_11modal.py"
MERGE_CLASSIFY="${PROJECT}/variant_pairs_classifier/merge_classify_fdr.py"
INPUT="${ROOT}/data/AG_classification/variant_pairs_10kb/same_tissue_pairs.parquet"
MODEL_RUN="${PROJECT}/results/GTEX_all_11"
OUT="${PROJECT}/results/variant_pairs_classifier_11modal_v1"
LOGS="${OUT}/logs"

# 每次流水线的worker日志与合并日志放在结果目录内。
mkdir -p "${LOGS}"
# 限制JAX显存预占，为两个独立worker保留稳定运行空间。
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_GPU_ALLOCATOR=cuda_malloc_async

echo "[start] $(date -Is)"
echo "[input] ${INPUT}"
echo "[model] ${MODEL_RUN}/models/ExtraTrees.joblib"
echo "[output] ${OUT}"
echo "[config] gpu_count=2 batch_size_per_gpu=8 resume=true"

# 两个worker按稳定partition拆分pair-tissue context，并启用shard resume。
CUDA_VISIBLE_DEVICES=0 "${AG_PY}" -u "${SCORER}" score \
  --input "${INPUT}" --output-dir "${OUT}" \
  --partition-index 0 --n-partitions 2 --batch-size 8 --resume \
  >"${LOGS}/gpu0.log" 2>&1 & PID0=$!
CUDA_VISIBLE_DEVICES=1 "${AG_PY}" -u "${SCORER}" score \
  --input "${INPUT}" --output-dir "${OUT}" \
  --partition-index 1 --n-partitions 2 --batch-size 8 --resume \
  >"${LOGS}/gpu1.log" 2>&1 & PID1=$!

# 保存worker PID，便于外部监控任务状态。
printf '%s\n' "gpu0=${PID0}" "gpu1=${PID1}" >"${OUT}/worker_pids.txt"
echo "[launch] gpu0_pid=${PID0} gpu1_pid=${PID1}"

# 两个worker都等待完成；任一非零退出都禁止进入不完整合并。
FAIL=0
wait "${PID0}" || FAIL=1
wait "${PID1}" || FAIL=1
if [[ "${FAIL}" -ne 0 ]]; then
  echo "[failed] one or more workers exited non-zero; existing shards are safe and --resume is enabled" >&2
  exit 1
fi

# 评分完整后依次执行shard合并、冻结GTEx分类器预测和三种null的FDR。
echo "[score-complete] $(date -Is); merging, classifying, and testing FDR"
"${ANALYSIS_PY}" -u "${MERGE_CLASSIFY}" \
  --input "${INPUT}" --score-dir "${OUT}" --model-run "${MODEL_RUN}" \
  >"${LOGS}/merge_classify_fdr.log" 2>&1
echo "[done] $(date -Is)"

