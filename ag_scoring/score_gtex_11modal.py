#!/usr/bin/env python3
"""使用 AlphaGenome 的 11 个官方 scorer 为 GTEx_self 单变异数据评分。

执行顺序：物理variant去重 -> 16,384 bp REF/ALT前向 -> gene/tissue规约
-> 原子Parquet shard落盘。每个variant只前向一次，但每个association
context独立保存分数和匹配审计字段。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from zlib import crc32

# 项目根目录；公开部署时应改为配置项或相对于当前文件解析。
REPO = Path("/vepfs-mlp2/xts001/400107")
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "code/alphagen"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 禁止JAX预占全部显存，使OOM后缩小batch重试成为可能。
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.99")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import score_cell_lines as scl
from alphagenome.data import genome
from alphagenome.models import variant_scorers as vs_lib
from alphagenome_research.model import dna_model
from alphagenome_research.model.metadata import metadata as metadata_lib
from tissue_track_matching import match_tissue_rows, tissue_system

# AlphaGenome权重、参考注释和hg38序列资源。
CKPT = REPO / "code/alphagen/model/alphagenome-jax-all_folds-v1"
REF = REPO / "code/alphagen/model/reference/hg38"
FASTA = REPO / "data/genome/hg38/hg38.fa"
DEFAULT_DATASET = (
    REPO
    / "code/AG_classification/GTEx_self/data/ag_scoring_input_16kb.parquet"
)
DEFAULT_OUTPUT_DIR = (
    REPO
    / "code/AG_classification/GTEx_self/results/ag_scoring/gtex_11modal_16kb_v1"
)
CONTEXT_LENGTH = 16_384

# 模态顺序同时决定输出列与分类器输入的固定顺序。
SCORER_NAMES = [
    "ATAC",
    "DNASE",
    "CHIP_TF",
    "CHIP_HISTONE",
    "CAGE",
    "PROCAP",
    "RNA_SEQ",
    "CONTACT_MAPS",
    "SPLICE_SITES",
    "SPLICE_SITE_USAGE",
    "SPLICE_JUNCTIONS",
]
# 这些track模态允许按器官系统和细胞谱系逐级模糊匹配。
PERMISSIVE_TISSUE_MODALITIES = {
    "ATAC", "DNASE", "CHIP_TF", "CHIP_HISTONE", "CAGE", "PROCAP",
    "CONTACT_MAPS",
}
MATCHING_POLICY = "permissive_system_lineage_generic_global_v1"

# 兼容不同JAX/XLA版本的显存错误文本。
OOM_MARKERS = (
    "oom",
    "out of memory",
    "resource exhausted",
    "resource_exhausted",
    "failed to allocate",
)

def normalize_gene(value: object) -> str:
    """去除Ensembl版本号并统一大小写。"""
    text = "" if value is None or pd.isna(value) else str(value).strip()
    return text.split(".", 1)[0].upper()


def reduce_score_for_sample(
    score_df: pd.DataFrame,
    sample: pd.Series,
    modality: str,
) -> dict[str, object]:
    """将官方长表规约为当前gene/tissue context的一个signed score。"""
    result: dict[str, object] = {
        "score": np.nan,
        "match_rule": "unprocessed",
        "winning_track": pd.NA,
        "winning_gene": pd.NA,
        "n_candidates": 0,
    }
    if score_df.empty or "raw_score" not in score_df:
        result["match_rule"] = "empty_scorer_output"
        return result

    # 先做精确target-gene过滤，再做tissue匹配。
    candidates = score_df.copy()
    has_gene_rows = "gene_id" in candidates and candidates["gene_id"].notna().any()
    if has_gene_rows and sample["gene_match_mode"] == "exact":
        target_gene = normalize_gene(sample["target_gene_id"])
        candidates = candidates[candidates["gene_id"].map(normalize_gene).eq(target_gene)]
        if candidates.empty:
            result["match_rule"] = "no_exact_gene"
            return result

    # RNA要求精确tissue；其他模态使用可审计的逐级回退。
    candidates, tissue_rule = match_tissue_rows(
        candidates, str(sample["target_tissue"]), modality
    )
    result["match_rule"] = tissue_rule
    if candidates.empty:
        return result

    candidates = candidates.assign(
        _numeric_score=pd.to_numeric(candidates["raw_score"], errors="coerce")
    ).dropna(subset=["_numeric_score"])
    if candidates.empty:
        result["match_rule"] = f"{tissue_rule}|no_numeric_score"
        return result

    # 只在同一context和同一模态内取最大绝对值；保留符号且不跨模态。
    winner = candidates.loc[candidates["_numeric_score"].abs().idxmax()]
    result.update(
        {
            "score": float(winner["_numeric_score"]),
            "winning_track": winner.get("track_name", pd.NA),
            "winning_gene": winner.get("gene_id", pd.NA),
            "n_candidates": int(len(candidates)),
        }
    )
    return result


def get_scorer_settings(names: list[str] | None = None) -> list:
    recommended = vs_lib.RECOMMENDED_VARIANT_SCORERS
    return [recommended[name] for name in (names or SCORER_NAMES)]


def stable_partition(variant_key: str, num_parts: int) -> int:
    """稳定hash保证同一variant在重启后仍属于同一worker。"""
    return crc32(variant_key.encode("utf-8")) % num_parts


def make_variant(row: pd.Series) -> scl.VariantRow:
    return scl.VariantRow(
        chromosome=str(row["chromosome"]),
        position=int(row["position"]),
        reference=str(row["reference"]),
        alternate=str(row["alternate"]),
    )


def run_model_batch(model, variants: list[scl.VariantRow], settings: list):
    """一次批量REF/ALT前向，并仅请求当前scorer需要的模型输出头。"""
    organism = dna_model.Organism.HOMO_SAPIENS
    requested = tuple(sorted({s.requested_output for s in settings}, key=lambda x: x.name))
    ref_b, alt_b, ss_b, gm_b, im_b, strand, org_idx, intervals = (
        scl._per_variant_batch_inputs(
            # Only SPLICE_JUNCTIONS needs these model-input annotation masks.
            # Other gene-aware official scorers build their own masks later.
            model,
            variants,
            CONTEXT_LENGTH,
            skip_annotation_masks=not any(
                s.requested_output.name == "SPLICE_JUNCTIONS" for s in settings
            ),
        )
    )
    junction_masks = dna_model._SpliceJunctionVariantMasks(
        splice_sites=ss_b, reference_genes=gm_b, indel_masks=im_b
    )
    track_meta = model._metadata[organism]
    track_masks = metadata_lib.create_track_masks(
        track_meta, requested_outputs=requested, requested_ontologies=None
    )
    # 显式放置到当前设备，禁止隐藏的主机到设备传输。
    with model._device_context as device, jax.transfer_guard("disallow"):
        ref_pred, alt_pred = model._predict_variant(
            model._params,
            model._state,
            jax.device_put(ref_b, device),
            jax.device_put(alt_b, device),
            jax.device_put(junction_masks, device),
            jax.device_put(org_idx, device),
            requested_outputs=requested,
            negative_strand_mask=jax.device_put(strand, device),
            strand_reindexing=jax.device_put(track_meta.strand_reindexing, device),
        )
        ref_pred, alt_pred = dna_model._filter_variant_predictions(
            ref_pred, alt_pred, track_masks=jax.device_put(track_masks, device)
        )
    return ref_pred, alt_pred, intervals


def slice_and_upcast_predictions(predictions: dict, batch_index: int) -> dict:
    """Select one batch member while preserving nested junction dictionaries.

    SPLICE_JUNCTIONS is a nested mapping with ``predictions`` and
    ``splice_site_positions`` leaves.  Converting the top-level value with
    ``np.asarray`` turns that mapping into an object array, which JAX cannot
    place on a device.  Mapping over PyTree leaves mirrors AlphaGenome's
    single-variant path and keeps every leaf numeric.
    """

    def _slice(leaf):
        value = leaf[batch_index]
        if jnp.issubdtype(value.dtype, jnp.floating) and value.dtype != jnp.float32:
            value = value.astype(jnp.float32)
        return value

    return jax.tree.map(_slice, predictions)


def score_batch(
    model,
    variants: list[scl.VariantRow],
    settings: list,
    samples_by_variant: dict[str, pd.DataFrame],
) -> tuple[list[dict], list[dict]]:
    """评分一个物理variant批次，返回context宽表和错误表。"""
    ref_pred, alt_pred, intervals = run_model_batch(model, variants, settings)
    organism = dna_model.Organism.HOMO_SAPIENS
    output_meta = model.output_metadata(organism)
    records: list[dict] = []
    errors: list[dict] = []

    # 每个物理variant的多个gene/tissue context在此分别规约。
    for batch_idx, variant_row in enumerate(variants):
        variant_key = (
            f"{variant_row.chromosome}:{variant_row.position}:"
            f"{variant_row.reference}:{variant_row.alternate}"
        )
        contexts = samples_by_variant[variant_key]
        base_records = {
            str(row["sample_id"]): {
                "sample_id": str(row["sample_id"]),
                "variant_key": variant_key,
                "label": int(row["label"]),
                "sample_source": str(row["sample_source"]),
                "target_tissue": str(row["target_tissue"]),
                "target_gene_id": row["target_gene_id"],
                "rf_split": str(row["rf_split"]),
                "context_length": CONTEXT_LENGTH,
            }
            for _, row in contexts.iterrows()
        }

        ref_one = slice_and_upcast_predictions(ref_pred, batch_idx)
        alt_one = slice_and_upcast_predictions(alt_pred, batch_idx)
        interval = intervals[batch_idx]
        variant = genome.Variant(
            chromosome=variant_row.chromosome,
            position=variant_row.position,
            reference_bases=variant_row.reference,
            alternate_bases=variant_row.alternate,
        )

        with model._device_context as device, jax.transfer_guard("disallow"):
            ref_input = jax.tree.map(
                lambda x: jax.device_put(x, device), ref_one
            )
            alt_input = jax.tree.map(
                lambda x: jax.device_put(x, device), alt_one
            )

            # 每个模态独立调用官方mask、score、finalize和tidy流程。
            for scorer_settings in settings:
                modality = scorer_settings.requested_output.name
                try:
                    scorer = model._variant_scorers[organism][
                        scorer_settings.base_variant_scorer
                    ]
                    masks, mask_meta = scorer.get_masks_and_metadata(
                        interval,
                        variant,
                        settings=scorer_settings,
                        track_metadata=output_meta,
                    )
                    device_masks = (
                        jax.device_put(masks, device) if masks is not None else None
                    )
                    scores = scorer.score_variant(
                        ref_input,
                        alt_input,
                        masks=device_masks,
                        settings=scorer_settings,
                        variant=variant,
                        interval=interval,
                    )
                    finalized = scorer.finalize_variant(
                        jax.device_get(scores),
                        track_metadata=output_meta,
                        mask_metadata=mask_meta,
                        settings=scorer_settings,
                    )
                    finalized.uns["interval"] = interval
                    finalized.uns["variant"] = variant
                    finalized.uns["variant_scorer"] = scorer_settings
                    tidy = vs_lib.tidy_anndata(
                        finalized,
                        match_gene_strand=True,
                        include_extended_metadata=True,
                    )
                except Exception as exc:
                    # JAX dispatch is asynchronous: an OOM from the batched
                    # forward may surface only when the first scorer consumes
                    # its predictions.  It must reach the outer batch retry,
                    # otherwise a failed batch would be saved as all-NaN.
                    if any(marker in str(exc).lower() for marker in OOM_MARKERS):
                        raise
                    tidy = pd.DataFrame()
                    errors.append(
                        {
                            "variant_key": variant_key,
                            "modality": modality,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:1000],
                        }
                    )

                # 不同context分别计算，绝不跨tissue或模态做idxmax。
                for _, sample in contexts.iterrows():
                    reduced = reduce_score_for_sample(tidy, sample, modality)
                    rec = base_records[str(sample["sample_id"])]
                    rec[modality] = reduced["score"]
                    rec[f"{modality}__match_rule"] = reduced["match_rule"]
                    rec[f"{modality}__winning_track"] = reduced["winning_track"]
                    rec[f"{modality}__winning_gene"] = reduced["winning_gene"]
                    rec[f"{modality}__n_candidates"] = reduced["n_candidates"]

        records.extend(base_records.values())

    return records, errors


def existing_completed_variants(part_dir: Path) -> set[str]:
    """读取完整shard中的variant，作为自动resume跳过集合。"""
    completed: set[str] = set()
    for path in sorted(part_dir.glob("scores.shard*.parquet")):
        values = pd.read_parquet(path, columns=["variant_key"])["variant_key"]
        completed.update(values.astype(str))
    return completed


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """先写临时文件再原子改名，避免resume读到半写Parquet。"""
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, compression="zstd", index=False)
    temporary.replace(path)


def validate_dataset(dataset: pd.DataFrame) -> None:
    """加载模型前检查列、tissue、唯一ID、窗口长度和split。"""
    required = {
        "sample_id", "variant_key", "chromosome", "position", "reference", "alternate",
        "target_gene_id", "target_tissue", "gene_match_mode", "label",
        "sample_source", "rf_split", "context_length",
    }
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    for tissue in sorted(dataset["target_tissue"].dropna().astype(str).unique()):
        tissue_system(tissue)
    if dataset["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique")
    if set(pd.to_numeric(dataset["context_length"], errors="coerce").dropna()) != {
        CONTEXT_LENGTH
    }:
        raise ValueError(f"All samples must use context_length={CONTEXT_LENGTH}")
    if not set(dataset["rf_split"].dropna().astype(str)).issubset(
        {"train", "valid", "test"}
    ):
        raise ValueError("Dataset contains non-canonical chromosome split values")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--part-index", type=int, default=0)
    parser.add_argument("--num-parts", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-GPU batch; OOM automatically halves it.",
    )
    parser.add_argument("--flush-batches", type=int, default=10)
    parser.add_argument("--limit-variants", type=int, default=0)
    parser.add_argument(
        "--scorers",
        nargs="+",
        choices=SCORER_NAMES,
        default=SCORER_NAMES,
        help="Official scorer subset; default is all 11.",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """执行校验、稳定分区、断点续跑、评分和shard落盘。"""
    args = parse_args()
    if args.num_parts < 1 or not 0 <= args.part_index < args.num_parts:
        raise ValueError("part-index must be in [0, num-parts)")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    # validate-only只进行数据与tissue映射审计，不加载模型。
    dataset = pd.read_parquet(args.dataset)
    validate_dataset(dataset)
    print(
        f"[dataset] samples={len(dataset):,}, variants={dataset.variant_key.nunique():,}, "
        f"tissues={dataset.target_tissue.nunique()}, context={CONTEXT_LENGTH}",
        flush=True,
    )
    if args.validate_only:
        print("[validate-only] dataset and all tissue mappings are valid", flush=True)
        return

    # 每个worker写独立part目录，避免并发覆盖。
    part_dir = args.output_dir / f"part{args.part_index}of{args.num_parts}"
    part_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "dataset": str(args.dataset),
        "part_index": args.part_index,
        "num_parts": args.num_parts,
        "initial_batch_size": args.batch_size,
        "context_length": CONTEXT_LENGTH,
        "scorers": args.scorers,
    }
    if PERMISSIVE_TISSUE_MODALITIES.intersection(args.scorers):
        run_config["matching_policy"] = MATCHING_POLICY
    config_path = part_dir / "run_config.json"
    # resume时拒绝更改会影响分数语义或分区的配置。
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        immutable_keys = [
            "dataset", "part_index", "num_parts", "context_length", "scorers"
        ]
        if "matching_policy" in run_config:
            immutable_keys.append("matching_policy")
        mismatched = [
            key for key in immutable_keys if previous.get(key) != run_config.get(key)
        ]
        if mismatched:
            raise ValueError(
                f"Resume configuration mismatch in {config_path}: {mismatched}"
            )
    config_path.write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )

    # 先按物理variant去重，再使用稳定hash选择当前分区。
    variants_df = dataset[
        ["variant_key", "chromosome", "position", "reference", "alternate"]
    ].drop_duplicates("variant_key")
    part_mask = variants_df["variant_key"].map(
        lambda x: stable_partition(str(x), args.num_parts)
    ).eq(args.part_index)
    variants_df = variants_df[part_mask].sort_values("variant_key", kind="stable")

    # 已完整落盘的variant不会再次前向计算。
    completed = existing_completed_variants(part_dir)
    variants_df = variants_df[~variants_df["variant_key"].isin(completed)]
    if args.limit_variants:
        variants_df = variants_df.head(args.limit_variants)
    print(
        f"[part] {args.part_index}/{args.num_parts}: todo={len(variants_df):,}, "
        f"already_complete={len(completed):,}",
        flush=True,
    )
    if variants_df.empty:
        print("[done] nothing to score", flush=True)
        return

    wanted_keys = set(variants_df["variant_key"])
    sample_subset = dataset[dataset["variant_key"].isin(wanted_keys)]
    samples_by_variant = {
        str(key): group.copy()
        for key, group in sample_subset.groupby("variant_key", sort=False)
    }

    # 每个worker仅加载一次模型，并在分区内复用。
    print("[model] loading AlphaGenome", flush=True)
    model = scl.load_model(
        argparse.Namespace(
            weights_dir=CKPT,
            weights_archive=CKPT.with_suffix(".tar.gz"),
            fasta_path=FASTA,
            reference_dir=REF,
            device="gpu",
        )
    )
    settings = get_scorer_settings(args.scorers)
    print(f"[model] ready with {len(settings)} scorers", flush=True)

    variant_rows = [make_variant(row) for _, row in variants_df.iterrows()]
    current_bs = args.batch_size
    pending_records: list[dict] = []
    pending_errors: list[dict] = []
    shard_idx = len(list(part_dir.glob("scores.shard*.parquet")))
    processed = 0
    batches_since_flush = 0
    start = time.time()

    while processed < len(variant_rows):
        # OOM后current_bs永久减半，避免大shape反复编译。
        batch_size = min(current_bs, len(variant_rows) - processed)
        batch = variant_rows[processed : processed + batch_size]
        gc.collect()
        try:
            records, errors = score_batch(model, batch, settings, samples_by_variant)
            pending_records.extend(records)
            pending_errors.extend(errors)
            processed += batch_size
            batches_since_flush += 1
        except Exception as exc:
            message = str(exc).lower()
            is_oom = any(marker in message for marker in OOM_MARKERS)
            if is_oom and current_bs > 1:
                next_bs = max(1, current_bs // 2)
                print(f"[oom] batch {current_bs} -> {next_bs}", flush=True)
                current_bs = next_bs
                # Drop the failed executable and its buffers before compiling
                # the smaller shape.  Do not grow the batch again in this run;
                # repeated 8 -> 4 -> 8 oscillation wastes both time and memory.
                jax.clear_caches()
                gc.collect()
                continue
            print(f"[batch-error] {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            raise

        # 定期原子落盘，缩小异常中断后的重算范围。
        should_flush = (
            batches_since_flush >= args.flush_batches
            or processed == len(variant_rows)
        )
        if should_flush and pending_records:
            score_path = part_dir / f"scores.shard{shard_idx:05d}.parquet"
            write_parquet_atomic(pd.DataFrame(pending_records), score_path)
            if pending_errors:
                error_path = part_dir / f"errors.shard{shard_idx:05d}.parquet"
                write_parquet_atomic(pd.DataFrame(pending_errors), error_path)
            shard_idx += 1
            pending_records.clear()
            pending_errors.clear()
            batches_since_flush = 0
            elapsed = time.time() - start
            print(
                f"[progress] variants={processed:,}/{len(variant_rows):,} "
                f"({processed / len(variant_rows):.1%}), "
                f"{processed / elapsed:.2f} variants/s, batch={current_bs}",
                flush=True,
            )

    elapsed = time.time() - start
    print(f"[done] {processed:,} variants in {elapsed / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
