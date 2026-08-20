#!/usr/bin/env python3
"""合并GTEx_self的AG评分shard，并执行覆盖率、重复和元数据一致性审计。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# 项目和默认输入输出位置；命令行参数可以覆盖默认值。
REPO = Path("/vepfs-mlp2/xts001/400107")
DEFAULT_RESULTS = (
    REPO
    / "code/AG_classification/GTEx_self/results/ag_scoring/gtex_11modal_16kb_v1"
)
DEFAULT_DATASET = (
    REPO
    / "code/AG_classification/GTEx_self/data/ag_scoring_input_16kb.parquet"
)
DEFAULT_SPLICE_SITES_BACKFILL = (
    REPO
    / "results/AG_classification/scores_tissue_aligned_16kb_splice_sites_backfill"
)
DEFAULT_PERMISSIVE_TRACK_BACKFILL = (
    REPO
    / "results/AG_classification/scores_tissue_aligned_16kb_permissive_track_backfill"
)

# 固定11模态顺序，合并检查与分类器特征顺序保持一致。
MODALITIES = [
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
# 这些track模态理论上可使用全局回退，单独输出仍缺失的审计表。
PERMISSIVE_TRACK_MODALITIES = [
    "ATAC",
    "DNASE",
    "CHIP_TF",
    "CHIP_HISTONE",
    "CAGE",
    "PROCAP",
    "CONTACT_MAPS",
]


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """临时文件写完后原子改名，避免得到半写的最终文件。"""
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, compression="zstd", index=False)
    temporary.replace(path)


def load_shards(results_dir: Path, pattern: str) -> tuple[pd.DataFrame, list[Path]]:
    """跨所有part目录按文件名稳定排序后加载同类shard。"""
    paths = sorted(results_dir.glob(f"part*of*/{pattern}"))
    if not paths:
        return pd.DataFrame(), []
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True), paths


def validate_scores(scores: pd.DataFrame, expected: pd.DataFrame) -> dict[str, object]:
    """验证列、sample一对一覆盖以及输入与分数元数据完全一致。"""
    required = {
        "sample_id", "variant_key", "label", "sample_source", "target_tissue",
        "target_gene_id", "rf_split", "context_length", *MODALITIES,
    }
    missing_columns = sorted(required - set(scores.columns))
    if missing_columns:
        raise ValueError(f"Merged scores are missing columns: {missing_columns}")

    # 先检查ID集合：不允许重复、漏样本或额外样本。
    duplicate_samples = int(scores["sample_id"].duplicated().sum())
    score_ids = set(scores["sample_id"].astype(str))
    expected_ids = set(expected["sample_id"].astype(str))
    missing_samples = expected_ids - score_ids
    extra_samples = score_ids - expected_ids
    if duplicate_samples or missing_samples or extra_samples:
        raise ValueError(
            "Score coverage failed: "
            f"duplicates={duplicate_samples}, missing={len(missing_samples)}, "
            f"extra={len(extra_samples)}"
        )

    # 再逐字段比较标签、来源、组织、基因和split，防止shard串行。
    expected_meta = expected[
        [
            "sample_id", "variant_key", "label", "sample_source", "target_tissue",
            "target_gene_id", "rf_split", "context_length",
        ]
    ].copy()
    actual_meta = scores[expected_meta.columns].copy()
    joined = expected_meta.merge(
        actual_meta, on="sample_id", how="inner", suffixes=("__expected", "__score")
    )
    metadata_mismatches = 0
    for column in expected_meta.columns.drop("sample_id"):
        left = joined[f"{column}__expected"].astype("string").fillna("<NA>")
        right = joined[f"{column}__score"].astype("string").fillna("<NA>")
        metadata_mismatches += int((left != right).sum())
    if metadata_mismatches:
        raise ValueError(f"Found {metadata_mismatches} input/score metadata mismatches")

    return {
        "duplicate_sample_ids": duplicate_samples,
        "missing_expected_samples": len(missing_samples),
        "extra_samples": len(extra_samples),
        "metadata_mismatches": metadata_mismatches,
    }


def build_reports(
    scores: pd.DataFrame,
    errors: pd.DataFrame,
    score_shards: list[Path],
    error_shards: list[Path],
    validation: dict[str, object],
    results_dir: Path,
    *,
    splice_sites_backfill_integrated: bool,
    permissive_track_backfill_integrated: bool,
) -> None:
    # 报告一：逐模态覆盖率和signed/absolute分数分布。
    n_rows = len(scores)
    completeness = []
    for modality in MODALITIES:
        values = pd.to_numeric(scores[modality], errors="coerce")
        valid = values.dropna()
        abs_valid = valid.abs()
        completeness.append(
            {
                "modality": modality,
                "n_non_null": int(valid.size),
                "n_missing": int(n_rows - valid.size),
                "coverage_pct": float(valid.size / n_rows * 100),
                "score_mean": float(valid.mean()) if not valid.empty else np.nan,
                "score_median": float(valid.median()) if not valid.empty else np.nan,
                "abs_score_median": float(abs_valid.median()) if not valid.empty else np.nan,
                "abs_score_p95": float(abs_valid.quantile(0.95)) if not valid.empty else np.nan,
                "abs_score_max": float(abs_valid.max()) if not valid.empty else np.nan,
            }
        )
    completeness_df = pd.DataFrame(completeness)
    completeness_df.to_csv(results_dir / "score_completeness_by_modality.csv", index=False)

    # 报告二：按正样本、GTEx负样本和control来源拆分覆盖率。
    by_source_rows = []
    for source, group in scores.groupby("sample_source", sort=True):
        for modality in MODALITIES:
            n_valid = int(pd.to_numeric(group[modality], errors="coerce").notna().sum())
            by_source_rows.append(
                {
                    "sample_source": source,
                    "modality": modality,
                    "n_samples": int(len(group)),
                    "n_non_null": n_valid,
                    "coverage_pct": float(n_valid / len(group) * 100),
                }
            )
    pd.DataFrame(by_source_rows).to_csv(
        results_dir / "score_completeness_by_source.csv", index=False
    )

    # 报告三：统计每个模态使用了哪一级tissue匹配规则。
    match_rows = []
    for modality in MODALITIES:
        column = f"{modality}__match_rule"
        counts = scores[column].fillna("<NA>").value_counts(dropna=False)
        for rule, count in counts.items():
            match_rows.append(
                {
                    "modality": modality,
                    "match_rule": rule,
                    "n_samples": int(count),
                    "pct": float(count / n_rows * 100),
                }
            )
    pd.DataFrame(match_rows).to_csv(results_dir / "match_rule_counts.csv", index=False)

    # 汇总每条context拥有多少个有效模态，并写机器可读JSON摘要。
    scored_per_sample = scores[MODALITIES].notna().sum(axis=1)
    coverage = dict(
        zip(completeness_df["modality"], completeness_df["coverage_pct"])
    )
    summary = {
        "status": (
            "complete_with_expected_track_gene_missingness"
            if completeness_df["n_non_null"].gt(0).all()
            else "quality_control_incomplete_zero_coverage_modality"
        ),
        "merged_score_file": str(results_dir / "tissue_aligned_scores_11scorer.parquet"),
        "n_score_shards": len(score_shards),
        "n_error_shards": len(error_shards),
        "n_rows": int(n_rows),
        "n_unique_sample_ids": int(scores["sample_id"].nunique()),
        "n_unique_variants": int(scores["variant_key"].nunique()),
        "n_tissues": int(scores["target_tissue"].nunique()),
        "n_positive": int((scores["label"] == 1).sum()),
        "n_negative": int((scores["label"] == 0).sum()),
        "n_scorer_errors": int(len(errors)),
        "scorer_error_modalities": (
            errors["modality"].value_counts().to_dict() if not errors.empty else {}
        ),
        "n_samples_with_at_least_one_score": int((scored_per_sample > 0).sum()),
        "n_samples_with_all_11_scores": int((scored_per_sample == len(MODALITIES)).sum()),
        "mean_non_null_modalities_per_sample": float(scored_per_sample.mean()),
        "zero_coverage_modalities": completeness_df.loc[
            completeness_df["n_non_null"].eq(0), "modality"
        ].tolist(),
        "splice_sites_backfill_integrated": splice_sites_backfill_integrated,
        "permissive_track_backfill_integrated": permissive_track_backfill_integrated,
        "permissive_track_remaining_missing": {
            modality: int(
                pd.to_numeric(scores[modality], errors="coerce").isna().sum()
            )
            for modality in PERMISSIVE_TRACK_MODALITIES
        },
        "coverage_pct_by_modality": coverage,
        "validation": validation,
    }
    (results_dir / "final_score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--splice-sites-backfill-dir",
        type=Path,
        default=DEFAULT_SPLICE_SITES_BACKFILL,
    )
    parser.add_argument("--use-splice-sites-backfill", action="store_true")
    parser.add_argument(
        "--permissive-track-backfill-dir",
        type=Path,
        default=DEFAULT_PERMISSIVE_TRACK_BACKFILL,
    )
    parser.add_argument("--use-permissive-track-backfill", action="store_true")
    args = parser.parse_args()

    # 临时文件说明某个shard未完成；拒绝合并以避免静默缺数据。
    temporary_files = sorted(args.results_dir.glob("part*of*/.*.tmp"))
    if temporary_files:
        raise ValueError(f"Found unfinished temporary shards: {temporary_files}")

    # 分数与错误shard分开加载，错误不影响成功样本的合并。
    scores, score_shards = load_shards(args.results_dir, "scores.shard*.parquet")
    errors, error_shards = load_shards(args.results_dir, "errors.shard*.parquet")
    expected = pd.read_parquet(args.dataset)
    # 可选backfill必须与完整输入sample集合一对一一致后才能覆盖原列。
    if args.use_splice_sites_backfill:
        backfill, _ = load_shards(
            args.splice_sites_backfill_dir, "scores.shard*.parquet"
        )
        if backfill.empty:
            raise ValueError("SPLICE_SITES backfill contains no score shards")
        if backfill["sample_id"].duplicated().any():
            raise ValueError("SPLICE_SITES backfill has duplicate sample_id values")
        if set(backfill["sample_id"].astype(str)) != set(expected["sample_id"].astype(str)):
            raise ValueError("SPLICE_SITES backfill does not cover the expected sample set")
        backfill_columns = [
            "sample_id",
            "SPLICE_SITES",
            "SPLICE_SITES__match_rule",
            "SPLICE_SITES__winning_track",
            "SPLICE_SITES__winning_gene",
            "SPLICE_SITES__n_candidates",
        ]
        scores = scores.drop(columns=backfill_columns[1:]).merge(
            backfill[backfill_columns], on="sample_id", how="left", validate="one_to_one"
        )
    # track backfill同时替换分数及其match/winner/n_candidates审计字段。
    if args.use_permissive_track_backfill:
        track_backfill, _ = load_shards(
            args.permissive_track_backfill_dir, "scores.shard*.parquet"
        )
        if track_backfill.empty:
            raise ValueError("Permissive track backfill contains no score shards")
        if track_backfill["sample_id"].duplicated().any():
            raise ValueError("Permissive track backfill has duplicate sample_id values")
        if set(track_backfill["sample_id"].astype(str)) != set(
            expected["sample_id"].astype(str)
        ):
            raise ValueError(
                "Permissive track backfill does not cover the expected sample set"
            )
        track_columns = [
            "sample_id",
            *[
                column
                for modality in PERMISSIVE_TRACK_MODALITIES
                for column in (
                    modality,
                    f"{modality}__match_rule",
                    f"{modality}__winning_track",
                    f"{modality}__winning_gene",
                    f"{modality}__n_candidates",
                )
            ],
        ]
        missing_columns = sorted(set(track_columns) - set(track_backfill.columns))
        if missing_columns:
            raise ValueError(
                f"Permissive track backfill missing columns: {missing_columns}"
            )
        scores = scores.drop(columns=track_columns[1:]).merge(
            track_backfill[track_columns],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
    # 所有可选替换完成后执行最终一对一覆盖和元数据校验。
    validation = validate_scores(scores, expected)

    # 使用稳定排序保证重复运行得到相同的最终行顺序。
    scores = scores.sort_values("sample_id", kind="stable").reset_index(drop=True)
    merged_path = args.results_dir / "tissue_aligned_scores_11scorer.parquet"
    write_parquet_atomic(scores, merged_path)
    scores.head(200).to_csv(args.results_dir / "final_scores_preview_first200.csv", index=False)
    # 把允许模糊匹配后仍无分数的context逐行导出，便于人工追踪。
    missing_track_rows = []
    for modality in PERMISSIVE_TRACK_MODALITIES:
        missing = pd.to_numeric(scores[modality], errors="coerce").isna()
        if missing.any():
            subset = scores.loc[
                missing,
                [
                    "sample_id",
                    "variant_key",
                    "target_tissue",
                    "target_gene_id",
                    f"{modality}__match_rule",
                ],
            ].copy()
            subset.insert(0, "modality", modality)
            subset = subset.rename(
                columns={f"{modality}__match_rule": "remaining_missing_reason"}
            )
            missing_track_rows.append(subset)
    missing_track_frame = (
        pd.concat(missing_track_rows, ignore_index=True)
        if missing_track_rows
        else pd.DataFrame(
            columns=[
                "modality",
                "sample_id",
                "variant_key",
                "target_tissue",
                "target_gene_id",
                "remaining_missing_reason",
            ]
        )
    )
    missing_track_frame.to_csv(
        args.results_dir / "permissive_track_remaining_unmatched.csv", index=False
    )
    if not errors.empty:
        write_parquet_atomic(errors, args.results_dir / "scorer_errors.parquet")
        errors.to_csv(args.results_dir / "scorer_errors.csv", index=False)
    build_reports(
        scores,
        errors,
        score_shards,
        error_shards,
        validation,
        args.results_dir,
        splice_sites_backfill_integrated=args.use_splice_sites_backfill,
        permissive_track_backfill_integrated=args.use_permissive_track_backfill,
    )
    print((args.results_dir / "final_score_summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
