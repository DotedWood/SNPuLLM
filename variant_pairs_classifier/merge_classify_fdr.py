#!/usr/bin/env python3
"""合并11模态pair shard，应用冻结GTEx分类器并完成经验p值和FDR检验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 默认输入、评分目录和冻结模型run；均可通过命令行覆盖。
ROOT = Path("/vepfs-mlp2/xts001/400107")
PROJECT = ROOT / "code/AG_classification/GTEx_self"
DEFAULT_INPUT = ROOT / "data/AG_classification/variant_pairs_10kb/same_tissue_pairs.parquet"
DEFAULT_SCORE_DIR = PROJECT / "results/variant_pairs_classifier_11modal_v1"
DEFAULT_MODEL_RUN = PROJECT / "results/GTEX_all_11"
# 顺序必须与pair评分输出和Train冻结feature schema完全一致。
MODALITIES = [
    "ATAC", "DNASE", "CHIP_TF", "CHIP_HISTONE", "CAGE", "PROCAP",
    "RNA_SEQ", "CONTACT_MAPS", "SPLICE_SITES", "SPLICE_SITE_USAGE",
    "SPLICE_JUNCTIONS",
]
# 三种零分布分别用于敏感性分析，BH校正在各定义内独立进行。
NULL_DEFINITIONS = {
    "pip_lt_0.01": ("PIP_lt_0.01",),
    "control": ("control",),
    "pip_lt_0.01_plus_control": ("PIP_lt_0.01", "control"),
}
FDR_THRESHOLD = 0.05


def merge_score_shards(score_dir: Path, source_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """校验并合并聚合/track明细shard，再生成每个pair一行的宽表。"""
    aggregate_paths = sorted((score_dir / "aggregate_shards").glob("part*_shard*.parquet"))
    detail_paths = sorted((score_dir / "track_shards").glob("part*_shard*.parquet"))
    if not aggregate_paths:
        raise FileNotFoundError(f"No aggregate shards in {score_dir}")

    source = pd.read_parquet(source_path)
    # 同一pair-modality若因resume重复出现，保留最后一个完整shard记录。
    aggregate = pd.concat(
        [pd.read_parquet(path) for path in aggregate_paths], ignore_index=True
    ).drop_duplicates(["pair_id", "modality"], keep="last")
    # 每个pair-tissue context必须恰好拥有11条模态聚合记录。
    expected = len(source) * len(MODALITIES)
    observed = len(aggregate)
    if observed != expected:
        missing = expected - observed
        raise RuntimeError(
            f"Scoring is incomplete: expected {expected:,} pair-context-modality rows, "
            f"observed {observed:,}, missing {missing:,}. Resume scoring before merge."
        )
    modality_count = aggregate.groupby("pair_id")["modality"].nunique()
    if not modality_count.eq(len(MODALITIES)).all():
        raise RuntimeError("At least one pair_id does not contain exactly 11 modalities")
    aggregate = aggregate.sort_values(["pair_id", "modality"], kind="stable")
    aggregate.to_parquet(
        score_dir / "pair_tissue_modality_scores_long.parquet",
        index=False, compression="zstd",
    )

    # track明细不是分类输入，但完整保存候选行用于解释winner。
    details = pd.concat(
        [pd.read_parquet(path) for path in detail_paths], ignore_index=True
    ) if detail_paths else pd.DataFrame()
    if not details.empty:
        dedup = ["pair_id", "modality", "scorer_row_index"]
        if "combo" in details:
            details["combo"] = details["combo"].fillna("aligned")
            dedup.append("combo")
        if "mask_anchor" in details:
            dedup.append("mask_anchor")
        details = details.drop_duplicates(dedup, keep="last").sort_values(
            dedup, kind="stable"
        )
    details.to_parquet(
        score_dir / "pair_tissue_modality_track_scores.parquet",
        index=False, compression="zstd",
    )

    # 把interaction、单倍型delta和审计字段分别pivot为11模态宽列。
    wide = source.copy()
    for value in (
        "interaction_score", "altref_score", "refalt_score", "altalt_score",
        "status", "gene_match_rule", "match_rule", "winning_track",
        "winning_gene", "winning_biosample", "n_matched_scorer_rows",
    ):
        if value not in aggregate:
            continue
        pivot = aggregate.pivot(index="pair_id", columns="modality", values=value)
        pivot = pivot.reindex(columns=MODALITIES)
        pivot.columns = [f"AGPAIR_TISSUE_{modality}_{value}" for modality in MODALITIES]
        wide = wide.merge(pivot.reset_index(), on="pair_id", how="left", validate="one_to_one")
    wide.to_parquet(
        score_dir / "same_tissue_pairs_with_tissue_matched_11modal_scores.parquet",
        index=False, compression="zstd",
    )

    # 输出tissue×modality×匹配状态覆盖率，便于定位低覆盖组织。
    coverage = (
        aggregate.groupby(
            ["target_tissue", "modality", "status", "gene_match_rule", "match_rule"],
            dropna=False,
        ).size().rename("n_pair_tissue_contexts").reset_index()
    )
    coverage.to_csv(score_dir / "score_coverage_by_tissue_modality.csv", index=False)
    # 机器可读摘要固定记录交互公式、窗口和禁止跨模态idxmax约束。
    scoring_summary = {
        "status": "complete",
        "n_input_pair_tissue_contexts": int(len(source)),
        "n_unique_physical_pairs": int(source["pair_key"].nunique()),
        "n_aggregate_rows": int(len(aggregate)),
        "n_track_detail_rows": int(len(details)),
        "modalities": MODALITIES,
        "interaction_formula": "altalt - altref - refalt",
        "within_context_reduction": "max absolute signed interaction after gene/tissue matching",
        "no_cross_tissue_or_cross_modality_idxmax": True,
        "context_length": 16_384,
        "center_mask_width": 10_001,
        "status_counts": {
            str(key): int(value) for key, value in aggregate["status"].value_counts().items()
        },
    }
    (score_dir / "scoring_summary.json").write_text(
        json.dumps(scoring_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return aggregate, wide


def build_frozen_features(
    wide: pd.DataFrame, schema: dict
) -> tuple[np.ndarray, pd.DataFrame, list[str], list[str]]:
    # 只接受训练时冻结的主配方，防止错误模型或列顺序被静默使用。
    if schema["recipe"] != "score33_plus_tissue":
        raise ValueError(f"Expected score33_plus_tissue, got {schema['recipe']}")
    if list(schema["modalities"]) != MODALITIES:
        raise ValueError("Model modality order differs from pair-scoring modality order")

    # 提取11个signed interaction；不可用模态暂时保留NaN。
    score_frame = pd.DataFrame(index=wide.index)
    for modality in MODALITIES:
        column = f"AGPAIR_TISSUE_{modality}_interaction_score"
        score_frame[modality] = pd.to_numeric(wide[column], errors="coerce")

    # 以下插补、均值和尺度全部读取Train schema，不在pair数据重新拟合。
    feature_blocks = []
    feature_names = []
    # 第一块：11个signed score标准化。
    for modality in MODALITIES:
        values = score_frame[modality]
        filled = values.fillna(float(schema["signed_median"][modality])).to_numpy(np.float64)
        feature_blocks.append(
            ((filled - float(schema["signed_mean"][modality]))
             / float(schema["signed_scale"][modality]))[:, None]
        )
        feature_names.append(f"signed__{modality}")
    # 第二块：11个absolute score使用Train冻结参数标准化。
    for modality in MODALITIES:
        values = score_frame[modality].abs()
        filled = values.fillna(float(schema["absolute_median"][modality])).to_numpy(np.float64)
        feature_blocks.append(
            ((filled - float(schema["absolute_mean"][modality]))
             / float(schema["absolute_scale"][modality]))[:, None]
        )
        feature_names.append(f"absolute__{modality}")
    # 第三块：11个missing indicator保留0/1。
    for modality in MODALITIES:
        missing = score_frame[modality].isna().astype(np.float64).to_numpy()
        feature_blocks.append(missing[:, None])
        feature_names.append(f"missing__{modality}")

    # 第四块：按Train冻结顺序生成tissue one-hot；未知tissue全0。
    tissue_categories = list(schema["tissue_categories"])
    tissues = wide["target_tissue"].astype(str)
    for category in tissue_categories:
        feature_blocks.append(tissues.eq(category).astype(np.float64).to_numpy()[:, None])
        feature_names.append(f"tissue__{category}")

    matrix = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    if feature_names != list(schema["feature_names"]):
        raise AssertionError("Reconstructed feature names do not match frozen Train schema")
    if matrix.shape[1] != int(schema["n_features"]):
        raise AssertionError((matrix.shape, schema["n_features"]))
    unseen = sorted(set(tissues) - set(tissue_categories))
    if unseen:
        unknown = tissues.isin(unseen).to_numpy()
        if not np.all(matrix[unknown, 33:] == 0):
            raise AssertionError("Unseen tissues are not encoded as all-zero")
    return matrix, score_frame, feature_names, unseen


def empirical_right_tail(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    """用有限样本加一修正计算经验右尾p值，避免得到精确0。"""
    finite_null = np.sort(null[np.isfinite(null)])
    if len(finite_null) == 0:
        return np.full(len(observed), np.nan)
    left = np.searchsorted(finite_null, observed, side="left")
    return (len(finite_null) - left + 1.0) / (len(finite_null) + 1.0)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """对有限p值执行Benjamini-Hochberg校正并恢复原始行顺序。"""
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(pvalues), np.nan)
    finite = np.isfinite(pvalues)
    if not finite.any():
        return adjusted
    values = pvalues[finite]
    order = np.argsort(values, kind="stable")
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty_like(ranked)
    restored[order] = np.minimum(ranked, 1.0)
    adjusted[finite] = restored
    return adjusted


def plot_probability_density(
    null: np.ndarray,
    high: np.ndarray,
    cutoff: float | None,
    null_name: str,
    n_significant: int,
    output: Path,
) -> None:
    # null与高PIP使用完全相同的固定bin，便于直接比较密度。
    fig, ax = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    bins = np.linspace(0.0, 1.0, 81)
    ax.hist(null, bins=bins, density=True, alpha=0.36, color="#4C78A8", label=f"{null_name} (n={len(null):,})")
    ax.hist(high, bins=bins, density=True, alpha=0.36, color="#E45756", label=f"PIP > 0.9 (n={len(high):,})")
    if cutoff is not None:
        ax.axvline(cutoff, color="#6A3D9A", linestyle="--", linewidth=2,
                   label=f"FDR <= 0.05 observed cutoff: p >= {cutoff:.4f}")
    else:
        ax.text(0.98, 0.95, "No FDR <= 0.05 pair-tissue contexts", transform=ax.transAxes,
                ha="right", va="top")
    ax.set(xlim=(0, 1), xlabel="GTEx ExtraTrees positive-class probability",
           ylabel="Density", title=f"Pair classifier score · {null_name} · significant={n_significant:,}")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def apply_classifier_and_fdr(
    wide: pd.DataFrame, score_dir: Path, model_run: Path
) -> None:
    # 分类器输入、预测、显著pair、摘要和图像集中写入独立子目录。
    output_dir = score_dir / "classifier_fdr"
    figure_dir = output_dir / "density_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # feature schema、主模型和阈值均来自GTEx Valid阶段的冻结文件。
    schema = json.loads((model_run / "feature_schema.json").read_text(encoding="utf-8"))
    lock = json.loads((model_run / "selection_lock_before_test.json").read_text(encoding="utf-8"))
    primary_model = str(lock["primary_model"])
    threshold = float(lock["primary_threshold_from_validation"])
    if primary_model != "ExtraTrees":
        raise ValueError(f"Locked primary model is {primary_model}, expected ExtraTrees")
    model_path = model_run / "models" / f"{primary_model}.joblib"
    model = joblib.load(model_path)

    # 每个pair-tissue context只预测一次正类概率。
    matrix, score_frame, feature_names, unseen = build_frozen_features(wide, schema)
    probability = model.predict_proba(matrix)[:, 1]
    predicted = (probability >= threshold).astype(np.int8)
    np.save(output_dir / "classifier_input_matrix.npy", matrix)

    feature_table = pd.DataFrame(matrix, columns=feature_names)
    feature_table.insert(0, "pair_id", wide["pair_id"].to_numpy())
    feature_table.to_parquet(
        output_dir / "classifier_features_82d.parquet", index=False, compression="zstd"
    )

    metadata = [
        "pair_id", "pair_key", "pair_class", "target_tissue", "Category",
        "chromosome", "position_1", "position_2", "variant_1", "variant_2",
        "pair_distance_bp", "PIP_v1", "PIP_v2", "Gene_v1", "Gene_v2",
        "Gene_shared", "Tissue_v1", "Tissue_v2", "Tissue_shared",
    ]
    # 预测表同时保留pair元数据、11模态交互值、缺失指示和模型概率。
    predictions = wide[[column for column in metadata if column in wide]].copy()
    for modality in MODALITIES:
        predictions[f"AGPAIR_{modality}_interaction_score"] = score_frame[modality].to_numpy()
        predictions[f"AGPAIR_{modality}_missing"] = score_frame[modality].isna().astype(np.int8).to_numpy()
        status_column = f"AGPAIR_TISSUE_{modality}_status"
        if status_column in wide:
            predictions[f"AGPAIR_{modality}_status"] = wide[status_column].to_numpy()
    predictions["classifier_probability_positive"] = probability
    predictions["classifier_predicted_label"] = predicted
    predictions["locked_validation_threshold"] = threshold
    predictions["classifier_model"] = primary_model
    predictions["classifier_model_run"] = model_run.name
    predictions["classifier_score_interpretation"] = "causal-like non-additive interaction profile"
    predictions.to_parquet(
        output_dir / "pair_classifier_predictions.parquet", index=False, compression="zstd"
    )
    predictions.to_csv(output_dir / "pair_classifier_predictions.csv", index=False)

    tested_parts = []
    summary_rows = []
    # 对三种null分别计算右尾经验p值，并在各自高PIP检验集合内做BH。
    for null_name, null_classes in NULL_DEFINITIONS.items():
        null = predictions.loc[
            predictions["pair_class"].isin(null_classes),
            "classifier_probability_positive",
        ].to_numpy(float)
        high = predictions[predictions["pair_class"].eq("PIP_gt_0.9")].copy()
        observed = high["classifier_probability_positive"].to_numpy(float)
        # 概率越高表示越像GTEx因果正样本，因此使用右尾检验。
        pvalues = empirical_right_tail(observed, null)
        fdr = benjamini_hochberg(pvalues)
        high["null_definition"] = null_name
        high["empirical_p_value_right_tail"] = pvalues
        high["fdr_bh"] = fdr
        high["significant_fdr_0_05"] = fdr <= FDR_THRESHOLD
        high["n_null"] = len(null)
        tested_parts.append(high)
        significant = high["significant_fdr_0_05"]
        cutoff = float(high.loc[significant, "classifier_probability_positive"].min()) if significant.any() else None
        summary_rows.append({
            "null_definition": null_name,
            "n_null": int(len(null)),
            "n_high_pip_tested": int(len(high)),
            "n_significant_pair_tissue_contexts": int(significant.sum()),
            "n_significant_unique_physical_pairs": int(high.loc[significant, "pair_key"].nunique()),
            "minimum_probability_among_significant": cutoff,
            "fdr_threshold": FDR_THRESHOLD,
        })
        plot_probability_density(
            null, observed, cutoff, null_name, int(significant.sum()),
            figure_dir / f"classifier_probability_{null_name}_density_fdr05.png",
        )

    # 同时保存全部高PIP检验行和FDR<=0.05子集，保证结果可追溯。
    tested = pd.concat(tested_parts, ignore_index=True)
    significant = tested[tested["significant_fdr_0_05"]].copy().sort_values(
        ["null_definition", "fdr_bh", "classifier_probability_positive"],
        ascending=[True, True, False], kind="stable",
    )
    summary = pd.DataFrame(summary_rows)
    by_tissue = (
        significant.groupby(["null_definition", "target_tissue"], dropna=False)
        .agg(
            n_significant_pair_tissue_contexts=("pair_id", "size"),
            n_significant_unique_physical_pairs=("pair_key", "nunique"),
        ).reset_index()
    )
    tested.to_parquet(output_dir / "all_high_pip_fdr_tests.parquet", index=False, compression="zstd")
    significant.to_parquet(output_dir / "significant_variant_pairs.parquet", index=False, compression="zstd")
    significant.to_csv(output_dir / "significant_variant_pairs.csv", index=False)
    summary.to_csv(output_dir / "significance_summary.csv", index=False)
    by_tissue.to_csv(output_dir / "significant_counts_by_tissue.csv", index=False)
    try:
        with pd.ExcelWriter(output_dir / "variant_pair_classifier_fdr_results.xlsx") as writer:
            summary.to_excel(writer, sheet_name="summary", index=False)
            significant.to_excel(writer, sheet_name="significant_pairs", index=False)
            by_tissue.to_excel(writer, sheet_name="counts_by_tissue", index=False)
    except ImportError:
        pass

    # JSON摘要记录统计单位、公式、null定义和全部显著性计数。
    audit = {
        "status": "complete",
        "statistical_unit": "pair_id x target_tissue",
        "model_run": str(model_run),
        "model": primary_model,
        "model_input": "11 signed + 11 absolute + 11 missing + 49 Train-frozen tissue one-hot",
        "locked_validation_threshold": threshold,
        "n_predictions": int(len(predictions)),
        "n_model_features": int(matrix.shape[1]),
        "unseen_pair_tissues_encoded_all_zero": unseen,
        "classifier_score": "positive-class probability; higher means the 11-modal interaction profile is more causal-like",
        "null_definitions": {key: list(value) for key, value in NULL_DEFINITIONS.items()},
        "primary_null_recommendation": "pip_lt_0.01_plus_control",
        "p_value": "(1 + count(null_probability >= observed_probability)) / (N_null + 1)",
        "multiple_testing": "BH independently for each null definition across PIP>0.9 pair x tissue contexts",
        "fdr_threshold": FDR_THRESHOLD,
        "summary": summary.to_dict(orient="records"),
    }
    (output_dir / "classifier_fdr_summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def main() -> None:
    """先完成shard合并，再重建冻结特征并执行分类/FDR分析。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    parser.add_argument("--model-run", type=Path, default=DEFAULT_MODEL_RUN)
    args = parser.parse_args()
    _, wide = merge_score_shards(args.score_dir, args.input)
    apply_classifier_and_fdr(wide, args.score_dir, args.model_run)


if __name__ == "__main__":
    main()
