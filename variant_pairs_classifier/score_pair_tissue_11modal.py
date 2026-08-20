#!/usr/bin/env python3
"""为同tissue variant pairs计算11模态AlphaGenome非加性交互分数。

This is an independent production entry point.  It reuses the validated four-
haplotype prediction/sharding implementation, but changes the context reduction
to match the GTEx binary-classifier feature semantics:

* one row is ``pair_id x target_tissue x modality``;
* CenterMask scorers use a 10,001 bp mask around the pair midpoint;
* gene-aware rows are filtered to the pair's known GTEx gene(s), when present;
* rows are tissue matched and the largest-absolute signed interaction is kept;
* no maximum is ever taken across tissues or across modalities.

The interaction is ``ALTALT - ALTREF - REFALT`` because every official delta is
already relative to REFREF.  All matched scorer rows are retained in track
detail shards; the winning row is additionally recorded in the aggregate row.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import traceback
from pathlib import Path

# 复用已验证的四单倍型前向、shard和resume实现。
ROOT = Path("/vepfs-mlp2/xts001/400107")
BASE_DIR = ROOT / "code/AG_classification/variant_pairs_tissue_matched"
LEGACY_DIR = ROOT / "code/AG_classification/variant_pairs"
for directory in (BASE_DIR, LEGACY_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import jax
import numpy as np
import pandas as pd

import score_tissue_matched_pairs_3modal as base
import score_variant_pairs_11modal as legacy
from alphagenome.models import variant_scorers as vs
from alphagenome_research.model import dna_model
from tissue_track_matching import match_tissue_rows


# 模态顺序必须与GTEx二分类器冻结schema一致。
MODALITIES = (
    "ATAC", "DNASE", "CHIP_TF", "CHIP_HISTONE", "CAGE", "PROCAP",
    "RNA_SEQ", "CONTACT_MAPS", "SPLICE_SITES", "SPLICE_SITE_USAGE",
    "SPLICE_JUNCTIONS",
)
OUTPUT = (
    ROOT / "code/AG_classification/GTEx_self/results/"
    "variant_pairs_classifier_11modal_v1"
)
# 仅在同一pair、tissue和模态的匹配行内选最大绝对交互，保留符号。
AGGREGATION_RULE = "max_abs_interaction_within_gene_and_tissue_matched_rows"
MATCHING_POLICY = (
    "pair_gene_exact_when_available;RNA_exact_GTEx;"
    "other_modalities_system_lineage_generic_global_v1"
)

GENE_PATTERN = re.compile(r"ENSG\d+(?:\.\d+)?", flags=re.IGNORECASE)


def parse_genes(value: object) -> list[str]:
    """兼容列表或字符串形式的gene字段，并移除Ensembl版本号。"""
    if value is None:
        return []
    # Parquet对象列可能已经是数组，也可能被序列化成字符串。
    if isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    else:
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        try:
            parsed = ast.literal_eval(text)
            raw = list(parsed) if isinstance(parsed, (list, tuple)) else [text]
        except (ValueError, SyntaxError):
            raw = GENE_PATTERN.findall(text)
    return sorted({str(item).strip().split(".", 1)[0].upper() for item in raw if str(item).strip()})


def context_gene_set(context: pd.Series) -> tuple[set[str], str]:
    """优先使用两个variant共享gene，否则使用两侧gene并集。"""
    shared = set(parse_genes(context.get("Gene_shared")))
    if shared:
        return shared, "shared_gene_exact"
    union = set(parse_genes(context.get("Gene_v1"))) | set(parse_genes(context.get("Gene_v2")))
    if union:
        return union, "variant_gene_union_exact"
    return set(), "no_pair_gene_available"


def normalize_gene(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip().split(".", 1)[0].upper()


def filter_gene_then_tissue(
    rows: pd.DataFrame, context: pd.Series, modality: str
) -> tuple[pd.DataFrame, str, str]:
    # gene过滤先于tissue过滤，确保winner属于pair已知基因。
    candidates = rows.copy()
    target_genes, gene_rule = context_gene_set(context)
    has_gene_rows = "gene_id" in candidates and candidates["gene_id"].notna().any()
    if has_gene_rows and target_genes:
        candidates = candidates[
            candidates["gene_id"].map(normalize_gene).isin(target_genes)
        ]
        if candidates.empty:
            return candidates, gene_rule, "no_exact_pair_gene"
    elif has_gene_rows:
        gene_rule = "gene_filter_not_available"
    else:
        gene_rule = "modality_has_no_gene_rows"

    # RNA精确匹配；其余模态复用分级tissue-track匹配策略。
    matched, tissue_rule = match_tissue_rows(
        candidates, str(context["target_tissue"]), modality
    )
    return matched, gene_rule, tissue_rule


def score_modality_rows(
    model,
    vp,
    interval,
    ref_prediction: dict,
    alt_predictions: dict[str, dict],
    setting,
) -> tuple[pd.DataFrame, str]:
    """计算一个模态的三种ALT单倍型delta，并返回可比较scorer行。"""
    organism = dna_model.Organism.HOMO_SAPIENS
    output_metadata = model.output_metadata(organism)
    modality = setting.requested_output.name
    scorer = model._variant_scorers[organism][setting.base_variant_scorer]
    # CenterMask以pair中点为锚；gene-aware scorer沿用v1锚点和注释。
    anchor_name = (
        "midpoint"
        if setting.base_variant_scorer == vs.BaseVariantScorer.CENTER_MASK
        else "v1"
    )
    anchor_variant = legacy.make_anchor(vp, anchor_name)

    try:
        # 三种ALT组合共享同一REFREF预测，并显式放到同一设备。
        with model._device_context as device, jax.transfer_guard("disallow"):
            ref_device = jax.tree.map(
                lambda value: jax.device_put(value, device), ref_prediction
            )
            alt_device = {
                key: jax.tree.map(lambda value: jax.device_put(value, device), prediction)
                for key, prediction in alt_predictions.items()
            }
            masks, mask_metadata = scorer.get_masks_and_metadata(
                interval, anchor_variant, settings=setting,
                track_metadata=output_metadata,
            )
            masks = jax.device_put(masks, device) if masks is not None else None
            frames = {
                combo: legacy.tidy_delta(
                    model, scorer, setting, interval, anchor_variant,
                    masks, mask_metadata, ref_device, alt_prediction,
                )
                for combo, alt_prediction in alt_device.items()
            }

        # Junction rows are sequence-dependent and need not align across the
        # three haplotypes.  Preserve them as combo-long rows and reduce each
        # combo after the same gene/tissue filters.
        # 剪接连接行会随序列改变，三种组合不能按行号强制对齐。
        if modality == "SPLICE_JUNCTIONS":
            long_parts = []
            for combo, frame in frames.items():
                if frame.empty:
                    continue
                part = frame.copy().reset_index(drop=True)
                part["combo"] = combo
                part["combo_score"] = pd.to_numeric(part["raw_score"], errors="coerce")
                part["scorer_row_index"] = np.arange(len(part), dtype=np.int32)
                part["mask_anchor"] = anchor_name
                part["modality"] = modality
                part["row_mode"] = "independent_combo_rows"
                long_parts.append(part)
            if not long_parts:
                return pd.DataFrame(), "empty_scorer_output"
            return pd.concat(long_parts, ignore_index=True), "independent_combo_rows"

        # 其他模态必须三表等长，才能逐track计算AA-AR-RA。
        lengths = {len(frame) for frame in frames.values()}
        if len(lengths) != 1:
            raise ValueError(f"combo tidy lengths differ: {lengths}")
        if not lengths or next(iter(lengths)) == 0:
            return pd.DataFrame(), "empty_scorer_output"

        result = frames["altalt"].copy().reset_index(drop=True)
        result["scorer_row_index"] = np.arange(len(result), dtype=np.int32)
        for combo, frame in frames.items():
            result[f"{combo}_score"] = pd.to_numeric(frame["raw_score"], errors="coerce")
        result["refref_score"] = 0.0
        # 官方每个delta已相对REFREF，因此交互项为AA-AR-RA。
        result["interaction_score"] = (
            result["altalt_score"] - result["altref_score"] - result["refalt_score"]
        )
        result["mask_anchor"] = anchor_name
        result["modality"] = modality
        result["row_mode"] = "aligned_rows"
        return result, ""
    except Exception as error:
        detail = f"{type(error).__name__}:{error}|{traceback.format_exc()[-2000:]}"
        return pd.DataFrame(), detail


def base_record(context: pd.Series, modality: str, rows: pd.DataFrame, error: str) -> dict:
    """为每个pair-tissue-modality创建固定字段的聚合结果骨架。"""
    return {
        "pair_id": context["pair_id"],
        "pair_key": context["pair_key"],
        "target_tissue": context["target_tissue"],
        "pair_class": context["pair_class"],
        "Category": context.get("Category", pd.NA),
        "modality": modality,
        "refref_score": np.nan,
        "altref_score": np.nan,
        "refalt_score": np.nan,
        "altalt_score": np.nan,
        "interaction_score": np.nan,
        "n_scorer_rows_total": int(len(rows)),
        "n_matched_scorer_rows": 0,
        "n_matched_tracks": 0,
        "n_matched_genes": 0,
        "aggregation_rule": AGGREGATION_RULE,
        "matching_policy": MATCHING_POLICY,
        "gene_match_rule": "unprocessed",
        "match_rule": "empty_scorer_output" if rows.empty else "unprocessed",
        "winning_track": pd.NA,
        "winning_gene": pd.NA,
        "winning_biosample": pd.NA,
        "winning_scorer_row_index": pd.NA,
        "status": "empty_or_error" if rows.empty else "unprocessed",
        "error": error,
        "context_length": base.CONTEXT_LENGTH,
        "center_mask_width": (
            base.CENTER_MASK_WIDTH
            if modality in {"ATAC", "DNASE", "CHIP_TF", "CHIP_HISTONE", "CAGE", "PROCAP"}
            else pd.NA
        ),
    }


def detail_records(
    rows: pd.DataFrame,
    context: pd.Series,
    modality: str,
    gene_rule: str,
    tissue_rule: str,
    winning_indices: set[tuple[str, int]],
) -> list[dict]:
    # 明细表保留所有匹配scorer行，并显式标记最终winner。
    keep = [column for column in base.TRACK_METADATA_COLUMNS if column in rows.columns]
    score_cols = [
        column for column in (
            "combo", "combo_score", "refref_score", "altref_score",
            "refalt_score", "altalt_score", "interaction_score",
        ) if column in rows.columns
    ]
    result = []
    for row in rows[["scorer_row_index", "mask_anchor"] + score_cols + keep].to_dict(orient="records"):
        combo = str(row.get("combo", "aligned"))
        index = int(row["scorer_row_index"])
        row.update({
            "pair_id": context["pair_id"],
            "pair_key": context["pair_key"],
            "target_tissue": context["target_tissue"],
            "pair_class": context["pair_class"],
            "Category": context.get("Category", pd.NA),
            "modality": modality,
            "gene_match_rule": gene_rule,
            "match_rule": tissue_rule,
            "is_winning_row": (combo, index) in winning_indices,
        })
        result.append(row)
    return result


def reduce_for_context(
    all_rows: pd.DataFrame,
    context: pd.Series,
    modality: str,
    scorer_error: str,
) -> tuple[dict, list[dict]]:
    # 一个context最终只产生一个模态分数，但所有候选行另存明细shard。
    result = base_record(context, modality, all_rows, scorer_error)
    if all_rows.empty:
        return result, []

    # Dynamic splice junctions: choose the same max-|score| feature within each
    # combo after context filtering, then apply AA - AR - RA.
    if modality == "SPLICE_JUNCTIONS" and "combo" in all_rows:
        values: dict[str, float] = {}
        winners: dict[str, pd.Series] = {}
        detail_parts = []
        gene_rules, tissue_rules = [], []
        # 每个组合分别做相同gene/tissue筛选，再各取最大绝对delta。
        for combo in ("altref", "refalt", "altalt"):
            combo_rows = all_rows[all_rows["combo"].eq(combo)].copy()
            # 当前junction组合独立执行相同的gene/tissue筛选。
            matched, gene_rule, tissue_rule = filter_gene_then_tissue(
                combo_rows, context, modality
            )
            gene_rules.append(gene_rule)
            tissue_rules.append(tissue_rule)
            matched["combo_score"] = pd.to_numeric(matched["combo_score"], errors="coerce")
            matched = matched.dropna(subset=["combo_score"])
            if matched.empty:
                result.update({
                    "gene_match_rule": "|".join(sorted(set(gene_rules))),
                    "match_rule": f"{combo}:{tissue_rule}|no_numeric_score",
                    "status": "no_context_matched_numeric_rows",
                })
                return result, []
            winner = matched.loc[matched["combo_score"].abs().idxmax()]
            values[combo] = float(winner["combo_score"])
            winners[combo] = winner
            detail_parts.append(matched)

        numeric = pd.concat(detail_parts, ignore_index=True)
        # 三个独立winner组成动态junction的非加性交互值。
        interaction = values["altalt"] - values["altref"] - values["refalt"]
        altalt_winner = winners["altalt"]
        winning_indices = {
            (combo, int(winner["scorer_row_index"])) for combo, winner in winners.items()
        }
        result.update({
            "refref_score": 0.0,
            "altref_score": values["altref"],
            "refalt_score": values["refalt"],
            "altalt_score": values["altalt"],
            "interaction_score": float(interaction),
            "n_matched_scorer_rows": int(len(numeric)),
            "n_matched_tracks": int(numeric["track_name"].nunique(dropna=True)) if "track_name" in numeric else int(len(numeric)),
            "n_matched_genes": int(numeric["gene_id"].nunique(dropna=True)) if "gene_id" in numeric else 0,
            "gene_match_rule": "|".join(sorted(set(gene_rules))),
            "match_rule": "|".join(sorted(set(tissue_rules))),
            "winning_track": altalt_winner.get("track_name", pd.NA),
            "winning_gene": altalt_winner.get("gene_id", pd.NA),
            "winning_biosample": altalt_winner.get("biosample_name", pd.NA),
            "winning_scorer_row_index": int(altalt_winner["scorer_row_index"]),
            "status": "ok_independent_junction_rows",
            "error": "junction rows reduced independently by haplotype",
        })
        return result, detail_records(
            numeric, context, modality, result["gene_match_rule"],
            result["match_rule"], winning_indices,
        )

    # 非junction模态已对齐三组合行，再在context内筛选候选。
    matched, gene_rule, tissue_rule = filter_gene_then_tissue(
        all_rows, context, modality
    )
    result["gene_match_rule"] = gene_rule
    result["match_rule"] = tissue_rule
    matched = matched.copy()
    for column in ("altref_score", "refalt_score", "altalt_score", "interaction_score"):
        matched[column] = pd.to_numeric(matched[column], errors="coerce")
    matched = matched.dropna(subset=["interaction_score"])
    if matched.empty:
        result["status"] = "no_context_matched_numeric_rows"
        return result, []

    # idxmax仅发生在同一pair、同一tissue、同一模态的候选行中。
    winner = matched.loc[matched["interaction_score"].abs().idxmax()]
    result.update({
        "refref_score": 0.0,
        "altref_score": float(winner["altref_score"]),
        "refalt_score": float(winner["refalt_score"]),
        "altalt_score": float(winner["altalt_score"]),
        "interaction_score": float(winner["interaction_score"]),
        "n_matched_scorer_rows": int(len(matched)),
        "n_matched_tracks": int(matched["track_name"].nunique(dropna=True)) if "track_name" in matched else int(len(matched)),
        "n_matched_genes": int(matched["gene_id"].nunique(dropna=True)) if "gene_id" in matched else 0,
        "winning_track": winner.get("track_name", pd.NA),
        "winning_gene": winner.get("gene_id", pd.NA),
        "winning_biosample": winner.get("biosample_name", pd.NA),
        "winning_scorer_row_index": int(winner["scorer_row_index"]),
        "status": "ok",
        "error": "",
    })
    winning = {("aligned", int(winner["scorer_row_index"]))}
    return result, detail_records(
        matched, context, modality, gene_rule, tissue_rule, winning
    )


def patched_merge(args) -> None:
    """复用基础合并器，并把历史3模态文件名改成11模态名称。"""
    base.run_merge(args)
    old = args.output_dir / "same_tissue_pairs_with_tissue_matched_3modal_scores.parquet"
    new = args.output_dir / "same_tissue_pairs_with_tissue_matched_11modal_scores.parquet"
    if old.exists():
        old.replace(new)
    summary_path = args.output_dir / "scoring_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "classifier_feature_compatibility": "GTEX_all_11 score33_plus_tissue",
        "within_context_reduction": AGGREGATION_RULE,
        "no_cross_tissue_or_cross_modality_idxmax": True,
        "gene_filter": "shared gene; otherwise union of per-variant GTEx genes; controls tissue-only",
    })
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    # 在调用基础CLI前注入本模块的11模态评分和context规约实现。
    base.MODALITIES = MODALITIES
    base.OUTPUT = OUTPUT
    base.AGGREGATION_RULE = AGGREGATION_RULE
    base.MATCHING_POLICY = MATCHING_POLICY
    base.score_modality_rows = score_modality_rows
    base.reduce_for_context = reduce_for_context
    base.run_merge = patched_merge
    # 基础CLI用模块docstring生成--help；替换为当前11模态max-abs实现的准确说明。
    base.__doc__ = __doc__
    base.main()


if __name__ == "__main__":
    main()
