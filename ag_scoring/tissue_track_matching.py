#!/usr/bin/env python3
"""AlphaGenome输出的分级tissue-to-track匹配。

以下实现保留每一级match_rule，使下游可以单独过滤弱匹配。

RNA-seq keeps an exact GTEx-tissue requirement.  Other tissue-aware outputs
progressively fall back from the same organ system to a related lineage, then
to generic cell models, and finally to all non-padding tracks.  The last tier
is deliberately explicit in ``match_rule`` so downstream analyses can filter
weak matches instead of confusing them with true tissue matches.
"""

from __future__ import annotations

import re

import pandas as pd


# 每个GTEx器官系统对应可在官方track元数据中出现的关键词。
SYSTEM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "adipose": ("adipose", "adipocyte", "fat"),
    "adrenal": ("adrenal",),
    "artery": (
        "artery", "aorta", "vascular", "endothelial", "smooth muscle",
        "umbilical vein", "huvec",
    ),
    "brain": (
        "brain", "neural", "neuron", "astrocyte", "oligodendrocyte",
        "microglia", "cerebell", "cortex", "spinal cord", "motor neuron",
        "sk n sh", "sk-n-sh",
    ),
    "breast": ("breast", "mammary", "mcf 7", "mcf-7", "mcf 10a", "mcf10a"),
    "fibroblast": (
        "fibroblast", "imr 90", "imr-90", "hffc6", "hff c6", "hff",
    ),
    "lymphocyte": (
        "lymphoblast", "lymphocyte", "b lymphocyte", "t lymphocyte",
        "gm12878", "k562",
        "kbm 7", "kbm-7", "gm25256",
    ),
    "colon": (
        "colon", "colonic", "intestin", "colorectal", "hct116",
        "caco 2", "caco-2",
    ),
    "esophagus": ("esophag",),
    "heart": ("heart", "cardiac", "cardiomyocyte", "myocard"),
    "kidney": ("kidney", "renal"),
    "liver": ("liver", "hepatic", "hepatocyte", "hepg2"),
    "lung": ("lung", "pulmonary", "bronch", "a549", "calu3", "calu 3"),
    "muscle": ("skeletal muscle", "myoblast", "myotube", "muscle cell"),
    "nerve": ("nerve", "neural", "neuron", "schwann"),
    "ovary": ("ovary", "ovarian"),
    "pancreas": ("pancreas", "pancreatic", "panc1"),
    "pituitary": ("pituitary",),
    "prostate": ("prostate", "pc 3", "pc-3"),
    "salivary": ("salivary",),
    "skin": (
        "skin", "keratinocyte", "epiderm", "melanocyte", "hffc6",
        "hff c6",
    ),
    "small_intestine": (
        "small intestine", "ileum", "intestinal", "caco 2", "caco-2",
    ),
    "spleen": (
        "spleen", "splenic", "k562", "gm12878", "kbm 7", "kbm-7",
    ),
    "stomach": ("stomach", "gastric"),
    "testis": ("testis", "testicular", "sertoli", "germ cell"),
    "thyroid": ("thyroid",),
    "uterus": ("uterus", "uterine", "endomet", "hela s3", "hela-s3"),
    "vagina": ("vagina", "vaginal", "cervical", "hela s3", "hela-s3"),
    "blood": (
        "blood", "hematopoietic", "erythroid", "leukocyte", "monocyte",
        "macrophage", "b lymphocyte", "t lymphocyte", "natural killer", "k562",
        "lymphoblast", "gm12878", "kbm 7", "kbm-7", "gm25256",
    ),
}


# 器官系统进一步归并为宽泛细胞谱系，用作第二级模糊回退。
SYSTEM_LINEAGE: dict[str, str] = {
    "adipose": "mesenchymal_vascular",
    "artery": "mesenchymal_vascular",
    "fibroblast": "mesenchymal_vascular",
    "heart": "mesenchymal_vascular",
    "muscle": "mesenchymal_vascular",
    "brain": "neuro_ectodermal",
    "nerve": "neuro_ectodermal",
    "pituitary": "neuro_ectodermal",
    "skin": "neuro_ectodermal",
    "blood": "hematopoietic",
    "lymphocyte": "hematopoietic",
    "spleen": "hematopoietic",
    "colon": "digestive_endodermal",
    "esophagus": "digestive_endodermal",
    "liver": "digestive_endodermal",
    "pancreas": "digestive_endodermal",
    "small_intestine": "digestive_endodermal",
    "stomach": "digestive_endodermal",
    "lung": "epithelial_endodermal",
    "salivary": "epithelial_endodermal",
    "thyroid": "epithelial_endodermal",
    "adrenal": "urogenital_endocrine",
    "kidney": "urogenital_endocrine",
    "ovary": "urogenital_endocrine",
    "prostate": "urogenital_endocrine",
    "testis": "urogenital_endocrine",
    "uterus": "urogenital_endocrine",
    "vagina": "urogenital_endocrine",
    "breast": "reproductive_epithelial",
}


# These labels are not claimed to be tissue matched.  They are only used after
# both same-system and same-lineage searches fail.
GENERIC_MODEL_KEYWORDS = (
    "embryonic stem", "pluripotent", "h1 hesc", "h1-hesc", "h9",
    "a673", "cyt49", "fibroblast",
)

# These entries are deliberate word stems. All other keywords are matched as
# complete normalized words/phrases, preventing e.g. ``fat`` from matching the
# transcription factor name ``NFAT5``.
PREFIX_KEYWORDS = {
    "bronch", "cerebell", "endomet", "epiderm", "esophag", "intestin",
    "lymphoblast", "myocard",
}


def tissue_system(tissue: str) -> str:
    """将49个标准GTEx tissue确定性映射到一个宽泛器官系统。"""
    t = str(tissue)
    # 规则按标准化GTEx标签精确判断；未覆盖标签必须显式报错。
    rules = [
        (t.startswith("Adipose_"), "adipose"),
        (t == "Adrenal_Gland", "adrenal"),
        (t.startswith("Artery_"), "artery"),
        (t.startswith("Brain_"), "brain"),
        (t == "Breast_Mammary_Tissue", "breast"),
        (t == "Cells_Cultured_fibroblasts", "fibroblast"),
        (t == "Cells_EBV-transformed_lymphocytes", "lymphocyte"),
        (t.startswith("Colon_"), "colon"),
        (t.startswith("Esophagus_"), "esophagus"),
        (t.startswith("Heart_"), "heart"),
        (t.startswith("Kidney_"), "kidney"),
        (t == "Liver", "liver"),
        (t == "Lung", "lung"),
        (t == "Minor_Salivary_Gland", "salivary"),
        (t == "Muscle_Skeletal", "muscle"),
        (t == "Nerve_Tibial", "nerve"),
        (t == "Ovary", "ovary"),
        (t == "Pancreas", "pancreas"),
        (t == "Pituitary", "pituitary"),
        (t == "Prostate", "prostate"),
        (t.startswith("Skin_"), "skin"),
        (t == "Small_Intestine_Terminal_Ileum", "small_intestine"),
        (t == "Spleen", "spleen"),
        (t == "Stomach", "stomach"),
        (t == "Testis", "testis"),
        (t == "Thyroid", "thyroid"),
        (t == "Uterus", "uterus"),
        (t == "Vagina", "vagina"),
        (t == "Whole_Blood", "blood"),
    ]
    for matched, system in rules:
        if matched:
            return system
    raise ValueError(f"No tissue system configured for {tissue!r}")


def normalize_text(value: object) -> str:
    """统一大小写和标点，避免metadata格式差异影响关键词匹配。"""
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def metadata_text(df: pd.DataFrame) -> pd.Series:
    """拼接可用track元数据列，形成每行唯一的可搜索文本。"""
    columns = [
        c for c in (
            "track_name", "name", "biosample_name", "biosample_type",
            "gtex_tissue", "ontology_curie", "Assay title", "data_source",
        ) if c in df.columns
    ]
    if not columns:
        return pd.Series("", index=df.index, dtype="string")
    return df[columns].fillna("").astype(str).agg(" ".join, axis=1).map(normalize_text)


def non_padding_rows(df: pd.DataFrame) -> pd.DataFrame:
    """移除AlphaGenome为对齐维度加入的padding track。"""
    candidates = df
    for column in ("track_name", "name"):
        if column in candidates.columns:
            candidates = candidates[
                ~candidates[column].map(normalize_text).eq("padding")
            ]
    return candidates


def _keyword_rows(df: pd.DataFrame, keywords: tuple[str, ...]) -> pd.DataFrame:
    """保留metadata命中任一关键词的scorer行。"""
    normalized = tuple(normalize_text(k) for k in keywords)
    text = metadata_text(df)

    def contains(value: str, keyword: str) -> bool:
        if not keyword:
            return False
        if keyword in PREFIX_KEYWORDS:
            return any(token.startswith(keyword) for token in value.split())
        return f" {keyword} " in f" {value} "

    mask = text.map(lambda value: any(contains(value, k) for k in normalized))
    return df[mask]


def match_tissue_rows(
    score_df: pd.DataFrame,
    tissue: str,
    modality: str,
) -> tuple[pd.DataFrame, str]:
    """Select tissue-related rows, continuing until a non-empty tier exists."""
    # 空输出直接返回，调用者会记录为缺失而不是伪造分数。
    if score_df.empty:
        return score_df, "empty_scorer_output"

    score_df = non_padding_rows(score_df)
    if score_df.empty:
        return score_df, "empty_scorer_output"

    # SPLICE_SITES本身不带tissue语义，不进行组织过滤。
    if modality == "SPLICE_SITES":
        return score_df, "tissue_agnostic"

    # 第一级：优先使用官方metadata中的精确GTEx tissue。
    target = normalize_text(tissue)
    if "gtex_tissue" in score_df.columns:
        exact = score_df[score_df["gtex_tissue"].map(normalize_text).eq(target)]
        if not exact.empty:
            return exact, "exact_gtex_tissue"

    # RNA-seq is the one modality for which the requested design requires an
    # exact GTEx tissue.  We never hide a missing RNA tissue behind a fallback.
    if modality == "RNA_SEQ":
        return score_df.iloc[0:0], "no_exact_gtex_tissue"

    # 第二级：在同器官系统关键词范围内模糊匹配。
    system = tissue_system(tissue)
    same_system = _keyword_rows(score_df, SYSTEM_KEYWORDS[system])
    if not same_system.empty:
        return same_system, f"fuzzy_system:{system}"

    # 第三级：若本器官无track，则扩大到同一细胞谱系。
    lineage = SYSTEM_LINEAGE[system]
    lineage_keywords = tuple(
        keyword
        for related_system, related_lineage in SYSTEM_LINEAGE.items()
        if related_lineage == lineage
        for keyword in SYSTEM_KEYWORDS[related_system]
    )
    same_lineage = _keyword_rows(score_df, lineage_keywords)
    if not same_lineage.empty:
        return same_lineage, f"fuzzy_lineage:{lineage}"

    # 第四级：使用通用细胞/干细胞模型，但明确标记为generic。
    generic = _keyword_rows(score_df, GENERIC_MODEL_KEYWORDS)
    if not generic.empty:
        return generic, "fuzzy_generic_model"

    # User-requested terminal fallback: a non-empty official scorer output must
    # produce a score.  This is intentionally labelled as global, not tissue
    # matched, so it can be excluded in sensitivity analyses.
    return score_df, "fuzzy_global_fallback"
