"""Score SNVs with AlphaGenome ATAC / DNase / RNA-seq tracks and collapse
the outputs to one scalar per (variant, modality, cell line).

GTEx_self说明：主评分器主要复用本文件的VariantRow、模型加载和批量输入构造。
本文件保留原始cell-line CLI，便于独立测试底层REF/ALT预测与shard机制。

Pipeline
--------
1. Read the Excel file and extract the ``Variant`` column.
2. De-duplicate variants (the same chr:pos:ref:alt can appear hundreds of times
   attached to different (gene, tissue) rows).
3. Load the AlphaGenome model once with local hg38 + Gencode assets.
4. For every unique variant, call ``model.predict_variant`` with the shortest
   supported sequence length (16 384 bp by default). We request
   ``ATAC``, ``DNASE`` and ``RNA_SEQ`` outputs only.
5. For each modality, group the returned tracks by ``biosample_name``
   (this is the cell line / sample label in AlphaGenome's metadata) and
   reduce the (positional_bins, num_tracks) tensors to:

   * ``ref_max``             - max of REF values across bins and tracks
   * ``alt_max``             - max of ALT values across bins and tracks
   * ``abs_delta_max``       - max of |ALT - REF| across bins and tracks
   * ``signed_delta_at_argmax_abs`` - signed delta at the argmax position
   * ``n_tracks``            - number of tracks that belong to this cell line

   The primary "functional score" is ``abs_delta_max`` - the strongest
   allele-induced change this variant can cause in the cell line's assay.

6. Results are written incrementally to partitioned parquet (one shard per
   chunk) so the job can be resumed after a crash.
7. After all variants have been scored, we produce:

   * a long-format parquet of every (variant, modality, cell line) row
   * a modality-level "summary" table (max abs-delta across all cell lines
     + the best cell line name), joined back to the original Excel
   * optional wide-format parquet per modality (variant x cell_line matrix)

Usage
-----
Quick smoke test on 20 unique variants, saving next to the input file::

    python code/alphagen/score_cell_lines.py --limit 20 \
        --output-dir results/alphagenome_celllines_test

Full run::

    python code/alphagen/score_cell_lines.py \
        --output-dir results/alphagenome_celllines
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Path defaults (relative to repo root)
# ---------------------------------------------------------------------------
# 本地模型与参考资源根目录；公开部署时应改成用户配置。
REPO_ROOT = Path("/vepfs-mlp2/xts001/400107")
DEFAULT_WORKBOOK = REPO_ROOT / "data/41586_2026_10121_MOESM3_ESM_hg38_clean.xlsx"
DEFAULT_SHEET = "S1.Overview.hg38"
DEFAULT_HEADER_SKIP = 1  # the real column header lives on the 2nd row
DEFAULT_CHECKPOINT = REPO_ROOT / "code/alphagen/model/alphagenome-jax-all_folds-v1"
DEFAULT_CHECKPOINT_ARCHIVE = REPO_ROOT / "code/alphagen/model/alphagenome-jax-all_folds-v1.tar.gz"
DEFAULT_FASTA = REPO_ROOT / "data/genome/hg38/hg38.fa"
DEFAULT_REF_DIR = REPO_ROOT / "code/alphagen/model/reference/hg38"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/alphagenome_celllines"
# 16 384 bp is AlphaGenome's smallest supported context - fastest to compute
# and has been shown to be enough for local ATAC/DNase/RNA-seq effects.
DEFAULT_CONTEXT_LENGTH = 16_384

MODALITY_KEYS = ("ATAC", "DNASE", "RNA_SEQ")


@dataclass(frozen=True)
class VariantRow:
    # 不可变坐标对象，确保variant key在分区和resume期间保持稳定。
    chromosome: str
    position: int
    reference: str
    alternate: str

    @property
    def key(self) -> str:
        return f"{self.chromosome}:{self.position}:{self.reference}:{self.alternate}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    p.add_argument("--sheet-name", type=str, default=DEFAULT_SHEET)
    p.add_argument("--header-skip", type=int, default=DEFAULT_HEADER_SKIP,
                   help="Rows to skip before pandas reads the header row (default 1).")
    p.add_argument("--variant-col", type=str, default="Variant")
    p.add_argument("--chrom-col", type=str, default="Chromosome")
    p.add_argument("--pos-col", type=str, default="Position")
    p.add_argument("--ref-col", type=str, default="Allele1")
    p.add_argument("--alt-col", type=str, default="Allele2")

    p.add_argument("--weights-dir", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--weights-archive", type=Path, default=DEFAULT_CHECKPOINT_ARCHIVE,
                   help="Fallback tar.gz that will be extracted if weights-dir is empty.")
    p.add_argument("--fasta-path", type=Path, default=DEFAULT_FASTA)
    p.add_argument("--reference-dir", type=Path, default=DEFAULT_REF_DIR,
                   help="Directory containing Gencode / splice / polyA feather files.")

    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH,
                   choices=[16_384, 131_072, 524_288, 1_048_576],
                   help="AlphaGenome sequence window (in bp). 16384 is smallest & fastest.")

    p.add_argument("--modalities", nargs="+", default=list(MODALITY_KEYS),
                   choices=list(MODALITY_KEYS),
                   help="Which AlphaGenome output types to score.")

    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of unique variants to score (for testing).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Number of variants per GPU forward pass. 1 = use AlphaGenome's "
                        "public single-variant API; >1 = batched path via internal "
                        "model._predict_variant. For A100 80GB at 16KB try 8-16.")
    p.add_argument("--skip-annotation-masks", action="store_true", default=True,
                   help="Skip per-variant gene_mask / splice_site extraction on the CPU. "
                        "These masks only feed SPLICE_JUNCTIONS outputs; they do NOT "
                        "affect ATAC / DNASE / RNA_SEQ predictions (verified in "
                        "dna_model._predict_variant). Default ON, cuts CPU time ~30x.")
    p.add_argument("--no-skip-annotation-masks", dest="skip_annotation_masks",
                   action="store_false",
                   help="Force exact parity with the public predict_variant API "
                        "(i.e. run gene_mask / splice_site extraction for every variant).")
    p.add_argument("--shard-size", type=int, default=200,
                   help="Write a parquet shard every N variants. Controls resume granularity.")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip variants already present in existing shards.")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--device", type=str, default="gpu", choices=["auto", "gpu", "cpu", "tpu"])
    p.add_argument("--no-final-merge", action="store_true",
                   help="Skip the wide-format pivot + xlsx/csv join step and keep only shards.")
    p.add_argument("--save-wide-parquet", action="store_true", default=True,
                   help="Also pivot long -> wide parquet per modality (variant x cell_line).")
    p.add_argument("--no-save-wide-parquet", dest="save_wide_parquet", action="store_false")
    p.add_argument("--joined-format", type=str, default="csv", choices=["csv", "parquet", "xlsx"],
                   help="Format for the final joined-to-original table. Excel does not support 630k rows well.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------
def load_workbook(args: argparse.Namespace) -> pd.DataFrame:
    """读取实验表并保留原始行号，便于最终一对多回填。"""
    t0 = time.perf_counter()
    df = pd.read_excel(args.workbook, sheet_name=args.sheet_name, skiprows=args.header_skip)
    # Record the original row index so we can join back even if we permute.
    df = df.reset_index(drop=False).rename(columns={"index": "__orig_row__"})
    print(f"[load] Excel -> {df.shape} in {time.perf_counter() - t0:.1f}s")
    return df


def parse_variant_text(text: object) -> VariantRow | None:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return None
    s = str(text).strip()
    if not s:
        return None
    parts = s.split(":")
    if len(parts) < 4:
        return None
    chrom, pos, ref, alt = parts[0], parts[1], parts[2], parts[3]
    try:
        return VariantRow(chromosome=chrom, position=int(pos),
                          reference=ref.upper(), alternate=alt.upper())
    except (TypeError, ValueError):
        return None


def parse_variant_from_row(row: pd.Series, args: argparse.Namespace) -> VariantRow | None:
    if args.variant_col in row.index:
        parsed = parse_variant_text(row.get(args.variant_col))
        if parsed is not None:
            return parsed
    # fallback to chrom/pos/ref/alt columns
    try:
        chrom = row.get(args.chrom_col)
        pos = row.get(args.pos_col)
        ref = row.get(args.ref_col)
        alt = row.get(args.alt_col)
        if pd.isna(chrom) or pd.isna(pos) or pd.isna(ref) or pd.isna(alt):
            return None
        return VariantRow(chromosome=str(chrom), position=int(pos),
                          reference=str(ref).upper(), alternate=str(alt).upper())
    except Exception:
        return None


def deduplicate_variants(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """将重复gene/tissue行折叠为唯一物理variant，减少重复前向计算。"""
    """Return a frame with one row per unique variant, indexed by variant_key."""
    variants: dict[str, VariantRow] = {}
    counts: dict[str, int] = {}
    for _, row in df.iterrows():
        v = parse_variant_from_row(row, args)
        if v is None:
            continue
        if v.key not in variants:
            variants[v.key] = v
        counts[v.key] = counts.get(v.key, 0) + 1
    uniq = pd.DataFrame(
        [{"variant_key": v.key, "chromosome": v.chromosome, "position": v.position,
          "reference": v.reference, "alternate": v.alternate, "n_rows_in_workbook": counts[v.key]}
         for v in variants.values()]
    )
    print(f"[dedup] {len(df)} workbook rows -> {len(uniq)} unique variants")
    return uniq


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def ensure_checkpoint(weights_dir: Path, archive: Path) -> Path:
    if (weights_dir / "_CHECKPOINT_METADATA").exists():
        return weights_dir
    if archive.exists():
        weights_dir.mkdir(parents=True, exist_ok=True)
        print(f"[model] extracting {archive} -> {weights_dir}")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(weights_dir)
        return weights_dir
    raise FileNotFoundError(f"No checkpoint at {weights_dir} and no archive at {archive}")


def pick_jax_device(name: str):
    import jax
    if name != "auto":
        devs = jax.devices(name)
        if not devs:
            raise RuntimeError(f"No JAX {name} device available")
        return devs[0]
    for candidate in ("gpu", "tpu"):
        devs = jax.devices(candidate)
        if devs:
            return devs[0]
    return jax.devices("cpu")[0]


def load_model(args: argparse.Namespace):
    """加载checkpoint、hg38 FASTA和参考注释，并绑定指定JAX设备。"""
    from alphagenome_research.model import dna_model
    t0 = time.perf_counter()
    checkpoint_path = ensure_checkpoint(args.weights_dir, args.weights_archive).resolve()
    device = pick_jax_device(args.device)
    ref_dir = args.reference_dir
    organism_settings = {
        dna_model.Organism.HOMO_SAPIENS: dna_model.OrganismSettings(
            fasta_path=str(args.fasta_path),
            gtf_feather_path=str(ref_dir / "gencode.v46.annotation.gtf.gz.feather"),
            pas_feather_path=str(ref_dir / "polyadb_human_v3_exon3_contiguous_gtfv46.feather"),
            splice_site_starts_feather_path=str(ref_dir / "gencode.v46.splice_sites_starts.feather"),
            splice_site_ends_feather_path=str(ref_dir / "gencode.v46.splice_sites_ends.feather"),
        )
    }
    model = dna_model.create(checkpoint_path=str(checkpoint_path),
                             organism_settings=organism_settings, device=device)
    print(f"[model] loaded on {device} in {time.perf_counter() - t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# Cell-line grouping
# ---------------------------------------------------------------------------
@dataclass
class ModalityGroup:
    """Pre-computed aggregation plan for a modality.

    ``groups`` maps biosample_name -> ndarray of track column indices in the
    modality's TrackData.values (shape: (bins, num_tracks)).
    ``ontology_per_group`` gives the first observed ontology CURIE per cell
    line, if available (useful to trace back to ENCODE/Roadmap IDs).
    """
    modality: str
    groups: dict[str, np.ndarray]
    ontology_per_group: dict[str, str | None]


def build_groupings(model, modalities: list[str]) -> dict[str, ModalityGroup]:
    """按biosample/cell-line预计算track索引，避免每个variant重复分组。"""
    from alphagenome_research.model import dna_model
    meta = model.output_metadata(dna_model.Organism.HOMO_SAPIENS)
    tables = {
        "ATAC": meta.atac,
        "DNASE": meta.dnase,
        "RNA_SEQ": meta.rna_seq,
    }
    out: dict[str, ModalityGroup] = {}
    for modality in modalities:
        tbl = tables[modality]
        # Cell line label lives in `biosample_name`. Fall back to ontology
        # curie if biosample_name is missing/empty.
        if "biosample_name" in tbl.columns:
            labels = tbl["biosample_name"].astype("string")
        else:
            labels = pd.Series([""] * len(tbl), dtype="string")
        if "ontology_curie" in tbl.columns:
            onto = tbl["ontology_curie"].astype("string")
        else:
            onto = pd.Series([""] * len(tbl), dtype="string")
        labels = labels.where(labels.notna() & (labels.str.len() > 0), other=onto)
        labels = labels.fillna("unknown_cell_line")

        groups: dict[str, np.ndarray] = {}
        onto_map: dict[str, str | None] = {}
        for idx, (cell_line, row) in enumerate(zip(labels.tolist(), tbl.itertuples(index=False))):
            groups.setdefault(cell_line, []).append(idx)
            if cell_line not in onto_map:
                onto_map[cell_line] = getattr(row, "ontology_curie", None)
        groups = {k: np.asarray(v, dtype=np.int32) for k, v in groups.items()}
        out[modality] = ModalityGroup(modality=modality, groups=groups,
                                      ontology_per_group=onto_map)
        print(f"[meta] {modality}: {len(tbl)} tracks -> {len(groups)} cell lines")
    return out


def aggregate_modality(values_ref: np.ndarray, values_alt: np.ndarray,
                       group: ModalityGroup) -> list[dict]:
    """Collapse (bins, n_tracks) REF/ALT tensors to one row per cell line.

    Kept for the batch_size=1 public-API path (score_one_variant), which still
    returns full per-bin TrackData. The batched path avoids this by using
    `aggregate_modality_from_pertrack` on already-reduced (n_tracks,) arrays.
    """
    if values_ref.ndim != 2 or values_alt.ndim != 2:
        raise ValueError(f"Unexpected shape ref={values_ref.shape} alt={values_alt.shape}")

    ref = values_ref.astype(np.float32, copy=False)
    alt = values_alt.astype(np.float32, copy=False)
    delta = alt - ref
    abs_delta = np.abs(delta)

    ref_max_pt = np.nanmax(ref, axis=0)
    alt_max_pt = np.nanmax(alt, axis=0)
    ref_mean_pt = np.nanmean(ref, axis=0)
    alt_mean_pt = np.nanmean(alt, axis=0)
    absd_max_pt = np.nanmax(abs_delta, axis=0)
    absd_argbin_pt = np.argmax(abs_delta, axis=0)
    signed_at_argbin_pt = delta[absd_argbin_pt, np.arange(delta.shape[1])]

    return aggregate_modality_from_pertrack(
        ref_max_pt=ref_max_pt, alt_max_pt=alt_max_pt,
        ref_mean_pt=ref_mean_pt, alt_mean_pt=alt_mean_pt,
        absd_max_pt=absd_max_pt, absd_argbin_pt=absd_argbin_pt,
        signed_at_argbin_pt=signed_at_argbin_pt, group=group,
    )


def aggregate_modality_from_pertrack(
    *, ref_max_pt: np.ndarray, alt_max_pt: np.ndarray,
    ref_mean_pt: np.ndarray, alt_mean_pt: np.ndarray,
    absd_max_pt: np.ndarray, absd_argbin_pt: np.ndarray,
    signed_at_argbin_pt: np.ndarray, group: ModalityGroup,
) -> list[dict]:
    """Per-cell-line aggregation from pre-computed per-track reductions.

    All inputs are 1-D numpy arrays of shape (num_tracks,). This is the hot
    path called once per (variant, modality) — the expensive bin-axis
    reductions (which produced these arrays) are done ONCE on the GPU by
    the batched scoring path.
    """
    rows: list[dict] = []
    for cell_line, track_idx in group.groups.items():
        if track_idx.size == 0:
            continue
        cell_absd = absd_max_pt[track_idx]
        local_argmax = int(np.argmax(cell_absd))
        best_track_global = int(track_idx[local_argmax])
        rows.append({
            "modality": group.modality,
            "cell_line": cell_line,
            "ontology_curie": group.ontology_per_group.get(cell_line),
            "n_tracks": int(track_idx.size),
            "ref_max": float(np.nanmax(ref_max_pt[track_idx])),
            "alt_max": float(np.nanmax(alt_max_pt[track_idx])),
            "ref_mean": float(np.nanmean(ref_mean_pt[track_idx])),
            "alt_mean": float(np.nanmean(alt_mean_pt[track_idx])),
            "abs_delta_max": float(cell_absd[local_argmax]),
            "signed_delta_at_argmax_abs": float(signed_at_argbin_pt[best_track_global]),
            "argmax_bin": int(absd_argbin_pt[best_track_global]),
            "argmax_track_idx": best_track_global,
        })
    return rows


def _cell_line_for_track(group: ModalityGroup, track_idx: int) -> str:
    for cell_line, indices in group.groups.items():
        if track_idx in indices:
            return cell_line
    return "unknown"


def aggregate_modality_global_max_from_pertrack(
    *, ref_max_pt: np.ndarray, alt_max_pt: np.ndarray,
    ref_mean_pt: np.ndarray, alt_mean_pt: np.ndarray,
    absd_max_pt: np.ndarray, absd_argbin_pt: np.ndarray,
    signed_at_argbin_pt: np.ndarray, group: ModalityGroup,
) -> list[dict]:
    """全模态所有 track 取 |delta| 最大的一条（不按 cell_line 分组）。"""
    if absd_max_pt.size == 0:
        return []
    best_track_global = int(np.argmax(absd_max_pt))
    winning_cl = _cell_line_for_track(group, best_track_global)
    return [{
        "modality": group.modality,
        "cell_line": winning_cl,
        "ontology_curie": group.ontology_per_group.get(winning_cl),
        "n_tracks": int(absd_max_pt.size),
        "ref_max": float(ref_max_pt[best_track_global]),
        "alt_max": float(alt_max_pt[best_track_global]),
        "ref_mean": float(ref_mean_pt[best_track_global]),
        "alt_mean": float(alt_mean_pt[best_track_global]),
        "abs_delta_max": float(absd_max_pt[best_track_global]),
        "signed_delta_at_argmax_abs": float(signed_at_argbin_pt[best_track_global]),
        "argmax_bin": int(absd_argbin_pt[best_track_global]),
        "argmax_track_idx": best_track_global,
        "aggregate_mode": "global_max_track",
    }]


def pick_aggregate_modality_fn(mode: str):
    if mode == "global_max_track":
        return aggregate_modality_global_max_from_pertrack
    return aggregate_modality_from_pertrack


# ---------------------------------------------------------------------------
# Scoring loop
# ---------------------------------------------------------------------------
def build_interval(v: VariantRow, context_length: int):
    from alphagenome.data import genome
    half = context_length // 2
    start = max(0, v.position - half)
    end = start + context_length
    return genome.Interval(chromosome=v.chromosome, start=start, end=end)


def score_one_variant(model, v: VariantRow, context_length: int,
                      modality_groups: dict[str, ModalityGroup]) -> list[dict]:
    from alphagenome.data import genome
    from alphagenome.models import dna_output
    from alphagenome_research.model import dna_model

    requested = [dna_output.OutputType[m] for m in modality_groups.keys()]
    interval = build_interval(v, context_length)
    variant = genome.Variant(
        chromosome=v.chromosome, position=v.position,
        reference_bases=v.reference, alternate_bases=v.alternate,
    )
    out = model.predict_variant(
        interval=interval,
        variant=variant,
        organism=dna_model.Organism.HOMO_SAPIENS,
        requested_outputs=requested,
        ontology_terms=None,
    )

    rows: list[dict] = []
    for output_type in requested:
        ref_track = out.reference.get(output_type)
        alt_track = out.alternate.get(output_type)
        if ref_track is None or alt_track is None:
            continue
        group = modality_groups[output_type.name]
        ref_vals = np.asarray(ref_track.values)
        alt_vals = np.asarray(alt_track.values)
        if ref_vals.ndim == 1:  # (bins,) -> (bins, 1)
            ref_vals = ref_vals[:, None]
            alt_vals = alt_vals[:, None]
        for r in aggregate_modality(ref_vals, alt_vals, group):
            r.update({
                "variant_key": v.key,
                "chromosome": v.chromosome,
                "position": v.position,
                "reference": v.reference,
                "alternate": v.alternate,
                "context_length": int(context_length),
            })
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Batched scoring path (bypasses model.predict_variant by calling the JIT-ed
# internal _predict_variant directly with a B>1 leading dim).
# ---------------------------------------------------------------------------
# 为每个variant构造中心化16kb REF/ALT one-hot及可选注释mask。
def _per_variant_batch_inputs(model, variants: list[VariantRow], context_length: int,
                              *, skip_annotation_masks: bool = True):
    """Build the numpy arrays that go into model._predict_variant for a batch.

    When `skip_annotation_masks=True` (default), the per-variant gene_mask and
    splice_site extraction (both CPU-bound pandas filters over GTF/feather
    files, ~300ms/variant) is skipped and replaced by all-zero masks. This is
    SAFE when none of the requested outputs are `SPLICE_JUNCTIONS`:

    - `_predict_variant` only consumes splice_junction_masks.{splice_sites,
      reference_genes, indel_masks} inside the splice-junction branch
      (dna_model._predict_variant, lines ~223-275). That branch feeds ONLY
      `reference_predictions['splice_sites_junction']` and the alt equivalent.
    - The ATAC / DNASE / RNA_SEQ outputs come directly out of `apply_fn`
      (lines ~211-222) before any mask is applied.
    - For SNVs, `variant_is_indel` is False, so `align_alternate` is a noop.

    Returns:
        ref_batch        : float32 (B, S, 4) one-hot REF
        alt_batch        : float32 (B, S, 4) one-hot ALT
        splice_sites     : bool    (B, S, 5) or None if no extractor available
        ref_gene_mask    : bool    (B, S, 1)
        indel_masks_tree : batched IndelMask (leading B dim)
        strand_mask      : bool    (B,) - True for intervals on the negative strand
        organism_indices : int32   (B,)
        intervals        : list[genome.Interval] (for bookkeeping / reverse mapping)
    """
    from alphagenome.data import genome
    from alphagenome_research.io import genome as genome_io
    from alphagenome_research.model import dna_model
    from alphagenome_research.model.variant_scoring import variant_scoring as vs_lib
    import jax

    organism = dna_model.Organism.HOMO_SAPIENS
    fasta_extractor = model._get_fasta_extractor(organism)
    if skip_annotation_masks:
        gene_mask_extractor = None
        splice_site_extractor = None
        has_splice_extractor = True
    else:
        gene_mask_extractor = model._gene_mask_extractors.get(organism)
        splice_site_extractor = model._splice_site_extractors.get(organism)
        has_splice_extractor = splice_site_extractor is not None

    ref_seqs: list[np.ndarray] = []
    alt_seqs: list[np.ndarray] = []
    gene_masks: list[np.ndarray] = []
    splice_sites_list: list[np.ndarray | None] = []
    indel_masks_list = []
    strands: list[bool] = []
    intervals: list = []

    # Zero-valued placeholders reused across the batch when we skip the
    # CPU-bound extractors. All variants in the batch share the same interval
    # width (context_length), so we can build the masks once.
    if skip_annotation_masks:
        zero_gene_mask = np.zeros((context_length, 1), dtype=bool)
        zero_splice_sites = np.zeros((context_length, 5), dtype=bool)

    for v in variants:
        interval = build_interval(v, context_length)
        gv = genome.Variant(
            chromosome=v.chromosome, position=v.position,
            reference_bases=v.reference, alternate_bases=v.alternate,
        )
        ref_seq, alt_seq = genome_io.extract_variant_sequences(interval, gv, fasta_extractor)
        ref_seqs.append(np.asarray(model._one_hot_encoder.encode(ref_seq)))
        alt_seqs.append(np.asarray(model._one_hot_encoder.encode(alt_seq)))

        if skip_annotation_masks:
            # Placeholders. Model only consumes these for splice-junction
            # outputs, which we never request here.
            gene_masks.append(zero_gene_mask)
            splice_sites_list.append(zero_splice_sites)
        else:
            # Reference gene mask mirrors the single-variant path in dna_model.
            gene_mask = np.ones((interval.width, 1), dtype=bool)
            if gene_mask_extractor is not None:
                mask, _ = gene_mask_extractor.extract(interval, gv)
                if mask.size > 0:
                    gene_mask = mask.max(-1, keepdims=True)
            gene_masks.append(gene_mask)

            ss: np.ndarray | None = None
            if splice_site_extractor is not None:
                ss = splice_site_extractor.extract(interval)
                ss = ss * gene_mask
            splice_sites_list.append(ss)

        indel_masks_list.append(vs_lib.IndelMask.from_variant(gv, interval))
        strands.append(bool(interval.negative_strand))
        intervals.append(interval)

    ref_batch = np.stack(ref_seqs, axis=0).astype(np.float32)
    alt_batch = np.stack(alt_seqs, axis=0).astype(np.float32)
    gene_mask_batch = np.stack(gene_masks, axis=0)
    if has_splice_extractor and all(ss is not None for ss in splice_sites_list):
        splice_sites_batch = np.stack(splice_sites_list, axis=0)
    else:
        splice_sites_batch = None

    # IndelMask has fields of shape (*B, ...) with empty leading batch in the
    # output of from_variant. Stack along axis 0 to add a batch dim of size B.
    indel_masks_batch = jax.tree.map(
        lambda *xs: np.stack(xs, axis=0), *indel_masks_list
    )

    strand_mask = np.asarray(strands, dtype=bool)
    organism_indices = np.full(
        (len(variants),), dna_model.convert_to_organism_index(organism), dtype=np.int32,
    )
    return (ref_batch, alt_batch, splice_sites_batch, gene_mask_batch,
            indel_masks_batch, strand_mask, organism_indices, intervals)


_PERTRACK_REDUCE_FN = None  # lazy-initialised jit-ed reducer
_PERTRACK_REDUCE_LOG2_FN = None


def _get_pertrack_reducer(*, log2_before_delta: bool = False):
    """Return a jit-compiled per-modality-output reducer.

    Input  : ref, alt each shape (B, bins, T) (any float dtype)
    Output : dict of 7 arrays each shape (B, T), dtypes:
             ref_max, alt_max, ref_mean, alt_mean  : float32
             absd_max, signed_at_argbin            : float32
             absd_argbin                            : int32

    If ``log2_before_delta=True``, apply ``log2(x+1)`` to ref/alt along bins
    before computing delta (variant-pair log2 差分任务).

    Doing this on GPU cuts device->host transfer from (B, bins, T) =
    ~37 MB per batch to (B, T) = ~37 KB per batch (1000x less). It also
    moves the bin-axis reductions off the CPU where they were costing
    ~1.7 s per batch of 8.
    """
    global _PERTRACK_REDUCE_FN, _PERTRACK_REDUCE_LOG2_FN
    cached = _PERTRACK_REDUCE_LOG2_FN if log2_before_delta else _PERTRACK_REDUCE_FN
    if cached is not None:
        return cached
    import jax
    import jax.numpy as jnp

    def _reduce(ref, alt):
        ref32 = ref.astype(jnp.float32)
        alt32 = alt.astype(jnp.float32)
        if log2_before_delta:
            ref32 = jnp.log2(ref32 + 1.0)
            alt32 = jnp.log2(alt32 + 1.0)
        delta = alt32 - ref32
        abs_delta = jnp.abs(delta)
        # Bin axis = axis 1 (shape is (B, bins, T))
        ref_max = jnp.max(ref32, axis=1)
        alt_max = jnp.max(alt32, axis=1)
        ref_mean = jnp.mean(ref32, axis=1)
        alt_mean = jnp.mean(alt32, axis=1)
        absd_max = jnp.max(abs_delta, axis=1)
        absd_argbin = jnp.argmax(abs_delta, axis=1).astype(jnp.int32)
        # signed delta at argmax bin per (batch, track):
        #   gather delta[b, absd_argbin[b, t], t]
        B, bins, T = delta.shape
        bs = jnp.arange(B)[:, None]
        ts = jnp.arange(T)[None, :]
        signed_at_argbin = delta[bs, absd_argbin, ts]
        return {
            "ref_max": ref_max, "alt_max": alt_max,
            "ref_mean": ref_mean, "alt_mean": alt_mean,
            "absd_max": absd_max, "absd_argbin": absd_argbin,
            "signed_at_argbin": signed_at_argbin,
        }

    fn = jax.jit(_reduce)
    if log2_before_delta:
        _PERTRACK_REDUCE_LOG2_FN = fn
    else:
        _PERTRACK_REDUCE_FN = fn
    return fn


def score_batch_of_variants(model, variants: list[VariantRow], context_length: int,
                            modality_groups: dict[str, ModalityGroup],
                            *, skip_annotation_masks: bool = True) -> list[list[dict]]:
    """Run B variants through a single _predict_variant forward pass.

    Returns a list of per-variant row-lists (same shape as calling
    score_one_variant in a loop, so downstream code is identical).
    """
    import jax
    from alphagenome.models import dna_output
    from alphagenome_research.model import dna_model
    from alphagenome_research.model.metadata import metadata as metadata_lib

    if len(variants) == 0:
        return []

    organism = dna_model.Organism.HOMO_SAPIENS
    # Preserve caller-specified ordering of modalities (and thus the order the
    # model returns outputs) via a stable tuple.
    requested_output_types = tuple(
        dna_output.OutputType[m] for m in modality_groups.keys()
    )

    (ref_batch, alt_batch, splice_sites_batch, gene_mask_batch,
     indel_masks_batch, strand_mask, organism_indices, _intervals) = (
        _per_variant_batch_inputs(model, variants, context_length,
                                  skip_annotation_masks=skip_annotation_masks)
    )

    splice_junction_masks = dna_model._SpliceJunctionVariantMasks(
        splice_sites=splice_sites_batch,
        reference_genes=gene_mask_batch,
        indel_masks=indel_masks_batch,
    )

    track_metadata = model._metadata[organism]
    track_masks = metadata_lib.create_track_masks(
        track_metadata,
        requested_outputs=requested_output_types,
        requested_ontologies=None,
    )

    reducer = _get_pertrack_reducer()

    with model._device_context as device, jax.transfer_guard("disallow"):
        ref_pred, alt_pred = model._predict_variant(
            model._params,
            model._state,
            jax.device_put(ref_batch, device),
            jax.device_put(alt_batch, device),
            jax.device_put(splice_junction_masks, device),
            jax.device_put(organism_indices, device),
            requested_outputs=requested_output_types,
            negative_strand_mask=jax.device_put(strand_mask, device),
            strand_reindexing=jax.device_put(track_metadata.strand_reindexing, device),
        )
        # Apply track-type masks in the same way the single-variant path does.
        ref_pred, alt_pred = dna_model._filter_variant_predictions(
            ref_pred, alt_pred, track_masks=jax.device_put(track_masks, device),
        )

        # --- GPU-side per-track reductions ----------------------------------
        # For each requested modality, reduce the (B, bins, tracks) tensors
        # along the bins axis ON THE GPU. Only the resulting (B, tracks)
        # summaries cross the PCIe bus back to the host.
        per_modality_host: dict = {}
        for ot in requested_output_types:
            rp = ref_pred.get(ot)
            ap = alt_pred.get(ot)
            if rp is None or ap is None:
                continue
            # SPLICE_JUNCTIONS returns a dict, not a tensor — we never
            # request it here, so skip if shapes don't match simple modalities.
            # After _filter_variant_predictions, ATAC/DNASE/RNA_SEQ are
            # still raw tensors of shape (B, bins, T_filtered).
            if not hasattr(rp, "ndim"):
                continue
            if rp.ndim == 2:
                rp = rp[:, :, None]
                ap = ap[:, :, None]
            reduced_gpu = reducer(rp, ap)
            # Block and transfer the tiny (B, T) arrays to host.
            reduced_host = {k: np.asarray(jax.device_get(v)) for k, v in reduced_gpu.items()}
            per_modality_host[ot] = reduced_host

    per_variant_rows: list[list[dict]] = []
    for b, v in enumerate(variants):
        rows: list[dict] = []
        for ot in requested_output_types:
            reduced = per_modality_host.get(ot)
            if reduced is None:
                continue
            group = modality_groups[ot.name]
            for r in aggregate_modality_from_pertrack(
                ref_max_pt=reduced["ref_max"][b],
                alt_max_pt=reduced["alt_max"][b],
                ref_mean_pt=reduced["ref_mean"][b],
                alt_mean_pt=reduced["alt_mean"][b],
                absd_max_pt=reduced["absd_max"][b],
                absd_argbin_pt=reduced["absd_argbin"][b],
                signed_at_argbin_pt=reduced["signed_at_argbin"][b],
                group=group,
            ):
                r.update({
                    "variant_key": v.key,
                    "chromosome": v.chromosome,
                    "position": v.position,
                    "reference": v.reference,
                    "alternate": v.alternate,
                    "context_length": int(context_length),
                })
                rows.append(r)
        per_variant_rows.append(rows)
    return per_variant_rows


# ---------------------------------------------------------------------------
# Shard management for resume support
# ---------------------------------------------------------------------------
def shard_dir(output_dir: Path) -> Path:
    d = output_dir / "shards_long"
    d.mkdir(parents=True, exist_ok=True)
    return d


def scored_keys_index_path(output_dir: Path) -> Path:
    return output_dir / "scored_variant_keys.parquet"


def already_scored_keys(output_dir: Path) -> set[str]:
    """优先读取sidecar索引，并回退扫描shard以恢复已完成variant集合。"""
    """Return variant_keys present in shards. Prefer sidecar index if present."""
    index_path = scored_keys_index_path(output_dir)
    if index_path.exists():
        try:
            tbl = pq.read_table(index_path, columns=["variant_key"])
            keys = set(tbl.column("variant_key").to_pylist())
            print(f"[resume] loaded {len(keys)} keys from {index_path.name}")
            return keys
        except Exception as e:
            print(f"[warn] could not read {index_path.name}: {e}; scanning shards")

    d = shard_dir(output_dir)
    seen: set[str] = set()
    for shard in sorted(d.glob("shard_*.parquet")):
        if shard.stat().st_size < 50:
            continue
        try:
            tbl = pq.read_table(shard, columns=["variant_key"])
            seen.update(tbl.column("variant_key").to_pylist())
        except Exception as e:
            print(f"[warn] could not read {shard.name}: {e}")
    return seen


def append_scored_keys_index(output_dir: Path, keys: list[str]) -> None:
    """Append newly scored variant_keys to the resume sidecar index."""
    if not keys:
        return
    index_path = scored_keys_index_path(output_dir)
    new_tbl = pa.table({"variant_key": list(dict.fromkeys(keys))})
    if index_path.exists():
        old = pq.read_table(index_path)
        merged = pa.concat_tables([old, new_tbl])
        # dedupe while preserving order
        df = merged.to_pandas().drop_duplicates(subset=["variant_key"], keep="first")
        merged = pa.Table.from_pandas(df[["variant_key"]], preserve_index=False)
    else:
        merged = new_tbl
    pq.write_table(merged, index_path, compression="zstd")


def write_shard(rows: list[dict], output_dir: Path, shard_idx: int) -> Path:
    """把当前缓存写成独立Parquet shard，控制中断后的重算粒度。"""
    path = shard_dir(output_dir) / f"shard_{shard_idx:06d}.parquet"
    df = pd.DataFrame(rows)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="zstd")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def score_all(model, unique_variants: pd.DataFrame,
              modality_groups: dict[str, ModalityGroup], args) -> Path:
    done_keys = already_scored_keys(args.output_dir) if args.resume else set()
    if done_keys:
        print(f"[resume] {len(done_keys)} variants already scored - will skip them")

    todo = unique_variants[~unique_variants["variant_key"].isin(done_keys)].reset_index(drop=True)
    if args.limit is not None:
        todo = todo.head(args.limit)
    print(f"[score] will score {len(todo)} variants (context_length={args.context_length})")

    # JIT + compile cost is paid on the first call, cached thereafter.
    shard_rows: list[dict] = []
    shard_variants_in_buffer = 0
    # Pick a shard index that does not collide with existing shards.
    existing = sorted(shard_dir(args.output_dir).glob("shard_*.parquet"))
    shard_index = 0
    for p in existing:
        try:
            shard_index = max(shard_index, int(p.stem.split("_")[-1]) + 1)
        except ValueError:
            pass
    scored = 0
    errors: list[dict] = []
    t_total = time.perf_counter()

    # graceful shutdown: flush partial shard on SIGTERM/SIGINT
    _interrupted = {"flag": False}
    def _handler(signum, frame):  # pragma: no cover
        _interrupted["flag"] = True
        print(f"[signal] received {signum}; will flush after current variant")
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    batch_size = max(1, int(args.batch_size))
    skip_masks = bool(getattr(args, "skip_annotation_masks", True))
    if batch_size > 1:
        print(f"[score] using batched path (_predict_variant) with batch_size={batch_size}"
              f" skip_annotation_masks={skip_masks}")
    elif skip_masks:
        # Even at B=1 we want to benefit from skipping the annotation extraction,
        # so we still take the batched path with B=1.
        print(f"[score] using batched path with batch_size=1 skip_annotation_masks=True "
              f"(set --no-skip-annotation-masks to use the public API path)")

    # Materialize the full variant list up front so we can slice by batch.
    all_variants: list[VariantRow] = [
        VariantRow(chromosome=str(r["chromosome"]), position=int(r["position"]),
                   reference=str(r["reference"]), alternate=str(r["alternate"]))
        for _, r in todo.iterrows()
    ]
    total = len(all_variants)

    def _process_batch(batch: list[VariantRow]) -> list[list[dict]]:
        """Score a batch, falling back to per-variant on GPU errors.

        Returns a list of per-variant row-lists (same length as `batch`).
        Records per-variant errors into the outer `errors` list.
        """
        # When user forces full-annotation parity at B=1, route through the
        # public predict_variant API (which rebuilds masks the "official" way).
        if batch_size == 1 and not skip_masks:
            out: list[list[dict]] = []
            for v in batch:
                try:
                    out.append(score_one_variant(model, v, args.context_length, modality_groups))
                except Exception as exc:  # pragma: no cover
                    errors.append({"variant_key": v.key,
                                   "error_type": type(exc).__name__,
                                   "error_message": str(exc)})
                    print(f"[err] {v.key}: {type(exc).__name__}: {exc}", flush=True)
                    out.append([])
            return out
        # Batched path (any B>=1). If the whole batch fails (e.g. OOM or one
        # bad variant), fall back to per-variant scoring via public API so we
        # don't lose B-1 otherwise-healthy variants.
        try:
            return score_batch_of_variants(model, batch, args.context_length,
                                           modality_groups,
                                           skip_annotation_masks=skip_masks)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] batch of {len(batch)} failed ({type(exc).__name__}: {exc}); "
                  f"falling back to per-variant scoring for this batch", flush=True)
            out = []
            for v in batch:
                try:
                    out.append(score_one_variant(model, v, args.context_length, modality_groups))
                except Exception as exc2:
                    errors.append({"variant_key": v.key,
                                   "error_type": type(exc2).__name__,
                                   "error_message": str(exc2)})
                    print(f"[err] {v.key}: {type(exc2).__name__}: {exc2}", flush=True)
                    out.append([])
            return out

    cursor = 0
    while cursor < total:
        batch = all_variants[cursor:cursor + batch_size]
        t0 = time.perf_counter()
        per_variant_rows = _process_batch(batch)
        dt = time.perf_counter() - t0
        # Each element of per_variant_rows maps 1:1 to a variant in batch.
        for v, rows in zip(batch, per_variant_rows):
            if rows:
                shard_rows.extend(rows)
                scored += 1
                shard_variants_in_buffer += 1
        cursor += len(batch)

        if cursor <= batch_size * 3 or (cursor // batch_size) % 10 == 0:
            elapsed = time.perf_counter() - t_total
            rate = scored / elapsed if elapsed > 0 else 0.0
            remaining = (total - scored) / rate if rate > 0 else float("nan")
            per_variant_ms = dt * 1000 / max(1, len(batch))
            print(f"[score] {scored}/{total} batch={len(batch)} "
                  f"batch_ms={dt*1000:.0f} per_variant_ms={per_variant_ms:.0f} "
                  f"rate={rate:.2f}/s eta={remaining/60:.1f}min", flush=True)

        if shard_variants_in_buffer >= max(1, args.shard_size) or _interrupted["flag"]:
            path = write_shard(shard_rows, args.output_dir, shard_index)
            print(f"[shard] wrote {len(shard_rows)} rows (variants={shard_variants_in_buffer}) -> {path.name}",
                  flush=True)
            shard_rows = []
            shard_variants_in_buffer = 0
            shard_index += 1
            if _interrupted["flag"]:
                break
    if shard_rows:
        path = write_shard(shard_rows, args.output_dir, shard_index)
        print(f"[shard] wrote {len(shard_rows)} rows -> {path.name}")

    if errors:
        err_df = pd.DataFrame(errors)
        err_path = args.output_dir / "scoring_errors.csv"
        (err_df if not err_path.exists()
         else pd.concat([pd.read_csv(err_path), err_df], ignore_index=True)).to_csv(err_path, index=False)
        print(f"[err] {len(errors)} errors -> {err_path.name}")
    total_time = time.perf_counter() - t_total
    print(f"[score] finished {scored} variants in {total_time/60:.2f}min")
    return shard_dir(args.output_dir)


def collect_long_table(output_dir: Path) -> pd.DataFrame:
    d = shard_dir(output_dir)
    shards = sorted(d.glob("shard_*.parquet"))
    if not shards:
        return pd.DataFrame()
    tables = [pq.read_table(p) for p in shards]
    long_df = pa.concat_tables(tables).to_pandas()
    return long_df


def pivot_wide_per_modality(long_df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    """Save variant x cell_line wide parquet files, one per modality."""
    if long_df.empty:
        return
    wide_dir = output_dir / "wide_by_modality"
    wide_dir.mkdir(parents=True, exist_ok=True)
    for modality, grp in long_df.groupby("modality"):
        wide = grp.pivot_table(index="variant_key", columns="cell_line",
                               values=metric, aggfunc="max")
        wide.columns = [f"{modality}__{c}__{metric}" for c in wide.columns]
        wide = wide.reset_index()
        out_path = wide_dir / f"{modality}_{metric}.parquet"
        wide.to_parquet(out_path, compression="zstd", index=False)
        print(f"[wide] {modality}: {wide.shape} -> {out_path.name}")


def summarize_per_modality(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per variant with modality-level summary columns."""
    if long_df.empty:
        return pd.DataFrame(columns=["variant_key"])
    parts = []
    for modality, grp in long_df.groupby("modality"):
        grp = grp.copy()
        # argmax over cell_lines: get the cell line with largest abs_delta_max
        idx = grp.groupby("variant_key")["abs_delta_max"].idxmax()
        best = grp.loc[idx, ["variant_key", "cell_line", "ontology_curie",
                              "ref_max", "alt_max", "abs_delta_max",
                              "signed_delta_at_argmax_abs"]].copy()
        best = best.rename(columns={
            "cell_line": f"{modality}_best_cell_line",
            "ontology_curie": f"{modality}_best_ontology",
            "ref_max": f"{modality}_best_ref_max",
            "alt_max": f"{modality}_best_alt_max",
            "abs_delta_max": f"{modality}_max_abs_delta",
            "signed_delta_at_argmax_abs": f"{modality}_signed_delta_at_best",
        })
        # also mean / median abs delta across cell lines
        agg = grp.groupby("variant_key").agg(
            **{
                f"{modality}_mean_abs_delta": ("abs_delta_max", "mean"),
                f"{modality}_median_abs_delta": ("abs_delta_max", "median"),
                f"{modality}_n_cell_lines": ("cell_line", "nunique"),
            }
        ).reset_index()
        parts.append(best.merge(agg, on="variant_key", how="outer"))
    summary = parts[0]
    for more in parts[1:]:
        summary = summary.merge(more, on="variant_key", how="outer")
    return summary


def join_back_to_workbook(original_df: pd.DataFrame, summary_df: pd.DataFrame,
                          args) -> Path:
    joined = original_df.merge(summary_df, left_on=args.variant_col,
                               right_on="variant_key", how="left")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.joined_format == "csv":
        path = out_dir / "workbook_with_alphagenome.csv"
        joined.to_csv(path, index=False)
    elif args.joined_format == "parquet":
        path = out_dir / "workbook_with_alphagenome.parquet"
        joined.to_parquet(path, compression="zstd", index=False)
    else:  # xlsx
        path = out_dir / "workbook_with_alphagenome.xlsx"
        # Excel rows limit is 1 048 576 - warn if exceeded.
        if len(joined) > 1_048_000:
            print(f"[warn] xlsx near row limit ({len(joined)}); consider --joined-format csv")
        joined.to_excel(path, index=False)
    print(f"[join] wrote {joined.shape} -> {path}")
    return path


def main() -> None:
    """执行旧cell-line入口；GTEx 11模态入口由score_gtex_11modal.py负责。"""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Persist the CLI for reproducibility
    (args.output_dir / "run_args.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    # 1. Parse input
    df = load_workbook(args)
    uniq = deduplicate_variants(df, args)
    uniq_path = args.output_dir / "unique_variants.parquet"
    uniq.to_parquet(uniq_path, compression="zstd", index=False)
    print(f"[dedup] saved {uniq_path}")

    # 2. Load model + cell-line groupings
    model = load_model(args)
    modality_groups = build_groupings(model, args.modalities)

    # 3. Score
    score_all(model, uniq, modality_groups, args)

    if args.no_final_merge:
        print("[done] scoring complete; skipping merge (--no-final-merge).")
        return

    # 4. Collect long, write wide, join back
    long_df = collect_long_table(args.output_dir)
    if long_df.empty:
        print("[merge] no rows in shards; aborting merge.")
        return
    long_path = args.output_dir / "alphagenome_cellline_long.parquet"
    long_df.to_parquet(long_path, compression="zstd", index=False)
    print(f"[merge] long table {long_df.shape} -> {long_path.name}")

    if args.save_wide_parquet:
        pivot_wide_per_modality(long_df, metric="abs_delta_max", output_dir=args.output_dir)

    summary_df = summarize_per_modality(long_df)
    summary_path = args.output_dir / "alphagenome_cellline_summary.parquet"
    summary_df.to_parquet(summary_path, compression="zstd", index=False)
    print(f"[merge] summary table {summary_df.shape} -> {summary_path.name}")

    join_back_to_workbook(df, summary_df, args)


if __name__ == "__main__":
    main()
