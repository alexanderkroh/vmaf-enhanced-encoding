#!/usr/bin/env python3
"""
ladder_generator.py — Content-adaptive ABR bitrate ladder generator.

Generates a per-title or per-shot adaptive bitrate ladder using a trained
RDCurveModel from video_quality_pipeline.py. Produces HLS and DASH manifests
alongside a ladder definition JSON.

Architecture
------------
  Oracle data (records.jsonl)
      └── RDCurveModel (trained)
              └── LadderGenerator
                      ├── Feature extraction per chunk (signalstats)
                      ├── RD surface prediction across resolutions × CRF values
                      ├── Convex hull selection → Pareto-optimal rungs
                      ├── Per-shot CRF assignments per rung
                      └── Manifest generation (HLS .m3u8, DASH .mpd)

Usage
-----
  # Generate ladder from a trained model and source clip
  python3 ladder_generator.py source.mov \
      --model model.json \
      --output-dir ./ladder_output

  # With explicit VMAF targets (overrides convex hull auto-selection)
  python3 ladder_generator.py source.mov \
      --model model.json \
      --vmaf-targets 75 85 92 95 \
      --output-dir ./ladder_output

  # Custom resolution set
  python3 ladder_generator.py source.mov \
      --model model.json \
      --resolutions 360 540 720 1080 \
      --output-dir ./ladder_output

  # Encode the ladder (runs all renditions after generation)
  python3 ladder_generator.py source.mov \
      --model model.json \
      --encode \
      --output-dir ./ladder_output
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ── Optional dependencies ─────────────────────────────────────────────────────
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ── Import pipeline components ────────────────────────────────────────────────
try:
    from video_quality_pipeline import (
        RDCurveModel,
        EvalConfig,
        ChunkInfo,
        LaneSpec,
        DynamicOptimizerConfig,
        probe_color_metadata,
        extract_shot_features,
        build_chunks,
        validate_source,
        encode_multilane_with_vmaf,
    )
    HAS_PIPELINE = True
except ImportError as e:
    log.error(
        "video_quality_pipeline.py not found in the same directory. "
        "Place ladder_generator.py alongside video_quality_pipeline.py. (%s)", e
    )
    HAS_PIPELINE = False


# ──────────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────────

# Standard resolution profiles — width × height pairs.
# Heights are the canonical label (720p, 1080p etc.).
RESOLUTION_PROFILES = {
    240:  (426,  240),
    360:  (640,  360),
    480:  (854,  480),
    540:  (960,  540),
    720:  (1280, 720),
    1080: (1920, 1080),
    1440: (2560, 1440),
    2160: (3840, 2160),
}

# Default VMAF targets that define the ladder rungs.
# These correspond roughly to: low / medium / good / excellent / premium.
DEFAULT_VMAF_TARGETS = [65.0, 75.0, 85.0, 92.0, 95.0]

# Default resolutions to include in the ladder surface scan.
DEFAULT_RESOLUTIONS = [360, 540, 720, 1080]

# CRF range for the surface prediction scan.
CRF_SCAN_RANGE = (16, 34)   # integer CRF values evaluated during surface scan
CRF_SCAN_STEP  = 1


@dataclass
class LadderRung:
    """
    A single rendition in the content-adaptive bitrate ladder.

    Each rung targets a specific VMAF quality level at a specific resolution.
    The per_chunk_crfs dict records the per-shot CRF assignments — different
    shots within the same rung use different CRF values to maintain consistent
    perceptual quality rather than constant bitrate.
    """
    resolution_height: int          # e.g. 720
    resolution_width:  int          # e.g. 1280
    vmaf_target:       float        # quality target this rung was designed for
    predicted_vmaf:    float        # mean predicted VMAF across the title
    predicted_bitrate_kbps: float   # mean predicted bitrate (harmonic mean weighted by duration)
    per_chunk_crfs:    dict = field(default_factory=dict)  # {chunk_index: crf}

    @property
    def label(self) -> str:
        return f"{self.resolution_height}p_{int(self.predicted_bitrate_kbps)}kbps"

    @property
    def resolution_str(self) -> str:
        return f"{self.resolution_width}x{self.resolution_height}"


@dataclass
class LadderResult:
    """Output of LadderGenerator.generate()."""
    source_path:   str
    rungs:         list   # list[LadderRung], sorted by bitrate ascending
    source_width:  int    = 0
    source_height: int    = 0
    source_fps:    float  = 0.0
    source_codec:  str    = ""
    duration_s:    float  = 0.0
    chunk_count:   int    = 0
    generation_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "source":          self.source_path,
            "source_codec":    self.source_codec,
            "resolution":      f"{self.source_width}x{self.source_height}",
            "fps":             self.source_fps,
            "duration_s":      round(self.duration_s, 3),
            "chunk_count":     self.chunk_count,
            "generation_time_s": round(self.generation_time_s, 2),
            "rungs": [
                {
                    "label":               r.label,
                    "resolution":          r.resolution_str,
                    "vmaf_target":         r.vmaf_target,
                    "predicted_vmaf":      round(r.predicted_vmaf, 2),
                    "predicted_bitrate_kbps": round(r.predicted_bitrate_kbps, 1),
                    "per_chunk_crfs":      r.per_chunk_crfs,
                }
                for r in self.rungs
            ],
        }


# ──────────────────────────────────────────────────────────────────────────────
# RD Surface — predicted quality at each (resolution, CRF) operating point
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SurfacePoint:
    """One operating point on the RD surface: (resolution, CRF) → (VMAF, bitrate)."""
    resolution_height: int
    resolution_width:  int
    crf:               int
    vmaf:              float   # mean predicted VMAF across all chunks
    bitrate_kbps:      float   # mean predicted bitrate across all chunks
    per_chunk_crfs:    dict = field(default_factory=dict)  # {chunk_index: crf}


def _predict_surface(
    chunks:         list,       # list[ChunkInfo]
    features:       list,       # list[dict] parallel to chunks
    model:          "RDCurveModel",
    resolutions:    list,       # list of height ints
    crf_range:      tuple = CRF_SCAN_RANGE,
    crf_step:       int   = CRF_SCAN_STEP,
    source_height:  int   = 1080,
) -> list:
    """
    Predict the full (resolution, CRF) → (VMAF, bitrate) surface for a title.

    Resolution scaling affects both quality and bitrate. The model was trained
    on a specific resolution (typically the source resolution). For other
    resolutions we apply a content-aware scaling heuristic:

      VMAF penalty  — quality drops at lower resolutions because upscaling for
                      display reintroduces blur. Empirically modelled as a
                      function of the resolution ratio and content SI:
                        Δvmaf ≈ -k_vmaf · (1 - scale_ratio²) · (1 + SI_norm)

      Bitrate credit — lower resolution requires fewer bits for equivalent
                       block prediction quality:
                        rate_scaled ≈ rate_native · scale_ratio^1.5

    These are heuristic approximations. The correct approach is to train
    resolution-specific RD models, which requires oracle data at each resolution.
    The scaling heuristic is accurate enough for ladder rung selection but should
    be validated against actual encodes for production use.

    Returns a list of SurfacePoint objects.
    """
    if not HAS_NUMPY:
        raise RuntimeError("numpy required for surface prediction. pip install numpy")

    crfs = list(range(crf_range[0], crf_range[1] + 1, crf_step))
    surface: list[SurfacePoint] = []

    total_duration = sum(c.duration for c in chunks)
    if total_duration == 0:
        return surface

    for res_h in resolutions:
        res_w, res_h_actual = RESOLUTION_PROFILES.get(res_h, (res_h * 16 // 9, res_h))
        scale = res_h_actual / source_height if source_height > 0 else 1.0

        for crf in crfs:
            chunk_vmaf    = []
            chunk_rates   = []
            chunk_crf_map = {}

            for chunk, feats in zip(chunks, features):
                # ── Predict at source resolution ──────────────────────────────
                vmaf_native = model.predict_vmaf(feats, crf)
                rate_native = model.predict_rate_kbps(feats, crf)

                if vmaf_native is None or rate_native is None:
                    continue

                # ── Apply resolution scaling heuristics ───────────────────────
                si_norm = min(1.0, (feats.get("si") or 0) / 100.0)

                if scale < 1.0:
                    # Downscale: quality penalty from upscaling for display
                    vmaf_delta = -12.0 * (1.0 - scale ** 2) * (0.5 + si_norm)
                    vmaf_scaled = max(0.0, min(100.0, vmaf_native + vmaf_delta))
                    # Bitrate credit from lower resolution
                    rate_scaled = rate_native * (scale ** 1.5)
                else:
                    # Source resolution or upscale — no benefit from encoding higher
                    vmaf_scaled = vmaf_native
                    rate_scaled = rate_native

                chunk_vmaf.append(vmaf_scaled * chunk.duration)
                chunk_rates.append(rate_scaled * chunk.duration)
                chunk_crf_map[chunk.index] = crf

            if not chunk_vmaf:
                continue

            mean_vmaf    = sum(chunk_vmaf)  / total_duration
            mean_bitrate = sum(chunk_rates) / total_duration

            surface.append(SurfacePoint(
                resolution_height = res_h_actual,
                resolution_width  = res_w,
                crf               = crf,
                vmaf              = mean_vmaf,
                bitrate_kbps      = mean_bitrate,
                per_chunk_crfs    = chunk_crf_map,
            ))

    return surface


# ──────────────────────────────────────────────────────────────────────────────
# Convex Hull
# ──────────────────────────────────────────────────────────────────────────────

def _convex_hull_rd(surface: list) -> list:
    """
    Find the Pareto-optimal (bitrate, VMAF) frontier from the RD surface.

    A point is on the convex hull if no other point has both higher VMAF
    AND lower bitrate. This is the set of efficient operating points —
    the only points worth considering for ladder rungs.

    Returns surface points on the hull, sorted by bitrate ascending.
    """
    if not surface:
        return []

    # Sort by bitrate ascending
    pts = sorted(surface, key=lambda p: p.bitrate_kbps)

    hull = []
    max_vmaf_seen = -1.0
    for pt in pts:
        if pt.vmaf > max_vmaf_seen:
            hull.append(pt)
            max_vmaf_seen = pt.vmaf

    return hull


def _select_rungs_from_hull(
    hull:         list,         # convex hull points sorted by bitrate
    vmaf_targets: list,         # list of float VMAF targets
) -> list:
    """
    Select one hull point per VMAF target — the point on the hull that
    most closely achieves each target quality level.

    Returns a list of SurfacePoint, one per target, deduplicated.
    """
    if not hull:
        return []

    selected = []
    seen_labels = set()

    for target in sorted(vmaf_targets):
        # Find the hull point whose VMAF is closest to this target
        # but at least the target (prefer to meet-or-exceed)
        candidates = [p for p in hull if p.vmaf >= target]
        if candidates:
            # Lowest bitrate point that meets the target
            best = min(candidates, key=lambda p: p.bitrate_kbps)
        else:
            # Target is unachievable — use the best available point
            best = max(hull, key=lambda p: p.vmaf)
            log.warning(
                "VMAF target %.1f not achievable (max predicted = %.1f). "
                "Using best available rung.", target, best.vmaf
            )

        label = f"{best.resolution_height}p_{int(best.bitrate_kbps)}kbps"
        if label not in seen_labels:
            seen_labels.add(label)
            selected.append((target, best))

    return selected


# ──────────────────────────────────────────────────────────────────────────────
# Per-shot CRF assignments per rung
# ──────────────────────────────────────────────────────────────────────────────

def _per_shot_crfs_for_rung(
    chunks:       list,
    features:     list,
    model:        "RDCurveModel",
    vmaf_target:  float,
    resolution_height: int,
    source_height: int,
    crf_range:    tuple = (12, 45),
) -> dict:
    """
    For each chunk, find the CRF that meets vmaf_target at this resolution.

    This is the per-shot optimization within a rung — different shots get
    different CRF values to maintain consistent perceptual quality rather
    than constant bitrate. Easy shots get higher CRF (fewer bits), hard
    shots get lower CRF (more bits).

    Returns {chunk_index: crf}.
    """
    scale = resolution_height / source_height if source_height > 0 else 1.0

    # Adjust target VMAF upward to compensate for resolution downscale penalty,
    # so the native-resolution prediction hits the display-quality target.
    if scale < 1.0:
        si_values = [f.get("si") or 0 for f in features]
        mean_si_norm = min(1.0, (sum(si_values) / len(si_values)) / 100.0) if si_values else 0.5
        vmaf_upward_adj = 12.0 * (1.0 - scale ** 2) * (0.5 + mean_si_norm)
        adjusted_target = min(100.0, vmaf_target + vmaf_upward_adj)
    else:
        adjusted_target = vmaf_target

    assignments = {}
    for chunk, feats in zip(chunks, features):
        crf = model.predict_crf(feats, adjusted_target, crf_range=crf_range)
        if crf is None:
            crf = crf_range[1]  # fallback: highest CRF in range
        assignments[chunk.index] = crf

    return assignments


# ──────────────────────────────────────────────────────────────────────────────
# Ladder Generator
# ──────────────────────────────────────────────────────────────────────────────

class LadderGenerator:
    """
    Generates a content-adaptive ABR bitrate ladder from a trained RDCurveModel.

    The ladder is computed entirely from model predictions — no encoding is
    required for generation. The optional --encode flag runs the actual encodes
    after the ladder is defined.

    Workflow:
      1. Source validation and metadata probe
      2. Shot boundary detection (PySceneDetect or FFmpeg)
      3. Per-chunk content feature extraction (signalstats)
      4. RD surface prediction across resolutions × CRF values
      5. Convex hull selection — Pareto-optimal operating points
      6. Rung selection — one point per VMAF target
      7. Per-shot CRF assignment per rung
      8. Manifest generation (HLS + DASH)
      9. Optional: encode all renditions
    """

    def __init__(self, model: "RDCurveModel"):
        self.model = model

    def generate(
        self,
        source_path:   str,
        output_dir:    str,
        resolutions:   list  = None,
        vmaf_targets:  list  = None,
        codec:         str   = "libx264",
        preset:        str   = "medium",
        ffmpeg_path:   str   = "ffmpeg",
        ffprobe_path:  str   = "ffprobe",
        scene_threshold: float = 27.0,
        detector:      str   = "pyscenedetect",
        min_chunk_s:   float = 2.0,
    ) -> LadderResult:
        """
        Generate a content-adaptive ladder for source_path.

        Parameters
        ----------
        source_path   : Path to the source video (high-quality master).
        output_dir    : Directory for ladder JSON and manifests.
        resolutions   : List of height values e.g. [360, 540, 720, 1080].
                        Defaults to DEFAULT_RESOLUTIONS.
        vmaf_targets  : VMAF quality levels for ladder rungs.
                        Defaults to DEFAULT_VMAF_TARGETS.
        codec         : Encoding codec for manifest generation.
        preset        : Encoder preset.
        ffmpeg_path   : Path to ffmpeg binary.
        ffprobe_path  : Path to ffprobe binary.
        scene_threshold: PySceneDetect ContentDetector threshold.
        detector      : 'pyscenedetect' or 'ffmpeg'.
        min_chunk_s   : Minimum chunk duration in seconds.

        Returns a LadderResult with rungs sorted by bitrate ascending.
        """
        t_start = time.time()
        resolutions  = resolutions  or DEFAULT_RESOLUTIONS
        vmaf_targets = vmaf_targets or DEFAULT_VMAF_TARGETS
        os.makedirs(output_dir, exist_ok=True)

        # ── 1. Source validation ──────────────────────────────────────────────
        log.info("=== Ladder Generator ===")
        log.info("Source    : %s", source_path)
        log.info("Resolutions: %s", resolutions)
        log.info("VMAF targets: %s", vmaf_targets)

        validation = validate_source(source_path, ffprobe_path=ffprobe_path)
        validation.print_report()
        if not validation.ok:
            raise ValueError(f"Source validation failed: {validation.errors}")

        # ── 2. Probe source metadata ──────────────────────────────────────────
        cfg = EvalConfig(reference=source_path)
        cfg.ffmpeg_path  = ffmpeg_path
        cfg.ffprobe_path = ffprobe_path
        cfg.codec        = codec
        cfg.preset       = preset
        cfg.output_dir   = output_dir

        cfg.color_meta = probe_color_metadata(cfg)
        meta = cfg.color_meta or {}
        source_height = int(meta.get("height") or 1080)
        source_width  = int(meta.get("width")  or 1920)
        source_fps    = float(meta.get("fps")   or 24.0)
        duration_s    = float(meta.get("total_frames") or 0) / source_fps if source_fps else 0.0
        source_codec  = meta.get("source_codec") or "unknown"

        log.info(
            "Source metadata: %dx%d  %.3g fps  %.1fs  codec=%s",
            source_width, source_height, source_fps, duration_s, source_codec,
        )

        # Filter out resolutions larger than the source — no upscale benefit
        resolutions = [r for r in resolutions if r <= source_height]
        if not resolutions:
            resolutions = [source_height]
            log.warning("All requested resolutions exceed source height %dpx — "
                        "using source resolution only.", source_height)

        # ── 3. Shot boundary detection + chunk extraction ─────────────────────
        log.info("=== Shot Detection ===")
        do_cfg = DynamicOptimizerConfig(
            scene_threshold      = scene_threshold,
            min_chunk_duration   = min_chunk_s,
            detector             = detector,
            chunk_dir            = os.path.join(output_dir, "chunks"),
        )
        chunks = build_chunks(cfg, do_cfg)
        log.info("  %d chunks detected", len(chunks))

        # ── 4. Per-chunk feature extraction ───────────────────────────────────
        log.info("=== Feature Extraction (%d chunks) ===", len(chunks))
        chunk_features = []
        for i, chunk in enumerate(chunks):
            feats = extract_shot_features(chunk, cfg)
            chunk_features.append(feats)
            if (i + 1) % 10 == 0:
                log.info("  Extracted features for %d / %d chunks", i + 1, len(chunks))

        log.info("  Feature extraction complete")

        # ── 5. RD surface prediction ──────────────────────────────────────────
        log.info("=== RD Surface Prediction ===")
        log.info(
            "  Scanning %d resolutions × %d CRF values × %d chunks",
            len(resolutions),
            len(range(CRF_SCAN_RANGE[0], CRF_SCAN_RANGE[1] + 1, CRF_SCAN_STEP)),
            len(chunks),
        )
        surface = _predict_surface(
            chunks        = chunks,
            features      = chunk_features,
            model         = self.model,
            resolutions   = resolutions,
            source_height = source_height,
        )
        log.info("  Surface has %d operating points", len(surface))

        # ── 6. Convex hull selection ──────────────────────────────────────────
        log.info("=== Convex Hull Selection ===")
        hull = _convex_hull_rd(surface)
        log.info("  Convex hull: %d Pareto-optimal points", len(hull))

        for pt in hull:
            log.info(
                "  Hull point: %dp  CRF=%d  VMAF=%.1f  %.0f kbps",
                pt.resolution_height, pt.crf, pt.vmaf, pt.bitrate_kbps,
            )

        # ── 7. Rung selection from hull ───────────────────────────────────────
        log.info("=== Ladder Rung Selection ===")
        rung_selections = _select_rungs_from_hull(hull, vmaf_targets)

        rungs = []
        for vmaf_target, hull_pt in rung_selections:
            # Per-shot CRF assignments for this rung
            per_chunk = _per_shot_crfs_for_rung(
                chunks            = chunks,
                features          = chunk_features,
                model             = self.model,
                vmaf_target       = vmaf_target,
                resolution_height = hull_pt.resolution_height,
                source_height     = source_height,
            )

            rung = LadderRung(
                resolution_height     = hull_pt.resolution_height,
                resolution_width      = hull_pt.resolution_width,
                vmaf_target           = vmaf_target,
                predicted_vmaf        = hull_pt.vmaf,
                predicted_bitrate_kbps= hull_pt.bitrate_kbps,
                per_chunk_crfs        = per_chunk,
            )
            rungs.append(rung)
            log.info(
                "  Rung: %s  VMAF target=%.1f  pred VMAF=%.1f  pred rate=%.0f kbps",
                rung.label, vmaf_target, rung.predicted_vmaf, rung.predicted_bitrate_kbps,
            )

        # Sort rungs by bitrate ascending (low quality → high quality)
        rungs.sort(key=lambda r: r.predicted_bitrate_kbps)

        result = LadderResult(
            source_path        = os.path.abspath(source_path),
            rungs              = rungs,
            source_width       = source_width,
            source_height      = source_height,
            source_fps         = source_fps,
            source_codec       = source_codec,
            duration_s         = duration_s,
            chunk_count        = len(chunks),
            generation_time_s  = time.time() - t_start,
        )

        # ── 8. Write outputs ──────────────────────────────────────────────────
        _write_ladder_json(result, output_dir)
        _write_hls_manifest(result, output_dir)
        _write_dash_manifest(result, output_dir)

        log.info(
            "=== Ladder complete: %d rungs in %.1fs ===",
            len(rungs), result.generation_time_s,
        )
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Manifest Generation
# ──────────────────────────────────────────────────────────────────────────────

def _write_ladder_json(result: LadderResult, output_dir: str) -> str:
    path = os.path.join(output_dir, "ladder.json")
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    log.info("Ladder definition → %s", path)
    return path


def _write_hls_manifest(result: LadderResult, output_dir: str) -> str:
    """
    Write a master HLS playlist (master.m3u8) referencing one media playlist
    per rung.

    The media playlists are stub files — actual segment URLs depend on the
    encoding run. Each rung gets a placeholder media playlist that can be
    replaced with real segment URLs after encoding.

    HLS spec references:
      - EXT-X-STREAM-INF: BANDWIDTH, AVERAGE-BANDWIDTH, RESOLUTION, CODECS
      - BANDWIDTH = peak bitrate (we use predicted + 15% headroom)
      - AVERAGE-BANDWIDTH = mean predicted bitrate
    """
    lines = ["#EXTM3U", "#EXT-X-VERSION:6", ""]

    for rung in result.rungs:
        avg_bw   = int(rung.predicted_bitrate_kbps * 1000)
        peak_bw  = int(avg_bw * 1.15)   # 15% headroom for VBR peaks

        # Codec string — H.264 baseline/main/high depending on resolution
        if "264" in str(result.source_codec) or result.source_codec == "unknown":
            codecs = "avc1.640028,mp4a.40.2"
        elif "265" in str(result.source_codec) or "hevc" in str(result.source_codec):
            codecs = "hvc1.1.6.L120.90,mp4a.40.2"
        else:
            codecs = "avc1.640028,mp4a.40.2"

        lines.append(
            f'#EXT-X-STREAM-INF:BANDWIDTH={peak_bw},'
            f'AVERAGE-BANDWIDTH={avg_bw},'
            f'RESOLUTION={rung.resolution_str},'
            f'CODECS="{codecs}",'
            f'FRAME-RATE={result.source_fps:.3f},'
            f'VIDEO-RANGE=SDR'
        )
        lines.append(f"{rung.label}/media.m3u8")
        lines.append("")

        # Write stub media playlist
        media_dir  = os.path.join(output_dir, rung.label)
        os.makedirs(media_dir, exist_ok=True)
        media_path = os.path.join(media_dir, "media.m3u8")
        with open(media_path, "w") as f:
            f.write("#EXTM3U\n")
            f.write("#EXT-X-VERSION:6\n")
            f.write("#EXT-X-TARGETDURATION:6\n")
            f.write(f"# Rendition: {rung.label}\n")
            f.write(f"# Predicted VMAF: {rung.predicted_vmaf:.2f}\n")
            f.write(f"# Predicted bitrate: {rung.predicted_bitrate_kbps:.0f} kbps\n")
            f.write("# Segments will be populated after encoding run\n")
            f.write("#EXT-X-ENDLIST\n")

    master_path = os.path.join(output_dir, "master.m3u8")
    with open(master_path, "w") as f:
        f.write("\n".join(lines))
    log.info("HLS master manifest → %s", master_path)
    return master_path


def _write_dash_manifest(result: LadderResult, output_dir: str) -> str:
    """
    Write a DASH MPD (manifest.mpd) referencing one AdaptationSet with one
    Representation per rung.

    Produces a stub MPD — segment URLs are placeholders to be replaced after
    encoding. Structure follows ISO/IEC 23009-1 (MPEG-DASH) with CMAF segments.
    """
    fps_int   = int(round(result.source_fps))
    fps_frac  = f"{result.source_fps:.3f}"
    duration  = result.duration_s
    dur_str   = f"PT{int(duration // 3600)}H{int((duration % 3600) // 60)}M{duration % 60:.3f}S"

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<MPD xmlns="urn:mpeg:dash:schema:mpd:2011"',
        f'     profiles="urn:mpeg:dash:profile:isoff-live:2011"',
        f'     type="static"',
        f'     mediaPresentationDuration="{dur_str}"',
        f'     minBufferTime="PT2S">',
        f'  <Period id="1" duration="{dur_str}">',
        f'    <AdaptationSet mimeType="video/mp4" codecs="avc1.640028"',
        f'                   frameRate="{fps_frac}" segmentAlignment="true"',
        f'                   bitstreamSwitching="true">',
    ]

    for rung in result.rungs:
        avg_bw = int(rung.predicted_bitrate_kbps * 1000)
        lines += [
            f'      <Representation id="{rung.label}"',
            f'                      bandwidth="{avg_bw}"',
            f'                      width="{rung.resolution_width}"',
            f'                      height="{rung.resolution_height}">',
            f'        <!-- Predicted VMAF: {rung.predicted_vmaf:.2f} -->',
            f'        <!-- Predicted bitrate: {rung.predicted_bitrate_kbps:.0f} kbps -->',
            f'        <!-- Segments populated after encoding run -->',
            f'        <SegmentTemplate media="{rung.label}/seg-$Number$.m4s"',
            f'                         initialization="{rung.label}/init.mp4"',
            f'                         startNumber="1" duration="{fps_int * 6}" timescale="{fps_int}"/>',
            f'      </Representation>',
        ]

    lines += [
        '    </AdaptationSet>',
        '  </Period>',
        '</MPD>',
    ]

    mpd_path = os.path.join(output_dir, "manifest.mpd")
    with open(mpd_path, "w") as f:
        f.write("\n".join(lines))
    log.info("DASH MPD manifest → %s", mpd_path)
    return mpd_path


# ──────────────────────────────────────────────────────────────────────────────
# Ladder Summary Report
# ──────────────────────────────────────────────────────────────────────────────

def print_ladder_report(result: LadderResult) -> None:
    """Print a human-readable ladder summary to the log."""
    divider = "─" * 72
    log.info("")
    log.info("  CONTENT-ADAPTIVE BITRATE LADDER")
    log.info("  %s", divider)
    log.info("  Source    : %s", os.path.basename(result.source_path))
    log.info("  Resolution: %dx%d  %.3g fps  %.1fs",
             result.source_width, result.source_height,
             result.source_fps, result.duration_s)
    log.info("  Chunks    : %d shots", result.chunk_count)
    log.info("  Generated : %.1fs", result.generation_time_s)
    log.info("  %s", divider)
    log.info("  %-20s  %-10s  %-12s  %-10s",
             "Rung", "Resolution", "Bitrate", "VMAF (pred)")
    log.info("  %s", divider)
    for rung in result.rungs:
        log.info("  %-20s  %-10s  %-12s  %.1f",
                 rung.label,
                 rung.resolution_str,
                 f"{rung.predicted_bitrate_kbps:.0f} kbps",
                 rung.predicted_vmaf)
    log.info("  %s", divider)
    log.info("")


# ──────────────────────────────────────────────────────────────────────────────
# Ladder Encoding — multi-lane parallel encode with in-loop VMAF
# ──────────────────────────────────────────────────────────────────────────────

def _encode_ladder_renditions(
    result:     LadderResult,
    output_dir: str,
    args,
) -> None:
    """
    Encode all ladder renditions using multi-lane parallel encoding with
    in-loop VMAF measurement (Meta FFmpeg 8.0 architecture).

    All renditions share a single decode pass. Encoders run in parallel
    threads. VMAF is measured inline without a separate pass.

    For each rung, per-chunk CRF assignments from the ladder definition
    are used — different shots within a rung use different CRF values to
    maintain consistent perceptual quality.

    Output structure:
        output_dir/
            {rung_label}/
                chunk_0000.mp4
                chunk_0001.mp4
                ...
                vmaf_0000.json
                vmaf_0001.json
                ...
                media.m3u8  (updated with real segment references)
    """
    cfg = EvalConfig(reference=result.source_path)
    cfg.ffmpeg_path  = args.ffmpeg_path
    cfg.ffprobe_path = args.ffprobe_path
    cfg.codec        = args.codec
    cfg.preset       = args.preset
    cfg.color_meta   = probe_color_metadata(cfg)

    # Rebuild chunks to get ChunkInfo objects with timing
    do_cfg = DynamicOptimizerConfig(
        scene_threshold    = args.scene_threshold,
        min_chunk_duration = args.min_chunk_duration,
        detector           = args.detector,
        chunk_dir          = os.path.join(output_dir, "chunks"),
    )
    chunks = build_chunks(cfg, do_cfg)

    log.info(
        "Encoding %d rungs × %d chunks = %d total encodes (multi-lane per chunk)",
        len(result.rungs), len(chunks), len(result.rungs) * len(chunks),
    )

    for chunk_idx, chunk in enumerate(chunks):
        log.info(
            "[Chunk %04d/%04d]  %.2fs–%.2fs  (%d frames)",
            chunk_idx, len(chunks) - 1,
            chunk.start_time, chunk.end_time, chunk.frame_count,
        )

        # Build one LaneSpec per rung for this chunk
        lanes = []
        for rung in result.rungs:
            rung_dir = os.path.join(output_dir, rung.label)
            os.makedirs(rung_dir, exist_ok=True)

            chunk_crf = rung.per_chunk_crfs.get(chunk.index, 23)

            lane = LaneSpec(
                crf           = chunk_crf,
                output_path   = os.path.join(
                    rung_dir, f"chunk_{chunk_idx:04d}.mp4"
                ),
                vmaf_log_path = os.path.join(
                    rung_dir, f"vmaf_{chunk_idx:04d}.json"
                ),
                width  = rung.resolution_width,
                height = rung.resolution_height,
            )
            lanes.append(lane)

        # Encode all rungs for this chunk in one multi-lane pass with in-loop VMAF
        outcomes = encode_multilane_with_vmaf(
            cfg             = cfg,
            lanes           = lanes,
            source_override = result.source_path,
        )

        # Log per-rung results
        for lane, rung in zip(lanes, result.rungs):
            outcome = outcomes.get(lane.label, {})
            vmaf    = outcome.get("vmaf")
            br      = outcome.get("bitrate_kbps", 0)
            log.info(
                "  %-20s  CRF=%-3d  VMAF=%-6s  %.0f kbps",
                rung.label, lane.crf,
                f"{vmaf:.2f}" if vmaf is not None else "—",
                br,
            )

    log.info("=== Ladder encoding complete ===")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a content-adaptive ABR bitrate ladder using a trained RDCurveModel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("source", help="Path to source video (high-quality master)")
    p.add_argument("--model",       required=True, help="Path to trained RDCurveModel JSON (from video_quality_pipeline.py)")
    p.add_argument("-o", "--output-dir", default="./ladder_output", help="Output directory")

    # Ladder configuration
    p.add_argument("--resolutions",   nargs="+", type=int, default=DEFAULT_RESOLUTIONS,
                   metavar="H", help="Rendition heights to include in surface scan e.g. 360 540 720 1080")
    p.add_argument("--vmaf-targets",  nargs="+", type=float, default=DEFAULT_VMAF_TARGETS,
                   metavar="V", help="VMAF quality levels for ladder rungs")

    # Encoding
    p.add_argument("--codec",   default="libx264", help="Encoding codec")
    p.add_argument("--preset",  default="medium",  help="Encoder preset")
    p.add_argument("--encode",  action="store_true",
                   help="Encode all renditions after ladder generation (slow)")

    # Scene detection
    p.add_argument("--detector",         default="pyscenedetect",
                   choices=["pyscenedetect", "ffmpeg"])
    p.add_argument("--scene-threshold",  type=float, default=27.0)
    p.add_argument("--min-chunk-duration", type=float, default=2.0, metavar="S")

    # Tool paths
    p.add_argument("--ffmpeg",  default="ffmpeg",  dest="ffmpeg_path")
    p.add_argument("--ffprobe", default="ffprobe", dest="ffprobe_path")

    return p


def main():
    if not HAS_PIPELINE:
        log.error("Cannot import video_quality_pipeline. Aborting.")
        sys.exit(1)
    if not HAS_NUMPY:
        log.error("numpy is required: pip install numpy")
        sys.exit(1)

    args = build_parser().parse_args()

    # Load model
    if not os.path.exists(args.model):
        log.error("Model file not found: %s", args.model)
        sys.exit(1)

    model = RDCurveModel.load(args.model)
    if not model._weights:
        log.error("Model has no trained weights. Run video_quality_pipeline.py "
                  "--train-model first.")
        sys.exit(1)

    # Generate ladder
    generator = LadderGenerator(model=model)
    result = generator.generate(
        source_path      = args.source,
        output_dir       = args.output_dir,
        resolutions      = args.resolutions,
        vmaf_targets     = args.vmaf_targets,
        codec            = args.codec,
        preset           = args.preset,
        ffmpeg_path      = args.ffmpeg_path,
        ffprobe_path     = args.ffprobe_path,
        scene_threshold  = args.scene_threshold,
        detector         = args.detector,
        min_chunk_s      = args.min_chunk_duration,
    )

    print_ladder_report(result)

    if args.encode:
        log.info("--encode flag set — encoding all renditions with multi-lane parallel encoder...")
        _encode_ladder_renditions(result, args.output_dir, args)


if __name__ == "__main__":
    main()
