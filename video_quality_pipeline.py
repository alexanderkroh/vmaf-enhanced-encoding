#!/usr/bin/env python3
"""
Video Quality Evaluation Pipeline
Supports: AVQT (Apple Video Quality Tool) and FFmpeg+VMAF
Generates: Bit-Distortion (Rate-Distortion) curve graphs
"""

import subprocess
import json
import csv
import os
import re
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import shutil
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Optional dependencies (graceful fallback messages) ---
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    HAS_PYSCENEDETECT = True
except ImportError:
    HAS_PYSCENEDETECT = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Standard FFmpeg default CRF (used as a last-resort fallback in learned controller)
DEFAULT_CRF = 23


def _safe_remove(path: str) -> None:
    """Delete a file, silently ignoring errors (file already gone, permission, etc.)."""
    try:
        os.remove(path)
    except OSError:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QualityResult:
    """Holds quality metrics for a single encoded variant."""
    label: str                     # e.g. "500kbps" or "crf18"
    bitrate_kbps: float            # measured bitrate
    target_bitrate_kbps: Optional[float] = None  # requested bitrate (if ABR)
    crf: Optional[int] = None      # CRF value (if CRF mode)

    # Per-metric scores (mean over all frames)
    vmaf: Optional[float] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    avqt: Optional[float] = None

    # Per-frame data for temporal plots
    vmaf_frames: list = field(default_factory=list)
    psnr_frames: list = field(default_factory=list)
    avqt_frames: list = field(default_factory=list)

    # File size
    file_size_bytes: int = 0
    output_path: str = ""


@dataclass
class EvalConfig:
    """Pipeline configuration."""
    reference: str                 # path to reference (original) video
    output_dir: str = "./eval_output"
    codec: str = "libx264"         # libx264, libx265, libvpx-vp9, libaom-av1
    preset: str = "medium"
    bitrates_kbps: list = field(default_factory=lambda: [
        200, 500, 1000, 2000, 4000, 8000
    ])
    crfs: list = field(default_factory=lambda: [])  # if set, use CRF sweep instead
    enable_vmaf: bool = True
    enable_psnr: bool = True
    enable_ssim: bool = True
    enable_avqt: bool = False       # requires Apple AVQT tool installed
    avqt_path: str = "avqt"        # path to avqt binary
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    vmaf_model: str = ""           # "" = default model; or full path to .json model
    threads: int = 0               # 0 = auto
    keep_encoded: bool = False     # keep encoded files after eval
    hw_accel: str = "none"         # "none", "videotoolbox", "nvenc", "qsv", "amf"
    parallel_jobs: int = 1         # parallel encode+evaluate workers
    keyframe_interval: int = 0     # 0 = encoder default; >0 forces -g N (e.g. 48 for 2s@24fps)
    force_pix_fmt: str = ""        # "" = auto (match source bit depth); or e.g. "yuv420p" / "yuv420p10le"
    preserve_hdr: bool = True      # carry through color primaries, transfer, mastering display, MaxCLL/MaxFALL
    hdr_vmaf_normalise: bool = False       # linearise PQ→BT.709 before VMAF scoring (HDR source accuracy)
    hdr_vmaf_normalise_locked: bool = False  # True when user explicitly set --hdr/--no-hdr-vmaf-normalise
    color_meta: dict = field(default_factory=dict)  # populated at runtime by probe_color_metadata()


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic Optimizer — Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkInfo:
    """
    One shot-boundary-delimited segment of the source video.

    Frame indices are the authoritative coordinates.  Timestamps are derived
    properties so that a single fps value is the sole conversion authority —
    no rounding drift accumulates across chunks.

    context_start_frame is keyframe-aligned and may be < start_frame to give
    the encoder a look-ahead preroll window without changing the logical chunk
    boundaries.  The final encoded output always contains exactly frame_count
    frames (start_frame … end_frame-1 inclusive).

    source_path is kept for the quality-sweep reference clip (stream-copy,
    approximate) and is NOT used for the final encode — that always reads from
    the original source file via cfg.reference.
    """
    index: int
    start_frame: int          # first frame owned by this chunk (inclusive)
    end_frame: int            # first frame of the NEXT chunk (exclusive)
    fps: float                # source frame rate — sole authority for time↔frame

    context_start_frame: int = 0   # keyframe-aligned decode start for encoder preroll;
                                   # equals start_frame when context is disabled
    source_path: str = ""     # sweep-only stream-copy reference (approximate boundary)

    # ── Derived properties (all computed from frame indices + fps) ────────────

    @property
    def start_time(self) -> float:
        return self.start_frame / self.fps if self.fps else 0.0

    @property
    def end_time(self) -> float:
        return self.end_frame / self.fps if self.fps else 0.0

    @property
    def duration(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps if self.fps else 0.0

    @property
    def frame_count(self) -> int:
        """Exact logical frame count — derived from integer arithmetic, never drifts."""
        return self.end_frame - self.start_frame

    @property
    def preroll_frames(self) -> int:
        """Context frames before the logical chunk start (encoder look-ahead preroll)."""
        return max(0, self.start_frame - self.context_start_frame)

    @property
    def context_start_time(self) -> float:
        """Decode-start timestamp aligned to the context keyframe."""
        return self.context_start_frame / self.fps if self.fps else 0.0


@dataclass
class ChunkOptimResult:
    """Encoding sweep + optimal selection for a single chunk."""
    chunk: ChunkInfo
    sweep_results: list = field(default_factory=list)   # list[QualityResult]
    optimal: Optional[QualityResult] = None
    final_encoded_path: str = ""
    vmaf: Optional[float] = None
    psnr: Optional[float] = None
    bitrate_kbps: float = 0.0
    features: dict = field(default_factory=dict)        # content-complexity features from extract_shot_features()
    predicted_vmaf: Optional[float] = None              # model VMAF prediction (learned-controller mode only)


@dataclass
class DynamicOptimizerConfig:
    """Settings for the Netflix-style per-shot dynamic optimizer."""
    scene_threshold: float = 27.0      # 0–100; maps to FFmpeg scene expr (/100) or PySceneDetect directly
    min_chunk_duration: float = 2.0    # merge chunks shorter than this (seconds)
    vmaf_target: float = 93.0          # VMAF floor for optimal-encode selection
    optimize_mode: str = "crf"         # "crf" or "bitrate"
    detector: str = "ffmpeg"           # "ffmpeg" or "pyscenedetect"
    no_concat: bool = False            # skip final FFmpeg concat step
    snap_to_keyframes: bool = True     # snap shot boundaries to source keyframes before extraction
    encoder_context_duration: float = 2.0  # seconds of pre-roll context for encoder look-ahead (0 = disabled)
    chunk_dir: str = ""                # populated at runtime (output_dir/chunks)


@dataclass
class DynamicOptimizerResult:
    """Top-level result of a full dynamic optimizer run."""
    chunk_results: list = field(default_factory=list)   # list[ChunkOptimResult]
    aggregate_vmaf: Optional[float] = None
    aggregate_psnr: Optional[float] = None
    aggregate_bitrate_kbps: float = 0.0
    final_video_path: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Encoding
# ──────────────────────────────────────────────────────────────────────────────

def encode_video(
    cfg: EvalConfig,
    output_path: str,
    bitrate_kbps: Optional[float] = None,
    crf: Optional[int] = None,
) -> bool:
    """Encode reference video at a given bitrate or CRF, with optional GPU acceleration."""
    hw = cfg.hw_accel
    encoder = _resolve_encoder(cfg.codec, hw)

    cmd = [cfg.ffmpeg_path, "-y"]
    cmd += _hwaccel_flags(hw)
    cmd += ["-i", cfg.reference, "-c:v", encoder, "-an"]
    cmd += _container_tag_flags(cfg.codec, encoder)
    cmd += _quality_flags(cfg.codec, hw, crf, bitrate_kbps)
    cmd += _preset_flags(cfg.codec, hw, cfg.preset)
    cmd += _resolve_pix_fmt(cfg.color_meta, encoder, cfg.force_pix_fmt)
    cmd += _build_color_flags(cfg.color_meta, encoder, cfg.preserve_hdr)

    # Keyframe interval — ensures regular GOP structure (critical for ABR streaming)
    if cfg.keyframe_interval > 0:
        cmd += ["-g", str(cfg.keyframe_interval)]
        # keyint_min prevents shorter GOPs between forced frames (software encoders only)
        if hw not in ("nvenc", "videotoolbox", "qsv", "amf"):
            cmd += ["-keyint_min", str(cfg.keyframe_interval)]

    if cfg.threads > 0:
        cmd += ["-threads", str(cfg.threads)]

    # Exact PTS timing: use fps_num as the MP4 timescale so each frame
    # occupies exactly fps_den ticks with no rounding.  For 59.94 fps
    # (60000/1001) this avoids the ~1 ms/frame drift that occurs when
    # libx264's default 12800 Hz timescale rounds 1001/60000 s per frame.
    fps_num = (cfg.color_meta or {}).get("fps_num", 0)
    if fps_num > 0 and output_path.lower().endswith(".mp4"):
        cmd += ["-video_track_timescale", str(fps_num)]

    cmd.append(output_path)

    log.info("Encoding [%s] → %s", encoder, output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("FFmpeg encode failed:\n%s", result.stderr[-2000:])
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Lane Encoding  (Meta FFmpeg 8.0 parallel encoder architecture)
# ──────────────────────────────────────────────────────────────────────────────
#
# Meta's March 2026 engineering post describes two features now upstream in
# FFmpeg 8.0 that were previously only in their internal fork:
#
#   1. Threaded multi-lane encoding — multiple encoders run in parallel sharing
#      a single decode pass.  Previously encoders ran serially per-frame even
#      within a single FFmpeg command.  The refactoring landed in FFmpeg 6.0
#      with finishing touches in 8.0.
#
#   2. In-loop quality metrics — a decoder is inserted after each encoder lane,
#      its frames compared against the pre-encode reference, producing VMAF /
#      PSNR / SSIM scores per lane in real time without a separate pass.
#
# encode_multilane() implements (1): one decode pass → N parallel encoders.
# encode_multilane_with_vmaf() implements (1)+(2): one decode pass → N
# parallel encoders → N inline decoders → N VMAF measurements.
#
# Both functions accept the same EvalConfig as encode_video() and reuse all
# existing helper functions for quality flags, presets, pixel formats, and
# colour metadata.  They are drop-in replacements for the per-CRF encode loop
# in run_pipeline() and _process_chunk().
#
# Filter graph topology (4-lane CRF sweep example):
#
#   Input → split=4 ─┬─ scale → encoder_0 → output_0.mp4
#                    ├─ scale → encoder_1 → output_1.mp4
#                    ├─ scale → encoder_2 → output_2.mp4
#                    └─ scale → encoder_3 → output_3.mp4
#
# With in-loop VMAF (encode_multilane_with_vmaf):
#
#   Input → split=4 ─┬─ scale → encoder_0 ─┬─ output_0.mp4
#           split=4  │                       └─ decode → [dist_0][ref_0] → libvmaf → vmaf_0.json
#                    ├─ scale → encoder_1 ─┬─ output_1.mp4
#                    │                     └─ decode → [dist_1][ref_1] → libvmaf → vmaf_1.json
#                    ...
#
# IMPORTANT: In-loop VMAF requires FFmpeg 7.0+ for the enc/dec filter
# capability.  The function checks for this and falls back to encode_multilane()
# + sequential VMAF if the FFmpeg version is < 7.0.
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class LaneSpec:
    """
    Specification for a single encoding lane in a multi-lane encode.

    Each lane encodes the same source at a different CRF or bitrate,
    optionally at a different resolution.
    """
    crf:            Optional[int]   = None   # CRF value (mutually exclusive with bitrate_kbps)
    bitrate_kbps:   Optional[float] = None   # target bitrate (mutually exclusive with crf)
    output_path:    str             = ""     # output file path
    vmaf_log_path:  str             = ""     # path for VMAF JSON log (in-loop mode only)
    width:          Optional[int]   = None   # output width  (None = source width)
    height:         Optional[int]   = None   # output height (None = source height)

    @property
    def label(self) -> str:
        if self.crf is not None:
            return f"crf{self.crf}"
        if self.bitrate_kbps is not None:
            return f"{int(self.bitrate_kbps)}kbps"
        return "unknown"


def _ffmpeg_version_tuple(ffmpeg_path: str) -> tuple:
    """Return (major, minor) FFmpeg version as integers, or (0, 0) on failure."""
    try:
        r = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True, text=True, timeout=10,
        )
        for line in r.stdout.splitlines():
            if line.startswith("ffmpeg version"):
                # "ffmpeg version 8.0.1 ..."  or  "ffmpeg version N-12345-g..."
                parts = line.split()
                if len(parts) >= 3:
                    ver = parts[2].split(".")
                    try:
                        return (int(ver[0]), int(ver[1]) if len(ver) > 1 else 0)
                    except ValueError:
                        pass
    except Exception:
        pass
    return (0, 0)


def _build_normalise_chain() -> str:
    """Return the HDR PQ→linear→BT.709 zscale chain for inline use."""
    return (
        "zscale=t=linear:npl=203,"
        "format=gbrpf32le,"
        "zscale=tin=linear:t=bt709:p=bt709:m=bt709:r=tv,"
        "format=yuv420p"
    )


def encode_multilane(
    cfg:    "EvalConfig",
    lanes:  list,           # list[LaneSpec]
    source_override: str = "",   # override cfg.reference (for chunk sweeps)
) -> dict:
    """
    Encode multiple CRF or bitrate variants in a single FFmpeg pass.

    Decodes the source once and fans out to N parallel encoder instances —
    the architecture Meta upstreamed into FFmpeg 8.0.  Encoders run in
    parallel threads sharing the decoded frame buffer rather than serially.

    Parameters
    ----------
    cfg    : EvalConfig with codec, preset, hw_accel, color_meta etc.
    lanes  : list of LaneSpec — one per output rendition.
    source_override : if set, use this path as -i instead of cfg.reference.
                      Used by the chunk sweep to point at the chunk clip.

    Returns
    -------
    dict mapping lane label → True/False (success/failure per lane).
    """
    if not lanes:
        return {}

    source = source_override or cfg.reference
    hw     = cfg.hw_accel
    encoder = _resolve_encoder(cfg.codec, hw)
    n = len(lanes)

    # ── Build filter_complex ─────────────────────────────────────────────────
    # [0:v] → split=N → [v0][v1]...[vN-1]
    # Each branch: optional scale → encoder flags applied as output options
    filter_parts = []
    split_labels = "".join(f"[v{i}]" for i in range(n))
    filter_parts.append(f"[0:v]split={n}{split_labels}")

    # Scale branches — only added when lane requests a different resolution
    for i, lane in enumerate(lanes):
        src_label = f"[v{i}]"
        if lane.width and lane.height:
            out_label = f"[vscaled{i}]"
            filter_parts.append(
                f"{src_label}scale={lane.width}:{lane.height}{out_label}"
            )
        # If no scale, the [v{i}] label feeds the encoder directly via -map

    filter_complex = ";".join(filter_parts) if len(filter_parts) > 1 else filter_parts[0]

    # ── Assemble FFmpeg command ───────────────────────────────────────────────
    cmd = [cfg.ffmpeg_path, "-y"]
    cmd += _hwaccel_flags(hw)
    cmd += ["-i", source]
    cmd += ["-filter_complex", filter_complex]

    fps_num = (cfg.color_meta or {}).get("fps_num", 0)

    for i, lane in enumerate(lanes):
        # Map the correct output from the filter graph
        if lane.width and lane.height:
            cmd += ["-map", f"[vscaled{i}]"]
        else:
            cmd += ["-map", f"[v{i}]"]

        cmd += ["-an"]                              # no audio
        cmd += ["-c:v", encoder]
        cmd += _container_tag_flags(cfg.codec, encoder)
        cmd += _quality_flags(cfg.codec, hw, lane.crf, lane.bitrate_kbps)
        cmd += _preset_flags(cfg.codec, hw, cfg.preset)
        cmd += _resolve_pix_fmt(cfg.color_meta, encoder, cfg.force_pix_fmt)
        cmd += _build_color_flags(cfg.color_meta, encoder, cfg.preserve_hdr)

        if cfg.keyframe_interval > 0:
            cmd += ["-g", str(cfg.keyframe_interval)]
            if hw not in ("nvenc", "videotoolbox", "qsv", "amf"):
                cmd += ["-keyint_min", str(cfg.keyframe_interval)]

        if cfg.threads > 0:
            cmd += ["-threads", str(cfg.threads)]

        if fps_num > 0 and lane.output_path.lower().endswith(".mp4"):
            cmd += ["-video_track_timescale", str(fps_num)]

        cmd.append(lane.output_path)

    log.info(
        "Multi-lane encode: %d lanes [%s] → %s …",
        n,
        ", ".join(l.label for l in lanes),
        os.path.dirname(lanes[0].output_path),
    )
    result = subprocess.run(cmd, capture_output=True, text=True)

    outcomes = {}
    if result.returncode != 0:
        log.error("Multi-lane FFmpeg encode failed:\n%s", result.stderr[-3000:])
        for lane in lanes:
            outcomes[lane.label] = False
        return outcomes

    # Verify each output file was created
    for lane in lanes:
        ok = os.path.exists(lane.output_path) and os.path.getsize(lane.output_path) > 0
        outcomes[lane.label] = ok
        if ok:
            log.info("  %-12s → %s", lane.label, lane.output_path)
        else:
            log.error("  %-12s → MISSING output %s", lane.label, lane.output_path)

    return outcomes


def encode_multilane_with_vmaf(
    cfg:    "EvalConfig",
    lanes:  list,           # list[LaneSpec] — each must have vmaf_log_path set
    source_override: str = "",
) -> dict:
    """
    Encode multiple CRF/bitrate variants with in-loop VMAF measurement.

    Implements the Meta in-loop quality metric architecture: a decoder is
    inserted after each encoder, its frames compared against the pre-encode
    reference frames, producing VMAF (and optionally PSNR/SSIM) scores per
    lane without a separate measurement pass.

    Requires FFmpeg 7.0+ for in-loop decode support.  Falls back to
    encode_multilane() + sequential run_ffmpeg_metrics() on older versions.

    Filter graph topology (2-lane example, SDR):
    ┌─ Input ──────────────────────────────────────────────────────────────┐
    │ [0:v] → split=2 → [v0][ref0]                                         │
    │                     [ref1]                                            │
    │ Encode branch 0: [v0] → encoder_0 → [enc0] → output_0.mp4           │
    │                                     [enc0] → dec0 → [dist0]          │
    │ VMAF 0:          [dist0][ref0] → libvmaf → vmaf_0.json               │
    │ Encode branch 1: similar ...                                          │
    └──────────────────────────────────────────────────────────────────────┘

    For HDR sources the normalise chain is applied to both [distN] and [refN]
    before libvmaf, matching the existing run_ffmpeg_metrics() HDR behaviour.

    Parameters
    ----------
    cfg   : EvalConfig — hdr_vmaf_normalise, vmaf_model, enable_psnr/ssim etc.
    lanes : list[LaneSpec] — each must have output_path AND vmaf_log_path set.
    source_override : path override for chunk sweep use.

    Returns
    -------
    dict mapping lane label → dict with keys:
        'encode_ok'  : bool
        'vmaf'       : float or None
        'psnr'       : float or None
        'ssim'       : float or None
        'bitrate_kbps': float
    """
    if not lanes:
        return {}

    # ── Version check — in-loop decode requires FFmpeg 7.0+ ──────────────────
    major, minor = _ffmpeg_version_tuple(cfg.ffmpeg_path)
    if major < 7:
        log.warning(
            "In-loop VMAF requires FFmpeg 7.0+ (found %d.%d). "
            "Falling back to encode_multilane() + sequential VMAF.", major, minor,
        )
        encode_multilane(cfg, lanes, source_override=source_override)
        results = {}
        for lane in lanes:
            ok = os.path.exists(lane.output_path)
            vmaf_data = {}
            if ok and lane.vmaf_log_path and cfg.enable_vmaf:
                vmaf_data = run_ffmpeg_metrics(cfg, lane.output_path, lane.vmaf_log_path)
            br = probe_bitrate(cfg, lane.output_path) if ok else 0.0
            results[lane.label] = {
                "encode_ok":    ok,
                "vmaf":         vmaf_data.get("vmaf"),
                "psnr":         vmaf_data.get("psnr"),
                "ssim":         vmaf_data.get("ssim"),
                "bitrate_kbps": br,
            }
        return results

    source  = source_override or cfg.reference
    hw      = cfg.hw_accel
    encoder = _resolve_encoder(cfg.codec, hw)
    n       = len(lanes)
    hdr     = cfg.hdr_vmaf_normalise
    normalise = _build_normalise_chain()

    # ── Build libvmaf filter string (shared options, lane-specific log path) ──
    def _vmaf_filter(log_path: str) -> str:
        opts = f"log_fmt=json:log_path={log_path}"
        if cfg.vmaf_model:
            opts += f":model=path={cfg.vmaf_model}"
        if cfg.enable_psnr:
            opts += ":feature=name=psnr"
        if cfg.enable_ssim:
            opts += ":feature=name=float_ssim"
        return f"libvmaf={opts}"

    # ── Filter graph ─────────────────────────────────────────────────────────
    #
    # We need 2N outputs from the input:
    #   N encoder inputs:  [v0]..[vN-1]
    #   N reference copies for VMAF: [ref0]..[refN-1]
    #
    # split=2N → [v0][ref0][v1][ref1]...[vN-1][refN-1]
    #
    filter_parts = []
    split_labels = "".join(f"[v{i}][ref{i}]" for i in range(n))
    filter_parts.append(f"[0:v]split={2*n}{split_labels}")

    for i, lane in enumerate(lanes):
        # Scale branch (optional)
        if lane.width and lane.height:
            filter_parts.append(f"[v{i}]scale={lane.width}:{lane.height}[vs{i}]")
            enc_in = f"[vs{i}]"
        else:
            enc_in = f"[v{i}]"

        # Encoder → enc_out label
        enc_out = f"[enc{i}]"

        # The enc filter wraps the encoder and exposes a decoded output.
        # Syntax: [in]enc=c=ENCODER:OPTION=VALUE[enc_out][dec_out]
        # where dec_out is the decoded (post-compress) frames for VMAF.
        quality_flags = _quality_flags(cfg.codec, hw, lane.crf, lane.bitrate_kbps)
        preset_flags  = _preset_flags(cfg.codec, hw, cfg.preset)

        # Build encoder option string for the enc filter
        enc_opts = f"c={encoder}"
        # Quality
        for j in range(0, len(quality_flags) - 1, 2):
            k = quality_flags[j].lstrip("-").replace(":", "_")
            enc_opts += f":{k}={quality_flags[j+1]}"
        # Preset
        for j in range(0, len(preset_flags) - 1, 2):
            k = preset_flags[j].lstrip("-").replace(":", "_")
            enc_opts += f":{k}={preset_flags[j+1]}"

        dec_out = f"[dec{i}]"
        filter_parts.append(f"{enc_in}enc={enc_opts}{enc_out}{dec_out}")

        # VMAF branch — apply HDR normalisation if needed
        if hdr:
            filter_parts.append(f"{dec_out}{normalise}[dist_n{i}]")
            filter_parts.append(f"[ref{i}]{normalise}[ref_n{i}]")
            filter_parts.append(
                f"[dist_n{i}][ref_n{i}]{_vmaf_filter(lane.vmaf_log_path)}"
            )
        else:
            bit_depth = (cfg.color_meta or {}).get("bit_depth", 8)
            if bit_depth > 8:
                filter_parts.append(f"{dec_out}format=yuv420p[dist_8{i}]")
                filter_parts.append(f"[ref{i}]format=yuv420p[ref_8{i}]")
                filter_parts.append(
                    f"[dist_8{i}][ref_8{i}]{_vmaf_filter(lane.vmaf_log_path)}"
                )
            else:
                filter_parts.append(
                    f"{dec_out}[ref{i}]{_vmaf_filter(lane.vmaf_log_path)}"
                )

    filter_complex = ";".join(filter_parts)

    # ── Assemble FFmpeg command ───────────────────────────────────────────────
    fps_num = (cfg.color_meta or {}).get("fps_num", 0)
    cmd = [cfg.ffmpeg_path, "-y"]
    cmd += _hwaccel_flags(hw)
    cmd += ["-i", source]
    cmd += ["-filter_complex", filter_complex]

    for i, lane in enumerate(lanes):
        cmd += ["-map", f"[enc{i}]"]
        cmd += ["-c:v", "copy"]         # stream copy — already encoded by enc filter
        cmd += _container_tag_flags(cfg.codec, encoder)
        cmd += _build_color_flags(cfg.color_meta, encoder, cfg.preserve_hdr)
        cmd += _resolve_pix_fmt(cfg.color_meta, encoder, cfg.force_pix_fmt)

        if fps_num > 0 and lane.output_path.lower().endswith(".mp4"):
            cmd += ["-video_track_timescale", str(fps_num)]

        cmd.append(lane.output_path)

    log_prefix = "(HDR PQ→linear→BT.709 normalisation) " if hdr else ""
    log.info(
        "Multi-lane encode + in-loop VMAF %s: %d lanes [%s] …",
        log_prefix, n, ", ".join(l.label for l in lanes),
    )

    result = subprocess.run(cmd, capture_output=True, text=True)

    # ── Parse results ─────────────────────────────────────────────────────────
    outcomes = {}

    if result.returncode != 0:
        log.warning(
            "In-loop VMAF encode failed (returncode=%d). "
            "Retrying with encode_multilane() + sequential VMAF.",
            result.returncode,
        )
        log.debug("FFmpeg stderr:\n%s", result.stderr[-3000:])
        # Graceful fallback to two-pass approach
        encode_multilane(cfg, lanes, source_override=source_override)
        for lane in lanes:
            ok = os.path.exists(lane.output_path)
            vmaf_data = {}
            if ok and lane.vmaf_log_path and cfg.enable_vmaf:
                vmaf_data = run_ffmpeg_metrics(cfg, lane.output_path, lane.vmaf_log_path)
            br = probe_bitrate(cfg, lane.output_path) if ok else 0.0
            outcomes[lane.label] = {
                "encode_ok":    ok,
                "vmaf":         vmaf_data.get("vmaf"),
                "psnr":         vmaf_data.get("psnr"),
                "ssim":         vmaf_data.get("ssim"),
                "bitrate_kbps": br,
            }
        return outcomes

    for lane in lanes:
        encode_ok = (
            os.path.exists(lane.output_path)
            and os.path.getsize(lane.output_path) > 0
        )
        vmaf_data = {}
        if encode_ok and lane.vmaf_log_path and os.path.exists(lane.vmaf_log_path):
            try:
                vmaf_data = parse_vmaf_log(lane.vmaf_log_path)
            except Exception as e:
                log.warning("Could not parse VMAF log %s: %s", lane.vmaf_log_path, e)

        br = probe_bitrate(cfg, lane.output_path) if encode_ok else 0.0

        vmaf_score = vmaf_data.get("vmaf")
        log.info(
            "  %-12s  encode_ok=%-5s  VMAF=%-6s  %.0f kbps",
            lane.label,
            str(encode_ok),
            f"{vmaf_score:.2f}" if vmaf_score is not None else "—",
            br,
        )

        outcomes[lane.label] = {
            "encode_ok":    encode_ok,
            "vmaf":         vmaf_score,
            "psnr":         vmaf_data.get("psnr"),
            "ssim":         vmaf_data.get("ssim"),
            "bitrate_kbps": br,
        }

    return outcomes


# ──────────────────────────────────────────────────────────────────────────────
# Probe
# ──────────────────────────────────────────────────────────────────────────────

def probe_bitrate(cfg: EvalConfig, video_path: str) -> float:
    """Return actual video bitrate in kbps using ffprobe."""
    cmd = [
        cfg.ffprobe_path, "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.warning("ffprobe failed for %s", video_path)
        return 0.0
    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            br = stream.get("bit_rate")
            if br:
                return float(br) / 1000.0
    # fallback: file size / duration
    size = os.path.getsize(video_path)
    for stream in data.get("streams", []):
        dur = stream.get("duration")
        if dur:
            return (size * 8) / (float(dur) * 1000.0)
    return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Source Validation
# ──────────────────────────────────────────────────────────────────────────────

# Codecs that indicate a high-quality master (pass without bitrate warning)
_MASTER_CODECS = {
    "prores", "dnxhd", "dnxhr", "v210", "v410",   # professional video
    "ffv1", "huffyuv", "utvideo",                   # lossless
    "mjpeg",                                         # motion JPEG (common in cameras)
    "cineform", "cfhd",                             # GoPro CineForm
}

# Codecs that are delivery formats — warn unless bitrate is very high
_DELIVERY_CODECS = {"h264", "hevc", "vp9", "av1", "mpeg4", "mpeg2video"}

# Minimum bitrate threshold below which an H.264/H.265 source is likely a
# delivery encode rather than a usable master (kbps)
_MIN_MASTER_BITRATE_KBPS = 30_000   # 30 Mbps


@dataclass
class ValidationResult:
    """Result of a source file pre-validation check."""
    path: str
    ok: bool                          # True = safe to process
    errors: list = field(default_factory=list)    # blocking issues
    warnings: list = field(default_factory=list)  # non-blocking concerns
    info: dict = field(default_factory=dict)      # metadata summary

    def print_report(self) -> None:
        """Print a human-readable validation report to the log."""
        status = "PASS" if self.ok else "FAIL"
        log.info("=== Source Validation: %s ===", status)
        log.info("  File     : %s", self.path)
        for k, v in self.info.items():
            log.info("  %-10s: %s", k, v)
        if self.errors:
            log.error("  ERRORS (blocking):")
            for e in self.errors:
                log.error("    ✗  %s", e)
        if self.warnings:
            log.warning("  WARNINGS (non-blocking):")
            for w in self.warnings:
                log.warning("    ⚠  %s", w)
        if self.ok and not self.warnings:
            log.info("  All checks passed — source looks good.")
        log.info("=" * 50)


def validate_source(path: str, ffprobe_path: str = "ffprobe") -> ValidationResult:
    """
    Pre-flight validation of a source video file before running the oracle.

    Checks:
      - File exists and is readable
      - Contains a valid video stream
      - Duration is sufficient for meaningful chunking (>= 60s)
      - Resolution is adequate for VMAF scoring (>= 720p)
      - Frame rate is in a reasonable range
      - Codec and bitrate suggest a high-quality master, not a delivery encode
      - HDR metadata is present and will trigger normalisation correctly
      - Pixel format is compatible with target encoding

    Returns a ValidationResult with ok=True if safe to process.
    Errors are blocking; warnings are informational.
    """
    result = ValidationResult(path=path, ok=True)

    # ── 1. File accessibility ─────────────────────────────────────────────────
    if not os.path.exists(path):
        result.errors.append(f"File not found: {path}")
        result.ok = False
        return result

    if not os.access(path, os.R_OK):
        result.errors.append(f"File is not readable (permission denied): {path}")
        result.ok = False
        return result

    file_size_gb = os.path.getsize(path) / 1_000_000_000
    result.info["file_size"] = f"{file_size_gb:.2f} GB"

    # ── 2. FFprobe — stream-level metadata ────────────────────────────────────
    r = subprocess.run(
        [ffprobe_path, "-v", "quiet",
         "-print_format", "json",
         "-show_streams", "-show_format",
         path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        result.errors.append(
            f"ffprobe failed — file may be corrupt or in an unsupported format. "
            f"stderr: {r.stderr.strip()[:200]}"
        )
        result.ok = False
        return result

    try:
        probe = json.loads(r.stdout)
    except json.JSONDecodeError:
        result.errors.append("ffprobe output could not be parsed — file may be corrupt.")
        result.ok = False
        return result

    # ── 3. Video stream presence ──────────────────────────────────────────────
    video_streams = [s for s in probe.get("streams", [])
                     if s.get("codec_type") == "video"]
    if not video_streams:
        result.errors.append("No video stream found in file.")
        result.ok = False
        return result

    vs = video_streams[0]
    codec = vs.get("codec_name", "unknown").lower()
    width  = int(vs.get("width",  0))
    height = int(vs.get("height", 0))
    pix_fmt = vs.get("pix_fmt", "unknown")
    color_trc = vs.get("color_transfer", "") or ""
    color_primaries = vs.get("color_primaries", "") or ""

    # Frame rate
    rfr = vs.get("r_frame_rate") or vs.get("avg_frame_rate") or "0/1"
    try:
        fn, fd = rfr.split("/")
        fps = int(fn) / int(fd) if int(fd) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    # Duration — prefer format-level, fall back to stream-level
    fmt = probe.get("format", {})
    try:
        duration = float(fmt.get("duration") or vs.get("duration") or 0)
    except (ValueError, TypeError):
        duration = 0.0

    # Bitrate — prefer stream-level, fall back to format-level
    try:
        bitrate_kbps = float(vs.get("bit_rate") or fmt.get("bit_rate") or 0) / 1000.0
    except (ValueError, TypeError):
        bitrate_kbps = 0.0

    # If stream bitrate missing, estimate from file size / duration
    if bitrate_kbps == 0 and duration > 0:
        bitrate_kbps = (os.path.getsize(path) * 8) / (duration * 1000.0)

    result.info["codec"]      = codec
    result.info["resolution"] = f"{width}x{height}"
    result.info["fps"]        = f"{fps:.3g}"
    result.info["duration"]   = f"{duration:.1f}s ({duration/60:.1f} min)"
    result.info["bitrate"]    = f"{bitrate_kbps:.0f} kbps ({bitrate_kbps/1000:.1f} Mbps)"
    result.info["pix_fmt"]    = pix_fmt
    result.info["color_trc"]  = color_trc or "unset"
    result.info["color_primaries"] = color_primaries or "unset"

    # ── 4. Duration check ─────────────────────────────────────────────────────
    if duration == 0:
        result.errors.append(
            "Could not determine video duration. File may be incomplete or corrupt."
        )
        result.ok = False
    elif duration < 60:
        result.warnings.append(
            f"Duration is only {duration:.0f}s. Clips shorter than 60s produce very "
            f"few chunks and limited training value. Consider using a longer clip."
        )

    # ── 5. Resolution check ───────────────────────────────────────────────────
    if width == 0 or height == 0:
        result.errors.append("Could not determine video resolution.")
        result.ok = False
    elif height < 720:
        result.warnings.append(
            f"Resolution {width}x{height} is below 720p. VMAF scores are less "
            f"reliable at low resolutions and the 4K model will not be selected."
        )

    # ── 6. Frame rate check ───────────────────────────────────────────────────
    if fps == 0:
        result.errors.append("Could not determine frame rate.")
        result.ok = False
    elif fps < 10:
        result.warnings.append(
            f"Frame rate is unusually low ({fps:.2g} fps). "
            f"Scene detection and VMAF may behave unexpectedly."
        )
    elif fps > 120:
        result.warnings.append(
            f"Frame rate is very high ({fps:.2g} fps). "
            f"Processing will be slow and chunk counts will be large."
        )

    # ── 7. Codec / master quality check ──────────────────────────────────────
    if codec in _MASTER_CODECS:
        result.info["source_type"] = "master (professional codec)"
    elif codec in _DELIVERY_CODECS:
        if bitrate_kbps >= _MIN_MASTER_BITRATE_KBPS:
            result.info["source_type"] = f"delivery codec ({codec}) at high bitrate — probably OK"
            result.warnings.append(
                f"Codec is {codec.upper()} (a delivery format), but bitrate is "
                f"{bitrate_kbps/1000:.1f} Mbps which suggests a high-quality intermediate. "
                f"Verify this is not a re-encode of a lossy source."
            )
        else:
            result.info["source_type"] = f"delivery encode ({codec} at {bitrate_kbps/1000:.1f} Mbps)"
            result.warnings.append(
                f"Source appears to be a delivery encode: codec={codec.upper()}, "
                f"bitrate={bitrate_kbps/1000:.1f} Mbps (below {_MIN_MASTER_BITRATE_KBPS//1000} Mbps "
                f"threshold). VMAF scores will measure re-encode quality, not master→delivery "
                f"quality. Training data from this source will have limited generalization value. "
                f"Use a ProRes, DNxHD, or lossless master if available."
            )
    else:
        result.info["source_type"] = f"unknown codec: {codec}"
        result.warnings.append(
            f"Unrecognized codec '{codec}'. Verify this is a high-quality source."
        )

    # ── 8. HDR check ─────────────────────────────────────────────────────────
    is_hdr = color_trc.lower() in {"smpte2084", "arib-std-b67", "smpte428"}
    bit_depth = 8
    pf = pix_fmt or ""
    if "12" in pf:
        bit_depth = 12
    elif "10" in pf:
        bit_depth = 10

    if is_hdr:
        result.info["hdr"] = f"YES — trc={color_trc}  (HDR normalisation will auto-enable)"
    elif bit_depth >= 10 and color_primaries.lower() in {"bt2020", "bt2020nc", "bt2020c"}:
        result.warnings.append(
            f"Source is {bit_depth}-bit with bt2020 primaries but trc='{color_trc}' "
            f"is not a recognised HDR transfer function. HDR normalisation will NOT "
            f"auto-enable. If this is HDR content, pass --hdr-vmaf-normalise explicitly."
        )
        result.info["hdr"] = f"UNCERTAIN — 10-bit bt2020 but trc={color_trc or 'unset'}"
    else:
        result.info["hdr"] = "no"

    # ── 9. Pixel format compatibility ─────────────────────────────────────────
    if pix_fmt and "yuv" not in pix_fmt and "rgb" not in pix_fmt:
        result.warnings.append(
            f"Unusual pixel format '{pix_fmt}'. FFmpeg may need to convert before "
            f"encoding. Verify output looks correct."
        )

    return result

def run_ffmpeg_metrics(
    cfg: EvalConfig,
    distorted_path: str,
    json_out: str,
) -> dict:
    """
    Run VMAF (+ optionally PSNR/SSIM) via ffmpeg's libvmaf filter.
    Returns dict with mean scores and per-frame lists.

    HDR normalisation (cfg.hdr_vmaf_normalise=True):
      Standard VMAF was trained on 8-bit SDR BT.709 content. Feeding it raw
      PQ/BT.2020 frames produces physically meaningless VIF/DLM feature values
      because the signal distribution is completely outside the training domain.

      With normalisation enabled, both streams are passed through a two-step
      zscale chain before VMAF scoring:
        Step 1 — PQ electro-optical → scene-linear light (float)
                 npl=203 sets the reference white anchor: the SDR reference
                 white (100 cd/m²) maps to approximately 0.5 in linear scale,
                 giving headroom for HDR highlights above reference white.
        Step 2 — linear → BT.709 gamma (8-bit yuv420p)
                 This re-encodes the linear signal into the display-referred
                 SDR domain that VMAF's SVM features were trained on.

      The resulting scores are comparable to SDR VMAF scores and monotonically
      track encode quality, making them valid for CRF sweep optimisation.
      They are NOT directly comparable to SDR VMAF scores from non-HDR content
      (the tone-mapping step compresses the HDR range non-linearly), but they
      are internally consistent — which is what the optimizer needs.
    """
    # Build libvmaf filter string
    vmaf_opts = "log_fmt=json:log_path={log}".format(log=json_out)
    if cfg.vmaf_model:
        vmaf_opts += f":model=path={cfg.vmaf_model}"
    if cfg.enable_psnr:
        vmaf_opts += ":feature=name=psnr"
    if cfg.enable_ssim:
        vmaf_opts += ":feature=name=float_ssim"
    libvmaf_filter = f"libvmaf={vmaf_opts}"

    if cfg.hdr_vmaf_normalise:
        # Two-step PQ → linear → BT.709 normalisation applied to both streams
        # independently before they reach libvmaf. Each step:
        #   zscale=t=linear:npl=203  — EOTF (PQ → linear, float, npl anchors ref white)
        #   format=gbrpf32le         — explicit float planar format for second zscale
        #   zscale=tin=linear:t=bt709:p=bt709:m=bt709:r=tv — OETF (linear → BT.709)
        #   format=yuv420p           — 8-bit output that libvmaf is calibrated for
        #
        # tin=linear on the second zscale explicitly signals the input transfer so
        # zscale doesn't re-read the (now stale) stream metadata from the container.
        normalise = (
            "zscale=t=linear:npl=203,"
            "format=gbrpf32le,"
            "zscale=tin=linear:t=bt709:p=bt709:m=bt709:r=tv,"
            "format=yuv420p"
        )
        filter_complex = (
            f"[0:v]{normalise}[dist_n];"
            f"[1:v]{normalise}[ref_n];"
            f"[dist_n][ref_n]{libvmaf_filter}"
        )
        log.info("Running FFmpeg VMAF metrics (HDR PQ→linear→BT.709 normalisation) …")
    else:
        # scale2ref is deprecated and broken in recent FFmpeg builds — remove it.
        # Distorted is always encoded from the reference at the same resolution,
        # so no scaling is needed.  For 10-bit sources, convert to yuv420p so
        # libvmaf's feature extractors receive the 8-bit format they expect.
        bit_depth = (cfg.color_meta or {}).get("bit_depth", 8)
        if bit_depth > 8:
            # Output label must follow the last filter directly — no comma separator.
            to8 = "format=yuv420p"
            filter_complex = (
                f"[0:v]{to8}[dist_8];"
                f"[1:v]{to8}[ref_8];"
                f"[dist_8][ref_8]{libvmaf_filter}"
            )
        else:
            filter_complex = f"[0:v][1:v]{libvmaf_filter}"
        log.info("Running FFmpeg VMAF metrics …")

    def _run_vmaf_cmd(fc: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [cfg.ffmpeg_path, "-y",
             "-i", distorted_path,
             "-i", cfg.reference,
             "-filter_complex", fc,
             "-f", "null", "-"],
            capture_output=True, text=True,
        )

    result = _run_vmaf_cmd(filter_complex)

    # If the run failed AND we used an explicit model path, retry without it.
    # This catches libvmaf version-mismatch errors (e.g. FFmpeg linked against
    # libvmaf 2.x but the model file is from a libvmaf 3.x installation — the
    # feature-extractor registry differs between major versions).
    if result.returncode != 0 and cfg.vmaf_model:
        if "could not initialize feature extractor" in result.stderr or \
           "problem during vmaf_use_features_from_model" in result.stderr:
            log.warning(
                "VMAF model '%s' failed to load (libvmaf version mismatch?). "
                "Retrying without explicit model — FFmpeg will use its built-in default.",
                cfg.vmaf_model,
            )
            log.warning(
                "To suppress this: either leave --vmaf-model empty (auto-selects built-in), "
                "or verify FFmpeg and libvmaf are the same major version "
                "('ffmpeg -version | grep libvmaf' vs 'brew list libvmaf')."
            )
            # Rebuild filter without model_path in libvmaf opts
            vmaf_opts_fallback = "log_fmt=json:log_path={log}".format(log=json_out)
            if cfg.enable_psnr:
                vmaf_opts_fallback += ":feature=name=psnr"
            if cfg.enable_ssim:
                vmaf_opts_fallback += ":feature=name=float_ssim"
            lv_fallback = f"libvmaf={vmaf_opts_fallback}"
            if cfg.hdr_vmaf_normalise:
                fc_fallback = (
                    f"[0:v]{normalise}[dist_n];"
                    f"[1:v]{normalise}[ref_n];"
                    f"[dist_n][ref_n]{lv_fallback}"
                )
            elif (cfg.color_meta or {}).get("bit_depth", 8) > 8:
                fc_fallback = (
                    f"[0:v]format=yuv420p[dist_8];"
                    f"[1:v]format=yuv420p[ref_8];"
                    f"[dist_8][ref_8]{lv_fallback}"
                )
            else:
                fc_fallback = f"[0:v][1:v]{lv_fallback}"
            result = _run_vmaf_cmd(fc_fallback)

    if result.returncode != 0:
        log.error("FFmpeg VMAF failed:\n%s", result.stderr)
        return {}

    if not os.path.exists(json_out):
        log.error("VMAF log not created at %s", json_out)
        return {}

    return parse_vmaf_log(json_out)


def parse_vmaf_log(json_path: str) -> dict:
    """Parse libvmaf JSON log → dict with mean scores and frame lists."""
    with open(json_path) as f:
        data = json.load(f)

    frames = data.get("frames", [])
    out = {
        "vmaf_frames": [],
        "psnr_frames": [],
        "ssim_frames": [],
        "vmaf": None,
        "psnr": None,
        "ssim": None,
    }

    for fr in frames:
        metrics = fr.get("metrics", {})
        v = metrics.get("vmaf")
        if v is not None:
            out["vmaf_frames"].append(float(v))
        p = metrics.get("psnr_y") or metrics.get("psnr")
        if p is not None:
            out["psnr_frames"].append(float(p))
        s = metrics.get("float_ssim") or metrics.get("ssim")
        if s is not None:
            out["ssim_frames"].append(float(s))

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else None

    out["vmaf"] = safe_mean(out["vmaf_frames"])
    out["psnr"] = safe_mean(out["psnr_frames"])
    out["ssim"] = safe_mean(out["ssim_frames"])

    # Also check pooled metrics at top level
    pooled = data.get("pooled_metrics", {})
    if out["vmaf"] is None:
        vmaf_pooled = pooled.get("vmaf", {})
        out["vmaf"] = vmaf_pooled.get("mean")
    if out["psnr"] is None:
        psnr_pooled = pooled.get("psnr_y", pooled.get("psnr", {}))
        out["psnr"] = psnr_pooled.get("mean")
    if out["ssim"] is None:
        ssim_pooled = pooled.get("float_ssim", pooled.get("ssim", {}))
        out["ssim"] = ssim_pooled.get("mean")

    return out


# ──────────────────────────────────────────────────────────────────────────────
# AVQT
# ──────────────────────────────────────────────────────────────────────────────

def run_avqt(cfg: EvalConfig, distorted_path: str, csv_out: str) -> dict:
    """
    Run Apple AVQT tool.
    Expects: avqt --reference <ref> --distorted <dist> --output <csv>
    Returns dict with mean score and frame list.
    """
    if not shutil.which(cfg.avqt_path) and not os.path.isfile(cfg.avqt_path):
        log.warning("AVQT binary not found at '%s'. Skipping.", cfg.avqt_path)
        return {}

    cmd = [
        cfg.avqt_path,
        "--reference", cfg.reference,
        "--distorted", distorted_path,
        "--output", csv_out,
    ]

    log.info("Running AVQT …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("AVQT failed:\n%s", result.stderr[-2000:])
        return {}

    return parse_avqt_csv(csv_out)


def parse_avqt_csv(csv_path: str) -> dict:
    """Parse AVQT CSV output → dict with mean score and frame list."""
    if not os.path.exists(csv_path):
        return {}

    frames = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # AVQT column name may vary; try common names
            for col in ("avqt", "AVQT", "score", "Score"):
                if col in row:
                    try:
                        frames.append(float(row[col]))
                    except ValueError:
                        pass
                    break

    if not frames:
        log.warning("Could not parse AVQT scores from %s", csv_path)
        return {}

    return {
        "avqt_frames": frames,
        "avqt": sum(frames) / len(frames),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Shot Boundary Detection & Chunking
# ──────────────────────────────────────────────────────────────────────────────

def probe_duration(cfg: EvalConfig, path: str = "") -> float:
    """Return video duration in seconds via ffprobe."""
    target = path or cfg.reference
    cmd = [
        cfg.ffprobe_path, "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        target,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(result.stdout)
        return float(data.get("format", {}).get("duration", 0))
    except (ValueError, json.JSONDecodeError):
        return 0.0


def probe_frame_count(cfg: EvalConfig, path: str) -> int:
    """Return frame count for a video file via ffprobe, with packet-count fallback."""
    cmd = [
        cfg.ffprobe_path, "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames",
        "-of", "csv=p=0",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except ValueError:
        pass
    # Fallback: count packets (slower but works for container formats without nb_frames)
    cmd2 = [
        cfg.ffprobe_path, "-v", "quiet",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0",
        path,
    ]
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    try:
        return int(result2.stdout.strip())
    except ValueError:
        return 0


def probe_presented_frame_count(cfg: EvalConfig, path: str) -> int:
    """
    Return the number of *presented* (decoded) video frames by actually decoding
    the stream.  This is authoritative for files where nb_frames in the container
    header is misleading — most notably HEVC, which stores buffered/delay frames
    in the container count but does not display them.

    Uses ffprobe -count_frames, which decodes every frame and tallies them.
    Slower than a header read (~2–5 s for a 1-minute clip) but the only reliable
    way to get the count that matches the presentation timeline.

    Falls back to 0 on any error so callers can use cheaper methods instead.
    """
    cmd = [
        cfg.ffprobe_path, "-v", "quiet",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def probe_color_metadata(cfg: EvalConfig, path: str = "") -> dict:
    """
    Probe color-space and HDR10 metadata from the video file.

    Returns a dict with:
      pix_fmt          — e.g. "yuv420p10le"
      bit_depth        — 8, 10, or 12 (inferred from pix_fmt if not explicit)
      color_primaries  — e.g. "bt2020"
      color_trc        — e.g. "smpte2084"  (PQ = HDR10)
      colorspace       — e.g. "bt2020nc"
      has_hdr_meta     — True if mastering display side-data was found
      mastering_display — x265-params string e.g. "G(13250,34500)B(…)R(…)WP(…)L(…)"
      content_light_level — x265-params string e.g. "1000,120"

    Two ffprobe passes are needed: one for stream-level color flags, one for
    first-frame side-data (mastering display + content light level).
    """
    target = path or cfg.reference
    meta = {
        "pix_fmt":             None,
        "bit_depth":           8,
        "width":               0,
        "height":              0,
        "fps":                 0.0,     # exact frame rate (from r_frame_rate rational)
        "fps_num":             0,       # r_frame_rate numerator   e.g. 60000 for 59.94 fps
        "fps_den":             1,       # r_frame_rate denominator e.g. 1001  for 59.94 fps
        "total_frames":        0,       # total video frame count
        "source_codec":        "",      # e.g. "h264", "hevc", "vp9", "av1"
        "color_primaries":     None,
        "color_trc":           None,
        "colorspace":          None,
        "has_hdr_meta":        False,
        "mastering_display":   None,
        "content_light_level": None,
    }

    # ── Pass 1: stream-level color fields ─────────────────────────────────────
    r1 = subprocess.run(
        [cfg.ffprobe_path, "-v", "quiet",
         "-select_streams", "v:0",
         "-show_streams", "-print_format", "json", target],
        capture_output=True, text=True,
    )
    try:
        stream = json.loads(r1.stdout).get("streams", [{}])[0]
        meta["pix_fmt"]        = stream.get("pix_fmt") or ""
        meta["width"]          = int(stream.get("width") or 0)
        meta["height"]         = int(stream.get("height") or 0)
        meta["source_codec"]   = stream.get("codec_name") or ""
        meta["color_primaries"] = stream.get("color_primaries") or ""
        meta["color_trc"]      = stream.get("color_transfer") or ""
        meta["colorspace"]     = stream.get("color_space") or ""

        # bit_depth: prefer explicit field, fall back to pix_fmt heuristic
        bps = stream.get("bits_per_raw_sample") or stream.get("bits_per_coded_sample")
        try:
            meta["bit_depth"] = int(bps)
        except (TypeError, ValueError):
            pf = meta["pix_fmt"] or ""
            meta["bit_depth"] = 12 if "12" in pf else (10 if "10" in pf else 8)

        # fps from r_frame_rate (exact rational e.g. "24/1" or "30000/1001").
        # We keep both the float and the integer num/den so callers can use
        # fps_num as the MP4 video_track_timescale without any float rounding.
        rfr = stream.get("r_frame_rate") or stream.get("avg_frame_rate") or "0/1"
        try:
            n, d = rfr.split("/")
            ni, di = int(n), int(d)
            meta["fps"]     = ni / di if di else 0.0
            meta["fps_num"] = ni
            meta["fps_den"] = di if di else 1
        except (ValueError, ZeroDivisionError):
            meta["fps"] = 0.0

        # total_frames: priority order
        #   1. nb_read_frames (-count_frames): actual decoded presentation frames.
        #      Authoritative for HEVC which inflates nb_frames with buffered/delay
        #      frames that never appear on the display timeline.
        #   2. round(duration * fps): reliable when the container duration is
        #      accurate (MP4/MOV/MKV with a clean edit list).
        #   3. nb_frames: last resort — often correct for H.264/VP9/AV1 but wrong
        #      for HEVC streams with encoder-delay frames.
        presented = probe_presented_frame_count(cfg, target)
        if presented > 0:
            meta["total_frames"] = presented
            log.debug("total_frames from decoded frame count: %d", presented)
        else:
            try:
                meta["total_frames"] = round(float(stream["duration"]) * meta["fps"])
                log.debug("total_frames from duration×fps: %d", meta["total_frames"])
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                try:
                    meta["total_frames"] = int(stream["nb_frames"])
                    log.debug("total_frames from container nb_frames: %d", meta["total_frames"])
                except (KeyError, TypeError, ValueError):
                    log.warning("Could not determine total_frames from any source; using 0")
                    meta["total_frames"] = 0
    except (json.JSONDecodeError, IndexError):
        pass

    # ── Pass 2: first-frame side-data (mastering display + MaxCLL) ────────────
    r2 = subprocess.run(
        [cfg.ffprobe_path, "-v", "quiet",
         "-select_streams", "v:0",
         "-show_frames", "-read_intervals", "%+#1",
         "-print_format", "json", target],
        capture_output=True, text=True,
    )
    try:
        frames = json.loads(r2.stdout).get("frames", [])
        if frames:
            for sd in frames[0].get("side_data_list", []):
                sd_type = sd.get("side_data_type", "").lower()

                if "mastering" in sd_type:
                    meta["has_hdr_meta"] = True

                    def _chroma(s: str) -> int:
                        """Rational "num/den" → integer in 1/50000 units."""
                        try:
                            n, d = map(int, str(s).split("/"))
                            return int(round(n * 50000 / d))
                        except Exception:
                            return 0

                    def _lum(s: str) -> int:
                        """Rational "num/den" → integer in 1/10000 cd/m² units."""
                        try:
                            n, d = map(int, str(s).split("/"))
                            return int(round(n * 10000 / d))
                        except Exception:
                            return 0

                    gx = _chroma(sd.get("green_x", "0/1"))
                    gy = _chroma(sd.get("green_y", "0/1"))
                    bx = _chroma(sd.get("blue_x", "0/1"))
                    by = _chroma(sd.get("blue_y", "0/1"))
                    rx = _chroma(sd.get("red_x", "0/1"))
                    ry = _chroma(sd.get("red_y", "0/1"))
                    wx = _chroma(sd.get("white_point_x", "0/1"))
                    wy = _chroma(sd.get("white_point_y", "0/1"))
                    max_l = _lum(sd.get("max_luminance", "0/1"))
                    min_l = _lum(sd.get("min_luminance", "0/1"))
                    meta["mastering_display"] = (
                        f"G({gx},{gy})B({bx},{by})R({rx},{ry})"
                        f"WP({wx},{wy})L({max_l},{min_l})"
                    )

                elif "content light" in sd_type:
                    maxcll  = sd.get("max_content", 0)
                    maxfall = sd.get("max_average", 0)
                    meta["content_light_level"] = f"{maxcll},{maxfall}"
    except (json.JSONDecodeError, IndexError, KeyError):
        pass

    log.info(
        "Source: %s  %dx%d  %d-bit  %.4g fps  %d frames  codec=%s  "
        "primaries=%s  trc=%s  HDR=%s",
        meta["pix_fmt"], meta["width"], meta["height"], meta["bit_depth"],
        meta["fps"], meta["total_frames"],
        meta["source_codec"] or "unknown",
        meta["color_primaries"] or "unset",
        meta["color_trc"] or "unset",
        "yes (MaxCLL=%s)" % meta["content_light_level"] if meta["has_hdr_meta"] else "no",
    )
    return meta


def _build_color_flags(color_meta: dict, codec: str, preserve_hdr: bool) -> list:
    """
    Build FFmpeg output flags to carry color-space and HDR10 metadata through an encode.

    Color primaries / transfer / colorspace — passed via standard AVOption flags for all
    codecs, so the container and codec signal the correct color interpretation to players.

    Mastering display + MaxCLL/MaxFALL — only libx265 supports full HDR10 metadata
    embedding via x265-params.  For H.264 the VUI carries color flags but has no
    mastering display SEI; those fields will be silently absent from H.264 output.

    10-bit H.264 warning: macOS VideoToolbox (and therefore Finder/QuickTime) cannot
    hardware-decode H.264 High 10 profile.  The caller should switch to libx265 or
    hevc_videotoolbox for HDR10 source material.
    """
    if not preserve_hdr or not color_meta:
        return []

    flags = []

    if color_meta.get("color_primaries"):
        flags += ["-color_primaries", color_meta["color_primaries"]]
    if color_meta.get("color_trc"):
        flags += ["-color_trc", color_meta["color_trc"]]
    if color_meta.get("colorspace"):
        flags += ["-colorspace", color_meta["colorspace"]]

    # x265-params for full HDR10 SEI (mastering display + content light level)
    if codec == "libx265" and color_meta.get("has_hdr_meta"):
        x265 = ["hdr-opt=1", "repeat-headers=1"]
        if color_meta.get("color_primaries"):
            x265.append(f"colorprim={color_meta['color_primaries']}")
        if color_meta.get("color_trc"):
            x265.append(f"transfer={color_meta['color_trc']}")
        if color_meta.get("colorspace"):
            x265.append(f"colormatrix={color_meta['colorspace']}")
        if color_meta.get("mastering_display"):
            x265.append(f"master-display={color_meta['mastering_display']}")
        if color_meta.get("content_light_level"):
            x265.append(f"max-cll={color_meta['content_light_level']}")
        flags += ["-x265-params", ":".join(x265)]

    return flags


def _resolve_pix_fmt(color_meta: dict, codec: str, force_pix_fmt: str = "") -> list:
    """
    Choose the output pixel format for an encode.

    - force_pix_fmt overrides everything (e.g. "yuv420p" to downgrade HDR to SDR preview).
    - 10-bit source + H.264 → keeps 10-bit (correct) but logs a warning about macOS
      preview incompatibility.  Recommend switching to libx265 or hevc_videotoolbox.
    - 10-bit source + HEVC / other → yuv420p10le.
    - 8-bit or unknown source → yuv420p.
    """
    if force_pix_fmt:
        return ["-pix_fmt", force_pix_fmt]

    bit_depth = (color_meta or {}).get("bit_depth", 8)

    if bit_depth >= 10:
        if codec in ("libx264", "h264_videotoolbox", "h264_nvenc", "h264_qsv", "h264_amf"):
            log.warning(
                "Source is %d-bit but codec is %s (H.264 High 10 profile). "
                "macOS QuickTime/Finder CANNOT preview H.264 High 10 — VideoToolbox only "
                "decodes H.264 up to the High (8-bit) profile. "
                "Switch to --codec libx265 (or hevc_videotoolbox) to preserve HDR and get "
                "Finder preview, or use --pix-fmt yuv420p to downgrade to 8-bit SDR.",
                bit_depth, codec,
            )
        return ["-pix_fmt", "yuv420p10le"]

    return ["-pix_fmt", "yuv420p"]


def get_keyframe_timestamps(cfg: EvalConfig) -> list[float]:
    """
    Return all keyframe (IDR/I-frame) timestamps from the reference video.
    Uses ffprobe packet flags — much faster than decoding frames since it only
    reads packet headers. A 'K' in the flags field indicates a keyframe.
    """
    cmd = [
        cfg.ffprobe_path, "-v", "quiet",
        "-select_streams", "v:0",
        "-show_packets",
        "-show_entries", "packet=pts_time,flags",
        "-of", "csv=p=0",
        cfg.reference,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    timestamps: list[float] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                pts = float(parts[0])
                if "K" in parts[1]:   # K flag = keyframe
                    timestamps.append(round(pts, 4))
            except (ValueError, IndexError):
                pass
    log.info("  Source keyframes: %d found", len(timestamps))
    return sorted(timestamps)


def get_keyframe_frames(cfg: EvalConfig) -> list[int]:
    """
    Return source keyframe positions as frame indices.
    Converts get_keyframe_timestamps() results using the fps from color_meta.

    Negative timestamps are filtered out: HEVC/H.264 streams often have an
    initial negative DTS (encoder delay buffer) that appears as a keyframe in
    ffprobe output but does not correspond to any presentable frame.  Using it
    as a context_start_frame would cause FFmpeg to seek to t<0, which either
    clamps to 0 (silently) or errors, and produces wrong preroll_frames counts.
    """
    fps = (cfg.color_meta or {}).get("fps", 0.0)
    if not fps:
        return []
    return sorted(
        round(t * fps)
        for t in get_keyframe_timestamps(cfg)
        if t >= 0.0   # drop negative-DTS sentinel frames common in HEVC streams
    )


def _context_start_frame(
    start_frame: int,
    context_frames: int,
    kf_frames: list[int],
) -> int:
    """
    Find the keyframe-aligned decode start for encoder preroll.

    Expands start_frame backward by context_frames, then snaps to the nearest
    keyframe at or before that position.  Always returns a value <= start_frame.

    If no keyframes exist or context is disabled (context_frames == 0),
    returns start_frame (no preroll).
    """
    if context_frames <= 0 or not kf_frames:
        return start_frame
    target = max(0, start_frame - context_frames)
    # Keyframes that fall at or before start_frame are candidates
    candidates = [kf for kf in kf_frames if kf <= start_frame]
    if not candidates:
        return start_frame
    # Prefer the largest keyframe that is at or before target
    at_or_before = [kf for kf in candidates if kf <= target]
    if at_or_before:
        return max(0, max(at_or_before))
    # All source keyframes are between target and start_frame — take the earliest
    return max(0, min(candidates))


def snap_boundaries_to_keyframes(
    boundaries: list[float],
    keyframe_times: list[float],
) -> list[float]:
    """
    Snap each detected shot boundary to the nearest source keyframe timestamp.

    Why this matters: split_chunk() uses stream-copy (-c copy) which can only
    start at keyframe boundaries. Without snapping, the seek lands at the nearest
    keyframe BEFORE the boundary, creating a duplicate-frame overlap between
    adjacent chunks in the final concat. Snapping eliminates that drift.
    """
    if not keyframe_times:
        log.warning("No keyframe timestamps available — boundaries not snapped.")
        return boundaries

    snapped: list[float] = []
    for t in boundaries:
        # Floor snap: always choose the keyframe AT OR BEFORE the boundary.
        # "Nearest" can snap forward into the next scene, causing the previous
        # chunk's stream-copy extraction to include frames from the wrong scene.
        # Floor snap ensures we never reach past the detected cut point.
        candidates = [k for k in keyframe_times if k <= t]
        nearest = max(candidates) if candidates else keyframe_times[0]
        drift = abs(nearest - t)
        if drift > 0.04:    # log only non-trivial snaps
            log.debug("  Snap %.4fs → %.4fs (drift %.4fs)", t, nearest, drift)
        snapped.append(nearest)

    unique = sorted(set(snapped))
    merged = len(boundaries) - len(unique)
    if merged > 0:
        log.info("  Keyframe snap: %d boundaries merged (two shots mapped to same keyframe)",
                 merged)
    return unique


def detect_boundaries_ffmpeg(cfg: EvalConfig, do_cfg: DynamicOptimizerConfig) -> list[int]:
    """
    Detect shot boundaries using FFmpeg's select+showinfo filter.
    Returns sorted list of boundary frame indices (not including frame 0).
    scene_threshold (0–100) is divided by 100 to match FFmpeg's 0.0–1.0 scale.
    """
    threshold = do_cfg.scene_threshold / 100.0
    cmd = [
        cfg.ffmpeg_path, "-i", cfg.reference,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-an", "-f", "null", "-",
    ]
    log.info("Shot detection: FFmpeg select filter (threshold=%.3f) …", threshold)
    result = subprocess.run(cmd, capture_output=True, text=True)

    boundaries: list[int] = []
    for line in result.stderr.splitlines():
        if "Parsed_showinfo" in line and "n:" in line:
            # showinfo emits "n:FRAME_NUM" — extract the decoded frame index directly.
            # This is the authoritative integer frame number, not a float timestamp.
            m = re.search(r"\bn:(\d+)\b", line)
            if m:
                frame = int(m.group(1))
                if frame > 0:           # exclude frame 0 (always the start, not a boundary)
                    boundaries.append(frame)

    unique = sorted(set(boundaries))
    log.info("  Detected %d shot boundaries (frame indices)", len(unique))
    return unique


def detect_boundaries_pyscenedetect(cfg: EvalConfig, do_cfg: DynamicOptimizerConfig) -> list[int]:
    """
    Detect shot boundaries using PySceneDetect ContentDetector.
    Returns sorted list of boundary frame indices (not including frame 0).
    Falls back gracefully to FFmpeg if PySceneDetect is not installed.
    """
    if not HAS_PYSCENEDETECT:
        log.warning("PySceneDetect not installed (pip install scenedetect). Falling back to FFmpeg.")
        return detect_boundaries_ffmpeg(cfg, do_cfg)

    log.info("Shot detection: PySceneDetect ContentDetector (threshold=%.1f) …", do_cfg.scene_threshold)
    video = open_video(cfg.reference)
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=do_cfg.scene_threshold))
    manager.detect_scenes(video)
    scenes = manager.get_scene_list()
    # Each scene is (start_tc, end_tc); FrameTimecode.get_frames() is the exact integer frame index.
    boundaries = [s[0].get_frames() for s in scenes[1:]]
    log.info("  Detected %d shot boundaries (frame indices)", len(boundaries))
    return sorted(set(boundaries))


def split_chunk(cfg: EvalConfig, start: float, end: float, out_path: str) -> bool:
    """
    Extract [start, end) from reference to out_path via FFmpeg stream copy.
    Uses input-side -ss for fast keyframe-accurate seek; -avoid_negative_ts
    resets PTS to zero so each chunk is self-contained.
    """
    cmd = [
        cfg.ffmpeg_path, "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", cfg.reference,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("Chunk split failed:\n%s", result.stderr[-800:])
        return False
    return True


def build_chunks(
    cfg: EvalConfig,
    do_cfg: DynamicOptimizerConfig,
    boundary_frames: list[int],
    kf_frames: list[int],
) -> list[ChunkInfo]:
    """
    Convert shot boundary frame indices into ChunkInfo objects.

    Frame indices are the authoritative coordinates throughout.  No stream-copy
    extraction is performed here — each ChunkInfo records the original source
    path, and the quality-sweep reference is extracted lazily in _process_chunk().

    The total frame count is conserved by construction:
        sum(chunk.frame_count for chunk in chunks) == total_frames
    because the chunks tile [0, total_frames) with no gaps or overlaps.
    """
    fps = (cfg.color_meta or {}).get("fps", 0.0)
    total_frames = (cfg.color_meta or {}).get("total_frames", 0)

    if not fps or not total_frames:
        raise ValueError(
            "build_chunks() requires probe_color_metadata() to have been called first "
            "(fps and total_frames must be set in cfg.color_meta)."
        )

    # Tile [0, total_frames) with the detected boundaries
    all_frames = sorted({0} | set(boundary_frames) | {total_frames})
    raw = [(all_frames[i], all_frames[i + 1]) for i in range(len(all_frames) - 1)]

    # Forward-merge: absorb chunks shorter than min_chunk_duration into the next one.
    # All arithmetic is integer (frames), not float (seconds), so no rounding drift.
    min_frames = max(1, round(do_cfg.min_chunk_duration * fps))
    merged: list[tuple[int, int]] = []
    i = 0
    while i < len(raw):
        start_f, end_f = raw[i]
        while (end_f - start_f) < min_frames and i + 1 < len(raw):
            i += 1
            end_f = raw[i][1]
        merged.append((start_f, end_f))
        i += 1

    context_frames = round(do_cfg.encoder_context_duration * fps)
    os.makedirs(do_cfg.chunk_dir, exist_ok=True)

    chunks: list[ChunkInfo] = []
    for idx, (start_f, end_f) in enumerate(merged):
        ctx_start = _context_start_frame(start_f, context_frames, kf_frames)
        chunk = ChunkInfo(
            index=idx,
            start_frame=start_f,
            end_frame=end_f,
            fps=fps,
            context_start_frame=ctx_start,
            source_path=cfg.reference,   # always the original; sweep ref extracted per-chunk
        )
        chunks.append(chunk)
        log.info(
            "  Chunk %04d: frames %d–%d  (%.3f–%.3f s  |  %d frames  |  "
            "context from frame %d, %d preroll frames)",
            idx, start_f, end_f - 1,
            chunk.start_time, chunk.end_time,
            chunk.frame_count,
            ctx_start, chunk.preroll_frames,
        )

    # Sanity-check: chunks must tile the full frame range exactly
    reconstructed = sum(c.frame_count for c in chunks)
    if reconstructed != total_frames:
        log.warning(
            "Frame-count mismatch after build_chunks: chunks sum to %d frames "
            "but source has %d.  Boundary rounding may have introduced drift.",
            reconstructed, total_frames,
        )
    else:
        log.info("  Frame conservation check: %d chunks × total = %d frames ✓",
                 len(chunks), total_frames)

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Orchestration
# ──────────────────────────────────────────────────────────────────────────────

def _process_single(
    cfg: EvalConfig,
    mode: str,
    value,
    ref_name: str,
) -> Optional[QualityResult]:
    """Encode and evaluate one variant. Safe to call from multiple threads."""
    if mode == "crf":
        label = f"crf{value}"
        enc_path = os.path.join(cfg.output_dir, f"{ref_name}_{label}.mp4")
        ok = encode_video(cfg, enc_path, crf=value)
    else:
        label = f"{int(value)}kbps"
        enc_path = os.path.join(cfg.output_dir, f"{ref_name}_{label}.mp4")
        ok = encode_video(cfg, enc_path, bitrate_kbps=value)

    if not ok:
        log.warning("Skipping %s (encode failed)", label)
        return None

    result = QualityResult(
        label=label,
        bitrate_kbps=probe_bitrate(cfg, enc_path),
        target_bitrate_kbps=value if mode == "bitrate" else None,
        crf=value if mode == "crf" else None,
        file_size_bytes=os.path.getsize(enc_path),
        output_path=enc_path,
    )

    # VMAF / PSNR / SSIM
    if cfg.enable_vmaf or cfg.enable_psnr or cfg.enable_ssim:
        vmaf_log = os.path.join(cfg.output_dir, f"{ref_name}_{label}_vmaf.json")
        metrics = run_ffmpeg_metrics(cfg, enc_path, vmaf_log)
        result.vmaf = metrics.get("vmaf")
        result.psnr = metrics.get("psnr")
        result.ssim = metrics.get("ssim")
        result.vmaf_frames = metrics.get("vmaf_frames", [])
        result.psnr_frames = metrics.get("psnr_frames", [])

    # AVQT
    if cfg.enable_avqt:
        avqt_csv = os.path.join(cfg.output_dir, f"{ref_name}_{label}_avqt.csv")
        avqt_data = run_avqt(cfg, enc_path, avqt_csv)
        result.avqt = avqt_data.get("avqt")
        result.avqt_frames = avqt_data.get("avqt_frames", [])

    log.info(
        "  %-12s  BR=%.0f kbps  VMAF=%s  PSNR=%s  AVQT=%s",
        label,
        result.bitrate_kbps,
        f"{result.vmaf:.2f}" if result.vmaf is not None else "N/A",
        f"{result.psnr:.2f}" if result.psnr is not None else "N/A",
        f"{result.avqt:.2f}" if result.avqt is not None else "N/A",
    )

    if not cfg.keep_encoded:
        _safe_remove(enc_path)

    return result


def run_pipeline(cfg: EvalConfig) -> list[QualityResult]:
    """Run full encode + evaluate pipeline. Returns list of QualityResult."""
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Probe source color/HDR metadata once — propagated to all encodes via cfg.color_meta.
    if not cfg.color_meta:
        cfg.color_meta = probe_color_metadata(cfg)
    _auto_configure_metrics(cfg)
    _warn_codec_mismatch(cfg)

    sweep = [("crf", v) for v in cfg.crfs] if cfg.crfs else [("bitrate", v) for v in cfg.bitrates_kbps]
    ref_name = Path(cfg.reference).stem

    results = []
    with ThreadPoolExecutor(max_workers=max(1, cfg.parallel_jobs)) as executor:
        futures = {
            executor.submit(_process_single, cfg, mode, value, ref_name): (mode, value)
            for mode, value in sweep
        }
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                results.append(r)

    results.sort(key=lambda r: r.bitrate_kbps)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic Optimizer — Per-Chunk Processing & Assembly
# ──────────────────────────────────────────────────────────────────────────────

def select_optimal(
    sweep_results: list,
    do_cfg: DynamicOptimizerConfig,
) -> Optional[QualityResult]:
    """
    Select the lowest-bitrate encode that meets do_cfg.vmaf_target.
    Falls back to the highest-VMAF result available if none reach the target.
    Assumes sweep_results is already sorted ascending by bitrate (run_pipeline guarantees this).
    """
    if not sweep_results:
        return None
    with_vmaf = [r for r in sweep_results if r.vmaf is not None]
    if not with_vmaf:
        return sweep_results[0]
    meeting = [r for r in with_vmaf if r.vmaf >= do_cfg.vmaf_target]
    if meeting:
        return meeting[0]   # already sorted ascending; first = lowest bitrate
    log.warning("No encode meets VMAF target %.1f; using best available.", do_cfg.vmaf_target)
    return max(with_vmaf, key=lambda r: r.vmaf)


def _resolve_quality_params(
    optimal: Optional[QualityResult],
    cfg: EvalConfig,
    do_cfg: DynamicOptimizerConfig,
) -> tuple[Optional[int], Optional[float]]:
    """
    Extract (crf, bitrate_kbps) from the optimal result.
    Returns the pair to pass to encode_video() or a raw FFmpeg command.
    """
    if optimal is None:
        return (cfg.crfs[0] if cfg.crfs else None,
                cfg.bitrates_kbps[0] if not cfg.crfs else None)
    if do_cfg.optimize_mode == "crf" and optimal.crf is not None:
        return (optimal.crf, None)
    if optimal.target_bitrate_kbps is not None:
        return (None, optimal.target_bitrate_kbps)
    if optimal.crf is not None:
        return (optimal.crf, None)
    return (None, optimal.bitrate_kbps)


def encode_chunk_final(
    chunk: ChunkInfo,
    optimal: Optional[QualityResult],
    cfg: EvalConfig,
    do_cfg: DynamicOptimizerConfig,
) -> str:
    """
    Final encode of a chunk at optimal quality settings.  Always reads from the
    original reference (cfg.reference) — never from the approximate sweep ref.

    No-preroll path  (chunk.preroll_frames == 0):
        Encodes [chunk.start_time … chunk.end_time] directly from the source.
        Forces an IDR at the very first frame (encoder always opens with IDR) so the concat demuxer
        gets a clean segment boundary.  Exactly chunk.frame_count frames are
        written via -frames:v.

    Context path  (chunk.preroll_frames > 0):
        Encodes [chunk.context_start_time … chunk.end_time] from the source so
        the encoder's look-ahead and rate-control see real prior frames.
        context_start_frame is already keyframe-aligned (computed by
        _context_start_frame()), so the input-side -ss lands on an I-frame and
        chunk.preroll_frames is exact integer arithmetic — no float rounding.
        An IDR is forced at frame index chunk.preroll_frames (= the logical
        chunk start).  The pre-roll is then stripped by a stream-copy trim that
        starts at preroll_frames / fps seconds and hard-stops at chunk.frame_count
        frames, leaving a clean segment that starts with an IDR and contains
        exactly chunk.frame_count frames.
    """
    out_path = os.path.join(do_cfg.chunk_dir, f"chunk_{chunk.index:04d}_final.mp4")
    crf_val, br_val = _resolve_quality_params(optimal, cfg, do_cfg)

    hw = cfg.hw_accel
    encoder = _resolve_encoder(cfg.codec, hw)

    def _encode_cmd(seek_time: float, duration: float) -> list[str]:
        """Shared FFmpeg encode command fragment (no keyframe / frame-count flags)."""
        cmd = [cfg.ffmpeg_path, "-y"]
        cmd += _hwaccel_flags(hw)
        cmd += ["-ss", str(seek_time),
                "-i", cfg.reference,
                "-t", str(duration),
                "-c:v", encoder, "-an"]
        cmd += _container_tag_flags(cfg.codec, encoder)
        cmd += _quality_flags(cfg.codec, hw, crf_val, br_val)
        cmd += _preset_flags(cfg.codec, hw, cfg.preset)
        cmd += _resolve_pix_fmt(cfg.color_meta, encoder, cfg.force_pix_fmt)
        cmd += _build_color_flags(cfg.color_meta, encoder, cfg.preserve_hdr)
        # -forced_idr 1 promotes forced I-frames to true IDRs so each segment
        # fully resets decoder state — required for clean concat demuxer output.
        # This is a *private encoder option* only understood by nvenc and qsv
        # hardware encoders.  Software encoders (libx264, libx265) and
        # VideoToolbox already produce IDRs from force_key_frames by default;
        # passing -forced_idr to them triggers "Unrecognized option" and aborts.
        if hw in ("nvenc", "qsv"):
            cmd += ["-forced_idr", "1"]
        if cfg.keyframe_interval > 0:
            cmd += ["-g", str(cfg.keyframe_interval)]
            if hw not in ("nvenc", "videotoolbox", "qsv", "amf"):
                cmd += ["-keyint_min", str(cfg.keyframe_interval)]
        if cfg.threads > 0:
            cmd += ["-threads", str(cfg.threads)]
        # Exact PTS timing: use fps_num as the MP4 timescale so each frame
        # occupies exactly fps_den ticks with no rounding.  For 59.94 fps
        # (60000/1001) this avoids the ~1 ms/frame drift that occurs when
        # libx264's default 12800 Hz timescale rounds 1001/60000 s per frame.
        fps_num = (cfg.color_meta or {}).get("fps_num", 0)
        if fps_num > 0 and out_path.lower().endswith(".mp4"):
            cmd += ["-video_track_timescale", str(fps_num)]
        return cmd

    # ── No-preroll path ────────────────────────────────────────────────────────
    # Flags appended to every no-preroll segment encode (shared between the main
    # no-preroll path and the context-encode fallback so they stay in sync).
    #
    # -force_key_frames is omitted here: the encoder always opens with an IDR
    # frame so the flag is redundant and avoids dealing with the expr: syntax
    # for the trivial frame-0 case.
    _no_preroll_tail = [
        "-frames:v", str(chunk.frame_count), # hard frame-count stop
        "-avoid_negative_ts", "make_zero",   # reset PTS for concat demuxer
    ]

    if chunk.preroll_frames == 0:
        cmd = _encode_cmd(chunk.start_time, chunk.duration) + _no_preroll_tail + [out_path]

        log.info("[Chunk %04d] No-preroll encode: %d frames from %.3fs",
                 chunk.index, chunk.frame_count, chunk.start_time)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("No-preroll encode failed:\n%s", result.stderr)
            return ""
        return out_path

    # ── Context path ───────────────────────────────────────────────────────────
    temp_path = out_path + "_ctx.mp4"
    ctx_duration = chunk.end_time - chunk.context_start_time

    cmd = _encode_cmd(chunk.context_start_time, ctx_duration)
    # Force IDR at the logical chunk start (integer frame count from the start
    # of the temp output).  Because context_start_frame is keyframe-aligned, the
    # decoder lands exactly there on the input-side seek, so preroll_frames is
    # the exact count of frames that precede the IDR in temp_path.
    #
    # -force_key_frames expr:EXPR is the only valid way to trigger on a frame
    # index; n is the 0-based output frame counter.  The other forms accepted by
    # -force_key_frames are timestamp lists and "chapters" — neither can address
    # an arbitrary frame index.
    cmd += ["-force_key_frames", f"expr:eq(n,{chunk.preroll_frames})"]
    cmd.append(temp_path)

    log.info("[Chunk %04d] Context encode: %d preroll frames from %.3fs (IDR @ frame %d)",
             chunk.index, chunk.preroll_frames, chunk.context_start_time,
             chunk.preroll_frames)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error("Context encode failed — falling back to no-preroll:\n%s", result.stderr)
        _safe_remove(temp_path)
        # Fallback: encode without pre-roll (cold start, but frame-exact output).
        cmd2 = _encode_cmd(chunk.start_time, chunk.duration) + _no_preroll_tail + [out_path]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode != 0:
            log.error("No-preroll fallback also failed:\n%s", result2.stderr)
            return ""
        return out_path

    # Stream-copy trim: discard the pre-roll, output exactly chunk.frame_count frames.
    #
    # -ss preroll_frames/fps  → seek to the IDR we forced at that frame index.
    #   Since context_start_frame is keyframe-aligned and the encode produced
    #   an IDR at expr:eq(n,preroll_frames), this seek lands precisely on that IDR.
    # -t chunk.duration       → timestamp-based upper bound (belt).
    # -frames:v frame_count   → frame-count hard stop (suspenders).
    # -avoid_negative_ts make_zero → resets PTS to 0 for clean concat input.
    trim_ss = chunk.preroll_frames / chunk.fps   # exact rational, no float drift
    trim_cmd = [
        cfg.ffmpeg_path, "-y",
        "-ss", str(trim_ss),
        "-t", str(chunk.duration),
        "-i", temp_path,
        "-c", "copy",
        "-frames:v", str(chunk.frame_count),
        "-avoid_negative_ts", "make_zero",
        out_path,
    ]
    result2 = subprocess.run(trim_cmd, capture_output=True, text=True)
    _safe_remove(temp_path)

    if result2.returncode != 0:
        log.error("Context trim failed:\n%s", result2.stderr)
        return ""

    return out_path


def extract_shot_features(chunk: ChunkInfo, cfg: EvalConfig) -> dict:
    """
    Extract cheap perceptual-complexity features from an already-extracted chunk file.
    Two lightweight FFmpeg filter passes — no encode required, typically 2–5× real-time.

    Pass 1 — signalstats:
        YAVG → mean_luma  (0–255 Y-channel mean)
        YDIF → ti         (mean absolute inter-frame Y diff — temporal complexity proxy)
        YMIN/YMAX → luma_range  (dynamic range proxy)

    Pass 2 — format=gray, sobel, signalstats:
        YAVG of Sobel-filtered frames → si  (mean Sobel edge energy — spatial complexity proxy)

    SI / TI are the ITU-R BT.1683 content-complexity metrics widely used in adaptive
    encoding research. Higher SI = more spatial detail; higher TI = more motion.

    Returns a flat dict suitable for JSON serialisation and use as an ML feature vector.
    All values are None if the source file is unavailable (safe to train/infer around).
    """
    features: dict = {
        "chunk_index": chunk.index,
        "start_time":  round(chunk.start_time, 4),
        "end_time":    round(chunk.end_time,   4),
        "duration":    round(chunk.duration,   4),
        "frame_count": chunk.frame_count,
        "si":          None,   # Spatial Information — mean Sobel energy (0–255 scale)
        "ti":          None,   # Temporal Information — mean inter-frame Y diff (0–255 scale)
        "mean_luma":   None,   # Mean Y-channel value (0–255)
        "luma_range":  None,   # Mean(YMAX) - Mean(YMIN) — dynamic range proxy
    }

    if not chunk.source_path or not os.path.exists(chunk.source_path):
        log.warning("[Chunk %04d] Feature extraction skipped — source path missing", chunk.index)
        return features

    def _parse_metadata(stdout: str) -> dict:
        """
        Parse signalstats metadata from FFmpeg metadata=mode=print:file=- stdout.

        FFmpeg 7+ writes signalstats via the metadata side-data mechanism rather
        than directly to stderr. Output format is one key=value per line:
            lavfi.signalstats.YAVG=128.45
            lavfi.signalstats.YDIF=3.21
            ...
        """
        result: dict = {"yavg": [], "ydif": [], "ymin": [], "ymax": []}
        key_map = {
            "lavfi.signalstats.YAVG": "yavg",
            "lavfi.signalstats.YDIF": "ydif",
            "lavfi.signalstats.YMIN": "ymin",
            "lavfi.signalstats.YMAX": "ymax",
        }
        for line in stdout.splitlines():
            line = line.strip()
            for prefix, bucket in key_map.items():
                if line.startswith(prefix + "="):
                    try:
                        result[bucket].append(float(line.split("=", 1)[1]))
                    except ValueError:
                        pass
        return result

    def _mean(lst):
        return sum(lst) / len(lst) if lst else None

    # ── Pass 1: base signalstats — TI + mean luma + dynamic range ─────────────
    # metadata=mode=print:file=- writes per-frame stats to stdout (FFmpeg 7+).
    # source_path is always the original full reference; -ss/-t restricts to
    # this chunk's time window so we don't analyse the whole file.
    r1 = subprocess.run(
        [cfg.ffmpeg_path,
         "-ss", str(chunk.start_time), "-t", str(chunk.duration),
         "-i", chunk.source_path,
         "-vf", "signalstats,metadata=mode=print:file=-",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    p1 = _parse_metadata(r1.stdout)
    if p1["yavg"]:
        features["mean_luma"]  = round(_mean(p1["yavg"]), 2)
    if p1["ydif"]:
        features["ti"]         = round(_mean(p1["ydif"]), 2)
    if p1["ymin"] and p1["ymax"]:
        features["luma_range"] = round(_mean(p1["ymax"]) - _mean(p1["ymin"]), 2)

    # ── Pass 2: Sobel-filtered signalstats — SI (mean edge energy) ─────────────
    # Converts to gray first so signalstats YAVG reflects Sobel magnitude on the
    # luma plane only (consistent with the ITU-R BT.1683 SI definition which
    # computes std-dev of the Sobel-filtered luma frame; mean is a practical proxy).
    r2 = subprocess.run(
        [cfg.ffmpeg_path,
         "-ss", str(chunk.start_time), "-t", str(chunk.duration),
         "-i", chunk.source_path,
         "-vf", "format=gray,sobel,signalstats,metadata=mode=print:file=-",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    p2 = _parse_metadata(r2.stdout)
    if p2["yavg"]:
        features["si"] = round(_mean(p2["yavg"]), 2)

    log.debug(
        "[Chunk %04d] Features  SI=%.1f  TI=%.1f  luma=%.1f  range=%.1f",
        chunk.index,
        features["si"] or 0.0, features["ti"] or 0.0,
        features["mean_luma"] or 0.0, features["luma_range"] or 0.0,
    )
    return features


def _process_chunk(
    chunk: ChunkInfo,
    cfg: EvalConfig,
    do_cfg: DynamicOptimizerConfig,
) -> ChunkOptimResult:
    """
    Full encode sweep + optimal selection + final encode for one chunk.
    Designed to be called concurrently from a ThreadPoolExecutor.

    Sweep reference extraction:
        A stream-copy clip is extracted here for use as the quality-sweep
        reference.  It is approximate (stream-copy snaps to the nearest source
        keyframe before the logical boundary), but that is acceptable because
        sweep VMAF scores are used only for relative CRF comparison within the
        same chunk.

    Final encode:
        encode_chunk_final() always reads from the original source and uses
        frame-count-based keyframe forcing (frames:N) so the output contains
        exactly chunk.frame_count frames regardless of any boundary drift.
    """
    sweep_dir  = os.path.join(do_cfg.chunk_dir, f"chunk_{chunk.index:04d}_sweep")
    sweep_ref  = os.path.join(do_cfg.chunk_dir, f"chunk_{chunk.index:04d}_ref.mp4")
    os.makedirs(sweep_dir, exist_ok=True)

    log.info("[Chunk %04d] Sweep  frames %d–%d  (%.3f–%.3f s  |  %d frames)",
             chunk.index, chunk.start_frame, chunk.end_frame - 1,
             chunk.start_time, chunk.end_time, chunk.frame_count)

    # Extract approximate stream-copy reference for quality sweep.
    # Boundary drift here is fine — sweep CRF selection is a relative comparison.
    split_chunk(cfg, chunk.start_time, chunk.end_time, sweep_ref)

    chunk_cfg = copy.copy(cfg)
    chunk_cfg.reference    = sweep_ref
    chunk_cfg.output_dir   = sweep_dir
    chunk_cfg.keep_encoded = False
    chunk_cfg.parallel_jobs = 1   # outer pool already parallelises across chunks

    # Propagate HDR settings explicitly — stream-copy chunk files often drop
    # color_trc metadata so _auto_configure_metrics would fail to re-detect HDR.
    # Lock the normalise flag so it isn't overridden by the per-chunk probe.
    chunk_cfg.hdr_vmaf_normalise        = cfg.hdr_vmaf_normalise
    chunk_cfg.hdr_vmaf_normalise_locked = True   # prevent re-probe from clearing it
    chunk_cfg.color_meta                = cfg.color_meta  # reuse parent probe result

    # Feature extraction for learned controller training (no encode required)
    features = extract_shot_features(chunk, cfg)

    sweep = run_pipeline(chunk_cfg)
    optimal = select_optimal(sweep, do_cfg)

    if optimal:
        log.info("[Chunk %04d] Optimal → %s  VMAF=%s  BR=%.0f kbps",
                 chunk.index, optimal.label,
                 f"{optimal.vmaf:.2f}" if optimal.vmaf is not None else "N/A",
                 optimal.bitrate_kbps)

    # Final encode always reads from original source — frame-exact output
    final_path = encode_chunk_final(chunk, optimal, cfg, do_cfg)

    if not final_path or not os.path.exists(final_path):
        log.error("[Chunk %04d] Final encode failed; skipping metrics for this chunk.",
                  chunk.index)
        _safe_remove(sweep_ref)
        return ChunkOptimResult(
            chunk=chunk,
            sweep_results=sweep,
            optimal=optimal,
            final_encoded_path="",
            vmaf=None,
            psnr=None,
            bitrate_kbps=0.0,
            features=features,
        )

    # Quality measurement: compare final encode against the (approx) sweep reference.
    # The final encode is frame-exact; the ref is approximate at the boundaries.
    # Scores are per-chunk relative measures used for reporting, not reselection.
    vmaf_json = os.path.join(do_cfg.chunk_dir, f"chunk_{chunk.index:04d}_final_vmaf.json")
    final_metrics: dict = {}
    if cfg.enable_vmaf or cfg.enable_psnr or cfg.enable_ssim:
        final_metrics = run_ffmpeg_metrics(chunk_cfg, final_path, vmaf_json)

    # Clean up temporary sweep reference (final encode no longer needs it)
    _safe_remove(sweep_ref)

    return ChunkOptimResult(
        chunk=chunk,
        sweep_results=sweep,
        optimal=optimal,
        final_encoded_path=final_path,
        vmaf=final_metrics.get("vmaf"),
        psnr=final_metrics.get("psnr"),
        bitrate_kbps=probe_bitrate(cfg, final_path),
        features=features,
    )


def aggregate_metrics_do(chunk_results: list) -> dict:
    """
    Compute frame-count-weighted aggregate VMAF, PSNR, and mean bitrate.
    Architecture choice: VMAF is measured per-chunk (in _process_chunk) then
    aggregated here — no full-video reconstruction is needed before measuring quality.
    """
    total_frames = sum(cr.chunk.frame_count for cr in chunk_results)
    if total_frames == 0:
        return {"vmaf": None, "psnr": None, "bitrate_kbps": 0.0}

    def weighted_mean(attr: str) -> Optional[float]:
        pairs = [
            (getattr(cr, attr), cr.chunk.frame_count)
            for cr in chunk_results
            if getattr(cr, attr) is not None
        ]
        if not pairs:
            return None
        total_w = sum(w for _, w in pairs)
        return sum(v * w for v, w in pairs) / total_w if total_w else None

    return {
        "vmaf":         weighted_mean("vmaf"),
        "psnr":         weighted_mean("psnr"),
        "bitrate_kbps": weighted_mean("bitrate_kbps") or 0.0,
    }


def concat_chunks(
    cfg: EvalConfig,
    do_cfg: DynamicOptimizerConfig,
    chunk_results: list,
) -> str:
    """
    Assemble final encoded chunks into one video via FFmpeg concat demuxer.
    Uses stream copy — no re-encode. Chunks are sorted by index before assembly.
    """
    ordered = sorted(chunk_results, key=lambda cr: cr.chunk.index)
    manifest = os.path.join(do_cfg.chunk_dir, "concat_manifest.txt")
    with open(manifest, "w") as f:
        for cr in ordered:
            f.write(f"file '{os.path.abspath(cr.final_encoded_path)}'\n")

    out_path = os.path.join(cfg.output_dir,
                            Path(cfg.reference).stem + "_dynamic_optimized.mp4")
    cmd = [
        cfg.ffmpeg_path, "-y",
        "-f", "concat", "-safe", "0",
        "-i", manifest,
        "-c", "copy",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("Concat failed:\n%s", result.stderr[-2000:])
        return ""
    log.info("Assembled %d chunks → %s", len(ordered), out_path)
    return out_path


# ── Shared setup helpers used by run_dynamic_optimizer + run_learned_controller ──

def _setup_run_dirs(cfg: EvalConfig, do_cfg: DynamicOptimizerConfig) -> None:
    """Create output and chunk working directories, setting do_cfg.chunk_dir."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    do_cfg.chunk_dir = os.path.join(cfg.output_dir, "chunks")
    os.makedirs(do_cfg.chunk_dir, exist_ok=True)


def _init_run_config(cfg: EvalConfig) -> None:
    """Probe color metadata, auto-configure metrics, and warn on codec mismatch."""
    if not cfg.color_meta:
        cfg.color_meta = probe_color_metadata(cfg)
    _auto_configure_metrics(cfg)
    _warn_codec_mismatch(cfg)


def _detect_shot_boundaries(cfg: EvalConfig, do_cfg: DynamicOptimizerConfig) -> list[int]:
    """Run shot boundary detection and return boundary frame indices."""
    if do_cfg.detector == "pyscenedetect":
        return detect_boundaries_pyscenedetect(cfg, do_cfg)
    return detect_boundaries_ffmpeg(cfg, do_cfg)


def _read_keyframe_info(cfg: EvalConfig) -> list[int]:
    """Return source keyframe frame indices, logging a warning if none are found."""
    log.info("Reading source keyframe positions …")
    kf_frames = get_keyframe_frames(cfg)
    if not kf_frames:
        log.warning("Could not read source keyframes; preroll windows will start at boundary.")
    return kf_frames


def run_dynamic_optimizer(
    cfg: EvalConfig,
    do_cfg: DynamicOptimizerConfig,
) -> DynamicOptimizerResult:
    """
    Netflix-style per-shot adaptive encoding pipeline.

    Architecture: parallel chunks → merge VMAF data (no full-video reconstruction)
      1. Shot boundary detection (FFmpeg or PySceneDetect)
      2. Chunk extraction at boundaries (stream copy, min_chunk_duration merging)
      3. Parallel per-chunk encode sweep + optimal selection + final encode
         - Each chunk is independently evaluated against its own reference slice
         - VMAF measured per-chunk; no dependency between chunks
      4. Frame-count-weighted metric aggregation
      5. FFmpeg concat demuxer assembly (stream copy, no re-encode)
    """
    _setup_run_dirs(cfg, do_cfg)
    _init_run_config(cfg)

    # ── 1. Shot boundary detection ────────────────────────────────────────────
    log.info("=== Dynamic Optimizer: Shot Detection (%s) ===", do_cfg.detector)
    boundaries = _detect_shot_boundaries(cfg, do_cfg)

    # ── 1b. Source keyframe positions (frame indices) ─────────────────────────
    # Logical shot boundaries are kept in frame-index space and are NOT snapped
    # to keyframes.  Instead, _context_start_frame() (called inside build_chunks)
    # snaps the encoder preroll window to the nearest source keyframe, giving the
    # encoder's lookahead the benefit of a clean I-frame start without shifting
    # the logical chunk boundaries at all.
    kf_frames = _read_keyframe_info(cfg)

    # ── 2. Build chunks (frame-exact, no extraction) ──────────────────────────
    chunks = build_chunks(cfg, do_cfg, boundaries, kf_frames)
    log.info("=== Dynamic Optimizer: %d chunks ready (parallel_jobs=%d) ===",
             len(chunks), cfg.parallel_jobs)

    # ── 3. Parallel per-chunk sweep + final encode ────────────────────────────
    # Each chunk's encode sweep and VMAF measurement are independent.
    # Aggregation happens after all futures complete — no reconstruction step needed.
    chunk_results: list[ChunkOptimResult] = []
    with ThreadPoolExecutor(max_workers=max(1, cfg.parallel_jobs)) as executor:
        futures = {
            executor.submit(_process_chunk, chunk, cfg, do_cfg): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            cr = future.result()
            if cr is not None:
                chunk_results.append(cr)

    chunk_results.sort(key=lambda cr: cr.chunk.index)

    # ── 4. Aggregate metrics (frame-count weighted) ───────────────────────────
    agg = aggregate_metrics_do(chunk_results)
    log.info("Aggregate  VMAF=%.2f  PSNR=%s  BR=%.0f kbps",
             agg["vmaf"] or 0.0,
             f"{agg['psnr']:.2f}" if agg["psnr"] else "N/A",
             agg["bitrate_kbps"])

    # ── 5. Concatenate final chunks ───────────────────────────────────────────
    final_path = ""
    if not do_cfg.no_concat:
        valid = [cr for cr in chunk_results if cr.final_encoded_path]
        if valid:
            final_path = concat_chunks(cfg, do_cfg, valid)

    return DynamicOptimizerResult(
        chunk_results=chunk_results,
        aggregate_vmaf=agg["vmaf"],
        aggregate_psnr=agg["psnr"],
        aggregate_bitrate_kbps=agg["bitrate_kbps"],
        final_video_path=final_path,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Learned Encoding Controller — Training Data, RD Model, Budget Allocator
# ──────────────────────────────────────────────────────────────────────────────

def export_training_data(chunk_results: list, path: str) -> None:
    """
    Export per-chunk oracle data to a JSON Lines file for model training.

    Each line is a self-contained JSON object:
      features         — content-complexity features from extract_shot_features()
      sweep            — list of {crf, bitrate_kbps, vmaf} RD-curve sample points
      optimal_crf      — CRF the oracle selected (int or null)
      optimal_vmaf     — measured VMAF of the final encode (float or null)
      optimal_bitrate_kbps — measured bitrate of the final encode

    Accumulate records from multiple titles across many encode sessions.
    Even ~50–200 diverse shots from 2–3 representative titles are enough to
    train a useful first-pass RDCurveModel for similar content.

    Training workflow:
      1. Run --dynamic-optimizer --export-training-data records.jsonl  (repeat for N titles)
      2. Run --train-model model.json --training-data records.jsonl
      3. Run --model-path model.json --bit-budget-kbps 2000            (O(1) encode/chunk)
    """
    n_written = 0
    with open(path, "w") as f:
        for cr in sorted(chunk_results, key=lambda x: x.chunk.index):
            if not cr.features:
                continue   # chunk had no source path — cannot train on it

            sweep_curve = []
            for r in cr.sweep_results:
                if r.crf is not None and r.vmaf is not None:
                    sweep_curve.append({
                        "crf":          r.crf,
                        "bitrate_kbps": round(r.bitrate_kbps, 2),
                        "vmaf":         round(r.vmaf, 4),
                    })
            if len(sweep_curve) < 2:
                continue   # need ≥2 RD points to fit a curve

            record = {
                "features":             cr.features,
                "sweep":                sorted(sweep_curve, key=lambda x: x["crf"]),
                "optimal_crf":          cr.optimal.crf if cr.optimal else None,
                "optimal_vmaf":         round(cr.vmaf, 4) if cr.vmaf is not None else None,
                "optimal_bitrate_kbps": round(cr.bitrate_kbps, 2),
            }
            f.write(json.dumps(record) + "\n")
            n_written += 1

    log.info("Training data → %s  (%d records written)", path, n_written)


class RDCurveModel:
    """
    Lightweight Rate-Distortion curve model for per-shot CRF prediction.

    Architecture (two-level):

      Level 1 — per-chunk curve fitting (from oracle sweep data):
        VMAF(CRF)  ≈ a + b·CRF       (linear in CRF; b < 0 — quality drops as CRF rises)
        log(rate)  ≈ c + d·CRF       (log-linear; d < 0 — rate drops exponentially with CRF)

      Level 2 — cross-chunk generalisation (requires numpy):
        Each of {a, b, c, d} is regressed against the feature vector
        [si, ti, mean_luma, luma_range, duration] using ordinary least squares.
        At inference time, a cheap FFmpeg feature pass replaces the encode sweep entirely.

    Accuracy improves with dataset size. Useful predictions emerge from as few as
    50 diverse shots; 500+ shots give broadcast-grade CRF estimates.

    Without numpy the model can still be loaded and used to return per-chunk
    curve parameters if they were saved from a training run that had numpy.
    """

    FEATURE_KEYS = ["si", "ti", "mean_luma", "luma_range", "duration"]

    def __init__(self):
        # Regression weights for each RD curve parameter: {param: [w_f0, w_f1, …, bias]}
        self._weights: dict[str, list] = {}
        self._training_count: int = 0

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, training_records: list[dict]) -> None:
        """
        Fit curve models from records loaded from a training JSONL file.
        Each record must have 'features' and 'sweep' keys (as written by
        export_training_data). Requires numpy.

        VMAF(CRF) model (quadratic):
            VMAF(CRF)  ≈ a + b·CRF + e·CRF²

        A quadratic fit captures the sigmoidal curvature of real VMAF-CRF
        curves (saturation near 100 at low CRF, floor at high CRF) much
        better than a linear fit, particularly near the high-quality operating
        point (target VMAF 90+) where linear models under-predict quality and
        recommend CRFs that are lower than necessary.

        Chunks with only 2 sweep points fall back to a linear fit (e=0); the
        regression still learns a valid predictor, it just has less curvature
        information for those records.

        Rate model (log-linear, unchanged):
            log(rate(CRF)) ≈ c + d·CRF
        """
        if not HAS_NUMPY:
            raise RuntimeError(
                "numpy is required for RDCurveModel.fit().  pip install numpy"
            )

        per_chunk: list[dict] = []
        for rec in training_records:
            feats = rec.get("features", {})
            sweep = rec.get("sweep", [])

            crfs = np.array([s["crf"]         for s in sweep], dtype=float)
            vmaf = np.array([s["vmaf"]         for s in sweep], dtype=float)
            rate = np.array([s["bitrate_kbps"] for s in sweep], dtype=float)

            if len(crfs) < 2:
                continue   # need ≥2 points to fit any curve

            # VMAF(CRF) ≈ a + b·CRF + e·CRF²
            # Degree: quadratic when ≥3 sweep points, linear for exactly 2
            # (polyfit with deg=2 returns [e, b, a]; deg=1 returns [b, a])
            vmaf_deg    = min(2, len(crfs) - 1)
            vmaf_coeffs = np.polyfit(crfs, vmaf, vmaf_deg)
            if vmaf_deg == 2:
                e_v = float(vmaf_coeffs[0])
                b_v = float(vmaf_coeffs[1])
                a_v = float(vmaf_coeffs[2])
            else:
                e_v = 0.0
                b_v = float(vmaf_coeffs[0])
                a_v = float(vmaf_coeffs[1])

            # log(rate(CRF)) = c + d·CRF  — guard against non-positive rates
            valid = rate > 0
            if valid.sum() < 2:
                continue
            rate_coeffs = np.polyfit(crfs[valid], np.log(rate[valid]), 1)  # [d, c]
            d_r, c_r = float(rate_coeffs[0]), float(rate_coeffs[1])

            fv = self._feature_vector(feats)
            if fv is None:
                continue   # a required feature was None — skip record

            per_chunk.append({"fv": fv, "a": a_v, "b": b_v, "e": e_v, "c": c_r, "d": d_r})

        if not per_chunk:
            raise ValueError(
                "No valid training records (each chunk needs ≥2 CRF sweep points "
                "and all feature keys present)."
            )

        # Build design matrix X: rows = chunks, cols = [features…, bias=1]
        # "e" is the new quadratic VMAF term; "c"/"d" remain the rate params
        X = np.array([p["fv"] for p in per_chunk])   # shape (N, n_features+1)
        for param in ("a", "b", "e", "c", "d"):
            y = np.array([p[param] for p in per_chunk])
            w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            self._weights[param] = w.tolist()

        quad_count = sum(1 for p in per_chunk if abs(p["e"]) > 1e-8)
        self._training_count = len(per_chunk)
        log.info(
            "RDCurveModel fitted: %d chunks  quadratic=%d  features=%s",
            self._training_count, quad_count, self.FEATURE_KEYS,
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_curve_params(self, features: dict) -> Optional[dict]:
        """
        Predict (a, b, e, c, d) RD curve parameters for a chunk described by features.

        VMAF params:  a (intercept), b (linear CRF coefficient), e (quadratic CRF coefficient)
        Rate params:  c (log-rate intercept), d (log-rate CRF coefficient)

        Returns None if the model has not been fitted or a required feature is missing.
        Version-1 models (no 'e' weights) return a dict without 'e'; callers use
        p.get('e', 0.0) so they degrade gracefully to linear VMAF prediction.
        """
        if not self._weights:
            return None
        fv = self._feature_vector(features)
        if fv is None:
            return None
        if not HAS_NUMPY:
            return None
        fv_np = np.array(fv)
        return {p: float(np.dot(np.array(w), fv_np)) for p, w in self._weights.items()}

    def predict_vmaf(self, features: dict, crf: float) -> Optional[float]:
        """Predict VMAF score for given content features at a given CRF.

        Uses the quadratic model: VMAF = a + b·CRF + e·CRF².
        Falls back to linear (e=0) for version-1 models that have no 'e' weights.
        """
        p = self.predict_curve_params(features)
        if p is None:
            return None
        e = p.get("e", 0.0)
        return p["a"] + p["b"] * crf + e * crf * crf

    def predict_rate_kbps(self, features: dict, crf: float) -> Optional[float]:
        """Predict bitrate (kbps) for given content features at a given CRF."""
        import math
        p = self.predict_curve_params(features)
        if p is None:
            return None
        try:
            return math.exp(p["c"] + p["d"] * crf)
        except OverflowError:
            return None

    def predict_crf(
        self,
        features: dict,
        vmaf_target: float,
        crf_range: tuple = (12, 45),
    ) -> Optional[int]:
        """
        Predict the lowest CRF that achieves vmaf_target.

        Inverts the quadratic VMAF(CRF) model:
            e·CRF² + b·CRF + (a − vmaf_target) = 0

        For version-1 models (e≈0) this reduces to the linear formula.
        When two real roots exist both candidates are evaluated; the one whose
        predicted VMAF is closest to the target is returned (handles the rare
        case where two CRF values on the curve produce similar quality scores).
        Result is clamped to crf_range and rounded to the nearest integer.
        """
        import math as _math
        p = self.predict_curve_params(features)
        if p is None:
            return None

        a, b, e = p["a"], p["b"], p.get("e", 0.0)
        lo, hi  = crf_range

        if abs(e) < 1e-8:
            # Linear model (version-1 weights or degenerate quadratic)
            if abs(b) < 1e-6:
                return None   # flat curve — cannot invert
            raw = (vmaf_target - a) / b
            return int(round(max(lo, min(hi, raw))))

        # Quadratic: e·CRF² + b·CRF + (a - vmaf_target) = 0
        disc = b * b - 4.0 * e * (a - vmaf_target)
        if disc < 0:
            # No real root — curve never crosses vmaf_target in the reals.
            # Clamp to the end of the range that maximises predicted quality.
            return lo if self.predict_vmaf(features, lo) >= vmaf_target else hi

        sqrt_disc = _math.sqrt(disc)
        root1 = (-b + sqrt_disc) / (2.0 * e)
        root2 = (-b - sqrt_disc) / (2.0 * e)

        # Pick the root closest to the target VMAF after clamping to [lo, hi]
        def _score(r: float) -> float:
            c = max(lo, min(hi, r))
            return abs(a + b * c + e * c * c - vmaf_target)

        raw = root1 if _score(root1) <= _score(root2) else root2
        return int(round(max(lo, min(hi, raw))))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialise model weights to a JSON file (format version 2)."""
        data = {
            "version":        2,   # v2 adds 'e' quadratic VMAF weight; v1 had a,b,c,d only
            "feature_keys":   self.FEATURE_KEYS,
            "weights":        self._weights,
            "training_count": self._training_count,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        log.info("RDCurveModel saved → %s  (trained on %d chunks)", path, self._training_count)

    @classmethod
    def load(cls, path: str) -> "RDCurveModel":
        """Deserialise model from a JSON file produced by save().

        Backward-compatible: version-1 files (no 'e' weight key) load without
        error.  predict_vmaf / predict_crf use p.get('e', 0.0) so they
        automatically fall back to linear VMAF prediction for v1 models.
        """
        with open(path) as f:
            data = json.load(f)
        m = cls()
        m._weights        = data["weights"]
        m._training_count = data.get("training_count", 0)
        file_ver          = data.get("version", 1)
        log.info(
            "RDCurveModel loaded: %s  version=%d  trained on %d chunks",
            path, file_ver, m._training_count,
        )
        if file_ver < 2:
            log.warning(
                "Model file is version %d (pre-quadratic). Predictions will use "
                "linear VMAF(CRF) only. Re-train with current code for quadratic accuracy.",
                file_ver,
            )
        return m

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _feature_vector(self, features: dict) -> Optional[list]:
        """
        Build [si, ti, mean_luma, luma_range, duration, 1.0] from a features dict.
        The trailing 1.0 is the bias term so the lstsq solution absorbs the intercept.
        Returns None if any required key is missing or None.
        """
        vals = []
        for key in self.FEATURE_KEYS:
            v = features.get(key)
            if v is None:
                return None   # incomplete feature vector — skip
            vals.append(float(v))
        vals.append(1.0)   # bias term
        return vals


class TitleBudgetAllocator:
    """
    Distribute a title-level bit budget across shots via Lagrange multiplier search.

    Solves the convex rate-distortion optimisation problem:
        max  Σ_i  VMAF_i(CRF_i)
        s.t. Σ_i  rate_i(CRF_i) · duration_i  ≤  total_budget_kilobits

    Via the Lagrangian decomposition (classic Everett / slope optimisation):
        For a fixed multiplier λ (price-per-kilobit):
          Each shot independently picks the CRF that maximises
            VMAF_i(CRF) − λ · rate_i(CRF) · duration_i
          (the sub-problems are independent → trivially parallel if desired)
        Binary search on λ finds the value that makes the budget constraint tight.

    The λ=0 solution ignores rate and maximises VMAF everywhere.
    Large λ forces aggressive rate reduction across all shots.
    The binary search finds the λ* that balances quality vs budget.

    Requires a fitted RDCurveModel whose predict_vmaf() and predict_rate_kbps()
    methods return sensible values for all chunks.
    """

    def allocate(
        self,
        chunks: list,                        # list[ChunkInfo]
        chunk_features: list[dict],          # parallel list of feature dicts
        model: "RDCurveModel",
        budget_kbps: float,                  # target mean bitrate across the title
        crf_range: tuple = (12, 45),
        lambda_search_iters: int = 64,
    ) -> dict[int, int]:
        """
        Returns {chunk_index: crf} for each chunk.

        budget_kbps is the target *mean* bitrate across the entire title.
        Total bit budget = budget_kbps × total_duration  (in kilobits).
        """
        import math

        total_duration = sum(c.duration for c in chunks)
        total_budget_kb = budget_kbps * total_duration   # kbps × s = kilobits

        crf_grid = list(range(crf_range[0], crf_range[1] + 1))

        def best_crf_for_lambda(feats: dict, duration: float, lam: float) -> int:
            """CRF that maximises VMAF − λ·rate·duration for this shot."""
            best_crf, best_score = crf_grid[-1], float("-inf")
            for crf in crf_grid:
                vmaf = model.predict_vmaf(feats, crf)
                rate = model.predict_rate_kbps(feats, crf)
                if vmaf is None or rate is None:
                    continue
                score = vmaf - lam * rate * duration
                if score > best_score:
                    best_score, best_crf = score, crf
            return best_crf

        def total_bits_for_lambda(lam: float) -> float:
            total = 0.0
            for chunk, feats in zip(chunks, chunk_features):
                crf  = best_crf_for_lambda(feats, chunk.duration, lam)
                rate = model.predict_rate_kbps(feats, crf)
                if rate:
                    total += rate * chunk.duration
            return total

        # ── Binary search on λ ────────────────────────────────────────────────
        # λ=0  → ignore rate → every shot picks lowest CRF → maximum bits
        # λ→∞  → rate penalty dominates → every shot picks highest CRF → minimum bits
        # Find the λ bracket where budget falls in [lam_lo, lam_hi]
        lam_lo, lam_hi = 0.0, 1.0
        # Expand upper bound until predicted rate at lam_hi ≤ budget
        for _ in range(24):
            if total_bits_for_lambda(lam_hi) <= total_budget_kb:
                break
            lam_hi *= 4.0

        for _ in range(lambda_search_iters):
            lam_mid = (lam_lo + lam_hi) / 2.0
            if total_bits_for_lambda(lam_mid) > total_budget_kb:
                lam_lo = lam_mid   # too many bits → increase penalty
            else:
                lam_hi = lam_mid   # within budget → can afford lower penalty

        lam_star = (lam_lo + lam_hi) / 2.0

        # ── Compute final per-chunk CRF assignments at λ* ─────────────────────
        assignments: dict[int, int] = {}
        pred_bits = 0.0
        for chunk, feats in zip(chunks, chunk_features):
            crf = best_crf_for_lambda(feats, chunk.duration, lam_star)
            assignments[chunk.index] = crf
            rate = model.predict_rate_kbps(feats, crf) or 0.0
            pred_bits += rate * chunk.duration
            log.debug(
                "[Chunk %04d] Allocator → CRF %d  pred VMAF=%.1f  pred rate=%.0f kbps",
                chunk.index, crf,
                model.predict_vmaf(feats, crf) or 0,
                rate,
            )

        pred_mean_rate = pred_bits / total_duration if total_duration > 0 else 0.0
        log.info(
            "TitleBudgetAllocator: λ*=%.6f  pred_mean_rate=%.0f kbps  budget=%.0f kbps",
            lam_star, pred_mean_rate, budget_kbps,
        )
        return assignments


def run_learned_controller(
    cfg: EvalConfig,
    do_cfg: DynamicOptimizerConfig,
    model: RDCurveModel,
    budget_kbps: Optional[float] = None,
) -> DynamicOptimizerResult:
    """
    Production-mode encoding pipeline driven by a trained RDCurveModel.

    Replaces the exhaustive per-chunk CRF sweep with two cheap passes:
      1. Shot boundary detection + chunk extraction  (same as oracle mode)
      2. FFmpeg signalstats feature extraction       (no encode — ~3–10× real-time)
      3a. TitleBudgetAllocator  (if budget_kbps is set — hierarchical, rate-constrained)
      3b. RDCurveModel.predict_crf()  (per-chunk VMAF-target prediction)
      4. One final encode per chunk at the predicted CRF
      5. VMAF measurement + metric aggregation + concat

    Speed vs oracle:  O(1) encode per chunk vs O(|crf_sweep|) encodes per chunk.
    For a typical 6-point CRF sweep this is a 6× wall-clock speedup on the encode
    phase, plus elimination of the sweep VMAF measurements.

    Quality vs oracle:  The model's prediction error adds ±1–2 CRF points of
    noise (for in-distribution content) — negligible in practice and improving
    with more training data.
    """
    _setup_run_dirs(cfg, do_cfg)
    _init_run_config(cfg)

    # ── 1. Shot detection + keyframe positions ────────────────────────────────
    log.info("=== Learned Controller: Shot Detection (%s) ===", do_cfg.detector)
    boundaries = _detect_shot_boundaries(cfg, do_cfg)
    kf_frames  = _read_keyframe_info(cfg)

    # ── 2. Build chunks (frame-exact, no extraction) ──────────────────────────
    chunks = build_chunks(cfg, do_cfg, boundaries, kf_frames)
    log.info("=== Learned Controller: %d chunks ready ===", len(chunks))

    # ── 3. Cheap feature extraction (no encode) ───────────────────────────────
    log.info("Extracting shot features (signalstats) …")
    chunk_features: list[dict] = []
    for chunk in chunks:
        feats = extract_shot_features(chunk, cfg)
        chunk_features.append(feats)
        log.info(
            "  [Chunk %04d] SI=%.1f  TI=%.1f  luma=%.1f  dur=%.2fs",
            chunk.index,
            feats.get("si") or 0.0,
            feats.get("ti") or 0.0,
            feats.get("mean_luma") or 0.0,
            chunk.duration,
        )

    # ── 4. CRF assignment: budget allocator or per-chunk VMAF-target ──────────
    if budget_kbps is not None:
        log.info("=== Title Budget Allocator: target %.0f kbps ===", budget_kbps)
        allocator = TitleBudgetAllocator()
        crf_assignments = allocator.allocate(
            chunks, chunk_features, model, budget_kbps
        )
    else:
        fallback_crf = cfg.crfs[0] if cfg.crfs else DEFAULT_CRF
        crf_assignments = {}
        for chunk, feats in zip(chunks, chunk_features):
            predicted = model.predict_crf(feats, do_cfg.vmaf_target)
            crf_val = predicted if predicted is not None else fallback_crf
            crf_assignments[chunk.index] = crf_val
            log.info(
                "  [Chunk %04d] Predicted CRF %d  (target VMAF %.1f)",
                chunk.index, crf_val, do_cfg.vmaf_target,
            )

    # ── 5. One encode per chunk at the predicted CRF ──────────────────────────
    chunk_results: list[ChunkOptimResult] = []
    for chunk, feats in zip(chunks, chunk_features):
        crf_val = crf_assignments.get(chunk.index, DEFAULT_CRF)

        # Capture the model's VMAF prediction *before* encoding so we can
        # compare it to the measured result later (prediction accuracy report).
        pred_vmaf = model.predict_vmaf(feats, crf_val)

        # Build a synthetic QualityResult so encode_chunk_final() has its interface
        predicted_optimal = QualityResult(
            label=f"crf{crf_val}",
            bitrate_kbps=model.predict_rate_kbps(feats, crf_val) or 0.0,
            crf=crf_val,
        )

        log.info(
            "[Chunk %04d] Encoding at predicted CRF %d  (predicted VMAF %.1f)",
            chunk.index, crf_val, pred_vmaf if pred_vmaf is not None else 0.0,
        )
        final_path = encode_chunk_final(chunk, predicted_optimal, cfg, do_cfg)

        # Measure actual quality of the predicted-CRF encode.
        # Extract a stream-copy reference clip for this chunk so the VMAF filter
        # compares the encoded frames against the matching source frames
        # (not against the start of the full original file).
        vmaf_json = os.path.join(
            do_cfg.chunk_dir, f"chunk_{chunk.index:04d}_final_vmaf.json"
        )
        final_metrics: dict = {}
        if cfg.enable_vmaf or cfg.enable_psnr or cfg.enable_ssim:
            ref_clip = os.path.join(do_cfg.chunk_dir,
                                    f"chunk_{chunk.index:04d}_lc_ref.mp4")
            split_chunk(cfg, chunk.start_time, chunk.end_time, ref_clip)
            chunk_cfg = copy.copy(cfg)
            chunk_cfg.reference             = ref_clip
            chunk_cfg.hdr_vmaf_normalise    = cfg.hdr_vmaf_normalise
            chunk_cfg.hdr_vmaf_normalise_locked = True
            chunk_cfg.color_meta            = cfg.color_meta
            final_metrics = run_ffmpeg_metrics(chunk_cfg, final_path, vmaf_json)
            _safe_remove(ref_clip)

        actual_vmaf = final_metrics.get("vmaf")
        actual_br   = probe_bitrate(cfg, final_path)

        # Per-chunk prediction error (positive = model was too conservative / under-predicted)
        if actual_vmaf is not None and pred_vmaf is not None:
            err = actual_vmaf - pred_vmaf
            log.info(
                "[Chunk %04d] Done  CRF=%d  pred_VMAF=%.2f  actual_VMAF=%.2f  "
                "error=%+.2f  BR=%.0f kbps",
                chunk.index, crf_val, pred_vmaf, actual_vmaf, err, actual_br,
            )
        else:
            log.info(
                "[Chunk %04d] Done  CRF=%d  actual_VMAF=%s  BR=%.0f kbps",
                chunk.index, crf_val,
                f"{actual_vmaf:.2f}" if actual_vmaf is not None else "N/A",
                actual_br,
            )

        chunk_results.append(ChunkOptimResult(
            chunk=chunk,
            sweep_results=[],            # no sweep in predict mode
            optimal=predicted_optimal,
            final_encoded_path=final_path,
            vmaf=actual_vmaf,
            psnr=final_metrics.get("psnr"),
            bitrate_kbps=actual_br,
            features=feats,
            predicted_vmaf=pred_vmaf,
        ))

    chunk_results.sort(key=lambda cr: cr.chunk.index)

    # Log aggregate prediction accuracy across all chunks
    pred_pairs = [
        (cr.predicted_vmaf, cr.vmaf)
        for cr in chunk_results
        if cr.predicted_vmaf is not None and cr.vmaf is not None
    ]
    if pred_pairs:
        import math as _math
        errs    = [a - p for p, a in pred_pairs]
        mae     = sum(abs(e) for e in errs) / len(errs)
        rmse    = _math.sqrt(sum(e * e for e in errs) / len(errs))
        pct_ok  = 100.0 * sum(1 for e in errs if abs(e) <= 2.0) / len(errs)
        log.info(
            "Prediction accuracy (n=%d)  MAE=%.2f  RMSE=%.2f  within±2VMAF=%.1f%%",
            len(errs), mae, rmse, pct_ok,
        )

    # ── 6. Aggregate metrics + concat ─────────────────────────────────────────
    agg = aggregate_metrics_do(chunk_results)
    log.info(
        "Aggregate  VMAF=%.2f  PSNR=%s  BR=%.0f kbps",
        agg["vmaf"] or 0.0,
        f"{agg['psnr']:.2f}" if agg["psnr"] else "N/A",
        agg["bitrate_kbps"],
    )

    final_path = ""
    if not do_cfg.no_concat:
        valid = [cr for cr in chunk_results if cr.final_encoded_path]
        if valid:
            final_path = concat_chunks(cfg, do_cfg, valid)

    return DynamicOptimizerResult(
        chunk_results=chunk_results,
        aggregate_vmaf=agg["vmaf"],
        aggregate_psnr=agg["psnr"],
        aggregate_bitrate_kbps=agg["bitrate_kbps"],
        final_video_path=final_path,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CSV Export
# ──────────────────────────────────────────────────────────────────────────────

def export_csv(results: list[QualityResult], path: str):
    """Write summary results to CSV."""
    fieldnames = [
        "label", "bitrate_kbps", "target_bitrate_kbps", "crf",
        "vmaf", "psnr", "ssim", "avqt", "file_size_bytes",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "label": r.label,
                "bitrate_kbps": round(r.bitrate_kbps, 2),
                "target_bitrate_kbps": r.target_bitrate_kbps,
                "crf": r.crf,
                "vmaf": round(r.vmaf, 4) if r.vmaf is not None else "",
                "psnr": round(r.psnr, 4) if r.psnr is not None else "",
                "ssim": round(r.ssim, 6) if r.ssim is not None else "",
                "avqt": round(r.avqt, 4) if r.avqt is not None else "",
                "file_size_bytes": r.file_size_bytes,
            })
    log.info("CSV saved → %s", path)


def export_chunk_csv(chunk_results: list, path: str):
    """Write per-chunk optimal-encode summary to CSV."""
    fieldnames = [
        "chunk_index", "start_time", "end_time", "duration_s", "frame_count",
        "optimal_label", "optimal_crf", "optimal_target_bitrate_kbps",
        "actual_bitrate_kbps", "vmaf", "psnr", "final_encoded_path",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cr in sorted(chunk_results, key=lambda x: x.chunk.index):
            opt = cr.optimal
            writer.writerow({
                "chunk_index":                 cr.chunk.index,
                "start_time":                  round(cr.chunk.start_time, 4),
                "end_time":                    round(cr.chunk.end_time, 4),
                "duration_s":                  round(cr.chunk.duration, 4),
                "frame_count":                 cr.chunk.frame_count,
                "optimal_label":               opt.label if opt else "",
                "optimal_crf":                 opt.crf if opt else "",
                "optimal_target_bitrate_kbps": opt.target_bitrate_kbps if opt else "",
                "actual_bitrate_kbps":         round(cr.bitrate_kbps, 2),
                "vmaf":                        round(cr.vmaf, 4) if cr.vmaf is not None else "",
                "psnr":                        round(cr.psnr, 4) if cr.psnr is not None else "",
                "final_encoded_path":          cr.final_encoded_path,
            })
    log.info("Per-chunk CSV → %s", path)


def export_prediction_accuracy_csv(chunk_results: list, path: str) -> None:
    """
    Write a per-chunk model accuracy report for a learned-controller run.

    Columns
    -------
    chunk_index       — 0-based chunk number
    start_time        — chunk start (seconds)
    duration_s        — chunk duration (seconds)
    predicted_crf     — CRF the model recommended
    predicted_vmaf    — VMAF the model expected at that CRF
    actual_vmaf       — VMAF measured after encoding
    vmaf_error        — actual_vmaf − predicted_vmaf  (+ve = model was too conservative)
    abs_vmaf_error    — |vmaf_error|
    actual_bitrate_kbps

    The final row is an aggregate summary (mean / RMSE / max).
    """
    import math

    fieldnames = [
        "chunk_index", "start_time", "duration_s",
        "predicted_crf", "predicted_vmaf", "actual_vmaf",
        "vmaf_error", "abs_vmaf_error", "actual_bitrate_kbps",
    ]

    rows = []
    for cr in sorted(chunk_results, key=lambda x: x.chunk.index):
        p_vmaf = cr.predicted_vmaf
        a_vmaf = cr.vmaf
        err    = (a_vmaf - p_vmaf) if (a_vmaf is not None and p_vmaf is not None) else None
        rows.append({
            "chunk_index":          cr.chunk.index,
            "start_time":           round(cr.chunk.start_time, 4),
            "duration_s":           round(cr.chunk.duration, 4),
            "predicted_crf":        cr.optimal.crf if cr.optimal else "",
            "predicted_vmaf":       round(p_vmaf, 4) if p_vmaf is not None else "",
            "actual_vmaf":          round(a_vmaf, 4) if a_vmaf is not None else "",
            "vmaf_error":           round(err, 4)    if err   is not None else "",
            "abs_vmaf_error":       round(abs(err), 4) if err is not None else "",
            "actual_bitrate_kbps":  round(cr.bitrate_kbps, 2),
        })

    # Compute aggregate stats from rows that have both predicted and actual VMAF
    errors = [r["vmaf_error"] for r in rows if isinstance(r["vmaf_error"], float)]
    if errors:
        mae    = sum(abs(e) for e in errors) / len(errors)
        rmse   = math.sqrt(sum(e * e for e in errors) / len(errors))
        mean_e = sum(errors) / len(errors)
        max_ae = max(abs(e) for e in errors)
        pct_ok = 100.0 * sum(1 for e in errors if abs(e) <= 2.0) / len(errors)
    else:
        mae = rmse = mean_e = max_ae = pct_ok = 0.0

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        # Append summary footer as a plain comment row
        writer.writerow({
            "chunk_index":    "SUMMARY",
            "start_time":     "",
            "duration_s":     f"n={len(errors)}",
            "predicted_crf":  "",
            "predicted_vmaf": f"MAE={mae:.3f}",
            "actual_vmaf":    f"RMSE={rmse:.3f}",
            "vmaf_error":     f"mean={mean_e:.3f}",
            "abs_vmaf_error": f"max={max_ae:.3f}",
            "actual_bitrate_kbps": f"within±2={pct_ok:.1f}%",
        })

    log.info(
        "Prediction accuracy → %s  MAE=%.2f  RMSE=%.2f  within±2VMAF=%.1f%%",
        path, mae, rmse, pct_ok,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Hardware Acceleration
# ──────────────────────────────────────────────────────────────────────────────

# Maps (software_codec, hw_accel) → FFmpeg hardware encoder name
HW_CODEC_MAP = {
    ("libx264", "videotoolbox"): "h264_videotoolbox",
    ("libx265", "videotoolbox"): "hevc_videotoolbox",
    ("libx264", "nvenc"):        "h264_nvenc",
    ("libx265", "nvenc"):        "hevc_nvenc",
    ("libx264", "qsv"):          "h264_qsv",
    ("libx265", "qsv"):          "hevc_qsv",
    ("libx264", "amf"):          "h264_amf",
    ("libx265", "amf"):          "hevc_amf",
}

# Maps libx264/libx265 preset names → NVENC p-scale (p1=fastest … p7=slowest)
NVENC_PRESET_MAP = {
    "ultrafast": "p1", "superfast": "p1", "veryfast": "p2",
    "faster":    "p3", "fast":      "p4", "medium":   "p4",
    "slow":      "p5", "slower":    "p6", "veryslow": "p7",
}


# ──────────────────────────────────────────────────────────────────────────────
# Codec Profile Table
# ──────────────────────────────────────────────────────────────────────────────
#
# One entry per software encoder.  Fields:
#   hw_variants    — {hw_accel_name: ffmpeg_encoder_name}  (empty = no HW variant)
#   qt_tag         — container tag required for QuickTime/Finder; None = not needed
#   max_bit_depth  — highest bit depth the encoder supports for yuv420p* output
#   hdr_metadata   — True if the encoder can carry full HDR10 SEI (master display, MaxCLL)
#   source_codec   — canonical ffprobe codec_name string for this codec family (for mismatch warn)
#   extra_crf_flags — additional flags required when using CRF mode (e.g. VP9 needs -b:v 0)
#   preset_flag    — FFmpeg flag name for preset-equivalent (None = uses -preset)
#   preset_map     — {x264_preset_name: encoder_value}  (empty = pass preset name as-is)
#
CODEC_PROFILES = {
    "libx264": {
        "hw_variants": {
            "videotoolbox": "h264_videotoolbox",
            "nvenc":        "h264_nvenc",
            "qsv":          "h264_qsv",
            "amf":          "h264_amf",
        },
        "qt_tag":        None,
        "max_bit_depth": 10,
        "hdr_metadata":  False,
        "source_codec":  "h264",
        "extra_crf_flags": [],
        "preset_flag":   None,
        "preset_map":    {},
    },
    "libx265": {
        "hw_variants": {
            "videotoolbox": "hevc_videotoolbox",
            "nvenc":        "hevc_nvenc",
            "qsv":          "hevc_qsv",
            "amf":          "hevc_amf",
        },
        "qt_tag":        "hvc1",
        "max_bit_depth": 12,
        "hdr_metadata":  True,
        "source_codec":  "hevc",
        "extra_crf_flags": [],
        "preset_flag":   None,
        "preset_map":    {},
    },
    "libvpx-vp9": {
        "hw_variants":   {},
        "qt_tag":        None,
        "max_bit_depth": 10,
        "hdr_metadata":  False,
        "source_codec":  "vp9",
        "extra_crf_flags": ["-b:v", "0"],   # VP9 CRF requires constrained-quality mode
        "preset_flag":   "cpu-used",
        "preset_map": {
            "ultrafast": "8", "superfast": "8", "veryfast": "7", "faster": "6",
            "fast": "5", "medium": "4", "slow": "3", "slower": "2", "veryslow": "1",
        },
    },
    "libaom-av1": {
        "hw_variants":   {},
        "qt_tag":        "av01",
        "max_bit_depth": 10,
        "hdr_metadata":  True,
        "source_codec":  "av1",
        "extra_crf_flags": [],
        "preset_flag":   "cpu-used",
        "preset_map": {
            "ultrafast": "8", "superfast": "7", "veryfast": "6", "faster": "5",
            "fast": "4", "medium": "3", "slow": "2", "slower": "1", "veryslow": "0",
        },
    },
    "libsvtav1": {
        "hw_variants":   {},
        "qt_tag":        "av01",
        "max_bit_depth": 10,
        "hdr_metadata":  True,
        "source_codec":  "av1",
        "extra_crf_flags": [],
        "preset_flag":   "preset",
        "preset_map": {
            "ultrafast": "13", "superfast": "12", "veryfast": "10", "faster": "9",
            "fast": "8", "medium": "6", "slow": "5", "slower": "4", "veryslow": "3",
        },
    },
}


def _resolve_encoder(sw_codec: str, hw: str) -> str:
    """Return the actual FFmpeg encoder name for a (software codec, hw backend) pair."""
    if hw == "none":
        return sw_codec
    profile = CODEC_PROFILES.get(sw_codec, {})
    return profile.get("hw_variants", {}).get(hw, sw_codec)


def _quality_flags(
    sw_codec: str, hw: str, crf: Optional[int], bitrate_kbps: Optional[float]
) -> list:
    """Build the quality/rate flags for the chosen encoder."""
    if crf is not None:
        if hw == "videotoolbox":
            qv = max(1, min(100, int((51 - crf) * 100 / 51)))
            return ["-q:v", str(qv)]
        if hw == "nvenc":
            return ["-cq", str(crf)]
        if hw == "qsv":
            return ["-global_quality", str(crf)]
        flags = ["-crf", str(crf)]
        flags += CODEC_PROFILES.get(sw_codec, {}).get("extra_crf_flags", [])
        return flags
    if bitrate_kbps is not None:
        return ["-b:v", f"{bitrate_kbps}k"]
    raise ValueError("Must specify either crf or bitrate_kbps")


def _preset_flags(sw_codec: str, hw: str, preset: str) -> list:
    """Build preset/speed flags, translating x264 preset names where needed."""
    if hw == "videotoolbox":
        return []   # VideoToolbox has no preset knob
    if hw == "nvenc":
        return ["-preset", NVENC_PRESET_MAP.get(preset, "p4")]
    profile = CODEC_PROFILES.get(sw_codec, {})
    flag_name = profile.get("preset_flag")
    if flag_name is None:
        # Standard encoders (-preset)
        return ["-preset", preset]
    pmap = profile.get("preset_map", {})
    return [f"-{flag_name}", pmap.get(preset, pmap.get("medium", preset))]


def _hwaccel_flags(hw: str) -> list:
    """Build input-side hardware acceleration flags."""
    if hw == "videotoolbox":
        return ["-hwaccel", "videotoolbox"]
    if hw == "nvenc":
        return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    if hw == "qsv":
        return ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
    return []


def _container_tag_flags(sw_codec: str, encoder: str) -> list:
    """Return [-tag:v, TAG] if the encoder needs a container tag for QuickTime compat."""
    qt_tag = CODEC_PROFILES.get(sw_codec, {}).get("qt_tag")
    if qt_tag:
        return ["-tag:v", qt_tag]
    return []


def _warn_codec_mismatch(cfg: EvalConfig) -> None:
    """Warn when the chosen output codec family doesn't match the source codec family."""
    src = (cfg.color_meta or {}).get("source_codec", "")
    if not src:
        return
    profile = CODEC_PROFILES.get(cfg.codec, {})
    expected_src = profile.get("source_codec", "")
    if expected_src and src != expected_src:
        log.warning(
            "Codec mismatch: source is %s but encoding with %s (%s family). "
            "This is intentional if you are cross-codec comparing, but HDR metadata "
            "and lossless passthrough are only guaranteed within the same codec family.",
            src, cfg.codec, expected_src,
        )


# ──────────────────────────────────────────────────────────────────────────────
# VMAF Auto-Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Transfer functions that indicate HDR content (PQ and HLG, plus SMPTE ST 428)
_HDR_TRANSFERS = {"smpte2084", "arib-std-b67", "smpte428"}

# Ordered search paths for VMAF model directories; glob patterns accepted.
# Homebrew installs are version-pinned; the wildcard matches any libvmaf version.
_VMAF_MODEL_DIRS = [
    "/opt/homebrew/Cellar/libvmaf/*/share/libvmaf/model",  # Homebrew (Apple Silicon / Intel)
    "/usr/local/Cellar/libvmaf/*/share/libvmaf/model",      # Homebrew (older Intel path)
    "/usr/share/libvmaf/model",                              # Linux system install
    "/usr/local/share/libvmaf/model",                        # Linux local install
]


def _find_vmaf_model(color_meta: dict) -> str:
    """
    Auto-select the best available VMAF model file based on source resolution.

    Selection logic:
      - 4K source (height >= 2160) → prefer vmaf_float_4k_v0.6.1.json
        (float-precision model calibrated for 4K viewing distance)
      - HD or below → prefer vmaf_float_v0.6.1.json
      - Fall back to vmaf_v0.6.1.json (integer arithmetic, compatible with older libvmaf)
      - Returns "" if nothing is found (FFmpeg uses its built-in default model)
    """
    import glob as _glob

    is_4k = (color_meta or {}).get("height", 0) >= 2160

    # Candidate filenames in preference order
    candidates = (
    ["vmaf_v0.6.1.json", "vmaf_4k_v0.6.1.json",
     "vmaf_float_v0.6.1.json", "vmaf_float_4k_v0.6.1.json"]
    if is_4k else
    ["vmaf_v0.6.1.json", "vmaf_float_v0.6.1.json"]
    )

    for dir_pattern in _VMAF_MODEL_DIRS:
        for model_dir in sorted(_glob.glob(dir_pattern), reverse=True):  # newest version first
            for candidate in candidates:
                path = os.path.join(model_dir, candidate)
                if os.path.isfile(path):
                    return path
    return ""


def _validate_vmaf_model(ffmpeg_path: str, model_path: str) -> bool:
    import tempfile
    tmp_log = os.path.join(tempfile.gettempdir(), "_vmaf_validate_tmp.json")
    cmd = [
        ffmpeg_path, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=64x64:r=1:d=0.1",
        "-f", "lavfi", "-i", "color=c=black:s=64x64:r=1:d=0.1",
        "-filter_complex",
        f"[0:v][1:v]libvmaf=log_fmt=json:log_path={tmp_log}:model=path={model_path}",
        "-f", "null", "-",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if r.returncode != 0:
            log.warning("VMAF model probe failed for %s", model_path)
            if r.stderr:
                log.warning("FFmpeg stderr: %s", r.stderr.strip())
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        log.warning("VMAF model probe timed out for %s", model_path)
        return False
    finally:
        _safe_remove(tmp_log)


def _auto_configure_metrics(cfg: EvalConfig) -> None:
    """
    Mutate cfg to auto-select VMAF model and enable HDR normalisation.

    Called after probe_color_metadata() has populated cfg.color_meta.
    Only fills in values the user has not explicitly set:
      - cfg.vmaf_model: set to best matching model file (if "" and file found
                        AND model passes a compatibility probe against FFmpeg's
                        linked libvmaf version)
      - cfg.hdr_vmaf_normalise: set to True for PQ/HLG sources (if not already True)

    Logs the decisions made so the user can override with explicit flags.
    """
    meta = cfg.color_meta or {}

    # ── VMAF model ────────────────────────────────────────────────────────────
    if not cfg.vmaf_model:
        model_path = _find_vmaf_model(meta)
        if model_path:
            res_tag = "4K" if meta.get("height", 0) >= 2160 else "HD"
            if _validate_vmaf_model(cfg.ffmpeg_path, model_path):
                cfg.vmaf_model = model_path
                log.info("Auto VMAF model (%s): %s", res_tag, model_path)
            else:
                log.warning(
                    "Auto VMAF model (%s) found at %s but is incompatible with "
                    "FFmpeg's linked libvmaf — using FFmpeg's built-in default instead.",
                    res_tag, model_path,
                )
                log.warning(
                    "Version mismatch likely.  Diagnose: "
                    "'ffmpeg -version | grep libvmaf' vs 'ls %s'",
                    os.path.dirname(model_path),
                )
        else:
            log.info("Auto VMAF model: no model file found, using FFmpeg built-in default")

    # ── HDR normalisation ─────────────────────────────────────────────────────
    if not cfg.hdr_vmaf_normalise_locked and not cfg.hdr_vmaf_normalise:
        trc = meta.get("color_trc") or ""
        if trc in _HDR_TRANSFERS:
            cfg.hdr_vmaf_normalise = True
            log.info(
                "Auto HDR VMAF normalise: enabled (source trc=%s). "
                "Disable with --no-hdr-vmaf-normalise.", trc
            )


# ──────────────────────────────────────────────────────────────────────────────
# Run Manifest — provenance record written alongside every results CSV
# ──────────────────────────────────────────────────────────────────────────────

def _probe_tool_versions(cfg: EvalConfig) -> dict:
    """Collect FFmpeg, FFprobe, and libvmaf version strings."""
    versions = {}

    # FFmpeg version line
    try:
        r = subprocess.run(
            [cfg.ffmpeg_path, "-version"],
            capture_output=True, text=True, timeout=10,
        )
        first_line = r.stdout.splitlines()[0] if r.stdout else ""
        versions["ffmpeg"] = first_line.strip()
        # libvmaf version embedded in build config
        for line in r.stdout.splitlines():
            if "libvmaf" in line.lower():
                versions["libvmaf_build"] = line.strip()
                break
    except Exception as e:
        versions["ffmpeg"] = f"probe failed: {e}"

    # FFprobe version line
    try:
        r = subprocess.run(
            [cfg.ffprobe_path, "-version"],
            capture_output=True, text=True, timeout=10,
        )
        first_line = r.stdout.splitlines()[0] if r.stdout else ""
        versions["ffprobe"] = first_line.strip()
    except Exception as e:
        versions["ffprobe"] = f"probe failed: {e}"

    return versions


def _get_git_hash() -> str:
    """Return the current git commit hash, or empty string if not in a repo."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def write_run_manifest(
    cfg: EvalConfig,
    args_namespace,
    mode: str,
    do_cfg=None,
    chunk_results=None,
    warnings: list = None,
) -> str:
    """
    Write a run_manifest.json alongside the results CSV.

    Records everything needed to reproduce or audit this run:
      - Exact command line arguments
      - Source file technical metadata (from probe_color_metadata)
      - Tool versions (FFmpeg, libvmaf)
      - Pipeline mode and key parameters
      - HDR normalisation decision and why
      - Per-chunk summary (dynamic optimizer mode only)
      - Any quality warnings (VMAF suspiciously low, etc.)
      - Timestamp and git hash

    Returns the path to the written manifest file.
    """
    import datetime
    import sys

    os.makedirs(cfg.output_dir, exist_ok=True)
    manifest_path = os.path.join(cfg.output_dir, "run_manifest.json")

    # ── Source metadata ───────────────────────────────────────────────────────
    meta = cfg.color_meta or {}
    source_info = {
        "path":              os.path.abspath(cfg.reference),
        "filename":          os.path.basename(cfg.reference),
        "resolution":        f"{meta.get('width', 0)}x{meta.get('height', 0)}",
        "fps":               meta.get("fps"),
        "total_frames":      meta.get("total_frames"),
        "codec":             meta.get("source_codec"),
        "pix_fmt":           meta.get("pix_fmt"),
        "bit_depth":         meta.get("bit_depth"),
        "color_primaries":   meta.get("color_primaries"),
        "color_trc":         meta.get("color_trc"),
        "colorspace":        meta.get("colorspace"),
        "has_hdr_meta":      meta.get("has_hdr_meta", False),
        "mastering_display": meta.get("mastering_display"),
        "content_light_level": meta.get("content_light_level"),
    }

    # ── HDR normalisation decision ────────────────────────────────────────────
    trc = meta.get("color_trc") or ""
    hdr_decision = {
        "hdr_vmaf_normalise":  cfg.hdr_vmaf_normalise,
        "explicitly_set":      cfg.hdr_vmaf_normalise_locked,
        "auto_triggered":      (trc in _HDR_TRANSFERS and not cfg.hdr_vmaf_normalise_locked),
        "source_trc":          trc,
        "note": (
            "HDR normalisation ON: PQ→linear→BT.709 applied before VMAF scoring. "
            "Scores are internally consistent but not directly comparable to SDR VMAF."
            if cfg.hdr_vmaf_normalise else
            "HDR normalisation OFF. If source is HDR (PQ/HLG), VMAF scores will be "
            "physically meaningless — use --hdr-vmaf-normalise or re-run."
        ),
    }

    # ── Encode configuration ──────────────────────────────────────────────────
    encode_cfg = {
        "codec":             cfg.codec,
        "preset":            cfg.preset,
        "hw_accel":          cfg.hw_accel,
        "parallel_jobs":     cfg.parallel_jobs,
        "threads":           cfg.threads,
        "vmaf_model":        cfg.vmaf_model or "FFmpeg built-in default",
        "keyframe_interval": cfg.keyframe_interval,
        "force_pix_fmt":     cfg.force_pix_fmt or "auto",
        "preserve_hdr":      cfg.preserve_hdr,
        "metrics": {
            "vmaf":  cfg.enable_vmaf,
            "psnr":  cfg.enable_psnr,
            "ssim":  cfg.enable_ssim,
            "avqt":  cfg.enable_avqt,
        },
    }

    if cfg.crfs:
        encode_cfg["sweep_mode"]   = "crf"
        encode_cfg["sweep_values"] = cfg.crfs
    else:
        encode_cfg["sweep_mode"]     = "bitrate"
        encode_cfg["sweep_values_kbps"] = cfg.bitrates_kbps

    # ── Dynamic optimizer parameters ─────────────────────────────────────────
    do_params = None
    if do_cfg is not None:
        do_params = {
            "scene_threshold":        do_cfg.scene_threshold,
            "min_chunk_duration_s":   do_cfg.min_chunk_duration,
            "vmaf_target":            do_cfg.vmaf_target,
            "optimize_mode":          do_cfg.optimize_mode,
            "detector":               do_cfg.detector,
            "snap_to_keyframes":      do_cfg.snap_to_keyframes,
            "encoder_context_s":      do_cfg.encoder_context_duration,
        }

    # ── Per-chunk summary ─────────────────────────────────────────────────────
    chunk_summary = None
    quality_warnings = list(warnings or [])

    if chunk_results:
        chunk_summary = []
        for cr in sorted(chunk_results, key=lambda x: x.chunk.index):
            entry = {
                "index":        cr.chunk.index,
                "start_s":      round(cr.chunk.start_time, 4),
                "end_s":        round(cr.chunk.end_time, 4),
                "duration_s":   round(cr.chunk.duration, 4),
                "frames":       cr.chunk.frame_count,
                "optimal_crf":  cr.optimal.crf if cr.optimal else None,
                "bitrate_kbps": round(cr.bitrate_kbps, 1),
                "vmaf":         round(cr.vmaf, 4) if cr.vmaf is not None else None,
                "psnr":         round(cr.psnr, 4) if cr.psnr is not None else None,
            }
            chunk_summary.append(entry)

            # Flag suspiciously low VMAF
            if cr.vmaf is not None and cr.vmaf < 40.0:
                quality_warnings.append(
                    f"Chunk {cr.chunk.index} VMAF={cr.vmaf:.2f} is critically low "
                    f"(< 40). Likely cause: HDR source without --hdr-vmaf-normalise, "
                    f"or frame count mismatch between encode and reference."
                )

            # Flag extreme bitrates (likely encoding or concat artefact)
            if cr.bitrate_kbps > 50_000:
                quality_warnings.append(
                    f"Chunk {cr.chunk.index} bitrate={cr.bitrate_kbps:.0f} kbps is "
                    f"unusually high (> 50 Mbps). Check for encoding configuration issues."
                )

    # ── Assemble manifest ─────────────────────────────────────────────────────
    manifest = {
        "pipeline_version":  "1.0",
        "git_hash":          _get_git_hash(),
        "timestamp":         datetime.datetime.now().isoformat(),
        "mode":              mode,
        "command_line":      " ".join(sys.argv),
        "source":            source_info,
        "hdr_vmaf":          hdr_decision,
        "encode_config":     encode_cfg,
        "dynamic_optimizer": do_params,
        "tool_versions":     _probe_tool_versions(cfg),
        "chunk_summary":     chunk_summary,
        "quality_warnings":  quality_warnings,
        "output_dir":        os.path.abspath(cfg.output_dir),
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("Run manifest → %s", manifest_path)

    # Surface warnings prominently
    if quality_warnings:
        log.warning("=== QUALITY WARNINGS ===")
        for w in quality_warnings:
            log.warning("  ⚠  %s", w)

    return manifest_path
# ──────────────────────────────────────────────────────────────────────────────

METRIC_META = {
    "vmaf":  {"label": "VMAF",  "color": "#2563eb", "ylim": (0, 100)},
    "psnr":  {"label": "PSNR (dB)", "color": "#16a34a", "ylim": (20, 50)},
    "ssim":  {"label": "SSIM", "color": "#9333ea", "ylim": (0, 1)},
    "avqt":  {"label": "AVQT", "color": "#ea580c", "ylim": (0, 100)},
}


def _rd_curve_data(results: list[QualityResult], metric: str):
    """Extract (bitrate, score) pairs, sorted by bitrate."""
    pairs = []
    for r in results:
        score = getattr(r, metric, None)
        if score is not None and r.bitrate_kbps > 0:
            pairs.append((r.bitrate_kbps, score, r.label))
    pairs.sort(key=lambda x: x[0])
    return pairs


def plot_rd_curves(
    results: list[QualityResult],
    output_dir: str,
    title_prefix: str = "",
    log_scale: bool = True,
):
    """
    Generate Rate-Distortion (bit distortion) curve plots.
    One combined multi-panel figure + individual per-metric PNGs.
    """
    if not HAS_MATPLOTLIB:
        log.error("matplotlib not installed. Cannot plot. Run: pip install matplotlib")
        return

    active_metrics = [
        m for m in ("vmaf", "psnr", "ssim", "avqt")
        if any(getattr(r, m) is not None for r in results)
    ]

    if not active_metrics:
        log.warning("No metric data available to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── Combined figure ──────────────────────────────────────────────────────
    ncols = min(2, len(active_metrics))
    nrows = (len(active_metrics) + 1) // 2
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 5 * nrows),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#0f172a")

    if len(active_metrics) == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    prefix = f"{title_prefix} – " if title_prefix else ""
    fig.suptitle(
        f"{prefix}Rate–Distortion Curves",
        color="white", fontsize=16, fontweight="bold", y=1.01,
    )

    for idx, metric in enumerate(active_metrics):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        _draw_single_rd(ax, results, metric, log_scale=log_scale)

    # Hide unused subplots
    total_cells = nrows * ncols
    for idx in range(len(active_metrics), total_cells):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    combined_path = os.path.join(output_dir, "rd_curves_combined.png")
    fig.savefig(combined_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Combined RD plot → %s", combined_path)

    # ── Individual per-metric figures ────────────────────────────────────────
    for metric in active_metrics:
        fig2, ax2 = plt.subplots(figsize=(9, 5.5))
        fig2.patch.set_facecolor("#0f172a")
        _draw_single_rd(ax2, results, metric, log_scale=log_scale, big=True)
        fig2.suptitle(
            f"{prefix}{METRIC_META[metric]['label']} Rate–Distortion",
            color="white", fontsize=14, fontweight="bold",
        )
        single_path = os.path.join(output_dir, f"rd_{metric}.png")
        fig2.savefig(single_path, dpi=150, bbox_inches="tight",
                     facecolor=fig2.get_facecolor())
        plt.close(fig2)
        log.info("  RD plot [%s] → %s", metric.upper(), single_path)


def _draw_single_rd(ax, results, metric: str, log_scale: bool = True, big: bool = False):
    """Draw one Rate-Distortion panel onto ax."""
    meta = METRIC_META[metric]
    pairs = _rd_curve_data(results, metric)
    if not pairs:
        ax.set_visible(False)
        return

    bitrates = [p[0] for p in pairs]
    scores   = [p[1] for p in pairs]
    labels   = [p[2] for p in pairs]

    ax.set_facecolor("#1e293b")
    ax.grid(color="#334155", linestyle="--", linewidth=0.6, alpha=0.7)

    # Line + markers
    ax.plot(
        bitrates, scores,
        color=meta["color"], linewidth=2.2,
        marker="o", markersize=7 if big else 5,
        markerfacecolor="white", markeredgecolor=meta["color"], markeredgewidth=1.5,
        zorder=5,
    )

    # Shade under curve
    ax.fill_between(bitrates, scores, alpha=0.10, color=meta["color"])

    # Data labels
    for br, sc, lbl in zip(bitrates, scores, labels):
        ax.annotate(
            f"{sc:.1f}",
            xy=(br, sc), xytext=(0, 9),
            textcoords="offset points",
            ha="center", fontsize=7.5 if big else 6.5,
            color="white", alpha=0.85,
        )

    if log_scale:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

    ax.set_xlabel("Bitrate (kbps)", color="#94a3b8", fontsize=10)
    ax.set_ylabel(meta["label"], color="#94a3b8", fontsize=10)
    ax.set_title(meta["label"], color="white", fontsize=11, fontweight="semibold")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    ymin, ymax = meta["ylim"]
    actual_min = min(scores)
    actual_max = max(scores)
    pad = (actual_max - actual_min) * 0.15 or 2
    ax.set_ylim(
        max(ymin, actual_min - pad),
        min(ymax, actual_max + pad) if ymax > 1 else min(ymax, actual_max + 0.02),
    )


def plot_temporal(results: list[QualityResult], output_dir: str, title_prefix: str = ""):
    """Per-frame temporal quality plot for each encoded variant."""
    if not HAS_MATPLOTLIB:
        return

    # Collect variants that have frame-level data
    frame_data = {}
    for r in results:
        if r.vmaf_frames:
            frame_data.setdefault("vmaf", []).append(r)
        if r.psnr_frames:
            frame_data.setdefault("psnr", []).append(r)
        if r.avqt_frames:
            frame_data.setdefault("avqt", []).append(r)

    for metric, variants in frame_data.items():
        meta = METRIC_META[metric]
        fig, ax = plt.subplots(figsize=(12, 4.5))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")
        ax.grid(color="#334155", linestyle="--", linewidth=0.5, alpha=0.6)

        cmap = plt.cm.get_cmap("plasma", len(variants))
        for i, r in enumerate(variants):
            frames = getattr(r, f"{metric}_frames")
            ax.plot(
                range(len(frames)), frames,
                label=r.label, color=cmap(i), linewidth=1.2, alpha=0.85,
            )

        ax.set_xlabel("Frame", color="#94a3b8", fontsize=10)
        ax.set_ylabel(meta["label"], color="#94a3b8", fontsize=10)
        prefix = f"{title_prefix} – " if title_prefix else ""
        ax.set_title(
            f"{prefix}Per-Frame {meta['label']}",
            color="white", fontsize=12, fontweight="bold",
        )
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

        legend = ax.legend(
            loc="lower right", fontsize=8,
            facecolor="#0f172a", edgecolor="#334155", labelcolor="white",
        )

        out_path = os.path.join(output_dir, f"temporal_{metric}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        log.info("Temporal plot [%s] → %s", metric.upper(), out_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Video quality evaluation pipeline (VMAF, PSNR, SSIM, AVQT) with RD-curve graphs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("reference", help="Path to reference (original) video file")
    p.add_argument("-o", "--output-dir", default="./eval_output",
                   help="Directory for outputs")
    p.add_argument("--validate", action="store_true",
                   help="Run source validation checks and exit without processing. "
                        "Use before batch runs to catch bad source files early.")
    p.add_argument("--codec", default="libx264",
                   choices=list(CODEC_PROFILES.keys()),
                   help="Video codec to use for encoding "
                        "(libsvtav1 is the fastest AV1 software encoder; "
                        "libaom-av1 is reference AV1 — very slow)")
    p.add_argument("--preset", default="medium", help="Encoder preset")
    p.add_argument("--bitrates", nargs="+", type=float,
                   default=[200, 500, 1000, 2000, 4000, 8000],
                   metavar="KBPS", help="Bitrate sweep values in kbps")
    p.add_argument("--crfs", nargs="+", type=int, default=[],
                   metavar="CRF", help="CRF sweep values (overrides --bitrates)")
    p.add_argument("--no-vmaf", action="store_true", help="Disable VMAF")
    p.add_argument("--no-psnr", action="store_true", help="Disable PSNR")
    p.add_argument("--no-ssim", action="store_true", help="Disable SSIM")
    p.add_argument("--avqt", action="store_true", help="Enable AVQT evaluation")
    p.add_argument("--avqt-path", default="avqt", help="Path to AVQT binary")
    p.add_argument("--vmaf-model", default="",
                   help="Path to custom VMAF model .json.  Auto-selected if not set: "
                        "4K source → vmaf_float_4k_v0.6.1.json, HD → vmaf_float_v0.6.1.json. "
                        "Omit to use the auto-detected model or FFmpeg's built-in default.")
    p.add_argument("--pix-fmt", default="",
                   help="Force output pixel format (e.g. 'yuv420p' to downgrade 10-bit HDR "
                        "source to 8-bit for Finder preview, or 'yuv420p10le' to preserve bit "
                        "depth). Default: auto-match source (10-bit source → yuv420p10le). "
                        "Note: 10-bit H.264 (High 10) is not previewed by macOS QuickTime — "
                        "use --codec libx265 to preserve HDR and get Finder preview.")
    p.add_argument("--no-preserve-hdr", action="store_true",
                   help="Disable HDR metadata passthrough (color primaries, transfer, "
                        "mastering display, MaxCLL/MaxFALL). Use when intentionally "
                        "converting HDR to SDR or the encoder rejects the metadata.")
    # HDR VMAF normalisation: auto-enabled for PQ/HLG sources; explicit flags override.
    hdr_norm = p.add_mutually_exclusive_group()
    hdr_norm.add_argument("--hdr-vmaf-normalise", action="store_true", default=False,
                          help="Force-enable HDR VMAF normalisation: PQ → scene-linear "
                               "(npl=203) → BT.709 gamma before scoring. Auto-enabled for "
                               "PQ/HLG sources; use this flag to apply it to SDR content too.")
    hdr_norm.add_argument("--no-hdr-vmaf-normalise", action="store_true", default=False,
                          help="Disable HDR VMAF normalisation even for PQ/HLG sources. "
                               "Use when comparing results to pre-normalised SDR VMAF scores "
                               "or when your FFmpeg build lacks zscale support.")
    p.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg binary")
    p.add_argument("--ffprobe", default="ffprobe", help="Path to ffprobe binary")
    p.add_argument("--keep-encoded", action="store_true",
                   help="Keep encoded video files after evaluation")
    p.add_argument("--linear-scale", action="store_true",
                   help="Use linear x-axis instead of log scale for RD curves")
    p.add_argument("--threads", type=int, default=0,
                   help="Number of encoder threads (0=auto)")
    p.add_argument("--hw-accel", default="none",
                   choices=["none", "videotoolbox", "nvenc", "qsv", "amf"],
                   help="GPU/hardware acceleration backend (videotoolbox=Apple, nvenc=NVIDIA, qsv=Intel, amf=AMD)")
    p.add_argument("--jobs", type=int, default=1,
                   help="Parallel encode+evaluate workers")
    p.add_argument("--keyframe-interval", type=int, default=0, metavar="FRAMES",
                   help="Force keyframe every N frames in all encodes (-g N). "
                        "Recommended for ABR streaming: 48 for 24fps/2s, 60 for 30fps/2s. "
                        "0 = encoder default.")

    # ── Dynamic Optimizer ─────────────────────────────────────────────────────
    do = p.add_argument_group("Dynamic Optimizer (Netflix-style per-shot encoding)")
    do.add_argument("--dynamic-optimizer", action="store_true",
                    help="Enable per-shot adaptive encode sweep")
    do.add_argument("--scene-threshold", type=float, default=27.0, metavar="N",
                    help="Shot detection sensitivity 0–100. FFmpeg: divided by 100 for "
                         "scene expr. PySceneDetect: used directly. Higher = fewer cuts. (default: 27)")
    do.add_argument("--min-chunk-duration", type=float, default=2.0, metavar="SECONDS",
                    help="Merge chunks shorter than this into the next one (default: 2.0s)")
    do.add_argument("--vmaf-target", type=float, default=93.0,
                    help="VMAF floor for optimal encode selection (default: 93.0)")
    do.add_argument("--optimize-mode", default="crf", choices=["crf", "bitrate"],
                    help="Whether to sweep and select using CRF or bitrate ladder (default: crf)")
    do.add_argument("--detector", default="ffmpeg", choices=["ffmpeg", "pyscenedetect"],
                    help="Scene detection backend; pyscenedetect requires: pip install scenedetect")
    do.add_argument("--no-concat", action="store_true",
                    help="Skip final FFmpeg concat; leave per-chunk encodes in place")
    do.add_argument("--no-snap-keyframes", action="store_true",
                    help="Disable keyframe boundary snapping. Without snapping, stream-copy "
                         "extraction may produce duplicate frames at chunk seams in the final concat.")
    do.add_argument("--encoder-context", type=float, default=2.0, metavar="SECONDS",
                    help="Pre-roll context duration for encoder look-ahead warm-up (default: 2.0s). "
                         "The encoder sees this many seconds of prior frames before the chunk "
                         "boundary, improving rate allocation at the start of each chunk. "
                         "Set to 0 to disable (faster but cold look-ahead at every boundary).")
    do.add_argument("--export-training-data", default="", metavar="PATH",
                    help="After a --dynamic-optimizer run, export oracle training data to this "
                         "JSON Lines file. Each line contains {features, sweep, optimal_crf, …} "
                         "for one chunk. Accumulate across titles, then train with --train-model.")

    # ── Learned Encoding Controller ───────────────────────────────────────────
    lc = p.add_argument_group(
        "Learned Encoding Controller",
        description=(
            "Train an RDCurveModel from oracle data, then run a predict-mode pipeline "
            "that replaces per-chunk CRF sweeps with a single cheap feature pass + model inference."
        ),
    )
    lc.add_argument("--train-model", default="", metavar="OUTPUT_JSON",
                    help="Train RDCurveModel from --training-data and save weights to OUTPUT_JSON. "
                         "Requires numpy.  Example: --train-model model.json --training-data records.jsonl")
    lc.add_argument("--training-data", default="", metavar="JSONL_PATH",
                    help="JSON Lines file produced by one or more --export-training-data runs. "
                         "Required when using --train-model.")
    lc.add_argument("--model-path", default="", metavar="JSON_PATH",
                    help="Path to a trained RDCurveModel JSON (from --train-model). "
                         "Activates predict mode: one encode per chunk instead of a full CRF sweep.")
    lc.add_argument("--bit-budget-kbps", type=float, default=0.0, metavar="KBPS",
                    help="Title-level mean bitrate target in kbps. Activates the "
                         "TitleBudgetAllocator which assigns per-chunk CRFs via Lagrange "
                         "optimisation to hit the budget while maximising aggregate VMAF. "
                         "Requires --model-path. If omitted, per-chunk prediction uses --vmaf-target.")
    return p


def main():
    args = build_parser().parse_args()

    cfg = EvalConfig(
        reference=args.reference,
        output_dir=args.output_dir,
        codec=args.codec,
        preset=args.preset,
        bitrates_kbps=args.bitrates,
        crfs=args.crfs,
        enable_vmaf=not args.no_vmaf,
        enable_psnr=not args.no_psnr,
        enable_ssim=not args.no_ssim,
        enable_avqt=args.avqt,
        avqt_path=args.avqt_path,
        ffmpeg_path=args.ffmpeg,
        ffprobe_path=args.ffprobe,
        vmaf_model=args.vmaf_model,
        keep_encoded=args.keep_encoded,
        threads=args.threads,
        hw_accel=args.hw_accel,
        parallel_jobs=args.jobs,
        keyframe_interval=args.keyframe_interval,
        force_pix_fmt=args.pix_fmt,
        preserve_hdr=not args.no_preserve_hdr,
        hdr_vmaf_normalise=args.hdr_vmaf_normalise,
        hdr_vmaf_normalise_locked=args.hdr_vmaf_normalise or args.no_hdr_vmaf_normalise,
    )

    log.info("=== Video Quality Evaluation Pipeline ===")
    log.info("Reference : %s", cfg.reference)
    log.info("Codec     : %s (preset=%s)", cfg.codec, cfg.preset)

    # ── Source validation (always runs; --validate exits after report) ────────
    validation = validate_source(args.reference, ffprobe_path=args.ffprobe)
    validation.print_report()
    if not validation.ok:
        log.error("Source validation failed — aborting. Fix errors above and retry.")
        return
    if args.validate:
        return  # --validate flag: report only, no processing
    if cfg.crfs:
        log.info("Sweep     : CRF %s", cfg.crfs)
    else:
        log.info("Sweep     : Bitrates %s kbps", cfg.bitrates_kbps)
    log.info("Metrics   : VMAF=%s  PSNR=%s  SSIM=%s  AVQT=%s",
             cfg.enable_vmaf, cfg.enable_psnr, cfg.enable_ssim, cfg.enable_avqt)
    log.info("HW Accel  : %s  Jobs=%d", cfg.hw_accel, cfg.parallel_jobs)

    # ── Train-model-only mode ─────────────────────────────────────────────────
    # python pipeline.py <any_ref> --train-model model.json --training-data records.jsonl
    if args.train_model:
        if not args.training_data:
            log.error("--train-model requires --training-data <path_to_jsonl>")
            return
        if not HAS_NUMPY:
            log.error("--train-model requires numpy.  pip install numpy")
            return
        log.info("=== Training RDCurveModel from %s ===", args.training_data)
        records = []
        with open(args.training_data) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        log.info("  Loaded %d training records", len(records))
        model = RDCurveModel()
        model.fit(records)
        model.save(args.train_model)
        log.info("=== Model saved → %s ===", args.train_model)
        return

    # Shared DO config builder (used by both oracle and learned-controller modes)
    def _build_do_cfg() -> DynamicOptimizerConfig:
        return DynamicOptimizerConfig(
            scene_threshold=args.scene_threshold,
            min_chunk_duration=args.min_chunk_duration,
            vmaf_target=args.vmaf_target,
            optimize_mode=args.optimize_mode,
            detector=args.detector,
            no_concat=args.no_concat,
            snap_to_keyframes=not args.no_snap_keyframes,
            encoder_context_duration=args.encoder_context,
        )

    # ── Learned Controller mode (predict — no full CRF sweep) ─────────────────
    # python pipeline.py ref.mp4 --model-path model.json [--bit-budget-kbps 2000]
    if args.model_path:
        do_cfg = _build_do_cfg()
        log.info("── Learned Controller (predict mode) ──────────────────────────")
        log.info("Model         : %s", args.model_path)
        if args.bit_budget_kbps > 0:
            log.info("Budget        : %.0f kbps (TitleBudgetAllocator)", args.bit_budget_kbps)
        else:
            log.info("VMAF target   : %.1f (per-chunk CRF prediction)", do_cfg.vmaf_target)

        model = RDCurveModel.load(args.model_path)
        budget = args.bit_budget_kbps if args.bit_budget_kbps > 0 else None
        do_result = run_learned_controller(cfg, do_cfg, model, budget_kbps=budget)

        if not do_result.chunk_results:
            log.error("No chunk results produced. Check errors above.")
            return

        chunk_csv    = os.path.join(cfg.output_dir, "chunks_results.csv")
        accuracy_csv = os.path.join(cfg.output_dir, "prediction_accuracy.csv")
        export_chunk_csv(do_result.chunk_results, chunk_csv)
        export_prediction_accuracy_csv(do_result.chunk_results, accuracy_csv)
        agg_result = QualityResult(
            label="learned_controller",
            bitrate_kbps=do_result.aggregate_bitrate_kbps,
            vmaf=do_result.aggregate_vmaf,
            psnr=do_result.aggregate_psnr,
            output_path=do_result.final_video_path,
        )
        export_csv([agg_result], os.path.join(cfg.output_dir, "results.csv"))
        write_run_manifest(cfg, args, mode="learned_controller",
                           do_cfg=_build_do_cfg(),
                           chunk_results=do_result.chunk_results)
        log.info("=== Learned Controller complete. Outputs in %s ===", cfg.output_dir)
        if do_result.final_video_path:
            log.info("Final video     : %s", do_result.final_video_path)
        log.info("Per-chunk CSV   : %s", chunk_csv)
        log.info("Prediction acc. : %s", accuracy_csv)
        return

    # ── Dynamic Optimizer mode (oracle — full CRF sweep) ──────────────────────
    if args.dynamic_optimizer:
        do_cfg = _build_do_cfg()
        log.info("── Dynamic Optimizer ──────────────────────────────────────────")
        log.info("Detector      : %s  threshold=%.1f", do_cfg.detector, do_cfg.scene_threshold)
        log.info("Min chunk     : %.1fs  VMAF target=%.1f  Mode=%s",
                 do_cfg.min_chunk_duration, do_cfg.vmaf_target, do_cfg.optimize_mode)

        do_result = run_dynamic_optimizer(cfg, do_cfg)

        if not do_result.chunk_results:
            log.error("No chunk results produced. Check errors above.")
            return

        title = Path(cfg.reference).stem
        chunk_csv = os.path.join(cfg.output_dir, "chunks_results.csv")
        export_chunk_csv(do_result.chunk_results, chunk_csv)

        # Optionally export oracle training records
        if args.export_training_data:
            export_training_data(do_result.chunk_results, args.export_training_data)

        # Build a synthetic aggregate QualityResult so existing CSV + plots still work
        agg_result = QualityResult(
            label="dynamic_optimized",
            bitrate_kbps=do_result.aggregate_bitrate_kbps,
            vmaf=do_result.aggregate_vmaf,
            psnr=do_result.aggregate_psnr,
            output_path=do_result.final_video_path,
        )
        export_csv([agg_result], os.path.join(cfg.output_dir, "results.csv"))

        write_run_manifest(cfg, args, mode="dynamic_optimizer",
                           do_cfg=do_cfg,
                           chunk_results=do_result.chunk_results)

        log.info("=== Dynamic Optimizer complete. Outputs in %s ===", cfg.output_dir)
        if do_result.final_video_path:
            log.info("Final video  : %s", do_result.final_video_path)
        log.info("Per-chunk CSV: %s", chunk_csv)
        if args.export_training_data:
            log.info("Training data: %s", args.export_training_data)
        return

    # ── Standard pipeline mode ────────────────────────────────────────────────
    results = run_pipeline(cfg)

    if not results:
        log.error("No results produced. Check errors above.")
        return

    title = Path(cfg.reference).stem

    # Export CSV
    csv_path = os.path.join(cfg.output_dir, "results.csv")
    export_csv(results, csv_path)

    # Write run manifest
    write_run_manifest(cfg, args, mode="standard")

    # Plot RD curves
    graphs_dir = os.path.join(cfg.output_dir, "graphs")
    plot_rd_curves(results, graphs_dir, title_prefix=title,
                   log_scale=not args.linear_scale)
    plot_temporal(results, graphs_dir, title_prefix=title)

    log.info("=== Done. Outputs in %s ===", cfg.output_dir)


if __name__ == "__main__":
    main()