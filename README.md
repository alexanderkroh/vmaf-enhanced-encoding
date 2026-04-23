# vmaf-enhanced-encoding

A content-aware video encoding optimizer built in Python. Takes a source video, detects shot boundaries, runs a per-shot CRF sweep with VMAF quality measurement, selects optimal encoding parameters for each scene, and trains a learned rate-distortion model that predicts optimal CRF from content complexity features — without trial encodes.

Built to explore the architecture Netflix and others use for per-title and per-shot adaptive encoding, implemented from first principles.

---

## What It Does

Standard video encoding applies a fixed bitrate or CRF uniformly across an entire title. This wastes bits on easy scenes and starves hard ones. The approach here adapts encoding parameters per shot:

```
Source video
    │
    ├── PySceneDetect → shot boundaries
    │
    └── Per-chunk oracle sweep  [video_quality_pipeline.py]
            ├── Multi-lane CRF sweep (parallel encoders, single decode pass)
            ├── In-loop VMAF measurement per lane (no separate pass)
            │     (HDR: PQ→linear→BT.709 normalisation)
            ├── Optimal CRF selection (first to exceed VMAF target)
            └── Final encode at optimal parameters
                    │
                    ├── Assembled output video
                    ├── Per-chunk results CSV
                    ├── Training records (JSONL)
                    └── Run manifest (provenance)

Training records (accumulated across titles)
    │
    └── RDCurveModel.train()
            ├── Fits RD curves per chunk
            ├── Extracts content features (SI, TI, luma, luma range)
            └── OLS regression: features → curve parameters
                    │
                    └── Inference: predict optimal CRF in seconds
                            (replaces hours of trial encodes)
                                    │
                                    └── LadderGenerator  [ladder_generator.py]
                                            ├── RD surface prediction
                                            │     (resolutions × CRF values)
                                            ├── Convex hull → Pareto-optimal rungs
                                            ├── Per-shot CRF per rung
                                            ├── HLS master.m3u8
                                            ├── DASH manifest.mpd
                                            └── --encode: multi-lane encode
                                                  all rungs × all chunks
```

**Four operating modes:**

**Standard pipeline** — encode at multiple bitrates or CRF values across the full source, measure VMAF/PSNR/SSIM at each point, produce rate-distortion curves and CSV.

**Dynamic Optimizer (oracle mode)** — per-shot CRF sweep with optimal selection using multi-lane parallel encoding and in-loop VMAF. Generates training data for the learned controller.

**Learned Controller** — uses a trained `RDCurveModel` to predict optimal CRF from content features without a full sweep. Includes `TitleBudgetAllocator` using Lagrange optimization for bitrate budget distribution across chunks.

**Ladder Generator** — generates a content-adaptive ABR bitrate ladder from a trained model. Predicts the full RD surface across resolutions and CRF values, finds the Pareto-optimal operating points via convex hull analysis, and produces HLS and DASH manifests. Optional `--encode` flag runs all renditions using multi-lane parallel encoding.

---

## Key Engineering

### Per-Shot Optimization

The fundamental insight is that encoding difficulty is not uniform across a title. A static interview scene can hit VMAF 93 at CRF 26. An LED wall or crowd scene may not reach VMAF 93 even at CRF 16. Applying a uniform encode to both wastes bits on the interview and delivers inadequate quality on the complex scene.

PySceneDetect identifies shot boundaries. Each chunk gets an independent CRF sweep with VMAF measurement. The optimal CRF — the lowest bitrate that meets the quality target — is selected per chunk. The encoder context system provides pre-roll frames before each chunk boundary so the encoder's rate control isn't cold at every cut.

### HDR VMAF Normalisation

Two approaches exist for measuring VMAF on HDR content:

**1. HDR-specific VMAF models** — Netflix has developed VMAF models trained on HDR content that operate natively in the PQ domain. The `vmaf_4k_v0.6.1` model included with libvmaf provides improved 4K accuracy but is still fundamentally an SDR-trained model. Netflix's dedicated HDR VMAF model, which scores directly against PQ-encoded references without tone mapping, is used internally but is not part of the standard public libvmaf distribution.

**2. PQ→BT.709 normalisation (this pipeline)** — the most widely accessible approach given current libvmaf distributions. Tone-maps the source and encoded files from PQ to a display-referred BT.709 signal before running the standard VMAF model:

```
Source TRC: smpte2084 (PQ) detected
→ FFmpeg filter: zscale=transfer=linear:npl=203,format=yuv444p10le,
                 zscale=transfer=bt709,format=yuv420p
→ VMAF runs on tone-mapped BT.709 signal
→ Scores are perceptually meaningful and internally consistent
```

The normalisation approach produces scores that are reliable for comparing encodes against each other on the same source but are not directly comparable to SDR VMAF scores — a 93 VMAF on normalised HDR content and a 93 VMAF on native SDR content are not equivalent. The run manifest records which approach was used so results are never ambiguously interpreted.

The pipeline auto-detects HDR via `color_transfer` metadata (`smpte2084`, `arib-std-b67`) and enables normalisation automatically. HDR normalisation propagates explicitly through the per-chunk processing pipeline — stream-copy chunk files often lose `color_trc` metadata, which would cause the re-probe in `_auto_configure_metrics` to silently disable normalisation mid-run. This was a diagnosed and fixed production bug.

### Content Feature Extraction

Per-chunk content features are extracted using FFmpeg `signalstats` filter with `metadata=mode=print` output (required for FFmpeg 7+, which moved per-frame stats from stderr to the metadata side-data mechanism):

| Feature | Description | Signal |
|---|---|---|
| SI (Spatial Information) | Mean Sobel edge magnitude on luma | Structural complexity |
| TI (Temporal Information) | Mean inter-frame luma difference | Motion complexity |
| Mean luma | Average frame brightness | Lighting characteristics |
| Luma range | Max − min brightness per frame | Contrast / HDR range |

These features feed the `RDCurveModel` regression — the model learns to predict the shape of the RD curve for new content from these four numbers, enabling fast inference without trial encodes.

### Multi-Lane Parallel Encoding with In-Loop VMAF

Meta's March 2026 engineering post describes two features now upstream in FFmpeg 8.0 that were previously only in their internal fork, both implemented in this pipeline:

**Multi-lane parallel encoding** — rather than encoding each CRF variant sequentially, a single FFmpeg filter graph fans out from one decode pass to N parallel encoder instances. For a 7-point CRF sweep this reduces total decode overhead by roughly 7×.

```
Input → split=7 ─┬─ encoder[CRF 16] → output_crf16.mp4
                 ├─ encoder[CRF 18] → output_crf18.mp4
                 ├─ encoder[CRF 20] → output_crf20.mp4
                 ├─ encoder[CRF 22] → output_crf22.mp4
                 ├─ encoder[CRF 24] → output_crf24.mp4
                 ├─ encoder[CRF 26] → output_crf26.mp4
                 └─ encoder[CRF 28] → output_crf28.mp4
```

**In-loop VMAF measurement** — a decoder is inserted after each encoder lane and its output compared against the pre-encode reference frames. VMAF scores are produced per lane as a byproduct of the encode, eliminating the separate measurement pass entirely. For a 7-CRF sweep this replaces 14 FFmpeg invocations (7 encodes + 7 VMAF passes) with a single command.

```
Input → split=14 ─┬─ encoder[CRF 16] ─┬─ output_crf16.mp4
                  │                    └─ decode → [dist][ref] → libvmaf → vmaf_crf16.json
                  ├─ encoder[CRF 18] ─┬─ output_crf18.mp4
                  │                   └─ decode → [dist][ref] → libvmaf → vmaf_crf18.json
                  └─ ...
```

For HDR sources the PQ→linear→BT.709 normalisation chain is applied inline to both the decoded distorted output and the reference branch before libvmaf, matching the existing HDR measurement behaviour exactly. Requires FFmpeg 7.0+ for in-loop decode; falls back gracefully to sequential encode + VMAF on older versions.

The `LaneSpec` dataclass provides the interface between oracle sweep logic and the encoder — CRF or bitrate, output path, optional VMAF log path, optional resolution override. All existing helper functions for quality flags, presets, pixel formats, and colour metadata are reused.

### Content-Adaptive Bitrate Ladder

`ladder_generator.py` generates a per-title ABR ladder from a trained `RDCurveModel`:

**RD surface prediction** — predicts the full (resolution, CRF) → (VMAF, bitrate) surface across all requested rendition resolutions and CRF values. Resolution scaling uses a content-aware heuristic: complex content (high SI) degrades more when downscaled than simple content, and lower resolution encoders receive a bitrate credit proportional to `scale^1.5`.

**Convex hull selection** — finds the Pareto-optimal frontier of the RD surface — the set of operating points where no other point delivers higher VMAF at lower bitrate. These are the only operating points worth considering as ladder rungs.

**Per-shot CRF per rung** — each rung uses per-shot CRF assignments from the RDCurveModel rather than a fixed CRF. Within rung 4 (720p, VMAF 85 target), an easy interview shot might use CRF 26 while a crowd scene uses CRF 18 — both targeting consistent perceptual quality rather than constant bitrate.

**Manifest generation** — produces a standards-compliant HLS `master.m3u8` with correct `BANDWIDTH`, `AVERAGE-BANDWIDTH`, `RESOLUTION`, and `CODECS` attributes, and a DASH `manifest.mpd` with `AdaptationSet` / `Representation` structure per rung.

**Multi-lane ladder encoding** — the `--encode` flag encodes all renditions using `encode_multilane_with_vmaf()`. Each chunk is encoded across all rungs simultaneously in one FFmpeg invocation, with VMAF measured inline per rung.

The `RDCurveModel` fits two parametric curves to each chunk's oracle sweep data:
- VMAF as a function of CRF
- log(bitrate) as a function of CRF

OLS regression maps the feature vector `[SI, TI, mean_luma, luma_range, duration]` to the curve parameters. At inference time, the model receives content features and solves for the CRF that produces the target VMAF.

The `TitleBudgetAllocator` uses Lagrange multiplier optimization to distribute a title-level bitrate budget across chunks to maximize aggregate VMAF — spending more bits where the content needs them.

### Run Manifest

Every run writes `run_manifest.json` to the output directory capturing:
- Exact command line
- Source technical metadata (resolution, fps, codec, color TRC, HDR metadata)
- HDR normalisation decision and reason
- Tool versions (FFmpeg, libvmaf build string)
- Git hash of the pipeline
- Per-chunk quality summary
- Quality warnings (VMAF < 40, bitrate > 50 Mbps)

This is directly analogous to PREMIS event records in archival preservation — full provenance for every encoding decision.

### Source Validation

Pre-flight validation runs automatically before any processing:
- File accessibility and video stream presence
- Duration adequacy (< 60s warns)
- Resolution adequacy (< 720p warns)
- Codec/bitrate analysis — detects delivery encodes masquerading as masters (H.264 < 30 Mbps triggers a warning; ProRes/DNxHD pass silently)
- HDR metadata completeness check

Use `--validate` to run validation only without encoding:

```bash
python3 video_quality_pipeline.py source.mov --validate
```

---

## Installation

```bash
git clone https://github.com/alexkroh/vmaf-enhanced-encoding.git
cd vmaf-enhanced-encoding
pip install -r requirements.txt
```

**Required:**
- Python 3.10+
- FFmpeg with libvmaf (`brew install ffmpeg-full` on macOS, or build from source with `--enable-libvmaf`)
- libvmaf 2.x or 3.x

**Optional:**
- `scenedetect[opencv]` — PySceneDetect for shot boundary detection (`pip install scenedetect[opencv]`)
- `matplotlib` + `numpy` — RD curve graphs and BD-rate tables
- `torch` / HuggingFace `transformers` — for future CNN/transformer feature extraction

Verify your FFmpeg has libvmaf:

```bash
ffmpeg -filters | grep vmaf
```

---

## Usage

### Standard Pipeline — bitrate or CRF sweep

```bash
# Bitrate sweep across full source
python3 video_quality_pipeline.py source.mov --output-dir ./results

# CRF sweep
python3 video_quality_pipeline.py source.mov \
  --crfs 18 20 22 24 26 28 \
  --output-dir ./results

# Specific codec
python3 video_quality_pipeline.py source.mov \
  --codec libx265 \
  --crfs 18 20 22 24 26 28 \
  --output-dir ./results
```

### Dynamic Optimizer — oracle mode, generates training data

```bash
# SDR source
python3 video_quality_pipeline.py source.mov \
  --dynamic-optimizer \
  --detector pyscenedetect \
  --crfs 16 18 20 22 24 26 28 \
  --codec libx264 \
  --preset medium \
  --jobs 2 \
  --export-training-data records.jsonl \
  --output-dir ./eval_output/$(basename source.mov .mov)

# HDR source (auto-detected, no flag needed)
python3 video_quality_pipeline.py hdr_source.mp4 \
  --dynamic-optimizer \
  --detector pyscenedetect \
  --crfs 16 18 20 22 24 26 28 \
  --codec libx265 \
  --jobs 2 \
  --export-training-data records.jsonl \
  --output-dir ./eval_output/hdr_clip
```

Run across multiple clips — records accumulate in the same JSONL file:

```bash
for clip in ./masters/*.mov; do
  python3 video_quality_pipeline.py "$clip" \
    --dynamic-optimizer \
    --detector pyscenedetect \
    --crfs 16 18 20 22 24 26 28 \
    --codec libx264 \
    --jobs 2 \
    --export-training-data records.jsonl \
    --output-dir "./eval_output/$(basename "$clip" .mov)"
done
```

### Train the RD model

```bash
python3 video_quality_pipeline.py placeholder.mov \
  --train-model model.json \
  --training-data records.jsonl
```

### Learned Controller — fast inference without full sweep

```bash
python3 video_quality_pipeline.py source.mov \
  --model-path model.json \
  --vmaf-target 93 \
  --output-dir ./results

# With title-level bitrate budget (Lagrange allocation)
python3 video_quality_pipeline.py source.mov \
  --model-path model.json \
  --bit-budget-kbps 3000 \
  --output-dir ./results
```

### Ladder Generator — content-adaptive ABR ladder

```bash
# Generate ladder from trained model (prediction only, no encoding)
python3 ladder_generator.py source.mov \
  --model model.json \
  --output-dir ./ladder_output

# Custom resolution set and VMAF targets
python3 ladder_generator.py source.mov \
  --model model.json \
  --resolutions 360 540 720 1080 \
  --vmaf-targets 65 75 85 92 95 \
  --output-dir ./ladder_output

# Generate and encode all renditions (multi-lane parallel)
python3 ladder_generator.py source.mov \
  --model model.json \
  --resolutions 360 540 720 1080 \
  --encode \
  --output-dir ./ladder_output
```

Outputs: `ladder.json` (rung definitions with per-chunk CRF assignments), `master.m3u8` (HLS), `manifest.mpd` (DASH), and per-rung stub media playlists.

### Plot RD curves

```bash
# Single run
python3 plot_rd.py eval_output/results.csv -o graphs/

# Codec comparison overlay
python3 plot_rd.py h264_results.csv \
  --compare h265_results.csv av1_results.csv \
  --labels "H.264" "H.265" "AV1" \
  -o graphs/

# With BD-rate table
python3 plot_rd.py h264_results.csv \
  --compare h265_results.csv \
  --labels "H.264 (baseline)" "H.265" \
  --bd-rate \
  -o graphs/
```

---

## Source Material Requirements

The oracle measures quality loss from master to delivery encode. The source must be a high-quality master — not a delivery encode re-encoded.

| Codec | Typical bitrate | Suitability |
|---|---|---|
| ProRes 422 / 4444 | 80–220 Mbps | Excellent |
| DNxHD / DNxHR | 60–175 Mbps | Excellent |
| Uncompressed / FFV1 | varies | Excellent |
| H.264 at 50+ Mbps | high | Acceptable |
| H.264 at < 20 Mbps | low | Poor — delivery encode |

The `--validate` flag will warn if source appears to be a delivery encode. A clip like `tears_of_steel.mov` at 6 Mbps H.264 will produce VMAF scores in the 20-40 range — measuring encode-against-encode, not master-against-delivery.

---

## Output Files

Each pipeline run produces in `--output-dir`:

| File | Description |
|---|---|
| `results.csv` | Aggregate quality results |
| `chunks_results.csv` | Per-chunk metrics (dynamic optimizer) |
| `run_manifest.json` | Full provenance record |
| `records.jsonl` | Training records for RDCurveModel (with `--export-training-data`) |
| `rd_curves.png` | Rate-distortion curve plots |
| `*_dynamic_optimized.mp4` | Final assembled output (dynamic optimizer) |

Ladder generator produces in `--output-dir`:

| File | Description |
|---|---|
| `ladder.json` | Rung definitions with per-chunk CRF assignments |
| `master.m3u8` | HLS master playlist |
| `manifest.mpd` | DASH MPD |
| `{rung_label}/media.m3u8` | Per-rung HLS media playlist (stub) |
| `{rung_label}/chunk_NNNN.mp4` | Encoded chunks per rung (with `--encode`) |
| `{rung_label}/vmaf_NNNN.json` | Per-chunk VMAF logs per rung (with `--encode`) |

---

## Configuration Reference

| Flag | Default | Description |
|---|---|---|
| `--codec` | `libx264` | Encoding codec |
| `--preset` | `medium` | Encoder preset |
| `--crfs` | — | CRF sweep values (overrides `--bitrates`) |
| `--bitrates` | 200–8000 kbps | Bitrate sweep values |
| `--jobs` | 1 | Parallel chunk sweeps |
| `--vmaf-target` | 93.0 | VMAF floor for optimal selection |
| `--detector` | `ffmpeg` | Scene detector: `ffmpeg` or `pyscenedetect` |
| `--scene-threshold` | 27.0 | Detection sensitivity |
| `--min-chunk-duration` | 2.0s | Minimum chunk length |
| `--encoder-context` | 2.0s | Pre-roll context for encoder warm-up |
| `--hdr-vmaf-normalise` | auto | Force HDR VMAF normalisation |
| `--validate` | — | Validate source and exit |
| `--export-training-data` | — | Path to accumulate oracle training records |

---

## Hardware Acceleration

```bash
# Apple VideoToolbox (macOS)
python3 video_quality_pipeline.py source.mov --hw-accel videotoolbox

# NVIDIA NVENC
python3 video_quality_pipeline.py source.mov --hw-accel nvenc

# Intel QSV
python3 video_quality_pipeline.py source.mov --hw-accel qsv
```

Note: hardware encoders produce different RD characteristics than software encoders. Don't mix hardware and software oracle runs in the same training dataset.

---

## Relationship to Industry Approaches

Netflix's per-title encoding (2015) and per-shot encoding (2018) established that per-content encoding optimization significantly reduces bitrate at equivalent quality — 20-50% in published results. The oracle mode in this pipeline implements the same core concept: find the optimal encoding parameters for each shot by measuring, then apply them.

The learned controller extends this by replacing the expensive per-shot sweep with a fast content analysis pass — the approach needed to make per-shot optimization practical at scale where running a full CRF sweep on every chunk of every title isn't economically feasible.

Meta's March 2026 engineering post on FFmpeg describes two features their team helped upstream into FFmpeg 7.0/8.0: threaded multi-lane encoding and in-loop real-time quality metrics. Both are implemented here. The multi-lane encoder shares a single decode pass across all CRF variants; the in-loop VMAF inserts a decoder after each encoder to measure quality without a separate pass. Together they implement the same efficiency architecture Meta uses to process over one billion video uploads per day.

The content feature extraction (SI, TI, luma, luma range) provides the basis for the current model. The natural extension is richer feature representations — CNN embeddings from pretrained visual models, video transformer features — that capture content complexity properties SI and TI cannot distinguish, particularly film grain vs. structured texture, and predictable vs. chaotic motion. CLIP embeddings are a tractable near-term upgrade that would improve prediction accuracy on the edge cases where the current linear model fails.

---

## Background

This project grew out of work managing video ingest pipelines for broadcast media archives, where the challenge of encoding heterogeneous archival content — everything from pristine studio footage to degraded field recordings — made the limitations of uniform encoding parameters concrete. Per-shot quality measurement and adaptive encoding are as relevant to a 50,000-hour archival digitization queue as they are to a streaming platform's title catalog.

---

## License

MIT