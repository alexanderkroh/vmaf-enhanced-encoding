#!/usr/bin/env python3
"""
plot_rd.py  –  Generate / regenerate RD-curve graphs from a results CSV.

Usage:
    python plot_rd.py results.csv -o graphs/
    python plot_rd.py results.csv --compare file_a.csv file_b.csv --labels "H.264" "H.265"
"""

import argparse
import csv
import os
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
METRIC_META = {
    "vmaf": {"label": "VMAF",      "color": "#2563eb", "ylim": (0, 100)},
    "psnr": {"label": "PSNR (dB)", "color": "#16a34a", "ylim": (20, 55)},
    "ssim": {"label": "SSIM",      "color": "#9333ea", "ylim": (0, 1)},
    "avqt": {"label": "AVQT",      "color": "#ea580c", "ylim": (0, 100)},
}

CODEC_COLORS = [
    "#2563eb", "#ea580c", "#16a34a", "#9333ea",
    "#e11d48", "#0891b2", "#ca8a04", "#64748b",
]


@dataclass
class Series:
    """One codec/config's worth of results."""
    label: str
    bitrates: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)   # metric → [float]


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def rows_to_series(rows: list[dict], label: str) -> Series:
    s = Series(label=label)
    for row in rows:
        try:
            br = float(row["bitrate_kbps"])
        except (KeyError, ValueError):
            continue
        s.bitrates.append(br)
        for m in METRIC_META:
            val = row.get(m, "").strip()
            if val:
                try:
                    s.scores.setdefault(m, []).append(float(val))
                except ValueError:
                    pass
    # sort by bitrate
    order = sorted(range(len(s.bitrates)), key=lambda i: s.bitrates[i])
    s.bitrates = [s.bitrates[i] for i in order]
    for m in s.scores:
        if len(s.scores[m]) == len(order):
            s.scores[m] = [s.scores[m][i] for i in order]
    return s


# ──────────────────────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor("#1e293b")
    ax.grid(color="#334155", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")


def plot_single_series(series: Series, output_dir: str, log_scale: bool = True):
    """Plot RD curves for a single codec/series."""
    active = [m for m in METRIC_META if m in series.scores]
    if not active:
        log.warning("No metric data in series '%s'", series.label)
        return

    ncols = min(2, len(active))
    nrows = (len(active) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows),
                              constrained_layout=True)
    fig.patch.set_facecolor("#0f172a")

    if len(active) == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    fig.suptitle(f"{series.label}  –  Rate–Distortion Curves",
                 color="white", fontsize=14, fontweight="bold")

    for idx, metric in enumerate(active):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        _style_ax(ax)
        meta = METRIC_META[metric]
        br = series.bitrates
        sc = series.scores[metric]

        ax.plot(br, sc, color=meta["color"], linewidth=2.2,
                marker="o", markersize=6,
                markerfacecolor="white", markeredgecolor=meta["color"],
                markeredgewidth=1.5, zorder=5)
        ax.fill_between(br, sc, alpha=0.10, color=meta["color"])

        for b, s_val in zip(br, sc):
            ax.annotate(f"{s_val:.1f}", xy=(b, s_val), xytext=(0, 9),
                        textcoords="offset points", ha="center",
                        fontsize=7, color="white", alpha=0.85)

        if log_scale:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

        ax.set_xlabel("Bitrate (kbps)", color="#94a3b8", fontsize=10)
        ax.set_ylabel(meta["label"],    color="#94a3b8", fontsize=10)
        ax.set_title(meta["label"],     color="white",   fontsize=11, fontweight="semibold")

        ymin, ymax = meta["ylim"]
        pad = (max(sc) - min(sc)) * 0.15 or 2
        ax.set_ylim(max(ymin, min(sc) - pad), min(ymax, max(sc) + pad)
                    if ymax > 1 else min(ymax, max(sc) + 0.02))

    # hide empty cells
    for idx in range(len(active), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "rd_curves.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Saved → %s", out)


def plot_comparison(series_list: list[Series], output_dir: str, log_scale: bool = True):
    """
    Overlay multiple codecs / configs on the same RD axes for direct comparison.
    One subplot per metric.
    """
    all_metrics = []
    for m in METRIC_META:
        if any(m in s.scores for s in series_list):
            all_metrics.append(m)

    if not all_metrics:
        log.warning("No metric data to compare.")
        return

    ncols = min(2, len(all_metrics))
    nrows = (len(all_metrics) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5.5 * nrows),
                              constrained_layout=True)
    fig.patch.set_facecolor("#0f172a")

    if len(all_metrics) == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    fig.suptitle("Codec Comparison – Rate–Distortion Curves",
                 color="white", fontsize=14, fontweight="bold")

    for idx, metric in enumerate(all_metrics):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        _style_ax(ax)
        meta = METRIC_META[metric]
        ax.set_title(meta["label"], color="white", fontsize=11, fontweight="semibold")
        ax.set_xlabel("Bitrate (kbps)", color="#94a3b8", fontsize=10)
        ax.set_ylabel(meta["label"],    color="#94a3b8", fontsize=10)

        all_scores = []
        for i, s in enumerate(series_list):
            if metric not in s.scores:
                continue
            color = CODEC_COLORS[i % len(CODEC_COLORS)]
            ax.plot(s.bitrates, s.scores[metric],
                    color=color, linewidth=2.0, marker="o", markersize=5,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=1.5, zorder=5, label=s.label)
            all_scores.extend(s.scores[metric])

        if log_scale:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

        if all_scores:
            ymin, ymax = meta["ylim"]
            pad = (max(all_scores) - min(all_scores)) * 0.12 or 2
            ax.set_ylim(max(ymin, min(all_scores) - pad),
                        min(ymax, max(all_scores) + pad)
                        if ymax > 1 else min(ymax, max(all_scores) + 0.02))

        ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="white",
                  fontsize=8, loc="lower right")

    for idx in range(len(all_metrics), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "rd_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("Comparison plot saved → %s", out)


def plot_bd_table(series_list: list[Series], output_dir: str):
    """
    Render a Bjøntegaard-Delta summary table as an image.
    (Requires numpy for interpolation.)
    """
    try:
        import numpy as np
    except ImportError:
        log.warning("numpy not available; skipping BD-rate table.")
        return

    if len(series_list) < 2:
        return

    def bd_rate(ref_br, ref_sc, tst_br, tst_sc):
        """Approximate BD-rate (%) using log-domain cubic spline interpolation."""
        from numpy.polynomial import polynomial as P
        log_ref_br = np.log(ref_br)
        log_tst_br = np.log(tst_br)
        min_sc = max(min(ref_sc), min(tst_sc))
        max_sc = min(max(ref_sc), max(tst_sc))
        if min_sc >= max_sc:
            return float("nan")
        sc_range = np.linspace(min_sc, max_sc, 100)
        ref_fit = np.interp(sc_range, ref_sc, log_ref_br)
        tst_fit = np.interp(sc_range, tst_sc, log_tst_br)
        avg = np.mean(tst_fit - ref_fit)
        return (np.exp(avg) - 1) * 100

    metrics_present = [m for m in ("vmaf", "psnr", "ssim", "avqt")
                       if any(m in s.scores for s in series_list)]
    ref = series_list[0]

    rows = []
    for tst in series_list[1:]:
        row = {"Codec": tst.label}
        for m in metrics_present:
            if m in ref.scores and m in tst.scores:
                r = ref.scores[m]
                t = tst.scores[m]
                rb = ref.bitrates[:len(r)]
                tb = tst.bitrates[:len(t)]
                if len(rb) >= 2 and len(tb) >= 2:
                    bd = bd_rate(rb, r, tb, t)
                    row[m.upper()] = f"{bd:+.1f}%" if not np.isnan(bd) else "N/A"
                else:
                    row[m.upper()] = "N/A"
            else:
                row[m.upper()] = "—"
        rows.append(row)

    if not rows:
        return

    cols = ["Codec"] + [m.upper() for m in metrics_present]
    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 1.8), len(rows) * 0.6 + 1.2))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    ax.axis("off")

    cell_data = [[r.get(c, "—") for c in cols] for r in rows]
    tbl = ax.table(
        cellText=cell_data,
        colLabels=cols,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1e293b" if r % 2 == 0 else "#0f172a")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#334155")
        if r == 0:
            cell.set_facecolor("#1e3a5f")
            cell.set_text_props(color="white", fontweight="bold")

    ax.set_title(f"BD-Rate relative to '{ref.label}'",
                 color="white", fontsize=12, fontweight="bold", pad=12)

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, "bd_rate_table.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    log.info("BD-rate table → %s", out)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if not HAS_MATPLOTLIB:
        log.error("matplotlib is required: pip install matplotlib")
        sys.exit(1)

    p = argparse.ArgumentParser(description="Plot RD curves from results CSV(s)")
    p.add_argument("csv", help="Primary results CSV (from video_quality_pipeline.py)")
    p.add_argument("--compare", nargs="+", metavar="CSV",
                   help="Additional CSV files to overlay for comparison")
    p.add_argument("--labels", nargs="+", metavar="LABEL",
                   help="Labels for each CSV (primary first)")
    p.add_argument("-o", "--output-dir", default="./graphs")
    p.add_argument("--linear-scale", action="store_true",
                   help="Use linear instead of log x-axis")
    p.add_argument("--bd-rate", action="store_true",
                   help="Compute and display approximate BD-rate table")
    args = p.parse_args()

    log_scale = not args.linear_scale
    csv_files = [args.csv] + (args.compare or [])
    default_labels = [Path(f).stem for f in csv_files]
    labels = args.labels or default_labels
    labels += default_labels[len(labels):]  # pad if not enough labels

    series_list = []
    for csv_file, lbl in zip(csv_files, labels):
        rows = load_csv(csv_file)
        series_list.append(rows_to_series(rows, lbl))

    os.makedirs(args.output_dir, exist_ok=True)

    if len(series_list) == 1:
        plot_single_series(series_list[0], args.output_dir, log_scale=log_scale)
    else:
        plot_comparison(series_list, args.output_dir, log_scale=log_scale)

    if args.bd_rate and len(series_list) >= 2:
        plot_bd_table(series_list, args.output_dir)

    log.info("Done. Graphs in %s", args.output_dir)


if __name__ == "__main__":
    main()
