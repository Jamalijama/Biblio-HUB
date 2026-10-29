# -*- coding: utf-8 -*-
"""
Supplementary Figure S17 - BERTopic semantic-clustering stability.
Final, tuned setting (UMAP n_neighbors=30; HDBSCAN min_cluster_size=max(60,0.006N),
eom; all-MiniLM-L6-v2; outliers reassigned by embedding similarity).
Panel A: cross-seed adjusted Rand index (seed 42 vs 123 and 2024) for each virus.
Panel B: final topic count and raw HDBSCAN unassigned fraction for each virus.
Reads the verified BERTopic_four_virus_summary.csv; no model is rerun.
"""
import ast
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": "#444444", "axes.linewidth": 1.0})

BLUE_D, BLUE_L, GREEN, ORANGE = "#2C5F8A", "#7FA8C9", "#39844A", "#D98E04"

df = pd.read_csv(DATA / "BERTopic_four_virus_summary.csv")
order = ["chik", "zika", "dengue", "wnv"]
labels = ["Chikungunya", "Zika", "Dengue", "West Nile\nvirus"]
df = df.set_index("virus").loc[order].reset_index()
df["ari"] = df["ARI_seed42_vs_123_2024"].apply(ast.literal_eval)

fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))

# ---------- Panel A: cross-seed ARI ----------
ax = axes[0]
x = np.arange(len(df))
for i, (a1, a2) in enumerate(df["ari"]):
    mean = np.mean([a1, a2])
    ax.plot([i - 0.16, i + 0.16], [mean, mean], color="#999999", lw=2.2, zorder=1)
    ax.scatter(i - 0.16, a1, s=96, color=BLUE_D, edgecolor="white", lw=0.8, zorder=3)
    ax.scatter(i + 0.16, a2, s=96, color=BLUE_L, edgecolor="white", lw=0.8, zorder=3)
    ax.text(i, mean + 0.012, f"{mean:.2f}", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#333333")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel("Adjusted Rand index vs seed 42", fontsize=11.5)
ax.set_ylim(0.55, 0.90)
ax.set_title("Cross-seed partition stability", fontsize=12.5, fontweight="bold", pad=8)
ax.grid(axis="y", color="#EFEBE3", lw=.7); ax.set_axisbelow(True)
ax.legend(handles=[
    Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE_D, markeredgecolor="white",
           markersize=11, label="seed 42 vs 123"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE_L, markeredgecolor="white",
           markersize=11, label="seed 42 vs 2024"),
    Line2D([0], [0], color="#999999", lw=2.2, label="pairwise mean"),
], loc="lower right", frameon=False, fontsize=9.5)

# ---------- Panel B: topics and raw outlier ----------
ax = axes[1]
ax2 = ax.twinx()
w = 0.38
topics = df["n_topics"].astype(float).values
outl = df["HDBSCAN_outlier_raw_%"].astype(float).values
b1 = ax.bar(x - w/2, topics, width=w, color=GREEN, edgecolor="white", lw=0.8,
            label="Final topics (left axis)", zorder=2)
b2 = ax2.bar(x + w/2, outl, width=w, color=ORANGE, alpha=.85, edgecolor="white", lw=0.8,
             label="HDBSCAN unassigned (right axis)", zorder=2)
for xi, v in zip(x - w/2, topics):
    ax.text(xi, v + 0.4, f"{int(v)}", ha="center", va="bottom", fontsize=10.5,
            fontweight="bold", color=GREEN)
for xi, v in zip(x + w/2, outl):
    ax2.text(xi, v + 0.7, f"{v:.1f}%", ha="center", va="bottom", fontsize=10.5,
             fontweight="bold", color="#9a6500")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5)
ax.set_ylabel("Number of topics", fontsize=11.5, color=GREEN)
ax2.set_ylabel("Raw HDBSCAN unassigned (%)", fontsize=11.5, color="#9a6500")
ax.set_ylim(0, 30); ax2.set_ylim(0, 55)
ax.set_title("Topic count and unassigned fraction", fontsize=12.5, fontweight="bold", pad=8)
ax.grid(axis="y", color="#EFEBE3", lw=.7); ax.set_axisbelow(True)
ax.tick_params(axis="y", colors=GREEN); ax2.tick_params(axis="y", colors="#9a6500")
ax.legend(handles=[
    Line2D([0], [0], marker="s", color="none", markerfacecolor=GREEN, markersize=11,
           label="Final topics (left axis)"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=ORANGE, markersize=11,
           label="Raw HDBSCAN unassigned (right axis)"),
], loc="upper left", frameon=False, fontsize=9.5)

fig.suptitle("Supplementary Figure S17. BERTopic stability across the four virus corpora and random seeds",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.97])
for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"FigS_bertopic_stability.{ext}", dpi=500,
                bbox_inches="tight", facecolor="white")
print("FigS_bertopic_stability written")
for _, r in df.iterrows():
    print(f"  {r['virus']:7s} n={int(r['n_docs(full)']):6d} topics={int(r['n_topics']):2d} "
          f"outlier={r['HDBSCAN_outlier_raw_%']:.1f}% ARI={r['ari']}")
