# -*- coding: utf-8 -*-
"""
FIG7 (final) — Per-capita robustness check, 4 viruses.
Axes are log10-transformed values on LINEAR axes (stable with adjust_text),
labelled with raw values. Point colour (research gap) comes from a multivariate
OLS  log(publications) ~ log(cases) + log(population) + log(GDP per capita),
so red points are high-incidence countries still below the output expected after
controlling for population and economic development. Companion to Fig6.
"""
import math
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from adjustText import adjust_text

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": "#444444", "axes.linewidth": 1.0})

RED, GREEN, LINE = "#B23A48", "#39844A", "#555555"
EPS = 1e-3

def raw_ticks(lo, hi):
    """positions in log10 and raw-value labels for integer decade powers within [lo,hi]."""
    pos, lab = [], []
    for p in range(math.floor(lo), math.ceil(hi) + 1):
        v = 10 ** p
        pos.append(p)
        if v >= 1000: lab.append(f"{v/1000:g}k")
        else: lab.append(f"{v:g}")
    return pos, lab

panels = [
    ("master_dengue_percapita.csv",  "Dengue — global", 10),
    ("master_chik_percapita.csv",    "Chikungunya — Americas", 9),
    ("master_zika_percapita.csv",    "Zika — Americas", 6),
    ("master_wnv_percapita.csv",     "West Nile virus — Europe + USA", 5),
]

fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.6))
for ax, (fname, title, kgap) in zip(axes.flat, panels):
    d = pd.read_csv(DATA / fname)
    lx = np.log10(d["case_rate"].values)
    ly = np.log10(d["pub_rate"].values + EPS)
    d["lx"], d["ly"] = lx, ly
    rr = spearmanr(d["case_rate"], d["pub_rate"])
    b, a = np.polyfit(lx, ly, 1)
    rstd = float((ly - (a + b * lx)).std())

    g = d[d["gap_mv"]]; ok = d[~d["gap_mv"]]
    ax.scatter(ok["lx"], ok["ly"], s=46, color=GREEN, alpha=.75,
               edgecolor="white", lw=.6, zorder=2)
    ax.scatter(g["lx"], g["ly"], s=62, color=RED, alpha=.88,
               edgecolor="white", lw=.6, zorder=3)

    xsg = np.linspace(lx.min(), lx.max(), 100)
    ysg = a + b * xsg
    ax.fill_between(xsg, ysg - 1.96 * rstd, ysg + 1.96 * rstd,
                    color="#555555", alpha=.08, zorder=0)
    ax.plot(xsg, ysg, "--", color=LINE, lw=1.4, zorder=1)

    gl = g.sort_values("resid_mv").head(kgap)
    anchors = ok.sort_values("pub_rate", ascending=False).head(4)
    texts = []
    for _, r in gl.iterrows():
        texts.append(ax.text(r["lx"], r["ly"], r["ISO3"], fontsize=7.8,
                             color="#8a1c28", fontweight="bold", zorder=5))
    for _, r in anchors.iterrows():
        texts.append(ax.text(r["lx"], r["ly"], r["ISO3"], fontsize=7.6,
                             color="#2f3a2f", fontstyle="italic", zorder=5))
    adjust_text(texts, ax=ax, lim=1500,
                expand_text=(1.2, 1.5), expand_points=(1.4, 1.4),
                force_text=(0.75, 1.0), force_points=(0.5, 0.6),
                arrowprops=dict(arrowstyle="-", color="#999999", lw=.45,
                                shrinkA=2, shrinkB=3))

    xp, xlab = raw_ticks(lx.min(), lx.max())
    yp, ylab = raw_ticks(ly.min(), ly.max())
    ax.set_xticks(xp); ax.set_xticklabels(xlab)
    ax.set_yticks(yp); ax.set_yticklabels(ylab)
    sig = "p < 0.001" if rr.pvalue < 0.001 else f"p = {rr.pvalue:.3f}"
    ax.set_title(f"{title}\nper-capita ρ = {rr.statistic:.2f}  ({sig})",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Cumulative cases per 100,000 population (log)", fontsize=10)
    ax.set_ylabel("WoS publications per 10 million population (log)", fontsize=10)
    ax.grid(color="#EFEBE3", lw=.6)
    ax.set_axisbelow(True)

legend_handles = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=RED, markeredgecolor="white",
           markersize=11, label="Research gap after GDP/population control"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markeredgecolor="white",
           markersize=11, label="Aligned"),
    Line2D([0], [0], linestyle="--", color=LINE, lw=1.4, label="OLS trend (log–log)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False,
           fontsize=10.5, bbox_to_anchor=(0.5, 0.002))
fig.suptitle("Supplementary Fig. Per-capita research-preparedness (population- and GDP-controlled robustness)",
             fontsize=14, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0.04, 1, 0.965])
fig.savefig(OUT / "FigS_robustness_percapita.png", dpi=500, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "FigS_robustness_percapita.pdf", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "FigS_robustness_percapita.svg", bbox_inches="tight", facecolor="white")
print("FigS_robustness_percapita written")
for fname, title, _ in panels:
    d = pd.read_csv(DATA / fname)
    rr = spearmanr(d["case_rate"], d["pub_rate"])
    print(f"  {title:40s} n={len(d):3d} gaps={int(d['gap_mv'].sum()):2d} rho={rr.statistic:.3f} p={rr.pvalue:.3g}")
