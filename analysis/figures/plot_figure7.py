# -*- coding: utf-8 -*-
"""
Figure 7 — three self-contained ROW files (1x4), identical column order and
margins so the user can stack them in Illustrator. Columns = Chik / Zika /
Dengue / WNV (matches manuscript caption order).

Unified gap rule used in BOTH row1 (scatter) and row3 (residual):
  flagged = cumulative Cases > within-virus median AND multivariate residual
            resid_mv < -0.25 (controls log-population and log-GDP; -0.25 log10
            ~= outputs below ~44% of capacity-expected level).
Zero-publication territories are drawn dark red.
Row 2 = standardized multivariate-OLS coefficients (no change to model).
"""
import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy import stats
from scipy.stats import spearmanr
from adjustText import adjust_text

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular",
    "axes.edgecolor": "#333333", "axes.linewidth": 1.4,
    "axes.grid": True, "grid.color": "#EFEBE3", "grid.linewidth": 1.0, "axes.axisbelow": True,
})

CASES_C = "#B23A48"; GDP_C = "#E08E45"; POP_C = "#3A6EA5"
GAP_C = "#B23A48"; ZERO_C = "#7E1F28"; GREEN = "#39844A"; LINE = "#555555"
RES_THR = -0.25

# caption order: chik, zika, dengue, wnv
PANELS = [
    ("master_chik_percapita.csv",   "Chikungunya (Americas, 2013–2023)", 6),
    ("master_zika_percapita.csv",   "Zika (Americas, 2016–2023)", 6),
    ("master_dengue_percapita.csv", "Dengue (global, 2008–2024)", 6),
    ("master_wnv_percapita.csv",    "WNV (Europe & USA, 2008–2023)", 6),
]
VARS = ["cases", "gdp", "pop"]
VAR_LABEL = {"cases": "Cases",
             "gdp": "GDP/capita",
             "pop": "Population"}
VAR_COLOR = {"cases": CASES_C, "gdp": GDP_C, "pop": POP_C}

DATA = []
for fname, title, _ in PANELS:
    df = pd.read_csv(DATA_DIR / fname)
    df = df[df["Cases"] > 0].copy()
    cmed = df["Cases"].median()
    # capacity-adjusted gap (used in full by row 3)
    df["gap_final"] = (df["Cases"] > cmed) & (df["resid_mv"] < RES_THR)
    # crude (unadjusted) trend residual, recomputed on this subset
    b0, a0 = np.polyfit(df["lc"], df["lp"], 1)
    df["resid_crude"] = df["lp"] - (a0 + b0*df["lc"])
    # row 1 flags only points that are below BOTH the crude trend and the
    # capacity-adjusted expectation -> always visually under the fitted line.
    df["gap_row1"] = df["gap_final"] & (df["resid_crude"] < 0)
    DATA.append((fname, title, df))

W = 16.6
L, R, WS = 0.07, 0.985, 0.24

def std_ols(df):
    d = pd.DataFrame({"cases": np.log10(df["Cases"]+1),
                      "pop":   np.log10(df["Population"]),
                      "gdp":   np.log10(df["GDP_pc"]),
                      "pubs":  np.log10(df["Publications"]+1)}).dropna()
    Z = (d - d.mean()) / d.std(ddof=1)
    X = np.column_stack([np.ones(len(Z)), Z[["cases","pop","gdp"]].values])
    y = Z["pubs"].values
    inv = np.linalg.inv(X.T @ X); beta = inv @ X.T @ y
    res = y - X @ beta; n, k = X.shape
    s2 = (res @ res)/(n-k); se = np.sqrt(np.diag(s2*inv))
    tc = stats.t.ppf(0.975, n-k)
    lo, hi = beta-tc*se, beta+tc*se; tv = beta/se
    p = 2*(1-stats.t.cdf(np.abs(tv), n-k))
    r2 = 1-(res@res)/((y-y.mean())@(y-y.mean()))
    keys = ["const","cases","pop","gdp"]
    return dict(n=n, r2=r2,
                beta={kk:beta[i] for i,kk in enumerate(keys)},
                lo={kk:lo[i] for i,kk in enumerate(keys)},
                hi={kk:hi[i] for i,kk in enumerate(keys)},
                p={kk:p[i] for i,kk in enumerate(keys)})

# ============================ ROW 1: scatter ============================
fig = plt.figure(figsize=(W, 4.8))
gs = GridSpec(1, 4, figure=fig, left=L, right=R, wspace=WS, top=0.80, bottom=0.21)
axs = [fig.add_subplot(gs[0, j]) for j in range(4)]
for j, (fname, title, df) in enumerate(DATA):
    ax = axs[j]
    rho = spearmanr(df["lc"], df["lp"])
    b, a = np.polyfit(df["lc"], df["lp"], 1)
    rsd = float((df["lp"]-(a+b*df["lc"])).std())
    g = df[df["gap_row1"]]; ok = df[~df["gap_row1"]]
    ax.scatter(ok["lc"], ok["lp"], s=52, color=GREEN, alpha=.72,
               edgecolor="white", lw=.6, zorder=2)
    ax.scatter(g["lc"], g["lp"], s=78, color=GAP_C, alpha=.9,
               edgecolor="white", lw=.7, zorder=3)
    xs = np.linspace(df["lc"].min(), df["lc"].max(), 100)
    ax.fill_between(xs, a+b*xs-1.96*rsd, a+b*xs+1.96*rsd, color="#555555", alpha=.08, zorder=0)
    ax.plot(xs, a+b*xs, "--", color=LINE, lw=1.5, zorder=1)
    sig = "p < 0.001" if rho.pvalue < .001 else f"p = {rho.pvalue:.3f}"
    ax.set_title(f"{title}\nρ = {rho.statistic:.2f} ({sig}), n = {len(df)}",
                 fontsize=12.5, fontweight="bold", pad=8)
    ax.set_xlabel("Cumulative reported cases, $\\log_{10}(x+1)$", fontsize=12)
    if j == 0:
        ax.set_ylabel("WoS publications (whole-counting), $\\log_{10}(x+1)$", fontsize=12)
    ax.tick_params(labelsize=10.5)
    gl = g.sort_values("resid_mv").head(6)
    anchors = ok.sort_values("Publications", ascending=False).head(3)
    txts = []
    for _, r in gl.iterrows():
        txts.append(ax.text(r["lc"], r["lp"], r["ISO3"], fontsize=9.2,
                            color="#8a1c28", fontweight="bold", zorder=5))
    for _, r in anchors.iterrows():
        txts.append(ax.text(r["lc"], r["lp"], r["ISO3"], fontsize=8.8,
                            color="#2f3a2f", fontstyle="italic", zorder=5))
    adjust_text(txts, ax=ax, lim=1200, expand_text=(1.15,1.4), expand_points=(1.35,1.35),
                force_text=(0.7,0.9), force_points=(0.5,0.6),
                arrowprops=dict(arrowstyle="-", color="#999999", lw=.45, shrinkA=2, shrinkB=3))
h = [Line2D([0],[0],marker="o",color="none",markerfacecolor=GAP_C,markeredgecolor="white",markersize=11,label="High-burden, under-researching (gap)"),
     Line2D([0],[0],marker="o",color="none",markerfacecolor=GREEN,markeredgecolor="white",markersize=11,label="Burden–research aligned"),
     Line2D([0],[0],ls="--",color=LINE,lw=1.5,label="OLS trend (log–log)")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.035))
for ext in ("png","pdf","svg"):
    fig.savefig(os.path.join(OUT, f"Figure7_row1_scatter.{ext}"), dpi=500, facecolor="white")
print("row1 done")

# ============================ ROW 2: forest ============================
fig = plt.figure(figsize=(W, 3.75))
gs = GridSpec(1, 4, figure=fig, left=L, right=R, wspace=WS, top=0.72, bottom=0.25)
axs = [fig.add_subplot(gs[0, j]) for j in range(4)]
for j, (fname, title, df) in enumerate(DATA):
    ax = axs[j]; m = std_ols(df)
    ypos = np.arange(len(VARS))[::-1]
    for y, vk in zip(ypos, VARS):
        b2, lo, hi = m["beta"][vk], m["lo"][vk], m["hi"][vk]
        s = m["p"][vk] < .05
        ax.plot([lo,hi],[y,y], color=VAR_COLOR[vk], lw=3.2, solid_capstyle="round", zorder=2, alpha=.95)
        ax.scatter([b2],[y], s=165, color=VAR_COLOR[vk], edgecolor="white", linewidth=1.3,
                   zorder=3, alpha=1.0 if s else 0.5)
        tag = "***" if m["p"][vk]<.001 else ("**" if m["p"][vk]<.01 else ("*" if m["p"][vk]<.05 else "ns"))
        ax.text(hi+0.03, y, tag, va="center", ha="left", fontsize=12.5,
                color=VAR_COLOR[vk] if s else "#888888", fontweight="bold" if s else "normal")
    ax.axvline(0, color="#555555", lw=1.3, ls=(0,(4,3)), zorder=1)
    ax.set_xlim(-0.38,1.30); ax.set_ylim(-0.6,len(VARS)-0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels([VAR_LABEL[v] for v in VARS] if j==0 else [], fontsize=12)
    ax.tick_params(axis="x", labelsize=10.5)
    ax.set_title(f"{title.split(' (')[0]}  (n={m['n']}, R²={m['r2']:.2f})", fontsize=12.5, fontweight="bold", pad=8)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
fig.text(0.5, 0.09, "Standardized regression coefficient (β) — multivariate OLS, bars = 95% CI",
         ha="center", va="center", fontsize=12)
fh = [Line2D([0],[0],marker="o",color="w",markerfacecolor=CASES_C,markersize=12,label="Disease burden (cases)"),
      Line2D([0],[0],marker="o",color="w",markerfacecolor=GDP_C,markersize=12,label="Economy (GDP/capita)"),
      Line2D([0],[0],marker="o",color="w",markerfacecolor=POP_C,markersize=12,label="Research size (population)"),
      Line2D([0],[0],color="#555555",lw=1.4,ls=(0,(4,3)),label="β = 0")]
fig.legend(handles=fh, loc="upper center", ncol=4, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5,0.975))
for ext in ("png","pdf","svg"):
    fig.savefig(os.path.join(OUT, f"Figure7_row2_forest.{ext}"), dpi=500, facecolor="white")
print("row2 done")

# ============================ ROW 3: residual gaps ============================
fig = plt.figure(figsize=(W, 4.8))
gs = GridSpec(1, 4, figure=fig, left=L, right=R, wspace=WS, top=0.82, bottom=0.16)
axs = [fig.add_subplot(gs[0, j]) for j in range(4)]
for j, (fname, title, df) in enumerate(DATA):
    ax = axs[j]
    g = df[df["gap_final"]].sort_values("resid_mv")  # all flagged (do not truncate)
    labels = g["ISO3"].tolist(); vals = g["resid_mv"].tolist()
    zero = (g["Publications"].values == 0).tolist()
    n = len(labels)
    y = np.arange(n)
    cols = [ZERO_C if z else GAP_C for z in zero]
    fs = 11.5 if n <= 7 else 9.0
    ax.barh(y, vals, color=cols, edgecolor="white", height=0.66, zorder=2)
    ax.axvline(0, color="#555555", lw=1.3, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=fs, fontweight="bold")
    ax.set_ylim(n-0.5, -0.5)
    ax.set_xlim(-1.25, 0.10)
    ax.tick_params(axis="x", labelsize=10.5)
    ax.set_title(title.split(" (")[0], fontsize=12.5, fontweight="bold", pad=8)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
gh = [Line2D([0],[0],marker="s",color="w",markerfacecolor=GAP_C,markersize=12,label="Under-researching gap"),
      Line2D([0],[0],marker="s",color="w",markerfacecolor=ZERO_C,markersize=12,label="Zero published outputs")]
fig.legend(handles=gh, loc="upper center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5,0.975))
fig.text(0.5, 0.055, "Residual log-publications after controlling for population & GDP (negative = below expected)",
         ha="center", va="center", fontsize=12)
for ext in ("png","pdf","svg"):
    fig.savefig(os.path.join(OUT, f"Figure7_row3_gap.{ext}"), dpi=500, facecolor="white")
print("row3 done")

# ---- verification print ----
print("\n=== verification (rho on percapita subset; gaps) ===")
for fname, title, df in DATA:
    rho = spearmanr(df["lc"], df["lp"])
    gg = df[df["gap_final"]].sort_values("resid_mv")
    print(f"{title.split(' (')[0]:12s} n={len(df):3d} rho={rho.statistic:.3f} p={rho.pvalue:.3g} gaps={len(gg):2d}: " +
          ",".join(gg['ISO3'].head(8)))
