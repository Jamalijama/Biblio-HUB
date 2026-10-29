# -*- coding: utf-8 -*-
"""
Supplementary figure: temporal juxtaposition of reported disease burden (bars)
and global WoS research output (line), four viruses. Descriptive only; the two
series are plotted on independent axes and no lead-lag or synchrony test is run.
"""
import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype":"none","pdf.fonttype":42,"ps.fonttype":42,"mathtext.default":"regular",
    "axes.edgecolor":"#333333","axes.linewidth":1.3,
    "axes.grid":True,"grid.color":"#EFEBE3","grid.linewidth":0.9,"axes.axisbelow":True})

COL = {"Chikungunya":"#B23A48","Zika":"#3A6EA5","Dengue":"#39844A","WNV":"#E08E45"}
BAR = "#C9C2B6"

# --- burden series ---
arca = pd.read_csv(DATA / "arca_country_year_clean.csv")
chik_c = arca[arca.virus=="Chikungunya"].groupby("year").cases.sum()
zik_c  = arca[arca.virus=="Zika"].groupby("year").cases.sum()
dg = pd.read_csv(DATA / "global_dengue_cases_country_year.csv")
den_c = dg.groupby("Year").Cases.sum()
wn = pd.read_csv(DATA / "USA_WNV_cases_by_year.csv")
wnv_c = wn.groupby("Year").Cases.sum()

pubs = pd.read_csv(DATA / "wos_global_pubs_by_year.csv")
def pubser(v):
    return pubs[pubs.virus==v].set_index("year").global_pubs

panels = [
    ("Chikungunya", chik_c, pubser("Chikungunya"), 2013, 2023,
     "Chikungunya — Americas cases vs. global output"),
    ("Zika", zik_c, pubser("Zika"), 2016, 2023,
     "Zika — Americas cases vs. global output"),
    ("Dengue", den_c, pubser("Dengue"), 2008, 2024,
     "Dengue — global cases vs. global output"),
    ("WNV", wnv_c, pubser("West Nile"), 2008, 2023,
     "West Nile — USA cases vs. global output"),
]

fig, axs = plt.subplots(2, 2, figsize=(13.2, 9.0))
for ax, (vk, cases, ps, y0, y1, title) in zip(axs.flat, panels):
    years = np.arange(y0, y1+1)
    cvals = [float(cases.get(y, np.nan)) for y in years]
    pvals = [float(ps.get(y, np.nan)) for y in years]
    ax.bar(years, cvals, color=BAR, edgecolor="#A9A194", lw=0.5, width=0.72, zorder=1,
           label="Reported cases")
    ax2 = ax.twinx()
    ax2.plot(years, pvals, "-o", color=COL[vk], lw=2.4, ms=4.5, zorder=3,
             label="Global WoS publications")
    ax.set_xticks(years[::2])
    ax.set_xlim(y0-0.6, y1+0.6)
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=9)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Reported cases", fontsize=12, color="#6f6859")
    ax2.set_ylabel("WoS publications", fontsize=12, color=COL[vk])
    ax.tick_params(axis="y", labelsize=10.5, colors="#6f6859")
    ax2.tick_params(axis="y", labelsize=10.5, colors=COL[vk])
    ax.tick_params(axis="x", labelsize=10.5)
    ax2.grid(False)
handles = [Patch(facecolor=BAR, edgecolor="#A9A194", label="Reported cases (region)"),
           Line2D([0],[0], color="#444444", marker="o", lw=2.4, ms=5, label="Global WoS publications")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=12,
           bbox_to_anchor=(0.5, -0.005))
fig.suptitle("Supplementary Figure. Temporal juxtaposition of reported disease burden and "
             "global scientific output", fontsize=15, fontweight="bold", y=0.985)
fig.text(0.5, 0.952, "Descriptive comparison on independent axes; bars are reported cases in the "
         "surveillance region and the line is global WoS publications; no lead-lag test is reported.",
         ha="center", fontsize=9.5, color="#555555")
fig.tight_layout(rect=[0,0.03,1,0.94])
for ext in ("png","pdf","svg"):
    fig.savefig(os.path.join(OUT, f"FigS_temporal_response.{ext}"), dpi=500,
                bbox_inches="tight", facecolor="white")
print("FigS_temporal_response done")
for vk, cases, ps, y0, y1, _ in panels:
    print(f"  {vk:12s} cases years {[y for y in range(y0,y1+1) if not np.isnan(cases.get(y,np.nan))][0]}-"
          f"{[y for y in range(y0,y1+1) if not np.isnan(cases.get(y,np.nan))][-1]} peak={max(cases.values):.0f}")
