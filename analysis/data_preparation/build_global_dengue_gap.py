# -*- coding: utf-8 -*-
"""Global dengue burden (OpenDengue, 2008-2024) vs WoS output (whole-counting).
Aligns mixed WoS codes (EU ISO2 + rest ISO3) to ISO3, fits log-log, computes
Spearman rho and regression-residual preparedness gap; emits table + figure.
"""
import numpy as np, pandas as pd
from pathlib import Path
import os
DATA_ROOT = Path(os.environ.get("BIBLIO_DATA_ROOT", Path.cwd())).resolve()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = DATA_ROOT / "06_全球分析"
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "pdf.fonttype": 42})

EU2_TO_3 = {"AT":"AUT","BE":"BEL","BG":"BGR","CY":"CYP","CZ":"CZE","DE":"DEU","DK":"DNK",
 "EE":"EST","EL":"GRC","ES":"ESP","FI":"FIN","FIN":"FIN","FR":"FRA","HR":"HRV","HU":"HUN",
 "IE":"IRL","IS":"ISL","IT":"ITA","LI":"LIE","LT":"LTU","LU":"LUX","LV":"LVA","MT":"MLT",
 "NL":"NLD","NO":"NOR","PL":"POL","PT":"PRT","RO":"ROU","SE":"SWE","SI":"SVN","SK":"SVK"}
def to_iso3(c): return EU2_TO_3.get(c, c)

cases = pd.read_csv(OUT / "global_dengue_cases_cumulative.csv").rename(
    columns={"ISO_A0":"ISO3","Cases_2008_2024":"Cases"})
pubs = pd.read_csv(OUT / "global_country_pubs_Dengue.csv")
pubs["ISO3"] = pubs["Code"].map(to_iso3)
pubs = pubs.groupby("ISO3", as_index=False)["Publications"].sum()

m = cases.merge(pubs, on="ISO3", how="outer")
m["Cases"] = m["Cases"].fillna(0)
m["Publications"] = m["Publications"].fillna(0)
m = m[~m["Country"].isna()]
m["lc"] = np.log10(m["Cases"]+1); m["lp"] = np.log10(m["Publications"]+1)

# fit on countries with any reported cases (burden-bearing sample)
fit = m[m["Cases"] > 0]
b, a = np.polyfit(fit["lc"], fit["lp"], 1)
m["resid"] = m["lp"] - (a + b*m["lc"])
rho = spearmanr(fit["lc"], fit["lp"])
cmed = fit["Cases"].median()
m["gap"] = (m["Cases"] > cmed) & (m["resid"] < 0)
m.to_csv(OUT / "global_dengue_gap.csv", index=False, encoding="utf-8-sig")
print(f"Fit countries (cases>0): {len(fit)}  Spearman rho={rho.statistic:.3f} p={rho.pvalue:.3g}")
print(f"slope={b:.2f} intercept={a:.2f} burden median={cmed:.0f}")
print("\nHIGH-BURDEN / UNDER-RESEARCHED (top 25 by residual):")
g = m[m["gap"]].sort_values("resid")
print(g[["ISO3","Country","Cases","Publications","resid"]].head(25).to_string(index=False))
print("\nOVER-SUPPLIED (high research, low burden, examples):")
o = m[(m["Cases"] <= cmed) & (m["resid"] > 0.6)].sort_values("resid", ascending=False)
print(o[["ISO3","Country","Cases","Publications","resid"]].head(12).to_string(index=False))

# figure
fig, ax = plt.subplots(figsize=(8.2, 6.4))
for _, r in m.iterrows():
    if r["Cases"] <= 0: continue
    col = "#B23A48" if r["gap"] else "#39844A"
    ax.scatter(r["lc"], r["lp"], s=44, color=col, alpha=0.8, edgecolor="white", lw=0.5, zorder=3 if r["gap"] else 2)
    label = r["gap"] or r["ISO3"] in ("USA","CHN","IND","BRA","GBR","DEU","FRA","AUS","JPN","THA","MEX","COL","SGP")
    if label:
        ax.annotate(r["ISO3"], (r["lc"], r["lp"]), xytext=(3,3),
                    textcoords="offset points", fontsize=6.5,
                    color="#8a1c28" if r["gap"] else "#333",
                    fontweight="bold" if r["gap"] else "normal")
xs = np.linspace(0, m["lc"].max(), 50)
ax.plot(xs, a+b*xs, "--", color="#555", lw=1.2, zorder=1)
ax.set_xlabel("Cumulative reported dengue cases 2008-2024, $\\log_{10}(x+1)$\n(OpenDengue v1.3)")
ax.set_ylabel("Dengue publications 2008-2024, $\\log_{10}(x+1)$\n(WoS, whole-counting)")
ax.set_title(f"Global dengue research preparedness gap\n"
             f"Spearman rho={rho.statistic:.2f} (p<1e-6); red = high-burden / under-researching countries",
             fontsize=10.5, fontweight="bold")
ax.grid(color="#EFEBE3", lw=0.6)
fig.tight_layout()
fig.savefig(OUT / "Fig4_global_dengue_gap.png", dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "Fig4_global_dengue_gap.pdf", bbox_inches="tight", facecolor="white")
print("\nFig4 written.")
