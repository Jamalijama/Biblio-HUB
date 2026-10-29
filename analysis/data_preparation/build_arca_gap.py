# -*- coding: utf-8 -*-
"""Americas chikungunya & zika burden (ARCA/PAHO) vs WoS output, country level."""
import numpy as np, pandas as pd
from pathlib import Path
import os
DATA_ROOT = Path(os.environ.get("BIBLIO_DATA_ROOT", Path.cwd())).resolve()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = DATA_ROOT / "06_全球分析"
plt.rcParams.update({"font.family":"Arial","font.size":9,"pdf.fonttype":42})

arca = pd.read_csv(OUT/"ARCA美洲虫媒"/"arca_americas_cumulative.csv")
# align WoS codes to ISO3 (Americas are already ISO3; fix a few territory rollups)
ALIAS = {"ANT":"CUW","ABW":"ABW","SXM":"SXM"}
def wos_iso(c):
    return ALIAS.get(c, c)

results = {}
for virus, f, label in [
    ("Chikungunya","global_country_pubs_Chikungunya.csv","chik"),
    ("Zika","global_country_pubs_Zika.csv","zika")]:
    pubs = pd.read_csv(OUT/f)
    pubs["ISO3"] = pubs["Code"].map(wos_iso)
    pubs = pubs.groupby("ISO3",as_index=False)["Publications"].sum()
    m = arca[["ISO3","country",virus,"is_territory"]].merge(pubs,on="ISO3",how="left")
    m["Publications"] = m["Publications"].fillna(0)
    m = m.rename(columns={virus:"Cases"})
    m = m[~m["is_territory"]].copy()          # sovereign countries only
    m["lc"]=np.log10(m["Cases"]+1); m["lp"]=np.log10(m["Publications"]+1)
    fit = m[m["Cases"]>0]
    b,a = np.polyfit(fit["lc"],fit["lp"],1)
    m["resid"]=m["lp"]-(a+b*m["lc"])
    cmed = fit["Cases"].median()
    m["gap"]=(m["Cases"]>cmed)&(m["resid"]<0)
    rho = spearmanr(fit["lc"],fit["lp"])
    results[label]=(m,rho,b,a)
    m.to_csv(OUT/f"global_{label}_americas_gap.csv",index=False,encoding="utf-8-sig")
    print("="*70)
    print(f"{virus}: n={len(fit)} sovereign countries, rho={rho.statistic:.3f} p={rho.pvalue:.3g} slope={b:.2f}")
    print("HIGH-BURDEN / UNDER-RESEARCH:")
    print(m[m["gap"]].sort_values("resid")[["ISO3","country","Cases","Publications","resid"]].head(12).to_string(index=False))
    print("OVER-SUPPLIED (regional context):")
    print(m[(m["Cases"]<=cmed)&(m["resid"]>0.3)].sort_values("resid",ascending=False)[["ISO3","country","Cases","Publications","resid"]].head(6).to_string(index=False))

# 2-panel figure
fig, axes = plt.subplots(1,2, figsize=(12,5.2))
for ax,(label,title) in zip(axes,[("chik","Chikungunya (Americas, 2013–2023)"),
                                  ("zika","Zika (Americas, 2016–2023)")]):
    m,rho,b,a = results[label]
    for _,r in m.iterrows():
        if r["Cases"]<=0: continue
        col="#B23A48" if r["gap"] else "#39844A"
        ax.scatter(r["lc"],r["lp"],s=44,color=col,alpha=.82,edgecolor="white",lw=.5)
        if r["gap"] or r["ISO3"] in ("USA","BRA","CAN","MEX","COL","ARG"):
            ax.annotate(r["ISO3"],(r["lc"],r["lp"]),xytext=(3,3),textcoords="offset points",
                        fontsize=6.5,color="#8a1c28" if r["gap"] else "#333",
                        fontweight="bold" if r["gap"] else "normal")
    xs=np.linspace(0,m["lc"].max(),50)
    ax.plot(xs,a+b*xs,"--",color="#555",lw=1.1)
    ax.set_title(f"{title}\nSpearman rho={rho.statistic:.2f} (p={rho.pvalue:.2g}); red=high-burden/under-research",fontsize=10,fontweight="bold")
    ax.set_xlabel("Reported cases, $\\log_{10}(x+1)$ (PAHO/ARCA)")
    ax.set_ylabel("Publications (WoS, whole-counting), $\\log_{10}(x+1)$")
    ax.grid(color="#EFEBE3",lw=.6)
fig.tight_layout()
fig.savefig(OUT/"Fig5_americas_chik_zika_gap.png",dpi=600,bbox_inches="tight",facecolor="white")
fig.savefig(OUT/"Fig5_americas_chik_zika_gap.pdf",bbox_inches="tight",facecolor="white")
print("\nFig5 written.")
