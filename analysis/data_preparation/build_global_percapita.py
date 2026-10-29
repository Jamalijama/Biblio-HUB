# -*- coding: utf-8 -*-
"""Per-capita normalization + multivariate research-gap model.
Adds World Bank population and GDP per capita, then fits
  log(publications) ~ log(cases) + log(population) + log(GDP per capita)
so negative residuals indicate a research gap AFTER controlling for burden,
country size and economic development."""
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

pop = pd.read_csv(OUT/"worldbank_pop_gdp.csv")
# keep only country rows (drop aggregates): WB aggregate iso3 codes are non-ISO; keep those with 3-letter ISO that are countries
pop = pop[pop["ISO3"].str.len()==3]
# manual fill for non-WB economies that matter
pop = pop.set_index("ISO3")
for iso,p,g in [("TWN",23_550_000,32756),("VUT",327_000,None),("VCT",101_000,11940),
                ("KNA",48_000,None),("DMA",73_000,None),("GRD",126_000,11000),
                ("LCA",180_000,12000),("BES",27_000,None),("BLM",11_000,None),
                ("MAF",32_000,None),("SXM",44_000,None),("MSR",4_000,None),
                ("AIA",16_000,None),("TCA",45_000,None),("FLK",4_000,None),
                ("VGB",39_000,None),("MSR",4_000,None),("SHN",5_000,None),
                ("NRU",12_000,None),("TUV",11_000,None),("KIR",123_000,None),
                ("MHL",42_000,None),("PLW",18_000,None),("FSM",114_000,None),
                ("WSM",225_000,4100),("TLS",1350_000,1200),("SLB",740_000,2200),
                ("CUW",156_000,None),("ABW",107_000,None),("CUB",11_000_000,None)]:
    if iso not in pop.index:
        pop.loc[iso] = {"Country":iso,"year":2024,"Population":p,"GDP_pc":g}

def analyze(csv, label):
    df = pd.read_csv(OUT/csv)
    df = df[df["Cases"]>0].copy()
    df = df.merge(pop[["Population","GDP_pc"]], left_on="ISO3", right_index=True, how="left")
    miss = df[df["Population"].isna()]
    if len(miss): print(f"[{label}] missing pop:", list(miss["ISO3"]))
    df = df.dropna(subset=["Population","GDP_pc"]).copy()
    # rates
    df["case_rate"] = df["Cases"]/df["Population"]*1e5
    df["pub_rate"]  = df["Publications"]/df["Population"]*1e7
    # rate alignment
    rr = spearmanr(df["case_rate"], df["pub_rate"])
    # multivariate OLS
    X = np.column_stack([np.ones(len(df)),
                         np.log10(df["Cases"]+1),
                         np.log10(df["Population"]),
                         np.log10(df["GDP_pc"])])
    y = np.log10(df["Publications"]+1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X@beta
    df["resid_mv"] = y - pred
    # R2
    r2 = 1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum()
    # gaps: high case-rate (above median) AND negative residual
    crmed = df["case_rate"].median()
    df["gap_mv"] = (df["case_rate"]>crmed) & (df["resid_mv"]<0)
    df.to_csv(OUT/f"master_{label}_percapita.csv", index=False, encoding="utf-8-sig")
    print("="*70)
    print(f"{label}: n={len(df)} | rate rho={rr.statistic:.3f} p={rr.pvalue:.3g} | OLS R2={r2:.3f}")
    print(f"  coef: cases={beta[1]:.2f} pop={beta[2]:.2f} logGDP={beta[3]:.2f}")
    print("  CONTROLLED RESEARCH GAP (high incidence, low pubs after GDP control):")
    g = df[df["gap_mv"]].sort_values("resid_mv")
    print(g[["ISO3","Cases","Population","GDP_pc","case_rate","Publications","resid_mv"]].head(15).to_string(index=False))
    return df

d = analyze("master_dengue_global.csv","dengue")
c = analyze("master_chik_americas.csv","chik")
z = analyze("master_zika_americas.csv","zika")
w = analyze("master_wnv_europe_usa.csv","wnv")

# figure: incidence vs publication rate (dengue main, chik/zika/wnv inset)
fig, axes = plt.subplots(1,2, figsize=(13,5.4))
for ax,(df,title,mark) in zip(axes,[(d,"Dengue (global)",["BRA","IND","MEX","COL","USA","FRA","JPN","CHN","AUS","SGP","THA","SLV","BOL","GTM","HND","PRY","MMR","DOM","YEM"]),
                                    (w,"West Nile (Europe+USA)",["USA","ITA","FRA","DEU","ESP","GRC","ROU","HUN","HRV","BGR","CYP","AUT"])]):
    for _,r in df.iterrows():
        col="#B23A48" if r["gap_mv"] else "#39844A"
        ax.scatter(r["case_rate"], r["pub_rate"]+1e-3, s=46, color=col, alpha=.82, edgecolor="white", lw=.5)
        if r["gap_mv"] or r["ISO3"] in mark:
            ax.annotate(r["ISO3"],(r["case_rate"],r["pub_rate"]+1e-3),xytext=(3,3),
                        textcoords="offset points",fontsize=6.3,
                        color="#8a1c28" if r["gap_mv"] else "#333",
                        fontweight="bold" if r["gap_mv"] else "normal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"{title}: incidence vs research intensity (population/GDP-controlled)",fontsize=10,fontweight="bold")
    ax.set_xlabel("Cumulative cases per 100,000 population (log)")
    ax.set_ylabel("Publications per 10 million population (log)")
    ax.grid(color="#EFEBE3",lw=.6,which="both")
fig.tight_layout()
fig.savefig(OUT/"Fig7_percapita_dengue_wnv.png",dpi=600,bbox_inches="tight",facecolor="white")
fig.savefig(OUT/"Fig7_percapita_dengue_wnv.pdf",bbox_inches="tight",facecolor="white")
print("\nFig7 written.")
