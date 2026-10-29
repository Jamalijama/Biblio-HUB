# -*- coding: utf-8 -*-
"""Global research-preparedness master analysis.
Dengue: OpenDengue v1.3 (global, 2008-2024).
Chikungunya / Zika: ARCA-PAHO (Americas; chik 2013-2023, zika 2016-2023).
West Nile: ECDC Europe + CDC USA (2008-2023).
Outputs: unified country tables, Spearman rho, regression-residual gap, 4-panel figure.
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
EU  = DATA_ROOT / "05_ECDC与lag分析" / "重跑统计_2026"
plt.rcParams.update({"font.family":"Arial","font.size":9,"pdf.fonttype":42})

EU2_TO_3 = {"AT":"AUT","BE":"BEL","BG":"BGR","CY":"CYP","CZ":"CZE","DE":"DEU","DK":"DNK",
 "EE":"EST","EL":"GRC","ES":"ESP","FI":"FIN","FIN":"FIN","FR":"FRA","HR":"HRV","HU":"HUN",
 "IE":"IRL","IS":"ISL","IT":"ITA","LI":"LIE","LT":"LTU","LU":"LUX","LV":"LVA","MT":"MLT",
 "NL":"NLD","NO":"NOR","PL":"POL","PT":"PRT","RO":"ROU","SE":"SWE","SI":"SVN","SK":"SVK"}
def to_iso3(c): return EU2_TO_3.get(c, c)

# Non-sovereign territories whose WoS output is attributed to the metropole (France/USA/NL/UK)
TERR = {"GUF","GLP","MTQ","MAF","BLM","REU","NCL","PYF","WLF","SPM","VGB","AIA","MSR","TCA",
 "CYM","BMU","FLK","SHN","IOT","PCN","SGS","ABW","CUW","SXM","BES","PRI","VIR","GUM","ASM",
 "MNP","UMI","HMD","BVT","ATA","ATF","SJM","NFK","CXR","CCK","IMN","GGY","JEY","VAT","ANT"}

def load_pubs(fname):
    p = pd.read_csv(OUT/fname)
    p["ISO3"] = p["Code"].map(to_iso3)
    return p.groupby("ISO3", as_index=False)["Publications"].sum()

def gap_frame(cases, iso_col, cname_col, case_col, pubs):
    m = cases.merge(pubs, on="ISO3", how="left")
    m["Publications"] = m["Publications"].fillna(0)
    m = m[~m["ISO3"].isin(TERR)].copy()
    m["Cases"] = m["Cases"].fillna(0)
    m["lc"] = np.log10(m["Cases"]+1); m["lp"] = np.log10(m["Publications"]+1)
    fit = m[m["Cases"]>0]
    b,a = np.polyfit(fit["lc"],fit["lp"],1)
    m["resid"] = m["lp"]-(a+b*m["lc"])
    cmed = fit["Cases"].median()
    m["gap"] = (m["Cases"]>cmed)&(m["resid"]<0)
    rho = spearmanr(fit["lc"],fit["lp"])
    return m, fit, rho, b, a, cmed

panels = []
summary = []

# ---- DENGUE (global) ----
den = pd.read_csv(OUT/"global_dengue_cases_cumulative.csv").rename(
    columns={"ISO_A0":"ISO3","Cases_2008_2024":"Cases"})
den_p = load_pubs("global_country_pubs_Dengue.csv")
m,fit,rho,b,a,cmed = gap_frame(den[["ISO3","Country","Cases"]],"ISO3","Country","Cases",den_p)
m.to_csv(OUT/"master_dengue_global.csv",index=False,encoding="utf-8-sig")
panels.append(("Dengue\n(global, 2008–2024)", m, fit, rho, b, a, ["USA","FRA","JPN","CHN","AUS","BRA","IND","THA","MEX","COL","SGP","GBR"]))
summary.append(("Dengue (global)", len(fit), rho, b))

# ---- CHIK / ZIKA (Americas) ----
arca = pd.read_csv(OUT/"ARCA美洲虫媒"/"arca_americas_cumulative.csv")
for virus, fname, title, period in [
    ("Chikungunya","global_country_pubs_Chikungunya.csv","Chikungunya\n(Americas, 2013–2023)","chik"),
    ("Zika","global_country_pubs_Zika.csv","Zika\n(Americas, 2016–2023)","zika")]:
    c = arca[["ISO3","country",virus]].rename(columns={"country":"Country",virus:"Cases"})
    m,fit,rho,b,a,cmed = gap_frame(c,"ISO3","Country","Cases",load_pubs(fname))
    m.to_csv(OUT/f"master_{period}_americas.csv",index=False,encoding="utf-8-sig")
    panels.append((title, m, fit, rho, b, a, ["USA","CAN","BRA","MEX","COL","ARG"]))
    summary.append((virus+" (Americas)", len(fit), rho, b))

# ---- WEST NILE (ECDC Europe + CDC USA) ----
eu = pd.read_csv(EU/"02b_country_level.csv")
wnv = eu[eu["Virus"].astype(str).str.contains("West Nile|WNV", case=False, regex=True)].copy()
wnv["ISO3"] = wnv["RegionCode"].map(to_iso3)
wnv = wnv.rename(columns={"Country":"Country","Cumulative_cases":"Cases_ecdc"})
# US CDC 2008-2023
usa = pd.read_csv(OUT/"CDC美国数据"/"USA_WNV_cases_by_year.csv")
usa_08 = usa[(usa["Year"]>=2008)&(usa["Year"]<=2023)]["Cases"].sum()
wnv_cases = wnv[["ISO3","Country","Cases_ecdc"]].rename(columns={"Cases_ecdc":"Cases"})
wnv_cases = pd.concat([wnv_cases, pd.DataFrame([{"ISO3":"USA","Country":"United States","Cases":usa_08}])], ignore_index=True)
wnv_p = load_pubs("global_country_pubs_WestNile.csv")
m,fit,rho,b,a,cmed = gap_frame(wnv_cases,"ISO3","Country","Cases",wnv_p)
m.to_csv(OUT/"master_wnv_europe_usa.csv",index=False,encoding="utf-8-sig")
panels.append(("West Nile virus\n(Europe+USA, 2008–2023)", m, fit, rho, b, a, ["USA","ITA","FRA","DEU","GRC","ROU","HUN","HRV","BGR","ESP","AUT"]))
summary.append(("West Nile (Europe+USA)", len(fit), rho, b))

# ---- summary ----
print("="*80)
print("GLOBAL BURDEN-OUTPUT ALIGNMENT (Spearman rho, log-log)")
print("="*80)
for name,n,rho,slope in summary:
    print(f"{name:28s} n={n:3d}  rho={rho.statistic:.3f}  p={rho.pvalue:.3g}  slope={slope:.2f}")
print("\nUS WNV cases 2008-2023:", int(usa_08))

# ---- 4-panel figure ----
fig, axes = plt.subplots(2,2, figsize=(12.5,10))
for ax,(title,m,fit,rho,b,a,always) in zip(axes.flat, panels):
    for _,r in m.iterrows():
        if r["Cases"]<=0: continue
        col="#B23A48" if r["gap"] else "#39844A"
        ax.scatter(r["lc"],r["lp"],s=40,color=col,alpha=.82,edgecolor="white",lw=.5,zorder=3 if r["gap"] else 2)
        if r["gap"] or r["ISO3"] in always:
            ax.annotate(r["ISO3"],(r["lc"],r["lp"]),xytext=(3,3),textcoords="offset points",
                        fontsize=6.3,color="#8a1c28" if r["gap"] else "#333",
                        fontweight="bold" if r["gap"] else "normal")
    xs=np.linspace(0,m["lc"].max(),50)
    ax.plot(xs,a+b*xs,"--",color="#555",lw=1.1,zorder=1)
    sig = "p<0.001" if rho.pvalue<0.001 else f"p={rho.pvalue:.3f}"
    ax.set_title(f"{title}   ρ={rho.statistic:.2f} ({sig})",fontsize=10.5,fontweight="bold")
    ax.set_xlabel("Reported cases, $\\log_{10}(x+1)$")
    ax.set_ylabel("Publications (WoS, whole-counting), $\\log_{10}(x+1)$")
    ax.grid(color="#EFEBE3",lw=.6)
fig.suptitle("Research preparedness alignment across arboviruses and scales\n(red = high-burden, under-researching countries)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(OUT/"Fig6_global_preparedness_master.png",dpi=600,bbox_inches="tight",facecolor="white")
fig.savefig(OUT/"Fig6_global_preparedness_master.pdf",bbox_inches="tight",facecolor="white")
print("\nFig6 written.")

# gap country lists for the paper
print("\n--- WNV gap (Europe+USA) ---")
print(m[m["gap"]].sort_values("resid")[["ISO3","Country","Cases","Publications","resid"]].head(12).to_string(index=False))
