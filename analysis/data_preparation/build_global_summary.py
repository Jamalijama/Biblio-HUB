# -*- coding: utf-8 -*-
"""Consolidated global preparedness results table (single source of truth for the paper)."""
import numpy as np, pandas as pd
from pathlib import Path
import os
DATA_ROOT = Path(os.environ.get("BIBLIO_DATA_ROOT", Path.cwd())).resolve()

from scipy.stats import spearmanr
OUT = DATA_ROOT / "06_全球分析"

FILES = {"dengue":"master_dengue_global","chik":"master_chik_americas",
         "zika":"master_zika_americas","wnv":"master_wnv_europe_usa"}
def load(b):
    abs_df = pd.read_csv(OUT/f"{FILES[b]}.csv")
    pc_df  = pd.read_csv(OUT/f"master_{b}_percapita.csv")
    return abs_df, pc_df

rows = []
gap_lists = {}
for b, scope, period in [
    ("dengue","Global (129 countries)","2008–2024"),
    ("chik","Americas (PAHO)","2013–2023"),
    ("zika","Americas (PAHO)","2016–2023"),
    ("wnv","Europe (ECDC)+USA (CDC)","2008–2023")]:
    a, pc = load(b)
    fit = a[a["Cases"]>0]
    rho_abs = spearmanr(fit["lc"],fit["lp"])
    rho_rate = spearmanr(pc["case_rate"],pc["pub_rate"])
    # OLS
    X = np.column_stack([np.ones(len(pc)),np.log10(pc["Cases"]+1),
                         np.log10(pc["Population"]),np.log10(pc["GDP_pc"])])
    y = np.log10(pc["Publications"]+1); beta,*_ = np.linalg.lstsq(X,y,rcond=None)
    r2 = 1-((y-X@beta)**2).sum()/((y-y.mean())**2).sum()
    gaps = pc[pc["gap_mv"]].sort_values("resid_mv")
    over = pc[(pc["Cases"]>0)].sort_values("resid_mv",ascending=False)
    gap_lists[b] = list(gaps["ISO3"].head(10))
    rows.append({
        "Virus":b,"Scope":scope,"Period":period,"n_sovereign":len(fit),
        "Total_cases":int(fit["Cases"].sum()),
        "rho_abs":round(rho_abs.statistic,3),"p_abs":(f"{rho_abs.pvalue:.2e}" if rho_abs.pvalue<1e-3 else round(rho_abs.pvalue,3)),
        "rho_rate":round(rho_rate.statistic,3),"p_rate":(f"{rho_rate.pvalue:.2e}" if rho_rate.pvalue<1e-3 else round(rho_rate.pvalue,3)),
        "OLS_R2":round(r2,3),"coef_cases":round(beta[1],2),"coef_pop":round(beta[2],2),"coef_logGDP":round(beta[3],2),
        "Top_gap_countries(GDP-controlled)":";".join(gaps["ISO3"].head(10)),
        "Top_over_supplied":";".join(over.head(5)["ISO3"])})
tab = pd.DataFrame(rows)
tab.to_csv(OUT/"GLOBAL_preparedness_summary.csv",index=False,encoding="utf-8-sig")
pd.set_option("display.width",250,"display.max_columns",30)
print(tab[["Virus","Scope","n_sovereign","Total_cases","rho_abs","p_abs","rho_rate","OLS_R2","coef_cases","coef_pop","coef_logGDP"]].to_string(index=False))
print("\nGap countries (GDP-controlled):")
for b in ["dengue","chik","zika","wnv"]:
    print(f"  {b:7s}:", ", ".join(gap_lists[b]))
