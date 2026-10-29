# -*- coding: utf-8 -*-
import sqlite3, pandas as pd
from pathlib import Path
import os
DATA_ROOT = Path(os.environ.get("BIBLIO_DATA_ROOT", Path.cwd())).resolve()

OUT = DATA_ROOT / "06_全球分析" / "ARCA美洲虫媒"
con = sqlite3.connect(":memory:")
con.executescript((OUT / "arcadb.sql").read_text(encoding="utf-8", errors="ignore"))

ISO3 = {
 "Canada":"CAN","United States of America":"USA","Belize":"BLZ","Costa Rica":"CRI",
 "El Salvador":"SLV","Guatemala":"GTM","Honduras":"HND","Mexico":"MEX","Nicaragua":"NIC",
 "Panama":"PAN","Bolivia":"BOL","Colombia":"COL","Ecuador":"ECU","Peru":"PER",
 "Venezuela":"VEN","Argentina":"ARG","Brazil":"BRA","Chile":"CHL","Paraguay":"PRY",
 "Cuba":"CUB","Dominican Republic":"DOM","Puerto Rico":"PRI","Anguilla":"AIA",
 "Antigua and Barbuda":"ATG","Aruba":"ABW","Bahamas":"BHS","Barbados":"BRB",
 "Bonaire, Saint Eustatius and Saba":"BES","Cayman Islands":"CYM","Curaçao":"CUW",
 "Curacao":"CUW","Dominica":"DMA","French Guiana":"GUF","Grenada":"GRD",
 "Guadeloupe":"GLP","Guyana":"GUY","Haiti":"HTI","Jamaica":"JAM","Martinique":"MTQ",
 "Montserrat":"MSR","Saint Barthelemy":"BLM","Saint Kitts and Nevis":"KNA",
 "Saint Lucia":"LCA","Saint Martin":"MAF","Saint Vincent and the Grenadines":"VCT",
 "Sint Maarten":"SXM","Suriname":"SUR","Trinidad and Tobago":"TTO",
 "Turks and Caicos Islands":"TCA","Virgin Islands (UK)":"VGB",
 "Virgin Islands (US)":"VIR","Uruguay":"URY","Bermuda":"BMU"}
TERR = {"PRI","AIA","ABW","BES","CYM","CUW","GUF","GLP","MTQ","MSR","BLM","MAF",
        "SXM","TCA","VGB","VIR","BMU"}

# annual = max(totalcases) per country x virus x year (country-level only, code='C')
q = """
SELECT c.country, v.virus, ca.year, MAX(ca.totalcases) AS cases
FROM Cases ca
JOIN Countries c ON c.idx=ca.country AND c.code='C'
JOIN Viruses v ON v.idx=ca.virus
GROUP BY c.country, v.virus, ca.year
"""
ann = pd.read_sql(q, con)
ann["ISO3"] = ann["country"].map(ISO3)
assert ann["ISO3"].notna().all(), ann[ann["ISO3"].isna()]["country"].unique()
ann = ann[["ISO3","country","virus","year","cases"]]
ann.to_csv(OUT/"arca_country_year_clean.csv", index=False, encoding="utf-8-sig")

cum = ann.groupby(["country","virus"])["cases"].sum().reset_index()
piv = cum.pivot(index="country", columns="virus", values="cases").fillna(0).reset_index()
for v in ["Chikungunya","Dengue","Zika"]:
    if v not in piv.columns: piv[v]=0
piv["ISO3"] = piv["country"].map(ISO3)
piv["is_territory"] = piv["ISO3"].isin(TERR)
piv = piv[["ISO3","country","Chikungunya","Dengue","Zika","is_territory"]]
piv = piv.sort_values("Chikungunya", ascending=False)
piv.to_csv(OUT/"arca_americas_cumulative.csv", index=False, encoding="utf-8-sig")

print("Period: chik 2013-2023, zika 2016-2023, dengue 2014-2023 (PAHO/ARCA, code=country)")
print("Totals (all countries): chik=%.0f zika=%.0f dengue=%.0f" % (
    piv["Chikungunya"].sum(), piv["Zika"].sum(), piv["Dengue"].sum()))
sov = piv[~piv["is_territory"]]
print("Totals (sovereign, territories excl): chik=%.0f zika=%.0f" % (
    sov["Chikungunya"].sum(), sov["Zika"].sum()))
print("\nTop chik countries:")
print(piv.head(15)[["ISO3","country","Chikungunya","Zika","is_territory"]].to_string(index=False))
print("\nTop zika countries:")
print(piv.sort_values("Zika",ascending=False).head(15)[["ISO3","country","Chikungunya","Zika","is_territory"]].to_string(index=False))
