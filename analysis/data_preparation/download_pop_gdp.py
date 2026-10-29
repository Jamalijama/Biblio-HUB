# -*- coding: utf-8 -*-
"""Download World Bank total population and GDP per capita (latest available year)."""
import json, urllib.request
import pandas as pd
from pathlib import Path
import os
DATA_ROOT = Path(os.environ.get("BIBLIO_DATA_ROOT", Path.cwd())).resolve()

OUT = DATA_ROOT / "06_全球分析"

def wb(indicator, year):
    u = (f"https://api.worldbank.org/v2/country/all/indicator/{indicator}"
         f"?format=json&date={year}&per_page=400")
    req = urllib.request.Request(u, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=40) as r:
        j = json.loads(r.read().decode("utf-8"))
    rows = []
    for rec in j[1]:
        if rec["value"] is not None and rec["countryiso3code"]:
            rows.append({"ISO3": rec["countryiso3code"], "Country": rec["country"]["value"],
                         "year": rec["date"], indicator: rec["value"]})
    return pd.DataFrame(rows)

# population: try 2024 then 2023
pop = None
for yr in ["2024","2023"]:
    pop = wb("SP.POP.TOTL", yr)
    if len(pop) > 150: break
pop = pop.rename(columns={"SP.POP.TOTL":"Population"})
gdp = wb("NY.GDP.PCAP.CD", "2023").rename(columns={"NY.GDP.PCAP.CD":"GDP_pc"})

m = pop.merge(gdp[["ISO3","GDP_pc"]], on="ISO3", how="left")
m.to_csv(OUT/"worldbank_pop_gdp.csv", index=False, encoding="utf-8-sig")
print("rows:", len(m), "| pop year:", pop["year"].iloc[0] if len(pop) else None)
print(m.sort_values("Population", ascending=False).head(8).to_string(index=False))
print("\nGDP available:", m["GDP_pc"].notna().sum())
