# -*- coding: utf-8 -*-
"""Aggregate OpenDengue national extract to country x year total cases.
Rule: for each country-year, sum within (case_definition, T_res) groups, then
take the MAX across groups to avoid double counting (Total/weekly is the most
complete, and a yearly source plus a weekly source must not be added).
"""
import pandas as pd
from pathlib import Path
import os
DATA_ROOT = Path(os.environ.get("BIBLIO_DATA_ROOT", Path.cwd())).resolve()


OUT = DATA_ROOT / "06_全球分析"
src = OUT / "OpenDengue全球登革热" / "National_extract_V1_3.csv"
df = pd.read_csv(src, low_memory=False)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["dengue_total"] = pd.to_numeric(df["dengue_total"], errors="coerce")
df = df[df["Year"].between(2008, 2024) & df["dengue_total"].notna()]

# BRA sanity check across groups
for iso in ["BRA"]:
    sub = df[df["ISO_A0"] == iso]
    chk = sub.groupby(["Year", "case_definition_standardised", "T_res"])["dengue_total"].sum().reset_index()
    print("BRA 2024 by group:\n", chk[chk["Year"] == 2024].to_string(index=False))

grp = df.groupby(["ISO_A0", "adm_0_name", "Year",
                  "case_definition_standardised", "T_res"])["dengue_total"].sum().reset_index()
# max across definition/resolution groups = most complete single source per country-year
cy = grp.groupby(["ISO_A0", "adm_0_name", "Year"])["dengue_total"].max().reset_index()
cy = cy.rename(columns={"adm_0_name": "Country", "dengue_total": "Cases"})
cy["Cases"] = cy["Cases"].round(0).astype(int)
cy.to_csv(OUT / "global_dengue_cases_country_year.csv", index=False, encoding="utf-8-sig")

cum = cy.groupby(["ISO_A0", "Country"])["Cases"].sum().reset_index().rename(columns={"Cases": "Cases_2008_2024"})
cum = cum.sort_values("Cases_2008_2024", ascending=False)
cum.to_csv(OUT / "global_dengue_cases_cumulative.csv", index=False, encoding="utf-8-sig")
print("\nTop 25 dengue countries 2008-2024:")
print(cum.head(25).to_string(index=False))
print("\nCountries:", cum["ISO_A0"].nunique(), "| rows:", len(cy))
