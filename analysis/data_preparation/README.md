# Rebuilding the aggregate data from raw inputs

> **NOTE — reference scripts, not turnkey.** The scripts in this folder are
> archived from the authors' working environment and resolve their data root
> from the `BIBLIO_DATA_ROOT` environment variable (default: the current
> working directory). They document the exact data transformations used in the
> paper but **do not run as-is on a fresh machine**, because the raw inputs
> (Web of Science exports and surveillance downloads) are not committed.
> Set `BIBLIO_DATA_ROOT` to the folder holding the raw data directories
> (e.g. `02_原始数据_WoS`) before running. The plotting scripts in
> `../figures/` already run without any of this, because they use the
> pre-computed aggregate CSVs committed in `../data/`.

## Why raw inputs are not all committed

Web of Science export files are subscription data and cannot be redistributed.
The public surveillance sources (OpenDengue, PAHO, CDC, ECDC) are free and can
be re-downloaded; see the table below. World Bank indicators are fetched by a
script.

## Pipeline

| Step | Script | Inputs (supply yourself) | Output |
|---|---|---|---|
| 1 | `download_pop_gdp.py` | none (World Bank API, standard library) | `worldbank_pop_gdp.csv` (population 2024, GDP per capita 2023) |
| 2 | `global_wos_matching.py` | Web of Science `*.txt` exports (subscription) | `global_country_pubs_*.csv` (country-level publication counts; 98.5–99.7% matched) |
| 3 | `build_global_dengue.py` | OpenDengue `National_extract_V1_3.csv` | `global_dengue_cases_country_year.csv`, `global_dengue_cases_cumulative.csv` |
| 4 | `build_arca_final.py` | PAHO ARBO Portal export (`arcadb.sql`, SQLite) | `arca_country_year_clean.csv` (chikungunya & Zika, Americas) |
| 5 | *(manual download)* | CDC ArboNET WNV surveillance table | `USA_WNV_cases_by_year.csv` |
| 6 | `build_global_master.py` | outputs of steps 1–5 | `master_chik_americas.csv`, `master_zika_americas.csv`, `master_dengue_global.csv`, `master_wnv_europe_usa.csv` |
| 7 | `build_global_percapita.py` | step 6 masters + step 1 covariates | `master_*_percapita.csv` (rates, OLS residuals `resid_mv`, gap flags `gap_mv`) |
| 8 | `build_global_dengue_gap.py`, `build_arca_gap.py` | step 6/7 outputs | `global_dengue_gap.csv`, `global_chik_americas_gap.csv`, `global_zika_americas_gap.csv` |
| 9 | `build_global_summary.py` | all masters/gaps | `GLOBAL_preparedness_summary.csv` (Table S19) |
| 10 | `run_bertopic_full.py`, `tune_bertopic.py` | four-virus Web of Science exports (subscription) | `BERTopic_four_virus_summary.csv` (topic counts, outlier fraction, cross-seed ARI) |

## Source data

* OpenDengue master dataset v1.3 (CC-BY): https://github.com/OpenDengue/master-dataset
* PAHO ARBO Portal (dengue/chikungunya/Zika in the Americas): https://www.paho.org/en/arbo-portal
* CDC ArboNET (United States, including WNV): https://www.cdc.gov/westnile/statsmaps/index.html
* ECDC surveillance (European WNV): https://www.ecdc.europa.eu/en/surveillance-atlas-infectious-diseases
* World Bank World Development Indicators: population (2024), GDP per capita (2023)
* Web of Science Core Collection (institutional subscription): 1985–2026, search date 2026-06-07

## Topic-modeling settings (steps 10)

Embeddings `all-MiniLM-L6-v2` (384-d); UMAP `n_neighbors=30`, cosine metric;
HDBSCAN `min_cluster_size=max(60, 0.006 N)`, `cluster_selection_method="eom"`;
class-based c-TF-IDF with MMR diversity 0.3; HDBSCAN outliers reassigned by
embedding similarity; random seeds 42, 123, 2024; cross-seed agreement
reported as the adjusted Rand index (ARI). Requires the (commented-out)
topic-modelling packages listed in `../requirements.txt`.
