# Biblio-HUB

`Biblio-HUB` is an integrated, reproducible bibliometric analysis platform built with `Python + Streamlit`. It covers the full workflow from data import, field cleaning and keyword curation to collaboration networks, topic evolution, structural indicators, forward signals and export.

This repository is the official code for the following paper and contains both the platform and the complete reproduction code for the four-virus application:

> **Biblio-HUB: a reproducible bibliometric workflow for aligning arbovirus research output with disease burden**
>
> The application uses four arboviruses — chikungunya, Zika, dengue and West Nile virus — to link country-level research output with disease burden and to identify countries with research gaps that remain after accounting for population size and economic capacity.

## Two parts

| Location | Contents | Web of Science data needed? |
|---|---|---|
| Repository root (`app.py`, `modules/`) | Integrated bibliometric platform (Streamlit app) | Import your own WoS text/CSV after launching |
| [`analysis/`](analysis/README.md) | Reproduction of the four-virus application: Fig 7, Fig S15–S17, Tables S19–S26 | Not for the figure scripts; only when rebuilding from raw data |

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements-runtime.txt   # to run the platform only
pip install -r requirements.txt           # full features (including topic modeling)
```

### 2. Launch the platform

```bash
streamlit run app.py --server.port 8501 --server.headless true
```

On Windows, you can also double-click `start.bat`, then open `http://localhost:8501`.

### 3. Reproduce the four-virus figures and tables

```bash
cd analysis
pip install -r requirements.txt
python figures/plot_figure7.py            # Figure 7
python figures/plot_figS15_temporal.py    # Figure S15
python figures/plot_figS16_percapita.py   # Figure S16
python figures/plot_figS17_bertopic.py    # Figure S17
```

See [`analysis/README.md`](analysis/README.md). These four scripts read only the country-level aggregated CSV files shipped with the repository and run without any subscription data.

## Features

- **Data import**: Web of Science plain-text exports (`.txt`) and standardized `.csv`, with automatic detection of UTF-8 / Latin-1 / GBK / CP1252 encodings
- **Data curation**: field standardization, year cleaning, duplicate handling, keyword filtering and synonym merging
- **Relational networks**: keyword co-occurrence, keyword-journal association, author/institution/country collaboration, document/author/journal co-citation, bibliographic coupling
- **Structural analysis**: descriptive summary, journal analysis, keyword analysis, thematic map, hierarchical cluster heatmap, top entities, authors over time, three-field plot, Lotka's law, language distribution
- **Temporal and forward signals**: keyword timelines, burst detection, alluvial topic flow, BERTopic workspace, forward signals
- **Innovation signals**: disruption index and brokerage
- **Export center**: figures, tables, citations, network files and one-click bundles

The eight main modules correspond to Methods 2.2: Dataset Overview; Relational Network Analysis; Temporal Trend & Topic Evolution (BERTopic); Bibliometric Structure & Performance; Citation, Category & Source Analysis; Forward Signals; Innovation Analysis (DI + brokerage); and Export Center.

> Note on the Disruption Index (DI): the index depends on complete forward and backward citation networks. Within a closed search subset, values are pulled toward ±1 and serve only as an internal-network exploratory proxy; they do not support a causal claim about the "most disruptive paper." The index has its full bibliometric meaning on the cross-disciplinary Web of Science citation network.

## Data formats

- **Web of Science text (`.txt`)**: parses `AU`, `TI`, `SO`, `AB`, `DE`, `ID`, `CR`, `TC`, `C1`, `PY`, `DI` and other fields
- **CSV (`.csv`)**: maps common column names (title, authors, journal, abstract, keywords, citation count, affiliations, country, etc.)

The platform does not bundle any Web of Science records. The `data/` directory contains only a placement note; import files in the formats above. The country-level aggregated statistics for the four-virus application (no WoS records) are in `analysis/data/`.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest          # or: python run_tests.py (run_tests.bat on Windows)
```

## Project structure

```text
Biblio-HUB/
├── app.py                 # Streamlit entry point
├── modules/               # Platform modules (24)
├── lib/                   # Front-end components (vis-network, tom-select)
├── assets/                # Branding and static assets
├── tests/                 # pytest unit tests
├── data/                  # Data placement note (no raw data bundled)
├── analysis/              # Four-virus application reproduction
│   ├── figures/           #   Self-contained plotting scripts (Fig 7 / S15-S17)
│   ├── data/              #   Country-level aggregated CSVs (source data for Tables S19-S26)
│   ├── data_preparation/  #   Reference scripts to rebuild from raw data
│   └── outputs/           #   Figures generated by the scripts (created at run time)
├── .streamlit/  .github/workflows/
├── requirements*.txt  pytest.ini  run_tests.py  start.bat
└── LICENSE  THIRD_PARTY_NOTICES.md
```

## Dependency layers

- `requirements-runtime.txt`: to run the platform
- `requirements-topic-modeling.txt`: BERTopic and semantic topic modeling (heavier)
- `requirements-dev.txt`: testing and development
- `analysis/requirements.txt`: to reproduce the four-virus figure scripts
- `requirements.txt`: complete set

## Tech stack

Streamlit; Pandas, NumPy; NetworkX, python-louvain; PyVis; Plotly, Matplotlib; WordCloud; BERTopic, sentence-transformers, UMAP, HDBSCAN; SciPy (for the four-virus analysis).

## Requirements

Python >= 3.10; 4 GB or more memory recommended; Chrome / Edge / Firefox.

## Citation

If you use the platform or reproduce the analysis, please cite the accompanying paper (the formal volume, issue and page information will be added after acceptance).

## License

This project is released under the **MIT License**. Third-party components and license notes are in `THIRD_PARTY_NOTICES.md`.
