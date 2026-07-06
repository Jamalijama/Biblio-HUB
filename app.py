import base64
import time
import matplotlib
import numpy as np
import pandas as pd
import re
import streamlit as st

matplotlib.use('Agg')
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud

from modules.advanced_visualizations import (
    build_keyword_circular_cluster_figure,
    render_ranked_lollipop_figure,
)
from modules.citation_analysis import (
    build_journal_cocitation_network,
    build_publication_citation_trend_frame,
    build_reference_burst_table,
    build_rpys_peak_table,
    clean_cited_reference,
    extract_cited_reference_statistics,
    extract_rpys_statistics,
    render_publication_citation_dual_axis_figure,
    render_reference_burst_figure,
    render_rpys_figure,
)
from modules.data_pipeline import (
    _extract_country_from_affiliation,
    _count_semicolon_terms,
    _get_data_dir_fingerprint,
    _get_local_paths_fingerprint,
    _parse_wos_authors,
    _parse_wos_keywords,
    clean_year_column,
    process_global_data,
    safe_year,
)
from modules.entity_analysis import (
    build_top_country_table,
    build_top_institution_table,
    build_top_journal_table,
    calculate_category_frequency,
    calculate_frequency,
    calculate_simple_frequency,
)
from modules.error_logging import log_exception
from modules.experiment_framework import (
    build_brokerage_baseline_comparison_report,
    build_brokerage_robustness_report,
    build_experiment_comparison_report,
    build_graph_from_cooccurrence,
    build_innovation_metrics_report,
    build_journal_submission_package_report,
    build_journal_submission_package_snapshot,
    build_parameterized_journal_template,
    build_research_report,
    build_research_report_snapshot,
    build_reviewer_response_report,
    build_reviewer_response_snapshot,
    build_submission_figure_package_report,
    build_submission_figure_package_snapshot,
    build_submission_result_report,
    build_submission_result_snapshot,
    calculate_network_metrics,
    compute_brokerage_baseline_comparison,
    compute_brokerage_robustness_experiment,
    compute_disruption_index_frame,
    compute_export_center_innovation_payload,
    compute_structural_hole_frame,
    discard_innovation_background_job,
    format_execution_policy_summary,
    get_baseline_comparison_data,
    get_innovation_background_job,
    rank_disruption_extremes,
    submit_innovation_background_job,
    summarize_disruption_index,
    summarize_structural_hole_frame,
)
from modules.export_bundle import (
    KEYWORD_MATRIX_SEQUENTIAL_SCALE,
    MORANDI_SEQUENTIAL_SCALE,
    PUBLICATION_EXPORT_FORMATS,
    SCIENTIFIC_COLORWAY,
    build_manuscript_case_report,
    build_manuscript_case_snapshot,
    build_manuscript_submission_report,
    build_manuscript_submission_snapshot,
    build_reproducibility_report,
    build_reproducibility_snapshot,
    get_plotly_static_export_status,
)
from modules.export_orchestrator import (
    build_figure_explanation_bundle,
    build_journal_submission_bundle,
    build_manuscript_case_bundle,
    build_methods_package_bundle,
    build_one_click_research_bundle,
    build_reviewer_response_bundle,
    build_submission_result_bundle,
    generate_export_zip,
    get_available_figure_options,
    group_figure_options,
)
from modules.figure_export_bundle import generate_all_figure_bundle
from modules.keyword_pipeline import extract_keywords_from_dataframe
from modules.network_builders import build_cooccurrence_network, build_journal_network
from modules.network_visualization import (
    render_network_publication_figure,
)
from modules.project_state import (
    render_project_management_sidebar,
)
from modules.structure_visualization import (
    build_author_production_over_time_frame,
    render_author_production_over_time,
    render_lotkas_law,
    render_thematic_map,
    render_three_field_plot,
)
from modules.temporal_analysis import (
    build_entity_leadership_shift_tables,
    build_entity_forecast_tables,
    build_keyword_burst_table,
    build_keyword_growth_table,
    build_keyword_opportunity_map_frame,
    build_publication_forecast_frame,
    build_selected_keyword_share_table,
    build_theme_migration_forecast_tables,
    parse_selected_keywords,
    render_entity_leadership_shift_figure,
    render_entity_leadership_trajectory_figure,
    render_entity_forecast_rank_figure,
    render_entity_forecast_trajectory_figure,
    render_alluvial_topic_flow,
    render_burst_detection,
    render_citespace_timeline,
    render_keyword_growth_leader_figure,
    render_keyword_growth_trend_figure,
    render_keyword_opportunity_map,
    render_publication_forecast_figure,
    render_selected_keyword_share_figure,
    render_theme_migration_opportunity_map,
    render_theme_migration_trajectory_figure,
    summarize_entity_leadership_shift,
    summarize_entity_forecast_signals,
    summarize_forward_signals_overview,
    summarize_keyword_opportunity_map,
    summarize_publication_forecast,
    summarize_theme_migration_signals,
)
from modules.topic_modeling import (
    discard_bertopic_background_job,
    get_bertopic_background_job,
    get_bertopic_profile_settings,
    render_bertopic_comparison,
    render_bertopic_evolution,
    render_bertopic_overview,
    run_bertopic_analysis,
    submit_bertopic_background_job,
)
from modules.ui_export_center import render_export_center, render_innovation_analysis_panel
from modules.ui_helpers import (
    apply_publication_style_with_overrides,
    download_matplotlib_button,
    download_plotly_button,
    integer_control,
    render_plot_style_controls,
    render_and_download_network_figure,
    render_html_iframe,
    render_plotly_chart,
    show_cluster_report,
)
from modules.ui_relational_network import render_relational_network_page

NETWORK_CLUSTER_PALETTE = [
    "#D45959",
    "#2F74B8",
    "#5E379D",
    "#507D39",
    "#F2B382",
    "#60B0F4",
    "#FF6B78",
    "#36663E",
    "#5A97D0",
    "#E2A479",
    "#8CB26C",
    "#88AFD8",
    "#F9B43F",
    "#5DCE9C",
    "#3A79C0",
]
MORANDI_WORDCLOUD = LinearSegmentedColormap.from_list(
    "morandi_wordcloud",
    [
        "#D45959",
        "#E2A479",
        "#F2B382",
        "#2F74B8",
        "#5A97D0",
        "#5E379D",
        "#60B0F4",
        "#507D39",
        "#8CB26C",
        "#88AFD8",
        "#FF6B78",
        "#36663E",
        "#F9B43F",
    ],
)
pio.templates.default = "plotly_white"
px.defaults.color_discrete_sequence = SCIENTIFIC_COLORWAY


@st.cache_data
def get_cached_language_freq(df):
    return calculate_simple_frequency(df, "Language")


@st.cache_data
def get_cached_publisher_freq(df):
    return calculate_simple_frequency(df, "Publisher")


@st.cache_data
def get_cached_category_freq(df):
    return calculate_category_frequency(df)


def display_performance_metrics(start_time, label="Generation"):
    duration = time.time() - start_time
    if st.session_state.get("show_perf_metrics", False):
        st.caption(f"{label} performance: {duration:.2f} seconds.")
        if "perf_logs" not in st.session_state:
            st.session_state.perf_logs = []
        st.session_state.perf_logs.append({
            "Module": label,
            "Duration_Seconds": round(duration, 3),
            "Records": len(st.session_state.get("loaded_df", pd.DataFrame())) if st.session_state.get("project_loaded_active") else 0, # Note: this might be inaccurate if df is global
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })


def _count_available_authors(df: pd.DataFrame) -> int:
    authors = set()
    for value in df.get("Authors", pd.Series(dtype=object)).dropna():
        authors.update(_parse_wos_authors(value))
    return len(authors)


def _count_available_institutions(df: pd.DataFrame) -> int:
    institutions = set()
    for value in df.get("Affiliations", pd.Series(dtype=object)).dropna():
        affiliation_text = str(value).strip()
        if not affiliation_text or affiliation_text.lower() == "nan":
            continue
        address_blocks = re.split(r";\s*(?=\[)", affiliation_text)
        if len(address_blocks) <= 1:
            address_blocks = [affiliation_text]
        for block in address_blocks:
            block = block.strip()
            if not block:
                continue
            institution_name = block.split("]", 1)[1].strip() if "]" in block else block
            institution_name = institution_name.rstrip(".")
            if institution_name and len(institution_name) > 1:
                institutions.add(institution_name)
    return len(institutions)


def _count_available_countries(df: pd.DataFrame) -> int:
    countries = set()
    for value in df.get("Affiliations", pd.Series(dtype=object)).dropna():
        extracted = _extract_country_from_affiliation(str(value))
        if not extracted:
            continue
        countries.update(country.strip() for country in extracted.split(";") if country.strip())
    return len(countries)


def _parse_local_file_paths_text(raw_text: str) -> list[str]:
    paths = []
    for line in str(raw_text or "").splitlines():
        candidate = line.strip().strip('"')
        if candidate:
            paths.append(candidate)
    return paths


def _collect_input_file_stats(uploaded_files=None, local_file_paths=None):
    items = []
    for uploaded_file in uploaded_files or []:
        items.append(
            {
                "name": str(getattr(uploaded_file, "name", "uploaded_file")),
                "path": None,
                "size_bytes": int(getattr(uploaded_file, "size", 0) or 0),
                "source": "upload",
            }
        )
    for raw_path in local_file_paths or []:
        normalized = str(raw_path or "").strip().strip('"')
        if not normalized:
            continue
        size_bytes = os.path.getsize(normalized) if os.path.exists(normalized) else 0
        items.append(
            {
                "name": os.path.basename(normalized) or normalized,
                "path": normalized,
                "size_bytes": int(size_bytes or 0),
                "source": "local",
            }
        )
    return items


@st.cache_data
def get_cached_funding_freq(df):
    return calculate_frequency(df, "Funding")


@st.cache_data
def get_cached_doctype_freq(df):
    return calculate_simple_frequency(df, "DocType")


@st.cache_data
def get_cached_annual_growth_rate(df):
    df_valid = clean_year_column(df)
    year_counts = df_valid['Year'].value_counts().sort_index()
    growth_data = []
    for i in range(1, len(year_counts)):
        prev = year_counts.iloc[i - 1]
        curr = year_counts.iloc[i]
        if prev > 0:
            growth = (curr - prev) / prev * 100
        else:
            growth = 0
        growth_data.append({'Year': year_counts.index[i], 'Growth Rate (%)': round(growth, 1)})
    return pd.DataFrame(growth_data) if growth_data else pd.DataFrame()


@st.cache_data
def get_cached_journal_counts(df):
    df_valid = clean_year_column(df)
    return df_valid['Journal'].value_counts()


@st.cache_data
def get_cached_lotkas_law(df):
    return render_lotkas_law(df)


@st.cache_data
def get_cached_thematic_map(keyword_freq, cooccurrence, top_n=20):
    return render_thematic_map(keyword_freq, cooccurrence, top_n=top_n)


@st.cache_data
def get_cached_three_field_plot(df, keywords_list, keyword_freq, n_auth, n_kw, n_jn):
    return render_three_field_plot(df, keywords_list, keyword_freq, n_auth, n_kw, n_jn)


@st.cache_data
def get_cached_author_production_over_time(df, top_n=10):
    return render_author_production_over_time(df, top_n=top_n)


KEYWORD_SOURCE_OPTIONS = {
    "DE+ID": "DE+ID",
    "DE": "DE",
    "ID": "ID",
}


@st.cache_data
def get_cached_heavy_analysis_artifacts(
    df,
    blocked_terms_text="",
    replacement_map_text="",
    keyword_source="DE+ID",
    enable_optional_domain_plugin=False,
):
    keywords = extract_keywords_from_dataframe(
        df,
        blocked_terms_text=blocked_terms_text,
        replacement_map_text=replacement_map_text,
        keyword_source=keyword_source,
        enable_optional_domain_plugin=enable_optional_domain_plugin,
    )
    keyword_counter, cooccurrence_counter = build_cooccurrence_network(keywords)
    journal_year_map = build_journal_network(df)
    return keywords, keyword_counter, cooccurrence_counter, journal_year_map


@st.cache_data
def get_cached_citation_by_year(df):
    if 'Times_Cited' not in df.columns or 'Year' not in df.columns:
        return pd.DataFrame()
    df_valid = clean_year_column(df)
    df_valid['Times_Cited'] = pd.to_numeric(df_valid['Times_Cited'], errors='coerce').fillna(0)
    cite_by_year = df_valid.groupby('Year')['Times_Cited'].mean().reset_index()
    cite_by_year.columns = ['Year', 'Avg Citations']
    return cite_by_year

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
BRANDING_DIR = os.path.join(APP_ROOT, "assets", "branding")
LOGO_HORIZONTAL_PNG = os.path.join(BRANDING_DIR, "bibliohub_logo_horizontal.png")
LOGO_ICON_PNG = os.path.join(BRANDING_DIR, "bibliohub_logo_icon.png")


def _image_to_data_uri(image_path: str) -> str | None:
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_brand_header(container, *, sidebar: bool = False) -> None:
    icon_data_uri = _image_to_data_uri(LOGO_ICON_PNG)
    if not icon_data_uri:
        if sidebar:
            container.title("Biblio-HUB")
        else:
            container.title("Biblio-HUB")
        return

    if sidebar:
        container.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:12px; margin:0.1rem 0 0.45rem 0;">
              <img src="{icon_data_uri}" style="width:36px; height:36px; border-radius:8px; flex:0 0 auto;" />
              <div style="display:flex; flex-direction:column; justify-content:center;">
                <div style="font-size:1.48rem; font-weight:800; color:var(--text-color); line-height:1.02; letter-spacing:0.2px;">Biblio-HUB</div>
                <div style="font-size:0.86rem; color:var(--text-color); opacity:0.72; margin-top:3px; line-height:1.15;">Bibliometric intelligence workflow</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        container.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:16px; margin:0.15rem 0 0.15rem 0;">
              <img src="{icon_data_uri}" style="width:58px; height:58px; border-radius:14px; flex:0 0 auto;" />
              <div style="display:flex; flex-direction:column; justify-content:center;">
                <div style="font-size:2.62rem; font-weight:800; color:var(--text-color); line-height:0.98; letter-spacing:0.2px;">Biblio-HUB</div>
                <div style="font-size:1.04rem; color:var(--text-color); opacity:0.72; margin-top:5px; line-height:1.15;">Bibliometric intelligence workflow</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def inject_global_ui_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            font-size: 1rem;
        }
        .block-container {
            padding-top: 2.6rem;
            padding-bottom: 1.2rem;
        }
        h1 {
            font-size: 2.2rem !important;
        }
        h2 {
            font-size: 1.72rem !important;
        }
        h3 {
            font-size: 1.36rem !important;
        }
        h4 {
            font-size: 1.14rem !important;
        }
        p,
        li,
        div[data-testid="stMarkdownContainer"] p {
            font-size: 1rem;
            line-height: 1.58;
        }
        div[data-testid="stCaptionContainer"] p,
        [data-testid="stSidebar"] div[data-testid="stCaptionContainer"] p {
            font-size: 0.98rem !important;
            line-height: 1.5;
        }
        button[role="tab"] {
            padding-top: 0.72rem !important;
            padding-bottom: 0.72rem !important;
        }
        button[role="tab"] p {
            font-size: 1.08rem !important;
            font-weight: 600 !important;
            line-height: 1.35 !important;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label p {
            font-size: 1.04rem !important;
            font-weight: 600 !important;
            line-height: 1.4 !important;
        }
        [data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p,
        label[data-testid="stWidgetLabel"] p {
            font-size: 1rem !important;
            font-weight: 600 !important;
        }
        [data-testid="stSidebar"] .stExpander summary p,
        .stExpander summary p {
            font-size: 1rem !important;
            font-weight: 600 !important;
        }
        div.stButton > button,
        div[data-testid="stDownloadButton"] > button {
            font-size: 1rem !important;
            font-weight: 600 !important;
            min-height: 2.7rem !important;
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div {
            font-size: 0.98rem !important;
        }
        textarea,
        input {
            font-size: 0.98rem !important;
        }
        [data-testid="stMetricLabel"] p {
            font-size: 1rem !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.58rem !important;
        }
        [data-testid="stDataFrame"] div,
        [data-testid="stTable"] div {
            font-size: 0.96rem !important;
        }
        div[data-testid="stVerticalBlock"] > div:has(> div .stAppToolbar) {
            margin-top: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.set_page_config(
    page_title="Biblio-HUB",
    page_icon=LOGO_ICON_PNG if os.path.exists(LOGO_ICON_PNG) else "\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_global_ui_css()
render_brand_header(st.sidebar, sidebar=True)

if st.sidebar.button("\U0001F9F9 Reset Session & Cache", help="Clear all temporary data, session state, and cached calculations. Use this if your data is not updating correctly."):
    st.cache_data.clear()
    st.cache_resource.clear()
    # Keep only essential keys if any, otherwise clear all
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.sidebar.markdown("---")

uploaded_files = st.sidebar.file_uploader("Upload Data Files", type=['csv', 'txt'], accept_multiple_files=True)

if "active_local_data_paths_text" not in st.session_state:
    st.session_state["active_local_data_paths_text"] = ""
st.sidebar.caption("For files larger than the Web upload limit, enter absolute local paths below. One file path per line.")
st.sidebar.caption("To leave local path mode, manually clear the text box.")
st.sidebar.text_area(
    "Local Data File Paths",
    key="active_local_data_paths_text",
    help="Local file paths take priority over browser uploads as soon as they are entered.",
)

local_file_paths = _parse_local_file_paths_text(st.session_state.get("active_local_data_paths_text", ""))
using_local_paths = bool(local_file_paths)
if using_local_paths:
    st.sidebar.info("Local file path mode is active. Browser uploads are ignored until local paths are cleared.")

input_file_stats = _collect_input_file_stats(
    uploaded_files=None if using_local_paths else uploaded_files,
    local_file_paths=local_file_paths if using_local_paths else None,
)
file_names = [item["name"] for item in input_file_stats] if input_file_stats else None
largest_input_size_mb = max((item["size_bytes"] for item in input_file_stats), default=0) / (1024 * 1024)
input_source_signature = (
    tuple((item["source"], item["name"], item["path"], item["size_bytes"]) for item in input_file_stats),
)
initial_defer_heavy_analysis = largest_input_size_mb >= 200
VERY_LARGE_FILE_THRESHOLD_MB = 1000
VERY_LARGE_RECORD_THRESHOLD = 120000
DEFERRED_SUBANALYSIS_STATE_KEYS = [
    "subanalysis_temporal_burst_ready",
    "subanalysis_temporal_alluvial_ready",
    "subanalysis_temporal_bertopic_ready",
    "subanalysis_temporal_forward_ready",
    "subanalysis_structure_matrix_ready",
    "subanalysis_structure_thematic_ready",
    "subanalysis_structure_author_time_ready",
    "subanalysis_structure_three_field_ready",
]
SUBANALYSIS_ESTIMATED_WAIT = {
    "Burst Detection": "about 5-20 seconds",
    "Alluvial Topic Flow": "about 8-25 seconds",
    "BERTopic Workspace": "about 20-120 seconds",
    "Forward Signals": "about 10-45 seconds",
    "Co-occurrence Matrix": "about 5-20 seconds",
    "Thematic Map": "about 5-20 seconds",
    "Authors Over Time": "about 3-15 seconds",
    "Three-Field Plot": "about 5-25 seconds",
}

# Check if uploaded files changed since last run to prevent stale data
if input_source_signature != st.session_state.get("last_uploaded_filenames"):
    st.session_state["last_uploaded_filenames"] = input_source_signature
    st.session_state["heavy_analysis_deferred"] = bool(initial_defer_heavy_analysis)
    for deferred_key in DEFERRED_SUBANALYSIS_STATE_KEYS:
        st.session_state.pop(deferred_key, None)
    # If new files are uploaded, we should probably disable the "loaded project" mode
    if st.session_state.get("project_loaded_active"):
        st.session_state["project_loaded_active"] = False
        st.sidebar.warning("New files detected. Switched from 'Loaded Project' to 'Uploaded Files' mode.")

with st.sidebar.expander("Vocabulary Governance", expanded=False):
    keyword_source = st.selectbox(
        "Keyword Source",
        list(KEYWORD_SOURCE_OPTIONS.keys()),
        index=list(KEYWORD_SOURCE_OPTIONS.keys()).index(st.session_state.get("keyword_source", "DE+ID"))
        if st.session_state.get("keyword_source", "DE+ID") in KEYWORD_SOURCE_OPTIONS
        else 0,
        key="keyword_source",
        help="Controls which WoS keyword fields feed keyword-driven analyses: DE (Author Keywords), ID (Keywords Plus), or both.",
    )
    st.caption("Benchmark recommendation: use `DE+ID` unless you are explicitly aligning against a tool run that uses only one metadata field.")
    st.caption("Manage blocked keyword terms and alias-to-canonical normalization rules.")
    enable_optional_domain_plugin = st.checkbox(
        "Enable Optional Domain Keyword Plugin",
        value=bool(st.session_state.get("enable_optional_domain_plugin", False)),
        key="enable_optional_domain_plugin",
        help="Disabled by default for cross-domain portability. Enable only when you intentionally want extra title/abstract term matching from an optional domain dictionary.",
    )
    vocab_blocked_terms_text = st.text_area(
        "Blocked Terms",
        value=st.session_state.get("vocab_blocked_terms", ""),
        key="vocab_blocked_terms",
        help="One term per line or separated by semicolons. Matching is applied after keyword normalization.",
    )
    vocab_replacement_map_text = st.text_area(
        "Alias => Canonical",
        value=st.session_state.get("vocab_replacement_map", ""),
        key="vocab_replacement_map",
        help="Example: covid 19 => COVID-19",
    )

using_loaded_project = bool(st.session_state.get("project_loaded_active"))
if using_loaded_project:
    df = st.session_state.get("loaded_df")
    dedup_report = st.session_state.get("loaded_dedup_report")
    keywords_list = st.session_state.get("loaded_keywords_list")
    keyword_freq = st.session_state.get("loaded_keyword_freq")
    cooccurrence = st.session_state.get("loaded_cooccurrence")
    journal_year = st.session_state.get("loaded_journal_year")
    heavy_analysis_deferred = bool(st.session_state.get("heavy_analysis_deferred", False))
else:
    try:
        load_status_box = st.empty()
        load_progress_bar = st.progress(0, text="Waiting to start data loading...")

        def _update_load_progress(progress_value, status_text):
            load_status_box.info(status_text)
            load_progress_bar.progress(int(max(0.0, min(1.0, float(progress_value))) * 100), text=status_text)

        _update_load_progress(0.02, "Starting data loading...")
        df, dedup_report, keywords_list, keyword_freq, cooccurrence, journal_year = process_global_data(
            None if using_local_paths else uploaded_files,
            _file_names=file_names,
            blocked_terms_text=vocab_blocked_terms_text,
            replacement_map_text=vocab_replacement_map_text,
            keyword_source=keyword_source,
            enable_optional_domain_plugin=enable_optional_domain_plugin,
            _dir_fingerprint=_get_data_dir_fingerprint(),
            local_file_paths=local_file_paths if using_local_paths else None,
            _local_path_fingerprint=_get_local_paths_fingerprint(local_file_paths if using_local_paths else None),
            _progress_callback=_update_load_progress,
            defer_heavy_analysis=initial_defer_heavy_analysis,
        )
        load_progress_bar.empty()
        load_status_box.empty()
        heavy_analysis_deferred = bool(initial_defer_heavy_analysis)
        st.session_state["heavy_analysis_deferred"] = heavy_analysis_deferred
    except Exception as exc:
        log_exception("process_global_data", exc)
        st.error(f"Failed to process dataset: {exc}")
        st.stop()

if df is None:
    st.error("No data file found. Please upload one or more CSV or WoS Tab-delimited TXT files.")
    st.stop()

st.sidebar.markdown(f"**Records loaded:** {len(df)}")
if dedup_report and dedup_report['removed'] > 0:
    st.sidebar.info(f"**Deduplication:** Merged {dedup_report['removed']} duplicate records.")

large_input_detected = largest_input_size_mb >= 200
large_record_detected = len(df) >= 50000
very_large_input_detected = largest_input_size_mb >= VERY_LARGE_FILE_THRESHOLD_MB
very_large_record_detected = len(df) >= VERY_LARGE_RECORD_THRESHOLD
subanalysis_lazy_enabled = bool(very_large_input_detected or very_large_record_detected)
auto_lightweight_signature = (input_source_signature, int(len(df)))
if auto_lightweight_signature != st.session_state.get("last_auto_lightweight_signature"):
    auto_lightweight_enabled = bool(large_input_detected or large_record_detected)
    st.session_state["global_lightweight_mode"] = auto_lightweight_enabled
    if auto_lightweight_enabled:
        st.session_state["export_lightweight_mode"] = True
        st.session_state["bt_selected_profile_name"] = "quick_preview"
    st.session_state["last_auto_lightweight_signature"] = auto_lightweight_signature

if large_input_detected or large_record_detected:
    warning_parts = []
    if large_input_detected:
        warning_parts.append(f"largest input file: {largest_input_size_mb:.1f} MB")
    if large_record_detected:
        warning_parts.append(f"records loaded: {len(df):,}")
    st.sidebar.warning(
        "Large-file lightweight mode is active. "
        + "; ".join(warning_parts)
        + ". Innovation/export heavy tasks default to lightweight settings, and BERTopic stays on the fast profile by default."
    )
    if heavy_analysis_deferred:
        st.sidebar.info(
            "Keyword, co-occurrence, and journal-year structures are deferred for faster entry. "
            "They will be computed only after you explicitly load a heavy analysis module."
        )
    if subanalysis_lazy_enabled:
        subanalysis_reason_parts = []
        if very_large_input_detected:
            subanalysis_reason_parts.append(f"largest input file: {largest_input_size_mb:.1f} MB")
        if very_large_record_detected:
            subanalysis_reason_parts.append(f"records loaded: {len(df):,}")
        st.sidebar.info(
            "Second-level lazy loading is active for very heavy temporal and structure sub-panels. "
            + "; ".join(subanalysis_reason_parts)
            + ". Those subsections will render only after you explicitly load them."
        )
        loaded_subanalysis_count = sum(1 for key in DEFERRED_SUBANALYSIS_STATE_KEYS if st.session_state.get(key, False))
        st.sidebar.caption(f"Loaded heavy sub-panels in this session: {loaded_subanalysis_count}/{len(DEFERRED_SUBANALYSIS_STATE_KEYS)}")
        st.sidebar.caption("Reset only clears loaded states. Cached computations may still be reused where available.")
        if st.sidebar.button("Reset Heavy Sub-panels", key="reset_heavy_subpanels", use_container_width=True):
            for deferred_key in DEFERRED_SUBANALYSIS_STATE_KEYS:
                st.session_state.pop(deferred_key, None)
            st.rerun()


def ensure_heavy_analysis_ready():
    global keywords_list, keyword_freq, cooccurrence, journal_year, heavy_analysis_deferred
    if not heavy_analysis_deferred:
        return
    with st.spinner("Computing deferred keyword and network structures for this module..."):
        (
            keywords_list,
            keyword_freq,
            cooccurrence,
            journal_year,
        ) = get_cached_heavy_analysis_artifacts(
            df,
            blocked_terms_text=vocab_blocked_terms_text or "",
            replacement_map_text=vocab_replacement_map_text or "",
            keyword_source=keyword_source,
            enable_optional_domain_plugin=enable_optional_domain_plugin,
        )
    heavy_analysis_deferred = False
    st.session_state["heavy_analysis_deferred"] = False


def request_heavy_analysis_unlock(module_title, button_key):
    if not heavy_analysis_deferred:
        return True
    st.warning(
        f"{module_title} uses deferred keyword and network structures. "
        "To keep very large datasets responsive, this page will not auto-compute on open."
    )
    st.info(
        "When you are ready, click the button below to build keyword extraction, co-occurrence, "
        "and journal-year support structures for the current dataset."
    )
    if st.button(f"Load {module_title}", key=button_key, use_container_width=True):
        ensure_heavy_analysis_ready()
        st.rerun()
    return False


def request_subanalysis_unlock(section_title, state_key, button_key):
    if not subanalysis_lazy_enabled:
        return True
    if st.session_state.get(state_key, False):
        return True
    estimated_wait = SUBANALYSIS_ESTIMATED_WAIT.get(section_title, "about 5-30 seconds")
    trigger_reason = []
    if very_large_input_detected:
        trigger_reason.append(f"largest file {largest_input_size_mb:.1f} MB")
    if very_large_record_detected:
        trigger_reason.append(f"{len(df):,} records")
    trigger_reason_text = "; ".join(trigger_reason) if trigger_reason else "very large dataset"
    st.warning(
        f"{section_title} is on-demand for this dataset to keep the page responsive."
    )
    st.caption(
        f"Trigger reason: {trigger_reason_text}. Estimated wait: {estimated_wait}. "
        "Once loaded, the result remains available in the current dataset session."
    )
    if st.button(f"Load {section_title} Now", key=button_key, use_container_width=True):
        st.session_state[state_key] = True
        st.rerun()
    return False


def render_subanalysis_status(section_title, state_key):
    if not subanalysis_lazy_enabled:
        return
    if st.session_state.get(state_key, False):
        st.caption(
            f"Status: Loaded. {section_title} is already available for the current dataset session."
        )
    else:
        estimated_wait = SUBANALYSIS_ESTIMATED_WAIT.get(section_title, "about 5-30 seconds")
        st.caption(
            f"Status: Not loaded. This subsection stays on-demand for very large datasets. "
            f"Estimated wait after loading: {estimated_wait}."
        )

has_authors = "Authors" in df.columns
has_times_cited = "Times_Cited" in df.columns
has_affiliations = "Affiliations" in df.columns
has_wos_cat = "WoS_Categories" in df.columns
has_research_areas = "Research_Areas" in df.columns
has_doctype = "DocType" in df.columns
has_funding = "Funding" in df.columns
has_cited_refs = "Cited_References" in df.columns
has_publisher = "Publisher" in df.columns
has_language = "Language" in df.columns

available_pages = ["Dataset Overview", "Relational Network Analysis", "Temporal Trend & Topic Evolution", "Bibliometric Structure & Performance"]
if has_times_cited or has_doctype or has_wos_cat or has_funding or has_cited_refs:
    available_pages.append("Citation, Category & Source Analysis")
available_pages.append("Forward Signals")
if has_cited_refs:
    available_pages.append("Innovation Analysis")
available_pages.append("Export Center")

query_page = None
try:
    query_page = st.query_params.get("module")
    if isinstance(query_page, list):
        query_page = query_page[0] if query_page else None
except Exception:
    query_page = None

current_page = st.session_state.get("active_analysis_module")
if current_page not in available_pages:
    current_page = None
if current_page is None:
    current_page = query_page if query_page in available_pages else available_pages[0]
elif (
    query_page in available_pages
    and query_page != current_page
    and query_page != st.session_state.get("_last_synced_module_query")
):
    current_page = query_page
st.session_state["active_analysis_module"] = current_page
page = st.sidebar.radio(
    "Analysis Module",
    available_pages,
    index=available_pages.index(st.session_state.get("active_analysis_module", current_page)),
    key="active_analysis_module",
)
try:
    if st.query_params.get("module") != page:
        st.query_params["module"] = page
    st.session_state["_last_synced_module_query"] = page
except Exception:
    pass

st.sidebar.markdown("---")
with st.sidebar.expander("Research & Benchmarking", expanded=False):
    show_perf = st.checkbox(
        "Show Performance Metrics",
        value=False,
        key="show_perf_metrics",
        help="Display and record rendering time for each chart. Useful for 'Performance Evaluation' sections in papers."
    )
    if show_perf:
        st.info("Performance timing is active. Navigate through tabs to collect data.")
        if "perf_logs" in st.session_state and st.session_state.perf_logs:
            perf_df = pd.DataFrame(st.session_state.perf_logs)
            csv_data = perf_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Performance Log (CSV)",
                data=csv_data,
                file_name="biblio_hub_performance_log.csv",
                mime="text/csv",
                help="Download recorded timings for statistical analysis in your paper."
            )
            if st.button("Clear Logs"):
                st.session_state.perf_logs = []
                st.rerun()

render_project_management_sidebar(
    st=st,
    page=page,
    df=df,
    keywords_list=keywords_list,
    keyword_freq=keyword_freq,
    cooccurrence=cooccurrence,
    journal_year=journal_year,
    dedup_report=dedup_report,
    log_exception_callback=log_exception,
)

if 'Year' not in df.columns or df['Year'].isna().all():
    st.warning("No valid 'Year' column found. Year-based analyses will not be available.")
if 'Journal' not in df.columns or df['Journal'].isna().all():
    st.warning("No valid 'Journal' column found. Journal-based analyses will not be available.")

if page == "Dataset Overview":
    render_brand_header(st, sidebar=False)
    st.caption("This overview page follows standard descriptive bibliometric dashboards and serves as the baseline entry point for the full workflow.")

    if dedup_report and dedup_report['removed'] > 0:
        st.success(f"**Data Fusion & Deduplication:** Successfully processed {dedup_report['original']} records, removed {dedup_report['removed']} duplicates, yielding {dedup_report['final']} unique records. Duplicates were identified via exact DOI match and fuzzy matching on Title + Year.")

    st.markdown("### Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        year_min = safe_year(df['Year'].dropna().apply(safe_year).min()) if 'Year' in df.columns else "N/A"
        year_max = safe_year(df['Year'].dropna().apply(safe_year).max()) if 'Year' in df.columns else "N/A"
        st.metric("Year Range", f"{year_min}-{year_max}")
    with col3:
        st.metric("Unique Journals", df['Journal'].nunique() if 'Journal' in df.columns else "N/A")
    with col4:
        st.metric("Unique Keywords", "Deferred" if heavy_analysis_deferred else len(keyword_freq))
    st.markdown("---")
    if heavy_analysis_deferred:
        st.info(
            "Large-file quick preview is active on this overview page. "
            "Keyword summaries and network-derived views are deferred until you open a heavy analysis module."
        )
    st.markdown("### Year Distribution")
    st.caption("This panel reports annual publication counts as a basic descriptive time-series summary of dataset coverage.")
    if 'Year' in df.columns:
        df_valid = clean_year_column(df)
        year_counts = df_valid['Year'].value_counts().sort_index()
        year_style = render_plot_style_controls(
            "overview_year_distribution",
            default_primary=SCIENTIFIC_COLORWAY[3],
            default_secondary="#7F8C8D",
            default_height=460,
            allow_color_controls=False,
            preserve_original_colors=True,
        )
        year_df = pd.DataFrame({"Year": year_counts.index, "Publications": year_counts.values})
        stacked_year_df = pd.DataFrame()
        if "DocType" in df_valid.columns and df_valid["DocType"].notna().any():
            doc_type_rows = []
            for _, row in df_valid[["Year", "DocType"]].dropna(subset=["Year"]).iterrows():
                raw_types = [
                    term.strip()
                    for term in str(row.get("DocType", "")).split(";")
                    if term and term.strip() and str(term).strip().lower() != "nan"
                ]
                if not raw_types:
                    raw_types = ["Unspecified"]
                for doc_type in raw_types:
                    doc_type_rows.append({"Year": int(row["Year"]), "DocType": doc_type})
            if doc_type_rows:
                doc_type_df = pd.DataFrame(doc_type_rows)
                top_doc_types = (
                    doc_type_df["DocType"].value_counts().head(5).index.tolist()
                )
                doc_type_df["DocType"] = doc_type_df["DocType"].where(
                    doc_type_df["DocType"].isin(top_doc_types),
                    "Others",
                )
                stacked_year_df = (
                    doc_type_df.groupby(["Year", "DocType"])
                    .size()
                    .reset_index(name="Publications")
                    .sort_values(["Year", "DocType"])
                )
        fig = go.Figure()
        if not stacked_year_df.empty:
            doc_type_order = [
                name
                for name in stacked_year_df["DocType"].drop_duplicates().tolist()
                if name != "Others"
            ]
            if "Others" in stacked_year_df["DocType"].values:
                doc_type_order.append("Others")
            for idx, doc_type in enumerate(doc_type_order):
                subset = stacked_year_df[stacked_year_df["DocType"] == doc_type]
                fig.add_bar(
                    x=subset["Year"],
                    y=subset["Publications"],
                    name=doc_type,
                    marker_color=SCIENTIFIC_COLORWAY[idx % len(SCIENTIFIC_COLORWAY)],
                    opacity=0.86,
                    width=0.72,
                )
        else:
            fig.add_bar(
                x=year_df["Year"],
                y=year_df["Publications"],
                name="Annual Publications",
                marker_color=SCIENTIFIC_COLORWAY[3],
                opacity=0.82,
                width=0.72,
            )
        fig.add_scatter(
            x=year_df["Year"],
            y=year_df["Publications"],
            mode="lines",
            name="Overall Trend",
            line=dict(color="rgba(79, 109, 138, 0.48)", width=1.5, shape="spline", smoothing=1.05),
        )
        fig.update_layout(
            title="Publications by Year",
            xaxis_title="Year",
            yaxis_title="Number of Publications",
            barmode="stack",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig.update_xaxes(type="linear", dtick=1)
        fig.update_yaxes(rangemode="tozero")
        fig = apply_publication_style_with_overrides(fig, year_style)
        render_plotly_chart(fig, width="stretch")
        download_plotly_button(fig, "publications_by_year.png", "Download Publications by Year")
    st.markdown("### Top Journals")
    if 'Journal' in df.columns:
        journal_counts = df['Journal'].value_counts()
        journal_display_limit = integer_control(
            st,
            "Top Journals to Display",
            1,
            min(30, len(journal_counts)),
            key="overview_top_journals_n",
            input_max=max(1, len(journal_counts)),
            slider_soft_cap=min(max(50, min(500, len(journal_counts))), len(journal_counts)),
        )
        journal_df = journal_counts.head(journal_display_limit).reset_index()
        journal_df.columns = ['Journal', 'Publications']
        with st.expander("View Top Journals Table", expanded=False):
            st.dataframe(journal_df, width="stretch", hide_index=True)
            st.download_button(
                "Download Top Journals Table (CSV)",
                journal_df.to_csv(index=False).encode("utf-8-sig"),
                "Biblio-HUB_top_journals.csv",
                "text/csv",
                key="overview_top_journals_csv",
                use_container_width=True,
            )
        journal_style = render_plot_style_controls(
            "overview_top_journals",
            default_primary=SCIENTIFIC_COLORWAY[0],
            default_height=500,
        )
        journal_plot_df = journal_df.sort_values("Publications", ascending=False).copy()
        fig = px.bar(
            journal_plot_df,
            x="Publications",
            y="Journal",
            orientation="h",
            title=f"Top {journal_display_limit} Journals"
        )
        fig.update_traces(marker_color=SCIENTIFIC_COLORWAY[0])
        fig = apply_publication_style_with_overrides(fig, journal_style)
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=list(reversed(journal_plot_df["Journal"].tolist())),
        )
        render_plotly_chart(fig, width="stretch")
        download_plotly_button(fig, "top_journals.png", "Download Top Journals")
    top_country_chart_df = build_top_country_table(df, top_n=20)
    if not top_country_chart_df.empty:
        st.markdown("### Country Publications Distribution")
        st.caption("This front-end panel matches the exportable country-publication bar chart and can be used directly for FIG2C preview.")
        country_style = render_plot_style_controls(
            "overview_country_publications",
            default_primary=SCIENTIFIC_COLORWAY[2],
            default_height=560,
        )
        country_plot_df = top_country_chart_df.sort_values("Papers", ascending=False).copy()
        fig_country = px.bar(
            country_plot_df,
            x="Papers",
            y="Country",
            orientation="h",
            title="Top Countries by Publications",
        )
        fig_country.update_traces(marker_color=SCIENTIFIC_COLORWAY[2])
        fig_country = apply_publication_style_with_overrides(fig_country, country_style)
        fig_country.update_yaxes(
            categoryorder="array",
            categoryarray=list(reversed(country_plot_df["Country"].tolist())),
        )
        render_plotly_chart(fig_country, width="stretch")
        download_plotly_button(fig_country, "country_publications.png", "Download Country Chart")
        with st.expander("Top Countries Table", expanded=False):
            st.dataframe(top_country_chart_df, width="stretch", hide_index=True)
    st.markdown("### Keyword Word Cloud")
    st.caption("This word cloud is a qualitative prominence display based on keyword frequency and should be interpreted as a visual summary rather than a formal inferential result.")
    if heavy_analysis_deferred:
        st.info("Keyword word cloud is deferred in quick preview mode. Open a heavy analysis module to compute keyword structures.")
    elif keyword_freq:
        wordcloud_col1, wordcloud_col2 = st.columns(2)
        with wordcloud_col1:
            max_words_wc = st.slider("Word Cloud Max Words", 20, 120, 80, key="overview_wordcloud_max_words")
        with wordcloud_col2:
            min_font_wc = st.slider("Word Cloud Min Font Size", 6, 24, 10, key="overview_wordcloud_min_font")
        wc = WordCloud(width=1200, height=400, background_color='white',
                       max_words=max_words_wc, colormap=MORANDI_WORDCLOUD,
                       prefer_horizontal=0.7, min_font_size=min_font_wc)
        wc.generate_from_frequencies(keyword_freq)
        wordcloud_display_col, top10_table_col = st.columns([5.2, 0.8])
        with wordcloud_display_col:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            download_matplotlib_button(fig, "keyword_wordcloud.png", "Download Word Cloud")
        with top10_table_col:
            st.markdown("#### Top 10 Keywords")
            top10_kw_df = pd.DataFrame(
                keyword_freq.most_common(10),
                columns=["Keyword", "Freq."],
            )
            st.dataframe(
                top10_kw_df,
                width=230,
                height=388,
                hide_index=True,
                column_config={
                    "Keyword": st.column_config.TextColumn("Keyword", width="medium"),
                    "Freq.": st.column_config.NumberColumn("Freq.", width="small", format="%d"),
                },
            )
    st.markdown("### Top Keywords")
    st.caption("This keyword frequency view follows standard metadata-based keyword summaries used in mainstream bibliometric software.")
    if heavy_analysis_deferred:
        st.info("Top keyword ranking is deferred in quick preview mode.")
    elif keyword_freq:
        keyword_display_limit = integer_control(
            st,
            "Top Keywords to Display",
            1,
            min(30, len(keyword_freq)),
            key="overview_top_keywords_n",
            input_max=max(1, len(keyword_freq)),
            slider_soft_cap=min(max(50, min(500, len(keyword_freq))), len(keyword_freq)),
        )
        top_kw = keyword_freq.most_common(keyword_display_limit)
        kw_df = pd.DataFrame(top_kw, columns=['Keyword', 'Frequency'])
        keyword_style = render_plot_style_controls(
            "overview_top_keywords",
            default_primary=SCIENTIFIC_COLORWAY[5],
            default_height=500,
        )
        kw_plot_df = kw_df.sort_values("Frequency", ascending=False).copy()
        fig = px.bar(
            kw_plot_df,
            x='Frequency',
            y='Keyword',
            orientation='h',
            title=f'Top {keyword_display_limit} Keywords'
        )
        fig.update_traces(marker_color=SCIENTIFIC_COLORWAY[5])
        fig = apply_publication_style_with_overrides(fig, keyword_style)
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=list(reversed(kw_plot_df["Keyword"].tolist())),
        )
        render_plotly_chart(fig, width="stretch")
        download_plotly_button(fig, "top_keywords.png", "Download Top Keywords")
        with st.expander("Top Keywords Table", expanded=False):
            st.dataframe(kw_df, width="stretch", hide_index=True)
            st.download_button(
                "Download Top Keywords Table (CSV)",
                kw_df.to_csv(index=False).encode("utf-8-sig"),
                "Biblio-HUB_top_keywords.csv",
                "text/csv",
                key="overview_top_keywords_csv",
                use_container_width=True,
            )

elif page == "Relational Network Analysis":
    if request_heavy_analysis_unlock("Relational Network Analysis", "load_relational_heavy"):
        t_start = time.time()
        render_relational_network_page(
            st=st,
            df=df,
            keywords_list=keywords_list,
            keyword_freq=keyword_freq,
            cooccurrence=cooccurrence,
            has_authors=has_authors,
            has_affiliations=has_affiliations,
            has_cited_refs=has_cited_refs,
            log_exception=log_exception,
            network_cluster_palette=NETWORK_CLUSTER_PALETTE,
        )
        display_performance_metrics(t_start, "Network analysis")

elif page == "Temporal Trend & Topic Evolution":
    st.title("Temporal Trend & Topic Evolution")
    if request_heavy_analysis_unlock("Temporal Trend & Topic Evolution", "load_temporal_heavy"):
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Temporal Keyword Evolution",
                "Burst Detection",
                "Alluvial Topic Flow",
                "Semantic Topic Modeling (BERTopic)",
            ]
        )
        with tab1:
            st.markdown("### Temporal Keyword Evolution")
            top_n_tl = integer_control(
                st,
                "Number of Keywords",
                5,
                15,
                key="cs_topn",
                input_max=max(5, len(keyword_freq)),
                slider_soft_cap=30,
            )
            timeline_style = render_plot_style_controls(
                "temporal_keyword_timeline",
                default_primary=SCIENTIFIC_COLORWAY[1],
                default_secondary=SCIENTIFIC_COLORWAY[5],
                default_height=620,
                show_legend_default=True,
                allow_color_controls=False,
                preserve_original_colors=True,
            )
            t_start = time.time()
            fig_tl = render_citespace_timeline(df, keywords_list, keyword_freq, top_n_tl)
            if fig_tl:
                fig_tl = apply_publication_style_with_overrides(fig_tl, timeline_style)
                render_plotly_chart(fig_tl, width="stretch")
                display_performance_metrics(t_start, "Keyword timeline")
                download_plotly_button(fig_tl, "temporal_keyword_evolution.png", "Download Temporal Keyword Evolution")
            else:
                st.warning("Not enough data for timeline analysis.")

            st.markdown("### Selected Keyword Share by Year")
            st.caption(
                "Enter comma-separated keywords such as `keyword1, keyword2, keyword3`. "
                "Within each year, only the selected keywords are normalized to 100%, so unselected keywords are excluded from the denominator."
            )
            selected_keyword_text = st.text_input(
                "Selected Keywords",
                key="selected_keyword_share_input",
                placeholder="keyword1, keyword2, keyword3",
            )
            if selected_keyword_text.strip():
                selected_keywords = parse_selected_keywords(selected_keyword_text)
                if not selected_keywords:
                    st.warning("No valid keywords were detected from the input. Please separate keywords with commas.")
                else:
                    matched_keywords = [kw for kw in selected_keywords if kw in keyword_freq]
                    missing_keywords = [kw for kw in selected_keywords if kw not in keyword_freq]
                    if matched_keywords:
                        st.caption("Matched keywords: " + ", ".join(matched_keywords))
                    if missing_keywords:
                        st.info("Not found in the current dataset: " + ", ".join(missing_keywords))

                    share_df = build_selected_keyword_share_table(df, keywords_list, matched_keywords)
                    if not share_df.empty:
                        share_style = render_plot_style_controls(
                            "temporal_selected_keyword_share",
                            default_primary=SCIENTIFIC_COLORWAY[2],
                            default_secondary=SCIENTIFIC_COLORWAY[6],
                            default_height=580,
                            show_legend_default=True,
                            allow_color_controls=False,
                            preserve_original_colors=True,
                        )
                        fig_share = render_selected_keyword_share_figure(share_df, matched_keywords)
                        if fig_share:
                            fig_share = apply_publication_style_with_overrides(fig_share, share_style)
                            render_plotly_chart(fig_share, width="stretch")
                            download_plotly_button(fig_share, "selected_keyword_share_by_year.png", "Download Selected Keyword Share")
                        with st.expander("Selected Keyword Share Table", expanded=False):
                            st.dataframe(share_df, width="stretch", hide_index=True)
                    elif matched_keywords:
                        st.info("These keywords were found, but they do not produce enough yearly observations for a stacked-share view.")
                    else:
                        st.warning("None of the input keywords matched the extracted keyword set in this dataset.")
            else:
                st.info("Enter a comma-separated keyword set such as `keyword1, keyword2, keyword3` to inspect how the selected terms divide yearly attention.")

            st.markdown("### Fastest-Growing Keywords")
            st.caption(
                "A ranking chart is clearer than a crowded multi-line plot for identifying leaders. "
                "The lollipop chart ranks keywords by latest-year growth rate, and the line chart below shows their yearly trajectories."
            )
            growth_top_n = integer_control(
                st,
                "Top Growth Keywords",
                5,
                10,
                key="keyword_growth_topn",
                input_max=max(5, min(len(keyword_freq), 25)),
                slider_soft_cap=20,
            )
            growth_df, growth_yearly_df = build_keyword_growth_table(
                df,
                keywords_list,
                keyword_freq,
                candidate_top_n=max(30, growth_top_n * 4),
                min_total_occurrences=3,
                min_latest_count=2,
            )
            if not growth_df.empty:
                growth_rank_style = render_plot_style_controls(
                    "temporal_keyword_growth_rank",
                    default_primary=SCIENTIFIC_COLORWAY[0],
                    default_height=560,
                )
                fig_growth_rank = render_keyword_growth_leader_figure(growth_df, top_n=growth_top_n)
                if fig_growth_rank:
                    fig_growth_rank = apply_publication_style_with_overrides(fig_growth_rank, growth_rank_style)
                    render_plotly_chart(fig_growth_rank, width="stretch")
                    download_plotly_button(fig_growth_rank, "fastest_growing_keywords.png", "Download Growth Ranking")

                growth_trend_style = render_plot_style_controls(
                    "temporal_keyword_growth_trend",
                    default_primary=SCIENTIFIC_COLORWAY[1],
                    default_secondary=SCIENTIFIC_COLORWAY[5],
                    default_height=560,
                    show_legend_default=True,
                    allow_color_controls=False,
                    preserve_original_colors=True,
                )
                fig_growth_trend = render_keyword_growth_trend_figure(growth_yearly_df, growth_df, top_n=min(6, growth_top_n))
                if fig_growth_trend:
                    fig_growth_trend = apply_publication_style_with_overrides(fig_growth_trend, growth_trend_style)
                    render_plotly_chart(fig_growth_trend, width="stretch")
                    download_plotly_button(fig_growth_trend, "growth_leader_keyword_trajectories.png", "Download Growth Trajectories")

                with st.expander("Fastest-Growing Keywords Table", expanded=False):
                    st.dataframe(growth_df.head(growth_top_n), width="stretch", hide_index=True)
            else:
                st.info("Not enough yearly keyword continuity is available to rank growth leaders.")

        with tab2:
            st.markdown("### Burst Detection")
            st.markdown("Keywords with sudden frequency increases (potential emerging trends). Uses Kleinberg's burst detection algorithm. *Inspired by CiteSpace's burst detection workflow.*")
            render_subanalysis_status("Burst Detection", "subanalysis_temporal_burst_ready")
            if request_subanalysis_unlock(
                "Burst Detection",
                "subanalysis_temporal_burst_ready",
                "load_temporal_burst",
            ):
                burst_year_df = clean_year_column(df)
                burst_year_min = int(burst_year_df["Year"].min()) if not burst_year_df.empty else safe_year(df["Year"].min())
                burst_year_max = int(burst_year_df["Year"].max()) if not burst_year_df.empty else safe_year(df["Year"].max())
                burst_control_col1, burst_control_col2 = st.columns([1.05, 1.35])
                with burst_control_col1:
                    top_n_burst = integer_control(
                        st,
                        "Keywords to Analyze",
                        10,
                        20,
                        key="burst_topn",
                        input_max=max(10, len(keyword_freq)),
                        slider_soft_cap=40,
                    )
                with burst_control_col2:
                    burst_year_range = st.slider(
                        "Burst Detection Year Range",
                        min_value=burst_year_min,
                        max_value=burst_year_max,
                        value=(burst_year_min, burst_year_max),
                        key="burst_year_range",
                    )
                burst_style = render_plot_style_controls(
                    "temporal_burst_detection",
                    default_primary=SCIENTIFIC_COLORWAY[4],
                    default_secondary=SCIENTIFIC_COLORWAY[11],
                    default_height=620,
                    show_legend_default=True,
                    allow_color_controls=False,
                    preserve_original_colors=True,
                )
                t_start = time.time()
                keyword_burst_df = build_keyword_burst_table(
                    df,
                    keywords_list,
                    keyword_freq,
                    top_n=top_n_burst,
                    start_year=burst_year_range[0],
                    end_year=burst_year_range[1],
                )
                if not keyword_burst_df.empty:
                    burst_table_col, burst_chart_col = st.columns([0.92, 3.08])
                    with burst_table_col:
                        burst_table_df = keyword_burst_df.copy()
                        burst_table_df = burst_table_df.rename(
                            columns={
                                "Burst Strength": "Raw Strength",
                                "Adjusted Burst Score": "Adjusted Score",
                            }
                        )[["Keyword", "Adjusted Score", "Raw Strength", "Start", "End", "Duration"]]
                        st.markdown("#### Top Burst Terms")
                        st.dataframe(
                            burst_table_df,
                            width="stretch",
                            height=min(500, 72 + len(burst_table_df) * 34),
                            hide_index=True,
                            column_config={
                                "Keyword": st.column_config.TextColumn("Keyword", width="small"),
                                "Adjusted Score": st.column_config.NumberColumn("Adjusted Score", width="small", format="%.2f"),
                                "Raw Strength": st.column_config.NumberColumn("Raw Strength", width="small", format="%.2f"),
                                "Start": st.column_config.NumberColumn("Start", width="small", format="%d"),
                                "End": st.column_config.NumberColumn("End", width="small", format="%d"),
                                "Duration": st.column_config.NumberColumn("Duration", width="small", format="%d"),
                            },
                        )
                        export_burst_df = keyword_burst_df.copy()
                        export_burst_df.insert(0, "Analysis Start Year", int(burst_year_range[0]))
                        export_burst_df.insert(1, "Analysis End Year", int(burst_year_range[1]))
                        st.download_button(
                            "Download Burst Table (CSV)",
                            export_burst_df.to_csv(index=False).encode("utf-8-sig"),
                            f"Biblio-HUB_keyword_burst_{burst_year_range[0]}_{burst_year_range[1]}.csv",
                            "text/csv",
                            key="download_keyword_burst_csv",
                            use_container_width=True,
                        )
                    with burst_chart_col:
                        fig_burst = render_burst_detection(
                            df,
                            keywords_list,
                            keyword_freq,
                            top_n_burst,
                            start_year=burst_year_range[0],
                            end_year=burst_year_range[1],
                        )
                        if fig_burst:
                            fig_burst = apply_publication_style_with_overrides(fig_burst, burst_style)
                            render_plotly_chart(fig_burst, width="stretch")
                            download_plotly_button(fig_burst, "burst_detection.png", "Download Burst Detection")
                    display_performance_metrics(t_start, "Burst detection")
                else:
                    st.info("Not enough temporal data for burst detection.")

        with tab3:
            st.markdown("### Alluvial Topic Flow")
            render_subanalysis_status("Alluvial Topic Flow", "subanalysis_temporal_alluvial_ready")
            if request_subanalysis_unlock(
                "Alluvial Topic Flow",
                "subanalysis_temporal_alluvial_ready",
                "load_temporal_alluvial",
            ):
                col_flow_1, col_flow_2 = st.columns(2)
                year_slice_max = max(2, clean_year_column(df)["Year"].nunique())
                with col_flow_1:
                    flow_slices = integer_control(
                        st,
                        "Time Slices",
                        2,
                        4,
                        key="alluvial_slices",
                        input_max=year_slice_max,
                        slider_soft_cap=8,
                    )
                with col_flow_2:
                    flow_topics = integer_control(
                        st,
                        "Max Topics per Slice",
                        3,
                        5,
                        key="alluvial_topics",
                        input_max=max(3, min(len(keyword_freq), 30)),
                        slider_soft_cap=10,
                    )
                flow_style = render_plot_style_controls(
                    "temporal_alluvial_flow",
                    default_primary=SCIENTIFIC_COLORWAY[2],
                    default_secondary=SCIENTIFIC_COLORWAY[6],
                    default_height=680,
                    show_legend_default=True,
                    allow_color_controls=False,
                    preserve_original_colors=True,
                )

                t_start = time.time()
                fig_flow = render_alluvial_topic_flow(
                    df,
                    keywords_list,
                    keyword_freq,
                    slice_count=flow_slices,
                    top_n_keywords=30,
                    max_topics_per_slice=flow_topics,
                    keywords_per_topic=3,
                )
                if fig_flow:
                    fig_flow = apply_publication_style_with_overrides(fig_flow, flow_style)
                    render_plotly_chart(fig_flow, width="stretch")
                    display_performance_metrics(t_start, "Alluvial flow")
                    download_plotly_button(fig_flow, "alluvial_topic_flow.png", "Download Alluvial Flow")
                else:
                    st.info("Not enough temporal keyword continuity to build the alluvial topic flow.")

        with tab4:
            st.markdown("### Semantic Topic Modeling with BERTopic")
            render_subanalysis_status("BERTopic Workspace", "subanalysis_temporal_bertopic_ready")
            if request_subanalysis_unlock(
                "BERTopic Workspace",
                "subanalysis_temporal_bertopic_ready",
                "load_temporal_bertopic",
            ):
                st.caption("Uses Sentence-BERT embeddings + UMAP dimensionality reduction + HDBSCAN clustering to discover semantic topics from Title & Abstract. Unlike keyword co-occurrence, this method captures semantic similarity (e.g., 'COVID-19' and 'SARS-CoV-2' are grouped together).")
                min_docs_bt = st.slider("Minimum Documents per Topic", 2, 10, 3, key="bt_min_docs")
                quick_profile = get_bertopic_profile_settings(len(df), "quick_preview")
                full_profile = get_bertopic_profile_settings(len(df), "full_analysis")
                st.caption(quick_profile["summary"])
                st.caption(full_profile["summary"])
                if len(df) > 3000:
                    st.warning("BERTopic remains computationally expensive on large corpora. Run the quick preview first, confirm the topic direction, and then launch the full analysis if needed.")

                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    run_bertopic_requested = st.button("Run Quick Preview", key="bt_run_quick")
                with action_col2:
                    run_full_bertopic_requested = st.button("Run Full Analysis", key="bt_run_full")

                requested_profile_name = None
                if run_bertopic_requested:
                    requested_profile_name = "quick_preview"
                elif run_full_bertopic_requested:
                    requested_profile_name = "full_analysis"

                cached_bertopic = st.session_state.get("bt_cached_result")
                if "bt_selected_profile_name" not in st.session_state:
                    st.session_state["bt_selected_profile_name"] = "quick_preview" if len(df) > 2000 else "full_analysis"
                active_profile_name = requested_profile_name or (
                    st.session_state.get("bt_selected_profile_name")
                    or (cached_bertopic.get("profile_name") if cached_bertopic else ("quick_preview" if len(df) > 2000 else "full_analysis"))
                )
                st.session_state["bt_selected_profile_name"] = active_profile_name
                active_profile = get_bertopic_profile_settings(len(df), active_profile_name)
                df_bertopic = df.sample(n=active_profile["doc_cap"], random_state=42) if active_profile["doc_cap"] < len(df) else df

                bertopic_signature = (
                    len(df),
                    active_profile["profile_name"],
                    len(df_bertopic),
                    min_docs_bt,
                    bool(active_profile["include_topics_over_time"]),
                )
                if requested_profile_name:
                    job_id = submit_bertopic_background_job(
                        df_bertopic,
                        min_docs=min_docs_bt,
                        include_topics_over_time=active_profile["include_topics_over_time"],
                        profile_name=active_profile["profile_name"],
                    )
                    st.session_state["bt_active_job_id"] = job_id
                    st.session_state["bt_active_signature"] = bertopic_signature
                    st.session_state["bt_cached_result"] = None
                    cached_bertopic = None

                active_job_state = None
                active_job_id = st.session_state.get("bt_active_job_id")
                if active_job_id and st.session_state.get("bt_active_signature") == bertopic_signature:
                    active_job_state = get_bertopic_background_job(active_job_id)
                    if active_job_state["status"] == "done":
                        st.session_state["bt_cached_result"] = {
                            "signature": bertopic_signature,
                            "data": active_job_state.get("result"),
                            "analyzed_records": active_job_state.get("analyzed_records", len(df_bertopic)),
                            "include_topics_over_time": active_job_state.get("include_topics_over_time", False),
                            "profile_name": active_job_state.get("profile_name", active_profile["profile_name"]),
                        }
                        cached_bertopic = st.session_state["bt_cached_result"]
                        discard_bertopic_background_job(active_job_id)
                        st.session_state["bt_active_job_id"] = None
                        st.session_state["bt_active_signature"] = None
                        active_job_state = None
                    elif active_job_state["status"] in {"error", "cancelled", "missing"}:
                        st.session_state["bt_active_job_id"] = None
                        st.session_state["bt_active_signature"] = None

                topic_model, topic_info, topics_over_time = None, None, None
                if cached_bertopic and cached_bertopic.get("signature") == bertopic_signature:
                    res = cached_bertopic.get("data")
                    if res and res[0] == "MISSING_LIB":
                        log_exception("bertopic_missing_lib", RuntimeError(res[1]))
                        st.error(f"Missing required libraries for BERTopic: {res[1]}")
                        st.info("Please install them using: `pip install bertopic sentence-transformers umap-learn hdbscan`")
                    elif res:
                        topic_model, topic_info, topics_over_time = res
                elif active_job_state and active_job_state["status"] == "running":
                    st.info(
                        f"BERTopic task is running in the background: {active_job_state.get('profile_name', active_profile['profile_name'])}, "
                        f"{active_job_state.get('analyzed_records', len(df_bertopic))} records."
                    )
                    status_col1, status_col2 = st.columns(2)
                    with status_col1:
                        st.button("Refresh BERTopic Status", key="bt_refresh_status")
                    with status_col2:
                        if st.button("Cancel BERTopic Task", key="bt_cancel_status"):
                            discard_bertopic_background_job(active_job_id)
                            st.session_state["bt_active_job_id"] = None
                            st.session_state["bt_active_signature"] = None
                            st.info("BERTopic task cancelled.")
                elif active_job_state and active_job_state["status"] == "error":
                    log_exception(
                        "bertopic_background_task",
                        RuntimeError(active_job_state.get("error", "Unknown error")),
                    )
                    st.error(f"BERTopic background task failed: {active_job_state.get('error', 'Unknown error')}")
                else:
                    st.info("Click `Run Quick Preview` or `Run Full Analysis` to start semantic topic modeling. Results are cached for the current parameter setting.")
                if topic_info is not None and len(topic_info) > 1:
                    n_topics = len(topic_info[topic_info['Topic'] != -1])
                    analyzed_records = cached_bertopic.get("analyzed_records", len(df_bertopic)) if cached_bertopic else len(df_bertopic)
                    profile_label = "Quick Preview" if cached_bertopic.get("profile_name") == "quick_preview" else "Full Analysis"
                    st.success(f"{profile_label}: discovered **{n_topics}** semantic topics from {analyzed_records} analyzed documents.")

                    t_start_render = time.time()
                    top_n_bt = integer_control(
                        st,
                        "Top N Topics to Display",
                        5,
                        10,
                        key="bt_topn",
                        input_max=max(5, n_topics),
                        slider_soft_cap=20,
                    )
                    st.markdown("#### Topic Distribution")
                    fig_bt_overview = render_bertopic_overview(topic_info, top_n_bt)
                    if fig_bt_overview:
                        render_plotly_chart(fig_bt_overview, width="stretch")
                        download_plotly_button(fig_bt_overview, "bertopic_overview.png", "Download Topic Distribution")
                    if topics_over_time is not None:
                        st.markdown("#### Topic Evolution Over Time")
                        top_n_bt_evo = integer_control(
                            st,
                            "Topics in Evolution Plot",
                            3,
                            8,
                            key="bt_evo_topn",
                            input_max=max(3, n_topics),
                            slider_soft_cap=15,
                        )
                        fig_bt_evo = render_bertopic_evolution(topics_over_time, top_n_bt_evo)
                        if fig_bt_evo:
                            render_plotly_chart(fig_bt_evo, width="stretch")
                            download_plotly_button(fig_bt_evo, "bertopic_evolution.png", "Download Topic Evolution")
                    elif not cached_bertopic.get("include_topics_over_time", True):
                        st.info("Current BERTopic run used the fast profile, so the topic evolution chart was skipped.")

                    display_performance_metrics(t_start_render, "BERTopic UI rendering")
                    st.markdown("#### Method Comparison: Co-occurrence vs. BERTopic")
                    st.caption(
                        "Left panel = shared topics, middle panel = topics found only by keyword co-occurrence, "
                        "right panel = topics identified only by BERTopic as semantic additions."
                    )
                    fig_bt_comp = render_bertopic_comparison(keyword_freq, topic_info, top_n_bt)
                    if fig_bt_comp:
                        render_plotly_chart(fig_bt_comp, width="stretch")
                        download_plotly_button(fig_bt_comp, "bertopic_comparison.png", "Download Method Comparison")
                    st.markdown("#### Topic Details")
                    display_info = topic_info[topic_info['Topic'] != -1].head(top_n_bt)
                    display_cols = [c for c in ['Topic', 'Count', 'Name'] if c in display_info.columns]
                    st.dataframe(display_info[display_cols], width="stretch", hide_index=True)
                elif cached_bertopic and cached_bertopic.get("signature") == bertopic_signature:
                    st.info("BERTopic could not identify semantic topics. This may be due to insufficient documents, missing Abstract data, or all documents being too similar. Try adjusting the minimum documents per topic.")

elif page == "Forward Signals":
    st.title("Forward Signals")
    if request_heavy_analysis_unlock("Forward Signals", "load_forward_heavy"):
        render_subanalysis_status("Forward Signals", "subanalysis_temporal_forward_ready")
        if request_subanalysis_unlock(
            "Forward Signals",
            "subanalysis_temporal_forward_ready",
            "load_temporal_forward",
        ):
            summary_forecast_horizon = int(st.session_state.get("publication_forecast_horizon", 4))
            summary_opportunity_top_n = int(st.session_state.get("keyword_opportunity_topn", 20))
            summary_entity_top_n = int(st.session_state.get("entity_forecast_topn", 10))
            summary_theme_slices = int(st.session_state.get("theme_migration_slices", 4))
            summary_theme_top_n = int(st.session_state.get("theme_migration_topn", 10))
            summary_entity_option_map = {
                "Top Journals": "journal",
                "Top Countries": "country",
                "Top Institutions": "institution",
            }
            summary_entity_options = ["Top Journals"]
            if has_affiliations:
                summary_entity_options.extend(["Top Countries", "Top Institutions"])
            summary_entity_choice = st.session_state.get("forward_entity_forecast_type", summary_entity_options[0])
            if summary_entity_choice not in summary_entity_options:
                summary_entity_choice = summary_entity_options[0]

            summary_forecast_df, summary_forecast_meta = build_publication_forecast_frame(
                df,
                forecast_horizon=summary_forecast_horizon,
            )
            summary_forecast_text = (
                summarize_publication_forecast(summary_forecast_df, summary_forecast_meta)
                if not summary_forecast_df.empty else ""
            )
            summary_opportunity_df = build_keyword_opportunity_map_frame(
                df,
                keywords_list,
                keyword_freq,
                top_n_keywords=max(30, summary_opportunity_top_n * 2),
                recent_year_window=4,
                min_total_occurrences=3,
            )
            summary_keyword_text = (
                summarize_keyword_opportunity_map(summary_opportunity_df, top_n=summary_opportunity_top_n)
                if not summary_opportunity_df.empty else ""
            )
            summary_entity_df, _, summary_entity_label = build_entity_forecast_tables(
                df,
                entity_type=summary_entity_option_map[summary_entity_choice],
                top_n_entities=summary_entity_top_n,
                forecast_horizon=summary_forecast_horizon,
                lookback_years=6,
                min_total_occurrences=3,
            )
            summary_entity_text = (
                summarize_entity_forecast_signals(summary_entity_df, summary_entity_label, top_n=min(5, summary_entity_top_n))
                if not summary_entity_df.empty else ""
            )
            summary_leadership_df, _, summary_leadership_label = build_entity_leadership_shift_tables(
                df,
                entity_type="country" if has_affiliations else "institution",
                top_n_entities=summary_entity_top_n,
                recent_year_window=4,
                min_total_occurrences=3,
            ) if has_affiliations else (pd.DataFrame(), pd.DataFrame(), "Entity")
            summary_leadership_text = (
                summarize_entity_leadership_shift(
                    summary_leadership_df,
                    summary_leadership_label,
                    top_n=min(5, summary_entity_top_n),
                )
                if not summary_leadership_df.empty else ""
            )
            summary_theme_df, _ = build_theme_migration_forecast_tables(
                df,
                keywords_list,
                keyword_freq,
                slice_count=summary_theme_slices,
                top_n_keywords=max(30, summary_theme_top_n * 4),
                max_topics_per_slice=min(6, max(3, summary_theme_top_n)),
                keywords_per_topic=3,
            )
            summary_theme_text = (
                summarize_theme_migration_signals(summary_theme_df, top_n=min(5, summary_theme_top_n))
                if not summary_theme_df.empty else ""
            )
            forward_overview_text = summarize_forward_signals_overview(
                publication_summary=summary_forecast_text,
                keyword_summary=summary_keyword_text,
                entity_summary=summary_entity_text,
                leadership_summary=summary_leadership_text,
                theme_summary=summary_theme_text,
            )

            st.markdown("### Forward Signals Summary")
            st.caption(
                "This integrated card combines publication trend, hotspot keywords, entity growth, and theme migration "
                "into one forward-looking assessment."
            )
            st.info(forward_overview_text)

            st.markdown("### Publication Forecast")
            st.caption(
                "This chart extrapolates the recent annual publication trend to the next few years. "
                "Treat it as a directional bibliometric signal rather than a precise forecast."
            )
            forecast_horizon = integer_control(
                st,
                "Forecast Horizon (Years)",
                2,
                4,
                key="publication_forecast_horizon",
                input_max=6,
                slider_soft_cap=8,
            )
            forecast_df, forecast_meta = build_publication_forecast_frame(df, forecast_horizon=forecast_horizon)
            if not forecast_df.empty:
                st.info(summarize_publication_forecast(forecast_df, forecast_meta))
                forecast_style = render_plot_style_controls(
                    "temporal_publication_forecast",
                    default_primary=SCIENTIFIC_COLORWAY[3],
                    default_secondary=SCIENTIFIC_COLORWAY[0],
                    default_height=560,
                    show_legend_default=True,
                    allow_color_controls=False,
                    preserve_original_colors=True,
                )
                fig_forecast = render_publication_forecast_figure(forecast_df)
                if fig_forecast:
                    fig_forecast = apply_publication_style_with_overrides(fig_forecast, forecast_style)
                    render_plotly_chart(fig_forecast, width="stretch")
                    download_plotly_button(fig_forecast, "publication_forecast.png", "Download Publication Forecast")
                if forecast_meta:
                    forecast_only = forecast_df[forecast_df["Series"] == "Forecast"]
                    next_projection = forecast_only.iloc[0]["Publications"] if not forecast_only.empty else 0
                    col_pf_1, col_pf_2, col_pf_3 = st.columns(3)
                    with col_pf_1:
                        st.metric("Latest Observed Year", forecast_meta["latest_year"])
                    with col_pf_2:
                        st.metric("Latest Publications", forecast_meta["latest_publications"])
                    with col_pf_3:
                        st.metric("Projected Next Year", f"{next_projection:.1f}")
            else:
                st.info("At least four valid publication years are needed to estimate a simple publication forecast.")

            st.markdown("### Keyword Opportunity Map")
            st.caption(
                "This map combines current keyword presence and recent growth to highlight potential future hotspots, "
                "stable core topics, and themes that may be cooling."
            )
            opportunity_top_n = integer_control(
                st,
                "Keywords in Opportunity Map",
                10,
                20,
                key="keyword_opportunity_topn",
                input_max=max(10, min(len(keyword_freq), 40)),
                slider_soft_cap=30,
            )
            opportunity_df = build_keyword_opportunity_map_frame(
                df,
                keywords_list,
                keyword_freq,
                top_n_keywords=max(30, opportunity_top_n * 2),
                recent_year_window=4,
                min_total_occurrences=3,
            )
            if not opportunity_df.empty:
                st.info(summarize_keyword_opportunity_map(opportunity_df, top_n=opportunity_top_n))
                opportunity_style = render_plot_style_controls(
                    "temporal_keyword_opportunity",
                    default_primary=SCIENTIFIC_COLORWAY[4],
                    default_secondary=SCIENTIFIC_COLORWAY[8],
                    default_height=620,
                    show_legend_default=True,
                    allow_color_controls=False,
                    preserve_original_colors=True,
                )
                fig_opportunity = render_keyword_opportunity_map(opportunity_df, top_n=opportunity_top_n)
                if fig_opportunity:
                    fig_opportunity = apply_publication_style_with_overrides(fig_opportunity, opportunity_style)
                    render_plotly_chart(fig_opportunity, width="stretch")
                    download_plotly_button(fig_opportunity, "keyword_opportunity_map.png", "Download Keyword Opportunity Map")
                with st.expander("Keyword Opportunity Table", expanded=False):
                    st.dataframe(opportunity_df.head(opportunity_top_n), width="stretch", hide_index=True)
            else:
                st.info("Not enough temporal keyword continuity is available to build the keyword opportunity map.")

            st.markdown("### Entity Growth Forecast")
            st.caption(
                "This panel projects which journals, countries, or institutions may grow faster in the next stage of the field. "
                "Use the ranking chart to spot leaders and the trajectory chart to inspect whether the forecast extends an existing trend."
            )
            entity_option_map = {
                "Top Journals": "journal",
                "Top Countries": "country",
                "Top Institutions": "institution",
            }
            available_entity_options = ["Top Journals"]
            if has_affiliations:
                available_entity_options.extend(["Top Countries", "Top Institutions"])
            selected_entity_option = st.selectbox(
                "Entity Type",
                available_entity_options,
                key="forward_entity_forecast_type",
            )
            entity_forecast_topn = integer_control(
                st,
                "Forecast Leaders",
                5,
                10,
                key="entity_forecast_topn",
                input_max=20,
                slider_soft_cap=20,
            )
            entity_summary_df, entity_series_df, entity_label = build_entity_forecast_tables(
                df,
                entity_type=entity_option_map[selected_entity_option],
                top_n_entities=entity_forecast_topn,
                forecast_horizon=forecast_horizon,
                lookback_years=6,
                min_total_occurrences=3,
            )
            if not entity_summary_df.empty:
                st.info(summarize_entity_forecast_signals(entity_summary_df, entity_label, top_n=min(5, entity_forecast_topn)))
                entity_rank_style = render_plot_style_controls(
                    "temporal_entity_growth_rank",
                    default_primary=SCIENTIFIC_COLORWAY[6],
                    default_height=560,
                )
                fig_entity_rank = render_entity_forecast_rank_figure(
                    entity_summary_df,
                    entity_label=entity_label,
                    top_n=entity_forecast_topn,
                )
                if fig_entity_rank:
                    fig_entity_rank = apply_publication_style_with_overrides(fig_entity_rank, entity_rank_style)
                    render_plotly_chart(fig_entity_rank, width="stretch")
                    download_plotly_button(fig_entity_rank, f"{entity_label.lower()}_growth_forecast_rank.png", f"Download {entity_label} Growth Forecast")

                entity_trend_style = render_plot_style_controls(
                    "temporal_entity_growth_trend",
                    default_primary=SCIENTIFIC_COLORWAY[7],
                    default_secondary=SCIENTIFIC_COLORWAY[1],
                    default_height=580,
                    show_legend_default=True,
                    allow_color_controls=False,
                    preserve_original_colors=True,
                )
                fig_entity_trend = render_entity_forecast_trajectory_figure(
                    entity_series_df,
                    entity_label=entity_label,
                    top_n=min(5, entity_forecast_topn),
                )
                if fig_entity_trend:
                    fig_entity_trend = apply_publication_style_with_overrides(fig_entity_trend, entity_trend_style)
                    render_plotly_chart(fig_entity_trend, width="stretch")
                    download_plotly_button(fig_entity_trend, f"{entity_label.lower()}_forecast_trajectories.png", f"Download {entity_label} Forecast Trajectories")
                with st.expander(f"{entity_label} Forecast Table", expanded=False):
                    st.dataframe(entity_summary_df, width="stretch", hide_index=True)
            else:
                st.info(f"Not enough valid year-by-{entity_label.lower()} continuity is available to build a forecast.")

            st.markdown("### Country / Institution Leadership Shift")
            st.caption(
                "This panel tracks whether leadership share is consolidating or shifting across countries or institutions. "
                "It focuses on relative presence in the literature, not only raw publication growth."
            )
            if has_affiliations:
                leadership_option_map = {
                    "Countries": "country",
                    "Institutions": "institution",
                }
                leadership_type = st.selectbox(
                    "Leadership Entity Type",
                    list(leadership_option_map.keys()),
                    key="forward_leadership_shift_type",
                )
                leadership_top_n = integer_control(
                    st,
                    "Leadership Shift Entities",
                    5,
                    10,
                    key="leadership_shift_topn",
                    input_max=20,
                    slider_soft_cap=20,
                )
                leadership_summary_df, leadership_series_df, leadership_label = build_entity_leadership_shift_tables(
                    df,
                    entity_type=leadership_option_map[leadership_type],
                    top_n_entities=leadership_top_n,
                    recent_year_window=4,
                    min_total_occurrences=3,
                )
                if not leadership_summary_df.empty:
                    st.info(
                        summarize_entity_leadership_shift(
                            leadership_summary_df,
                            leadership_label,
                            top_n=min(5, leadership_top_n),
                        )
                    )
                    leadership_shift_style = render_plot_style_controls(
                        "temporal_entity_leadership_shift",
                        default_primary=SCIENTIFIC_COLORWAY[5],
                        default_secondary=SCIENTIFIC_COLORWAY[8],
                        default_height=560,
                        show_legend_default=True,
                        allow_color_controls=False,
                        preserve_original_colors=True,
                    )
                    fig_leadership_shift = render_entity_leadership_shift_figure(
                        leadership_summary_df,
                        leadership_label,
                        top_n=leadership_top_n,
                    )
                    if fig_leadership_shift:
                        fig_leadership_shift = apply_publication_style_with_overrides(
                            fig_leadership_shift,
                            leadership_shift_style,
                        )
                        render_plotly_chart(fig_leadership_shift, width="stretch")
                        download_plotly_button(
                            fig_leadership_shift,
                            f"{leadership_label.lower()}_leadership_shift.png",
                            f"Download {leadership_label} Leadership Shift",
                        )

                    leadership_traj_style = render_plot_style_controls(
                        "temporal_entity_leadership_traj",
                        default_primary=SCIENTIFIC_COLORWAY[2],
                        default_secondary=SCIENTIFIC_COLORWAY[6],
                        default_height=580,
                        show_legend_default=True,
                        allow_color_controls=False,
                        preserve_original_colors=True,
                    )
                    fig_leadership_traj = render_entity_leadership_trajectory_figure(
                        leadership_series_df,
                        leadership_summary_df,
                        leadership_label,
                        top_n=min(5, leadership_top_n),
                    )
                    if fig_leadership_traj:
                        fig_leadership_traj = apply_publication_style_with_overrides(
                            fig_leadership_traj,
                            leadership_traj_style,
                        )
                        render_plotly_chart(fig_leadership_traj, width="stretch")
                        download_plotly_button(
                            fig_leadership_traj,
                            f"{leadership_label.lower()}_leadership_share_over_time.png",
                            f"Download {leadership_label} Leadership Share",
                        )
                    with st.expander(f"{leadership_label} Leadership Shift Table", expanded=False):
                        st.dataframe(leadership_summary_df, width="stretch", hide_index=True)
                else:
                    st.info("Not enough year-by-entity continuity is available to estimate leadership shifts.")
            else:
                st.info("Affiliation data are required to analyze leadership shifts across countries or institutions.")

            st.markdown("### Theme Cluster Migration Forecast")
            st.caption(
                "This analysis tracks topic clusters across time slices and estimates which theme chains are likely to keep strengthening, "
                "stabilize, or cool down in the next stage."
            )
            theme_slice_count = integer_control(
                st,
                "Theme Time Slices",
                2,
                4,
                key="theme_migration_slices",
                input_max=max(2, clean_year_column(df)["Year"].nunique()),
                slider_soft_cap=8,
            )
            theme_top_n = integer_control(
                st,
                "Theme Chains to Inspect",
                5,
                10,
                key="theme_migration_topn",
                input_max=15,
                slider_soft_cap=20,
            )
            theme_summary_df, theme_chain_df = build_theme_migration_forecast_tables(
                df,
                keywords_list,
                keyword_freq,
                slice_count=theme_slice_count,
                top_n_keywords=max(30, theme_top_n * 4),
                max_topics_per_slice=min(6, max(3, theme_top_n)),
                keywords_per_topic=3,
            )
            if not theme_summary_df.empty:
                st.info(summarize_theme_migration_signals(theme_summary_df, top_n=min(5, theme_top_n)))
                theme_traj_style = render_plot_style_controls(
                    "temporal_theme_migration_traj",
                    default_primary=SCIENTIFIC_COLORWAY[9],
                    default_secondary=SCIENTIFIC_COLORWAY[2],
                    default_height=600,
                    show_legend_default=True,
                    allow_color_controls=False,
                    preserve_original_colors=True,
                )
                fig_theme_traj = render_theme_migration_trajectory_figure(
                    theme_chain_df,
                    theme_summary_df,
                    top_n=min(6, theme_top_n),
                )
                if fig_theme_traj:
                    fig_theme_traj = apply_publication_style_with_overrides(fig_theme_traj, theme_traj_style)
                    render_plotly_chart(fig_theme_traj, width="stretch")
                    download_plotly_button(fig_theme_traj, "theme_cluster_migration_trajectories.png", "Download Theme Migration Trajectories")

                theme_map_style = render_plot_style_controls(
                    "temporal_theme_migration_map",
                    default_primary=SCIENTIFIC_COLORWAY[10],
                    default_secondary=SCIENTIFIC_COLORWAY[4],
                    default_height=620,
                    show_legend_default=True,
                    allow_color_controls=False,
                    preserve_original_colors=True,
                )
                fig_theme_map = render_theme_migration_opportunity_map(theme_summary_df, top_n=theme_top_n)
                if fig_theme_map:
                    fig_theme_map = apply_publication_style_with_overrides(fig_theme_map, theme_map_style)
                    render_plotly_chart(fig_theme_map, width="stretch")
                    download_plotly_button(fig_theme_map, "theme_cluster_future_hotspot_map.png", "Download Theme Hotspot Map")
                with st.expander("Theme Cluster Forecast Table", expanded=False):
                    st.dataframe(theme_summary_df.head(theme_top_n), width="stretch", hide_index=True)
            else:
                st.info("Not enough stable topic-slice continuity is available to estimate theme migration and future hotspot shifts.")

elif page == "Bibliometric Structure & Performance":
    st.title("Bibliometric Structure & Performance")
    if request_heavy_analysis_unlock("Bibliometric Structure & Performance", "load_structure_heavy"):
        bib_tabs = ["Descriptive", "Journal Analysis", "Keyword Analysis", "Thematic Map", "Hierarchical Cluster Heatmap"]
        if has_affiliations or "Journal" in df.columns:
            bib_tabs.append("Top Entities")
        if has_authors:
            bib_tabs.append("Authors Over Time")
            bib_tabs.append("Three-Field Plot")
            bib_tabs.append("Lotka's Law")
        if has_language:
            bib_tabs.append("Language Distribution")
        bib_tab_objs = st.tabs(bib_tabs)
        tab1 = bib_tab_objs[0]
        tab2 = bib_tab_objs[1]
        tab3 = bib_tab_objs[2]
        tab4 = bib_tab_objs[3]  # Thematic Map
        tab_heatmap = bib_tab_objs[4]  # Hierarchical Cluster Heatmap

        current_tab_idx = 5
        if has_affiliations or "Journal" in df.columns:
            tab_entities = bib_tab_objs[current_tab_idx]
            current_tab_idx += 1
        else:
            tab_entities = None
        if has_authors:
            tab_author_time = bib_tab_objs[current_tab_idx]
            current_tab_idx += 1
            tab6 = bib_tab_objs[current_tab_idx]  # Three-Field Plot
            current_tab_idx += 1
            tab_lotka = bib_tab_objs[current_tab_idx]
            current_tab_idx += 1
        else:
            tab_author_time = None
            tab6 = None
            tab_lotka = None

        if has_language:
            tab_lang = bib_tab_objs[current_tab_idx]
        else:
            tab_lang = None

        with tab1:
            st.markdown("### Descriptive Statistics")
            df_valid = clean_year_column(df)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Publications", len(df_valid))
                st.metric("Year Range", f"{df_valid['Year'].min()}-{df_valid['Year'].max()}")
            with col2:
                years_span = df_valid['Year'].max() - df_valid['Year'].min() + 1
                avg_per_year = len(df_valid) / years_span if years_span > 0 else 0
                st.metric("Avg Publications/Year", f"{avg_per_year:.1f}")
                st.metric("Unique Journals", df_valid['Journal'].nunique())
            with col3:
                if 'DOI' in df_valid.columns:
                    doi_coverage = df_valid['DOI'].notna().sum() / len(df_valid) * 100
                    st.metric("DOI Coverage", f"{doi_coverage:.1f}%")
                if 'Abstract' in df_valid.columns:
                    abs_coverage = df_valid['Abstract'].notna().sum() / len(df_valid) * 100
                    st.metric("Abstract Coverage", f"{abs_coverage:.1f}%")
            st.markdown("### Annual Growth Rate")
            growth_df = get_cached_annual_growth_rate(df)
            if not growth_df.empty:
                growth_style = render_plot_style_controls(
                    "structure_growth_rate",
                    default_primary=SCIENTIFIC_COLORWAY[0],
                    default_height=500,
                )
                fig = px.line(
                    growth_df,
                    x='Year',
                    y='Growth Rate (%)',
                    title='Annual Growth Rate',
                    markers=True,
                )
                fig = apply_publication_style_with_overrides(fig, growth_style)
                fig.update_traces(line_color=growth_style['primary_color'], marker_color=growth_style['primary_color'])
                fig.add_hline(y=0, line_dash="dash", line_color="#666666", opacity=0.6)
                render_plotly_chart(fig, width="stretch")
                download_plotly_button(fig, "annual_growth_rate.png", "Download Growth Rate")

        with tab2:
            st.markdown("### Journal Analysis")
            journal_counts = get_cached_journal_counts(df)
            st.markdown("#### Top 20 Journals by Publication Count")
            top_journals = journal_counts.head(20)
            journal_style = render_plot_style_controls(
                "structure_top_journals",
                default_primary=SCIENTIFIC_COLORWAY[1],
                default_height=600,
            )
            fig = render_ranked_lollipop_figure(
                top_journals.rename_axis("Journal").reset_index(name="Count"),
                label_col="Journal",
                value_col="Count",
                title="Top 20 Journals",
                marker_color=SCIENTIFIC_COLORWAY[1],
                line_color=SCIENTIFIC_COLORWAY[1],
                xaxis_title="Count",
                yaxis_title="Journal",
            )
            fig = apply_publication_style_with_overrides(fig, journal_style)
            render_plotly_chart(fig, width="stretch")
            download_plotly_button(fig, "top20_journals.png", "Download Top 20 Journals")
            st.markdown("#### Bradford's Law Analysis")
            sorted_journals = journal_counts.sort_values(ascending=False)
            cumulative = sorted_journals.cumsum()
            total = cumulative.iloc[-1]
            one_third = total / 3
            core_zone = cumulative[cumulative <= one_third]
            if len(core_zone) == 0:
                core_journals = [sorted_journals.index[0]]
            else:
                core_journals = core_zone.index.tolist()
            st.info(f"Core Zone (Bradford's Law): **{len(core_journals)}** journals contain approximately 1/3 of all publications.")
            if core_journals:
                st.write(", ".join(core_journals[:10]))

        with tab3:
            st.markdown("### Keyword Frequency Analysis")
            keyword_frequency_limit = integer_control(
                st,
                "Keyword Frequency Rows to Display",
                1,
                min(30, len(keyword_freq)),
                key="structure_keyword_frequency_n",
                input_max=max(1, len(keyword_freq)),
                slider_soft_cap=min(max(50, min(500, len(keyword_freq))), len(keyword_freq)),
            )
            kw_df = pd.DataFrame(
                keyword_freq.most_common(keyword_frequency_limit),
                columns=['Keyword', 'Frequency'],
            )
            st.dataframe(kw_df, width="stretch")
            st.markdown("### Keyword Frequency Distribution")
            st.caption("Distribution of extracted keyword frequencies. This panel is separate from author productivity and should not be interpreted as Lotka's Law.")
            if keyword_freq:
                col_x_max = st.slider(
                    "Max Frequency to Show (cut long tail)",
                    1,
                    max(keyword_freq.values()) if keyword_freq else 100,
                    min(50, max(keyword_freq.values()) if keyword_freq else 100),
                    key="kw_freq_x_max",
                )
                kw_freq_style = render_plot_style_controls(
                    "structure_kw_freq",
                    default_primary=SCIENTIFIC_COLORWAY[2],
                    default_height=500,
                )
                freq_dist = Counter(keyword_freq.values())
                freq_df = pd.DataFrame(sorted(freq_dist.items()), columns=['Frequency', 'Number of Keywords'])
                freq_df = freq_df[freq_df['Frequency'] <= col_x_max]
                fig = px.area(
                    freq_df,
                    x='Frequency',
                    y='Number of Keywords',
                    title='Keyword Frequency Distribution'
                )
                fig = apply_publication_style_with_overrides(fig, kw_freq_style)
                fig.update_traces(line_color=kw_freq_style['primary_color'], fillcolor="rgba(47, 116, 184, 0.18)")
                render_plotly_chart(fig, width="stretch")
                download_plotly_button(fig, "keyword_freq_distribution.png", "Download Frequency Distribution")
            st.markdown("### Keyword Circular Cluster Map")
            st.caption(
                "This circular cluster map complements the standard co-occurrence network by presenting "
                "keyword communities in a chord-style layout that is useful for cross-tool comparison figures."
            )
            circular_col1, circular_col2 = st.columns(2)
            with circular_col1:
                kw_circular_topn = integer_control(
                    st,
                    "Keywords in Circular Map",
                    15,
                    30,
                    key="kw_circular_topn",
                    input_max=max(15, len(keyword_freq)),
                    slider_soft_cap=60,
                )
            with circular_col2:
                kw_circular_minw = st.slider(
                    "Min Co-occurrence for Circular Map",
                    1,
                    10,
                    2,
                    key="kw_circular_minw",
                )
            kw_circular_style = render_plot_style_controls(
                "structure_kw_circular",
                default_primary=SCIENTIFIC_COLORWAY[4],
                default_height=920,
                show_legend_default=True,
                allow_color_controls=False,
                preserve_original_colors=True,
            )
            fig_kw_circular = build_keyword_circular_cluster_figure(
                keyword_freq,
                cooccurrence,
                top_n=kw_circular_topn,
                min_weight=kw_circular_minw,
            )
            if fig_kw_circular is not None:
                fig_kw_circular = apply_publication_style_with_overrides(fig_kw_circular, kw_circular_style)
                render_plotly_chart(fig_kw_circular, width="stretch")
                download_plotly_button(
                    fig_kw_circular,
                    "keyword_circular_cluster_map.png",
                    "Download Keyword Circular Map",
                )
            else:
                st.info("The current keyword network is too sparse to build the circular cluster map.")

            st.markdown("---")
            st.markdown("### Keyword Co-occurrence Matrix (Jaccard Index)")
            st.caption("Diagonal is fixed at 1.00 to represent self-similarity. Lower triangle shows Jaccard similarity = co-occurrence / (A union B). Keyword frequency is listed separately in the side table.")
            render_subanalysis_status("Co-occurrence Matrix", "subanalysis_structure_matrix_ready")
            if request_subanalysis_unlock(
                "Co-occurrence Matrix",
                "subanalysis_structure_matrix_ready",
                "load_structure_matrix",
            ):
                top_n_mat = integer_control(
                    st,
                    "Matrix Size (Top N Keywords)",
                    10,
                    15,
                    key="mat_topn",
                    input_max=max(10, len(keyword_freq)),
                    slider_soft_cap=40,
                )
                top_kws = [kw for kw, _ in keyword_freq.most_common(top_n_mat)]
                n_kws = len(top_kws)
                matrix = np.full((n_kws, n_kws), np.nan)
                for i, k1 in enumerate(top_kws):
                    matrix[i][i] = 1.0
                    for j in range(i):
                        k2 = top_kws[j]
                        w = cooccurrence.get((k1, k2), cooccurrence.get((k2, k1), 0))
                        freq_a = keyword_freq[k1]
                        freq_b = keyword_freq[k2]
                        union = freq_a + freq_b - w
                        jaccard = w / union if union > 0 else 0
                        matrix[i][j] = jaccard
                mat_df = pd.DataFrame(matrix, index=top_kws, columns=top_kws)
                mat_style = render_plot_style_controls(
                    "structure_cooccurrence_matrix",
                    default_primary=SCIENTIFIC_COLORWAY[3],
                    default_height=700,
                    allow_color_controls=False,
                    preserve_original_colors=True,
                )
                fig = px.imshow(mat_df, text_auto='.2f', aspect='auto',
                                title='Co-occurrence Matrix (Jaccard Index)',
                                color_continuous_scale=KEYWORD_MATRIX_SEQUENTIAL_SCALE,
                                range_color=[0, 1])
                fig = apply_publication_style_with_overrides(fig, mat_style)
                fig.update_traces(textfont=dict(size=14))
                matrix_col, freq_col = st.columns([4, 1.2])
                with matrix_col:
                    render_plotly_chart(fig, width="stretch")
                with freq_col:
                    freq_table_df = pd.DataFrame(
                        {"Keyword": top_kws, "Frequency": [keyword_freq[kw] for kw in top_kws]}
                    )
                    st.markdown("#### Keyword Frequency")
                    st.dataframe(freq_table_df, width="stretch", hide_index=True)
                download_plotly_button(fig, "cooccurrence_matrix.png", "Download Co-occurrence Matrix")

        with tab4:
            st.markdown("### Thematic Map")
            st.caption("Clusters are positioned by Callon centrality and density. *Inspired by the Bibliometrix/BiblioShiny thematic map approach.*")
            render_subanalysis_status("Thematic Map", "subanalysis_structure_thematic_ready")
            if request_subanalysis_unlock(
                "Thematic Map",
                "subanalysis_structure_thematic_ready",
                "load_structure_thematic",
            ):
                top_n_tm = integer_control(
                    st,
                    "Number of Keywords",
                    10,
                    20,
                    key="tm_topn",
                    input_max=max(10, len(keyword_freq)),
                    slider_soft_cap=50,
                )

                t_start = time.time()
                fig_tm = get_cached_thematic_map(keyword_freq, cooccurrence, top_n_tm)
                if fig_tm:
                    tm_style = render_plot_style_controls(
                        "structure_thematic_map",
                        default_primary=SCIENTIFIC_COLORWAY[4],
                        default_height=600,
                        allow_color_controls=False,
                        preserve_original_colors=True,
                    )
                    fig_tm = apply_publication_style_with_overrides(fig_tm, tm_style)
                    render_plotly_chart(fig_tm, width="stretch")
                    display_performance_metrics(t_start, "Thematic map")
                    download_plotly_button(fig_tm, "thematic_map.png", "Download Thematic Map")
                    st.markdown("#### Quadrant Interpretation")
                    st.markdown("""
                    | Quadrant | Meaning |
                    |----------|---------|
                    | **Motor themes** (upper-right) | High centrality & density: well-developed, core research themes driving the field |
                    | **Basic themes** (lower-right) | High centrality, low density: fundamental, transversal topics important across clusters |
                    | **Niche themes** (upper-left) | Low centrality, high density: specialized, internally cohesive but isolated topics |
                    | **Emerging/Declining** (lower-left) | Low centrality & density: marginal or newly appearing themes |
                    """)
                else:
                    st.info("Thematic mapping is unavailable because the current keyword co-occurrence structure is insufficient for stable clustering.")

        with tab_heatmap:
            st.markdown("### Hierarchical Cluster Heatmap")
            st.caption("Hierarchical clustering of keywords based on co-occurrence patterns.")
            
            top_n_hm = integer_control(
                st,
                "Number of Keywords",
                10,
                16,
                key="hm_topn",
                input_max=max(10, len(keyword_freq)),
                slider_soft_cap=30,
            )
            
            use_generic_filter = st.checkbox(
                "Filter Generic Terms",
                value=True,
                key="hm_filter_generic",
                help="Exclude common generic terms like 'study', 'research', 'analysis', etc.",
            )
            
            generic_exclusions = set()
            if use_generic_filter:
                generic_exclusions = {
                    "study", "studies", "research", "analysis", "method", "methods",
                    "approach", "approaches", "result", "results", "finding", "findings",
                    "data", "information", "model", "models", "system", "systems",
                    "application", "applications", "use", "using", "used", "based",
                    "new", "novel", "effective", "efficient", "performance", "evaluation",
                }
            
            t_start = time.time()
            from modules.structure_visualization import render_hierarchical_cluster_heatmap
            fig_hm, matrix_df = render_hierarchical_cluster_heatmap(
                keywords_list,
                keyword_freq,
                cooccurrence,
                top_n=top_n_hm,
                generic_exclusions=generic_exclusions,
            )
            
            if fig_hm:
                hm_style = render_plot_style_controls(
                    "structure_hierarchical_heatmap",
                    default_primary=SCIENTIFIC_COLORWAY[0],
                    default_height=800,
                    allow_color_controls=False,
                    preserve_original_colors=True,
                )
                fig_hm = apply_publication_style_with_overrides(fig_hm, hm_style)
                render_plotly_chart(fig_hm, width="stretch")
                display_performance_metrics(t_start, "Hierarchical cluster heatmap")
                download_plotly_button(fig_hm, "hierarchical_cluster_heatmap.png", "Download Hierarchical Cluster Heatmap")
                
                st.markdown("#### Co-occurrence Matrix")
                with st.expander("View Matrix Data", expanded=False):
                    st.dataframe(matrix_df, width="stretch")
                    csv_data = matrix_df.to_csv().encode("utf-8-sig")
                    st.download_button(
                        "Download Matrix (CSV)",
                        csv_data,
                        "hierarchical_cluster_matrix.csv",
                        "text/csv",
                    )
            else:
                st.info("Insufficient keyword data for hierarchical clustering heatmap.")

        if tab_entities is not None:
            with tab_entities:
                st.markdown("### Top Entities")
                entity_topn = integer_control(
                    st,
                    "Rows per Table",
                    5,
                    15,
                    key="entity_topn",
                    input_max=max(
                        5,
                        build_top_country_table(df, top_n=500).shape[0],
                        build_top_institution_table(df, top_n=500).shape[0],
                        build_top_journal_table(df, top_n=500).shape[0],
                    ),
                    slider_soft_cap=30,
                )
                entity_col1, entity_col2, entity_col3 = st.columns(3)

                top_country_df = build_top_country_table(df, top_n=entity_topn)
                with entity_col1:
                    st.markdown("#### Top Countries")
                    if not top_country_df.empty:
                        st.dataframe(top_country_df, width="stretch", hide_index=True)
                        st.download_button(
                            "Download Top Countries (CSV)",
                            top_country_df.to_csv(index=False).encode("utf-8-sig"),
                            "Biblio-HUB_top_countries.csv",
                            "text/csv",
                            key="download_top_countries",
                        )
                    else:
                        st.info("No affiliation-derived country data available.")

                top_institution_df = build_top_institution_table(df, top_n=entity_topn)
                with entity_col2:
                    st.markdown("#### Top Institutions")
                    if not top_institution_df.empty:
                        st.dataframe(top_institution_df, width="stretch", hide_index=True)
                        st.download_button(
                            "Download Top Institutions (CSV)",
                            top_institution_df.to_csv(index=False).encode("utf-8-sig"),
                            "Biblio-HUB_top_institutions.csv",
                            "text/csv",
                            key="download_top_institutions",
                        )
                    else:
                        st.info("No affiliation-derived institution data available.")

                top_journal_df = build_top_journal_table(df, top_n=entity_topn)
                with entity_col3:
                    st.markdown("#### Top Journals")
                    if not top_journal_df.empty:
                        st.dataframe(top_journal_df, width="stretch", hide_index=True)
                        st.download_button(
                            "Download Top Journals (CSV)",
                            top_journal_df.to_csv(index=False).encode("utf-8-sig"),
                            "Biblio-HUB_top_journals.csv",
                            "text/csv",
                            key="download_top_journals_table",
                        )
                    else:
                        st.info("No journal metadata available.")

        if tab_author_time is not None:
            with tab_author_time:
                st.markdown("### Authors' Production Over Time")
                render_subanalysis_status("Authors Over Time", "subanalysis_structure_author_time_ready")
                if request_subanalysis_unlock(
                    "Authors Over Time",
                    "subanalysis_structure_author_time_ready",
                    "load_structure_author_time",
                ):
                    top_n_author_time = integer_control(
                        st,
                        "Authors in Timeline",
                        5,
                        10,
                        key="author_time_topn",
                        input_max=max(5, _count_available_authors(df)),
                        slider_soft_cap=25,
                    )

                    t_start = time.time()
                    fig_author_time = get_cached_author_production_over_time(df, top_n=top_n_author_time)
                    if fig_author_time:
                        author_time_style = render_plot_style_controls(
                            "structure_author_time",
                            default_primary=SCIENTIFIC_COLORWAY[5],
                            default_height=600,
                            show_legend_default=True,
                            allow_color_controls=False,
                            preserve_original_colors=True,
                        )
                        fig_author_time = apply_publication_style_with_overrides(fig_author_time, author_time_style)
                        render_plotly_chart(fig_author_time, width="stretch")
                        display_performance_metrics(t_start, "Author production")
                        download_plotly_button(fig_author_time, "authors_production_over_time.png", "Download Author Timeline")
                        author_time_df = build_author_production_over_time_frame(df, top_n=top_n_author_time)
                        if not author_time_df.empty:
                            summary_df = (
                                author_time_df.groupby("Author", as_index=False)
                                .agg(**{"Total Papers": ("Total Papers", "max"), "Active Years": ("Year", "nunique")})
                                .sort_values(["Total Papers", "Author"], ascending=[False, True])
                            )
                            st.markdown("#### Top Authors Summary")
                            st.dataframe(summary_df, width="stretch", hide_index=True)
                    else:
                        st.info("Not enough author-year data to render author production over time.")

        if tab6 is not None:
            with tab6:
                st.markdown("### Three-Field Plot")
                st.caption("This Sankey diagram summarizes the linkage structure among authors, keywords, and journals. *Inspired by the Bibliometrix/BiblioShiny implementation.*")
                render_subanalysis_status("Three-Field Plot", "subanalysis_structure_three_field_ready")
                if request_subanalysis_unlock(
                    "Three-Field Plot",
                    "subanalysis_structure_three_field_ready",
                    "load_structure_three_field",
                ):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        n_auth = integer_control(
                            st,
                            "Top Authors",
                            5,
                            10,
                            key="3f_auth",
                            input_max=max(5, _count_available_authors(df)),
                            slider_soft_cap=25,
                        )
                    with col_b:
                        n_kw = integer_control(
                            st,
                            "Top Keywords",
                            5,
                            15,
                            key="3f_kw",
                            input_max=max(5, len(keyword_freq)),
                            slider_soft_cap=30,
                        )
                    with col_c:
                        n_jn = integer_control(
                            st,
                            "Top Journals",
                            5,
                            10,
                            key="3f_jn",
                            input_max=max(5, int(df["Journal"].dropna().nunique()) if "Journal" in df.columns else 5),
                            slider_soft_cap=25,
                        )

                    t_start = time.time()
                    fig_3f = get_cached_three_field_plot(df, keywords_list, keyword_freq, n_auth, n_kw, n_jn)
                    if fig_3f:
                        three_field_style = render_plot_style_controls(
                            "structure_three_field",
                            default_primary=SCIENTIFIC_COLORWAY[6],
                            default_height=700,
                            show_legend_default=True,
                            allow_color_controls=False,
                            preserve_original_colors=True,
                        )
                        fig_3f = apply_publication_style_with_overrides(fig_3f, three_field_style)
                        render_plotly_chart(fig_3f, width="stretch")
                        display_performance_metrics(t_start, "Three-field plot")
                        download_plotly_button(fig_3f, "three_field_plot.png", "Download Three-Field Plot")
                    else:
                        st.info("Insufficient data for three-field plot. Requires Authors, Keywords, and Journal fields.")

        if tab_lotka is not None:
            with tab_lotka:
                st.markdown("### Lotka's Law (Author Productivity)")
                st.caption("Lotka's Law describes the frequency of publication by authors in a given field. *Classical bibliometric indicator implemented in many tools like Bibliometrix.*")
                st.caption("It states that the number of authors making $x$ contributions is about $1/x^2$ of those making one contribution.")

                t_start = time.time()
                res = get_cached_lotkas_law(df)
                if res:
                    fig_lotka, lotka_stats = res
                    lotka_style = render_plot_style_controls(
                        "structure_lotka",
                        default_primary=SCIENTIFIC_COLORWAY[7],
                        default_secondary=SCIENTIFIC_COLORWAY[8],
                        default_height=500,
                        show_legend_default=True,
                        allow_color_controls=False,
                        preserve_original_colors=True,
                    )
                    fig_lotka = apply_publication_style_with_overrides(fig_lotka, lotka_style)
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        render_plotly_chart(fig_lotka, width="stretch")
                        display_performance_metrics(t_start, "Lotka's law")
                        download_plotly_button(fig_lotka, "lotkas_law.png", "Download Lotka's Law Plot")
                    with col2:
                        st.markdown("#### Productivity Stats")
                        st.metric("Total Authors", lotka_stats['total_authors'])
                        st.metric("Max Papers by One Author", lotka_stats['max_papers'])
                        st.metric("Authors with 1 Paper", f"{lotka_stats['authors_with_1_paper']} ({lotka_stats['percent_with_1']:.1f}%)")
                        st.metric("Theoretical Constant (C)", f"{lotka_stats['theoretical_c']:.4f}")
                else:
                    st.warning("Not enough author data to calculate Lotka's Law.")

        if tab_lang is not None and has_language:
            with tab_lang:
                st.markdown("### Language Distribution")
                lang_freq = get_cached_language_freq(df)
                if lang_freq:
                    lang_df = pd.DataFrame(lang_freq.most_common(20), columns=['Language', 'Papers'])
                    lang_pie_style = render_plot_style_controls(
                        "structure_lang_pie",
                        default_primary=SCIENTIFIC_COLORWAY[9],
                        default_height=500,
                        show_legend_default=True,
                        allow_color_controls=False,
                        preserve_original_colors=True,
                    )
                    fig_lang = px.pie(lang_df, values='Papers', names='Language',
                                     title='Publication Language Distribution')
                    fig_lang = apply_publication_style_with_overrides(fig_lang, lang_pie_style)
                    render_plotly_chart(fig_lang, width="stretch")
                    download_plotly_button(fig_lang, "language_distribution.png", "Download Language Chart")
                    lang_bar_style = render_plot_style_controls(
                        "structure_lang_bar",
                        default_primary=SCIENTIFIC_COLORWAY[7],
                        default_height=400,
                    )
                    lang_plot_df = lang_df.sort_values("Papers", ascending=False).copy()
                    fig_lang_bar = px.bar(
                        lang_plot_df,
                        x='Papers',
                        y='Language',
                        orientation='h',
                        title='Papers by Language'
                    )
                    fig_lang_bar = apply_publication_style_with_overrides(fig_lang_bar, lang_bar_style)
                    fig_lang_bar.update_yaxes(
                        categoryorder="array",
                        categoryarray=list(reversed(lang_plot_df["Language"].tolist())),
                    )
                    render_plotly_chart(fig_lang_bar, width="stretch")
                    download_plotly_button(fig_lang_bar, "language_distribution_bar.png", "Download Language Bar Chart")
                else:
                    st.info("No language data available.")

elif page == "Citation, Category & Source Analysis":
    st.title("Citation, Category & Source Analysis")
    tab_list = ["Citation Analysis", "Subject Categories", "Journal Distribution", "Document Types", "Funding Analysis"]
    if has_cited_refs:
        tab_list.append("Reference Analysis")
    if has_publisher:
        tab_list.append("Publisher Analysis")
    tabs = st.tabs(tab_list)
    tab_cite = tabs[0]
    tab_cat = tabs[1]
    tab_journal = tabs[2]
    tab_type = tabs[3]
    tab_fund = tabs[4]
    with tab_cite:
        st.markdown("### Highly Cited Papers")
        if has_times_cited:
            df_cite = df.copy()
            df_cite['Times_Cited'] = pd.to_numeric(df_cite['Times_Cited'], errors='coerce').fillna(0).astype(int)
            top_cited = df_cite.nlargest(20, 'Times_Cited')
            display_cols = [c for c in ['Title', 'Authors', 'Journal', 'Year', 'Times_Cited'] if c in top_cited.columns]
            st.dataframe(top_cited[display_cols], width="stretch", hide_index=True)

            st.markdown("---")
            st.markdown("### Disruption Index ($DI_1$)")
            st.caption("The Disruption Index (Funk & Owen-Smith, 2017; Wu et al., Nature 2019) measures whether a paper is 'disruptive' (breaks from the past) or 'consolidating' (builds on the past). Values range from -1 to +1.")
            if st.button("Calculate Disruption Index (Internal Network)"):
                with st.spinner("Calculating internal citation network and DI scores..."):
                    df_di = compute_disruption_index_frame(df_cite)
                    if df_di is not None and 'Disruption_Index' in df_di.columns:
                        di_summary = summarize_disruption_index(df_di)
                        di_col1, di_col2, di_col3, di_col4 = st.columns(4)
                        with di_col1:
                            st.metric("Analyzed Papers", di_summary["papers"])
                        with di_col2:
                            st.metric("Mean DI", f"{di_summary['mean_di']:.4f}")
                        with di_col3:
                            st.metric("Disruptive (>0)", di_summary["positive_count"])
                        with di_col4:
                            st.metric("Consolidating (<0)", di_summary["negative_count"])

                        top_disruptive = rank_disruption_extremes(df_di, kind="disruptive", top_n=10)
                        st.markdown("#### Top 10 Most Disruptive Papers (Internal)")
                        st.dataframe(
                            top_disruptive[[c for c in ['Title', 'Year', 'Disruption_Index', 'Support', 'Internal_Citers', 'DI_nd', 'DI_nc', 'DI_na'] if c in top_disruptive.columns]],
                            hide_index=True,
                            width="stretch",
                        )

                        top_consolidating = rank_disruption_extremes(df_di, kind="consolidating", top_n=10)
                        st.markdown("#### Top 10 Most Consolidating Papers (Internal)")
                        st.dataframe(
                            top_consolidating[[c for c in ['Title', 'Year', 'Disruption_Index', 'Support', 'Internal_Citers', 'DI_nd', 'DI_nc', 'DI_na'] if c in top_consolidating.columns]],
                            hide_index=True,
                            width="stretch",
                        )

                        di_style = render_plot_style_controls(
                            "cite_di_distribution",
                            default_primary=SCIENTIFIC_COLORWAY[4],
                            default_height=500,
                        )
                        fig_di = px.histogram(df_di, x='Disruption_Index', nbins=30,
                                             title='Disruption Index Distribution',
                                             labels={'Disruption_Index': 'DI Score', 'count': 'Number of Papers'})
                        fig_di = apply_publication_style_with_overrides(fig_di, di_style)
                        fig_di.update_traces(marker_color=di_style['primary_color'])
                        render_plotly_chart(fig_di, width="stretch")
                        download_plotly_button(fig_di, "disruption_index_distribution.png", "Download DI Distribution")
                        export_cols = [c for c in ['Title', 'Authors', 'Journal', 'Year', 'Disruption_Index', 'DI_nd', 'DI_nc', 'DI_na', 'Internal_References', 'Internal_Citers'] if c in df_di.columns]
                        st.download_button(
                            "Download DI Scores (CSV)",
                            df_di[export_cols].to_csv(index=False).encode("utf-8-sig"),
                            "Biblio-HUB_disruption_index_scores.csv",
                            "text/csv",
                        )
                    else:
                        st.warning("Disruption Index estimation is unavailable because the internal citation network is too sparse or incomplete.")

            st.markdown("---")
            st.markdown("### Citation Distribution")
            cite_dist_style = render_plot_style_controls(
                "cite_distribution",
                default_primary=SCIENTIFIC_COLORWAY[11],
                default_height=500,
            )
            cite_values = df_cite['Times_Cited'].dropna()
            x_max = int(np.quantile(cite_values, 0.99)) if len(cite_values) > 0 else 50
            x_max = max(x_max, 20)
            tick_step = 25
            fig_cite = px.histogram(df_cite, x='Times_Cited', nbins=50,
                                   title='Citation Distribution',
                                   labels={'Times_Cited': 'Times Cited', 'count': 'Number of Papers'})
            fig_cite = apply_publication_style_with_overrides(fig_cite, cite_dist_style)
            fig_cite.update_traces(marker_color=cite_dist_style['primary_color'])
            fig_cite.update_xaxes(range=[0, x_max], tick0=0, dtick=tick_step)
            render_plotly_chart(fig_cite, width="stretch")
            download_plotly_button(fig_cite, "citation_distribution.png", "Download Citation Distribution")
            st.markdown("### Annual Publications and Average Citations")
            if 'Year' in df_cite.columns:
                trend_df = build_publication_citation_trend_frame(df_cite)
                fig_dual_axis = render_publication_citation_dual_axis_figure(trend_df)
                if fig_dual_axis is not None:
                    dual_axis_style = render_plot_style_controls(
                        "cite_dual_axis",
                        default_primary=SCIENTIFIC_COLORWAY[12],
                        default_secondary=SCIENTIFIC_COLORWAY[13],
                        default_height=600,
                        show_legend_default=True,
                        allow_color_controls=False,
                        preserve_original_colors=True,
                    )
                    fig_dual_axis = apply_publication_style_with_overrides(fig_dual_axis, dual_axis_style)
                    render_plotly_chart(fig_dual_axis, width="stretch")
                    download_plotly_button(fig_dual_axis, "annual_publications_avg_citations.png", "Download Dual-Axis Trend")
                    st.caption("Recent-year citation counts are right-censored because newer papers have had less time to accumulate citations.")
                    st.dataframe(trend_df, width="stretch", hide_index=True)
            st.markdown("### Citation by Year")
            cite_by_year = get_cached_citation_by_year(df)
            if not cite_by_year.empty:
                cite_year_style = render_plot_style_controls(
                    "cite_by_year",
                    default_primary=SCIENTIFIC_COLORWAY[13],
                    default_height=500,
                )
                fig_cy = px.line(cite_by_year, x='Year', y='Avg Citations',
                               title='Average Citations per Paper by Year', markers=True)
                fig_cy = apply_publication_style_with_overrides(fig_cy, cite_year_style)
                fig_cy.update_traces(marker_color=cite_year_style['primary_color'], line_color=cite_year_style['primary_color'])
                render_plotly_chart(fig_cy, width="stretch")
                download_plotly_button(fig_cy, "citations_by_year.png", "Download Citations by Year")
        else:
            st.info("No citation data (TC field) available in this dataset.")
    with tab_cat:
        st.markdown("### Subject Category Distribution")
        cat_freq = get_cached_category_freq(df)
        if cat_freq:
            cat_style = render_plot_style_controls(
                "cite_subject_categories",
                default_primary=SCIENTIFIC_COLORWAY[6],
                default_height=600,
            )
            cat_df = pd.DataFrame(cat_freq.most_common(25), columns=['Category', 'Count'])
            cat_plot_df = cat_df.sort_values("Count", ascending=False).copy()
            fig_cat = px.bar(
                cat_plot_df,
                x='Count',
                y='Category',
                orientation='h',
                title='Top 25 Subject Categories'
            )
            fig_cat = apply_publication_style_with_overrides(fig_cat, cat_style)
            fig_cat.update_traces(marker_color=cat_style['primary_color'])
            fig_cat.update_yaxes(
                categoryorder="array",
                categoryarray=list(reversed(cat_plot_df["Category"].tolist())),
            )
            render_plotly_chart(fig_cat, width="stretch")
            download_plotly_button(fig_cat, "subject_categories.png", "Download Categories")
        else:
            st.info("No subject category data (WC/SC fields) available.")
    with tab_journal:
        st.markdown("### Journal Distribution")
        if 'Journal' in df.columns:
            journal_style = render_plot_style_controls(
                "cite_journal_distribution",
                default_primary=SCIENTIFIC_COLORWAY[0],
                default_height=600,
            )
            journal_counts = df['Journal'].value_counts()
            journal_df = journal_counts.head(25).reset_index()
            journal_df.columns = ['Journal', 'Publications']
            journal_plot_df = journal_df.sort_values("Publications", ascending=False).copy()
            fig_journal = px.bar(
                journal_plot_df,
                x='Publications',
                y='Journal',
                orientation='h',
                title='Top 25 Journals by Publication Count'
            )
            fig_journal = apply_publication_style_with_overrides(fig_journal, journal_style)
            fig_journal.update_traces(marker_color=journal_style['primary_color'])
            fig_journal.update_yaxes(
                categoryorder="array",
                categoryarray=list(reversed(journal_plot_df["Journal"].tolist())),
            )
            render_plotly_chart(fig_journal, width="stretch")
            download_plotly_button(fig_journal, "journal_distribution.png", "Download Journal Distribution")
        else:
            st.info("No journal data (SO field) available in this dataset.")
    with tab_type:
        st.markdown("### Document Type Distribution")
        if has_doctype:
            type_style = render_plot_style_controls(
                "cite_document_types",
                default_primary=SCIENTIFIC_COLORWAY[10],
                default_height=500,
                show_legend_default=True,
                allow_color_controls=False,
                preserve_original_colors=True,
            )
            type_freq = get_cached_doctype_freq(df)
            fig_type = px.pie(values=list(type_freq.values()), names=list(type_freq.keys()),
                             title='Document Types')
            fig_type = apply_publication_style_with_overrides(fig_type, type_style)
            render_plotly_chart(fig_type, width="stretch")
            download_plotly_button(fig_type, "document_types.png", "Download Document Types")
        else:
            st.info("No document type data (DT field) available.")
    with tab_fund:
        st.markdown("### Funding Agency Analysis")
        if has_funding:
            fund_freq = get_cached_funding_freq(df)
            if fund_freq:
                fund_style = render_plot_style_controls(
                    "cite_funding_agencies",
                    default_primary=SCIENTIFIC_COLORWAY[8],
                    default_height=500,
                )
                fund_df = pd.DataFrame(fund_freq.most_common(20), columns=['Funding Agency', 'Count'])
                fund_plot_df = fund_df.sort_values("Count", ascending=False).copy()
                fig_fund = px.scatter(
                    fund_plot_df,
                    x='Count',
                    y='Funding Agency',
                    size='Count',
                    size_max=24,
                    title='Top 20 Funding Agencies'
                )
                fig_fund = apply_publication_style_with_overrides(fig_fund, fund_style)
                fig_fund.update_traces(marker_color=fund_style['primary_color'], line=dict(color="white", width=1))
                fig_fund.update_yaxes(
                    categoryorder="array",
                    categoryarray=list(reversed(fund_plot_df["Funding Agency"].tolist())),
                )
                render_plotly_chart(fig_fund, width="stretch")
                download_plotly_button(fig_fund, "funding_agencies.png", "Download Funding")
            else:
                st.info("No funding data parsed from this dataset.")
        else:
            st.info("No funding data (FU field) available.")

    if has_cited_refs and len(tab_list) > 5:
        with tabs[5]:
            st.markdown("### Most Cited References")
            ref_freq, ref_year_freq, ref_counts_per_paper = extract_cited_reference_statistics(df)
            if ref_freq:
                top_ref_df = pd.DataFrame(ref_freq.most_common(30), columns=['Reference', 'Citations'])
                st.dataframe(top_ref_df, width="stretch", hide_index=True)
                if ref_year_freq:
                    ref_year_df = pd.DataFrame(sorted(ref_year_freq.items()), columns=['Year', 'Reference Count'])
                    peak_reference_year = int(ref_year_df.loc[ref_year_df["Reference Count"].idxmax(), "Year"])
                    col_ref_y1, col_ref_y2, col_ref_y3 = st.columns(3)
                    with col_ref_y1:
                        st.metric("Reference Year Span", f"{int(ref_year_df['Year'].min())}-{int(ref_year_df['Year'].max())}")
                    with col_ref_y2:
                        st.metric("Peak Reference Year", peak_reference_year)
                    with col_ref_y3:
                        st.metric("Peak Year Count", int(ref_year_df["Reference Count"].max()))
                st.markdown("### Reference Publication Year Spectroscopy (RPYS)")
                st.caption("This RPYS view highlights years whose cited-reference counts rise above the local five-year median. *Inspired by CiteSpace's RPYS implementation.*")
                rpys_df, rpys_reference_counter = extract_rpys_statistics(df)
                if not rpys_df.empty:
                    rpys_style = render_plot_style_controls(
                        "cite_rpys",
                        default_primary=SCIENTIFIC_COLORWAY[2],
                        default_secondary=SCIENTIFIC_COLORWAY[3],
                        default_height=600,
                        allow_color_controls=False,
                        preserve_original_colors=True,
                    )
                    fig_rpys = render_rpys_figure(rpys_df)
                    fig_rpys = apply_publication_style_with_overrides(fig_rpys, rpys_style)
                    render_plotly_chart(fig_rpys, width="stretch")
                    download_plotly_button(fig_rpys, "reference_publication_year_spectroscopy.png", "Download RPYS")

                    peak_df = build_rpys_peak_table(rpys_df, rpys_reference_counter, top_n=10, refs_per_year=3)
                    if not peak_df.empty:
                        st.markdown("#### Top RPYS Peak Years")
                        st.dataframe(peak_df, width="stretch", hide_index=True)
                else:
                    st.info("RPYS could not be generated because no valid cited-reference years were parsed.")
                st.markdown("### Reference Burst Detection")
                st.caption("This panel highlights references that receive concentrated attention during specific publication periods.")
                reference_burst_df = build_reference_burst_table(df, top_n=20)
                if not reference_burst_df.empty:
                    ref_burst_style = render_plot_style_controls(
                        "cite_ref_burst",
                        default_primary=SCIENTIFIC_COLORWAY[4],
                        default_height=600,
                        allow_color_controls=False,
                        preserve_original_colors=True,
                    )
                    fig_ref_burst = render_reference_burst_figure(reference_burst_df)
                    fig_ref_burst = apply_publication_style_with_overrides(fig_ref_burst, ref_burst_style)
                    render_plotly_chart(fig_ref_burst, width="stretch")
                    download_plotly_button(fig_ref_burst, "reference_burst_detection.png", "Download Reference Burst")
                    st.markdown("#### Top Reference Burst Events")
                    st.dataframe(reference_burst_df, width="stretch", hide_index=True)
                else:
                    st.info("Reference burst detection could not be generated because valid cited-reference year series were insufficient.")
                st.markdown("### Journal Co-citation Network")
                st.info(
                    "To avoid duplicate network panels, the full Journal Co-citation Network is now maintained "
                    "only in `Relational Network Analysis`, where threshold controls and publication-ready exports "
                    "are already grouped with the other network views."
                )
                st.markdown("### Reference Diversity per Paper")
                if ref_counts_per_paper:
                    ref_hist_style = render_plot_style_controls(
                        "cite_ref_hist",
                        default_primary=SCIENTIFIC_COLORWAY[6],
                        default_height=500,
                    )
                    fig_ref_hist = px.histogram(x=ref_counts_per_paper, nbins=30,
                                               title='Number of References per Paper',
                                               labels={'x': 'Number of References', 'y': 'Count'})
                    fig_ref_hist = apply_publication_style_with_overrides(fig_ref_hist, ref_hist_style)
                    fig_ref_hist.update_traces(marker_color=ref_hist_style['primary_color'])
                    render_plotly_chart(fig_ref_hist, width="stretch")
                    download_plotly_button(fig_ref_hist, "refs_per_paper.png", "Download Refs per Paper")
            else:
                st.info("No cited reference data available.")

    if has_publisher and len(tab_list) > 5 + (1 if has_cited_refs else 0):
        pub_tab_idx = 5 + (1 if has_cited_refs else 0)
        with tabs[pub_tab_idx]:
            st.markdown("### Publisher Analysis")
            pub_freq = get_cached_publisher_freq(df)
            if pub_freq:
                pub_style = render_plot_style_controls(
                    "cite_publisher",
                    default_primary=SCIENTIFIC_COLORWAY[9],
                    default_height=600,
                )
                pub_df = pd.DataFrame(pub_freq.most_common(25), columns=['Publisher', 'Papers'])
                pub_plot_df = pub_df.sort_values("Papers", ascending=False).copy()
                fig_pub = px.scatter(
                    pub_plot_df,
                    x='Papers',
                    y='Publisher',
                    size='Papers',
                    size_max=24,
                    title='Top 25 Publishers by Publication Count'
                )
                fig_pub = apply_publication_style_with_overrides(fig_pub, pub_style)
                fig_pub.update_traces(marker_color=pub_style['primary_color'], line=dict(color="white", width=1))
                fig_pub.update_yaxes(
                    categoryorder="array",
                    categoryarray=list(reversed(pub_plot_df["Publisher"].tolist())),
                )
                render_plotly_chart(fig_pub, width="stretch")
                download_plotly_button(fig_pub, "publisher_analysis.png", "Download Publisher Chart")
                st.markdown("### Publisher Concentration")
                total_papers = sum(pub_freq.values())
                top3 = sum(v for _, v in pub_freq.most_common(3))
                top5 = sum(v for _, v in pub_freq.most_common(5))
                top10 = sum(v for _, v in pub_freq.most_common(10))
                col_c1, col_c2, col_c3 = st.columns(3)
                with col_c1:
                    st.metric("Top 3 Publishers", f"{top3/total_papers*100:.1f}%")
                with col_c2:
                    st.metric("Top 5 Publishers", f"{top5/total_papers*100:.1f}%")
                with col_c3:
                    st.metric("Top 10 Publishers", f"{top10/total_papers*100:.1f}%")
            else:
                st.info("No publisher data available.")

# Moved generate_export_zip to top level

elif page == "Innovation Analysis":
    st.title("Innovation Analysis")
    st.caption("Dedicated workspace for disruption, structural-hole brokerage, and robustness metrics.")
    render_innovation_analysis_panel(df)
elif page == "Export Center":
    st.title("Export Center")
    st.caption("Prepare archives and publication bundles from the current processed dataset.")
    if request_heavy_analysis_unlock("Export Center", "load_export_heavy"):
        render_export_center(df, keywords_list, keyword_freq, cooccurrence, dedup_report)
 
