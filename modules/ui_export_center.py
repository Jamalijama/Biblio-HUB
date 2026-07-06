import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import zipfile
import plotly.express as px
from modules.export_background import (
    submit_export_background_job,
    get_export_job_status,
    discard_export_job,
)
from modules.export_bundle import (
    build_manuscript_case_report,
    build_manuscript_case_snapshot,
    build_manuscript_submission_report,
    build_manuscript_submission_snapshot,
    build_reproducibility_report,
    build_reproducibility_snapshot,
    get_plotly_static_export_status,
    PUBLICATION_EXPORT_FORMATS,
    plotly_figure_to_bytes,
    sanitize_filename,
    SCIENTIFIC_COLORWAY,
)
from modules.export_orchestrator import (
    build_figure_explanation_bundle,
    build_journal_submission_bundle,
    build_manuscript_case_bundle,
    build_methods_package_bundle,
    build_one_click_research_bundle,
    build_reviewer_response_bundle,
    build_submission_result_bundle,
    get_available_figure_options,
    group_figure_options,
)
from modules.data_pipeline import safe_year
from modules.entity_analysis import (
    build_top_country_table,
    build_top_institution_table,
    build_top_journal_table,
)
from modules.experiment_framework import (
    DEFAULT_DI_EXTREMES_MIN_INTERNAL_CITERS,
    DEFAULT_DI_EXTREMES_MIN_INTERNAL_REFERENCES,
    DEFAULT_DI_EXTREMES_MIN_SUPPORT,
    build_journal_submission_package_report,
    build_journal_submission_package_snapshot,
    build_parameterized_journal_template,
    build_reviewer_response_report,
    build_reviewer_response_snapshot,
    build_submission_figure_package_report,
    build_submission_figure_package_snapshot,
    format_execution_policy_summary,
    submit_innovation_background_job,
    get_innovation_background_job,
    discard_innovation_background_job,
    build_submission_result_snapshot,
    build_submission_result_report,
    build_research_report_snapshot,
    build_research_report,
    rank_disruption_extremes,
    render_brokerage_robustness_summary,
    render_structural_hole_brokerage_profile,
)
from modules.ui_helpers import (
    apply_publication_style_with_overrides,
    integer_control,
    render_plot_style_controls,
    render_plotly_chart,
)

def _get_export_time_hint_base(profile, record_count=0, lightweight_mode=False, scenario_count=1, sample_size=0):
    if profile in {"innovation", "submission", "journal", "research"}:
        if record_count > 2000 and not lightweight_mode:
            return (
                "Recommended: lightweight mode",
                "This export includes robustness analysis or multi-report aggregation; lightweight mode is recommended for larger datasets.",
            )
        if record_count > 2000:
            return (
                "Slow",
                f"The current run is expected to evaluate {scenario_count} robustness scenarios "
                f"over {sample_size}/{record_count} records.",
            )
        return ("Moderate", "Includes multiple reports, snapshots, or result summaries.")
    if profile in {"figure_bundle", "figure_selected"}:
        if record_count > 5000:
            return ("Slow", "Batch figure rendering and network packaging usually take longer for large datasets.")
        return ("Moderate", "Most of the time is spent on figure rendering and archive packaging.")
    if profile == "copyright_package":
        return ("Slow", "Collects core code snippets, technical documents, and the draft registration materials.")
    if profile in {"full_export", "methods", "case_package", "figure_package", "reviewer"}:
        if record_count > 5000:
            return ("Slow", "Packages multiple file types, so wait time becomes more noticeable for large datasets.")
        return ("Moderate", "Mainly bundles text, tables, and packaged outputs.")
    return ("Moderate", "Mostly single-file export, so wait time is usually manageable.")


def _render_export_download_button_base(profile, *args, record_count=0, lightweight_mode=False, scenario_count=1, sample_size=0, **kwargs):
    tier, note = _get_export_time_hint_base(
        profile,
        record_count=record_count,
        lightweight_mode=lightweight_mode,
        scenario_count=scenario_count,
        sample_size=sample_size,
    )
    st.caption(f"Estimated time: {tier} | {note}")
    return st.download_button(*args, **kwargs)


def _csv_filename_from_title(title: str) -> str:
    return f"Biblio-HUB_{sanitize_filename(title)}.csv"


def _render_named_export_download(label, data, file_name, mime, *, speed_label, key):
    st.caption(f"Estimated time: {speed_label}")
    return st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=mime,
        key=key,
    )


def _render_plotly_multiformat_downloads(profile, fig, filename_stem, *, width, height, key_prefix):
    export_cols = st.columns(len(PUBLICATION_EXPORT_FORMATS))
    exported = False
    for idx, export_format in enumerate(PUBLICATION_EXPORT_FORMATS):
        try:
            mime = (
                "image/png"
                if export_format == "png"
                else "image/svg+xml" if export_format == "svg" else "application/pdf"
            )
            export_cols[idx].download_button(
                label=f"Download {export_format.upper()}",
                data=plotly_figure_to_bytes(fig, export_format, width=width, height=height),
                file_name=f"{filename_stem}.{export_format}",
                mime=mime,
                key=f"{key_prefix}_{export_format}",
            )
            exported = True
        except Exception:
            continue
    if not exported:
        tier, note = _get_export_time_hint_base(profile)
        st.caption(f"Estimated time: {tier} | {note}")
        st.caption("Install `kaleido` to enable PNG/SVG/PDF export for this figure.")


def _collect_innovation_analysis_parameters(df, bundle_format_value):
    export_record_count = len(df)
    export_lightweight_mode_value = bool(
        st.session_state.get("export_lightweight_mode", export_record_count > 2000)
    )
    if export_lightweight_mode_value:
        export_robustness_sample_size = min(export_record_count, 3000) if export_record_count > 5000 else export_record_count
        export_robustness_scenario_count = 1
    elif export_record_count > 5000:
        export_robustness_sample_size = min(export_record_count, 5000)
        export_robustness_scenario_count = 4
    else:
        export_robustness_sample_size = export_record_count
        export_robustness_scenario_count = 9

    return [
        {"key": "keyword_source", "label": "Keyword Source", "value": st.session_state.get("keyword_source", "DE+ID"), "default": "DE+ID", "group": "Network"},
        {"key": "overview_top_journals_n", "label": "Overview Top Journals", "value": st.session_state.get("overview_top_journals_n", 30), "default": 30, "group": "Overview"},
        {"key": "vos_topn", "label": "Keyword Network Top N", "value": st.session_state.get("vos_topn", 30), "default": 30, "group": "Network"},
        {"key": "vos_minw", "label": "Keyword Network Min Co-occurrence", "value": st.session_state.get("vos_minw", 2), "default": 2, "group": "Network"},
        {"key": "kj_topn_kw", "label": "Keyword-Journal Top Keywords", "value": st.session_state.get("kj_topn_kw", 15), "default": 15, "group": "Network"},
        {"key": "kj_topn_jn", "label": "Keyword-Journal Top Journals", "value": st.session_state.get("kj_topn_jn", 10), "default": 10, "group": "Network"},
        {"key": "auth_min_papers", "label": "Author Collaboration Min Papers", "value": st.session_state.get("auth_min_papers", 2), "default": 2, "group": "Collaboration"},
        {"key": "auth_topn", "label": "Author Collaboration Max Authors", "value": st.session_state.get("auth_topn", 50), "default": 50, "group": "Collaboration"},
        {"key": "inst_topn", "label": "Institution Network Top N", "value": st.session_state.get("inst_topn", 25), "default": 25, "group": "Collaboration"},
        {"key": "bc_topn", "label": "Bibliographic Coupling Top Papers", "value": st.session_state.get("bc_topn", 30), "default": 30, "group": "Citation"},
        {"key": "bc_min_shared", "label": "Bibliographic Coupling Min Shared References", "value": st.session_state.get("bc_min_shared", 2), "default": 2, "group": "Citation"},
        {"key": "cocite_topn", "label": "Co-citation Top References", "value": st.session_state.get("cocite_topn", 20), "default": 20, "group": "Citation"},
        {"key": "cocite_minw", "label": "Co-citation Minimum Weight", "value": st.session_state.get("cocite_minw", 2), "default": 2, "group": "Citation"},
        {"key": "auth_cocite_topn", "label": "Author Co-citation Top Authors", "value": st.session_state.get("auth_cocite_topn", 20), "default": 20, "group": "Citation"},
        {"key": "auth_cocite_minw", "label": "Author Co-citation Minimum Weight", "value": st.session_state.get("auth_cocite_minw", 2), "default": 2, "group": "Citation"},
        {"key": "j_cocite_topn", "label": "Journal Co-citation Top Journals", "value": st.session_state.get("j_cocite_topn", 20), "default": 20, "group": "Citation"},
        {"key": "j_cocite_minw", "label": "Journal Co-citation Minimum Weight", "value": st.session_state.get("j_cocite_minw", 2), "default": 2, "group": "Citation"},
        {"key": "di_extremes_min_support", "label": "DI Extremes Min Support", "value": st.session_state.get("di_extremes_min_support", DEFAULT_DI_EXTREMES_MIN_SUPPORT), "default": DEFAULT_DI_EXTREMES_MIN_SUPPORT, "group": "Innovation"},
        {"key": "di_extremes_min_internal_citers", "label": "DI Extremes Min Internal Citers", "value": st.session_state.get("di_extremes_min_internal_citers", DEFAULT_DI_EXTREMES_MIN_INTERNAL_CITERS), "default": DEFAULT_DI_EXTREMES_MIN_INTERNAL_CITERS, "group": "Innovation"},
        {"key": "di_extremes_min_internal_references", "label": "DI Extremes Min Internal References", "value": st.session_state.get("di_extremes_min_internal_references", DEFAULT_DI_EXTREMES_MIN_INTERNAL_REFERENCES), "default": DEFAULT_DI_EXTREMES_MIN_INTERNAL_REFERENCES, "group": "Innovation"},
        {"key": "di_extremes_support_filter_mode", "label": "DI Extremes Support Rule", "value": st.session_state.get("di_extremes_support_filter_mode", "any"), "default": "any", "group": "Innovation"},
        {"key": "di_extremes_require_topic_match", "label": "DI Extremes Require Topic Match", "value": st.session_state.get("di_extremes_require_topic_match", False), "default": False, "group": "Innovation"},
        {"key": "cs_topn", "label": "Timeline Top Keywords", "value": st.session_state.get("cs_topn", 15), "default": 15, "group": "Temporal"},
        {"key": "burst_topn", "label": "Burst Detection Keyword Count", "value": st.session_state.get("burst_topn", 20), "default": 20, "group": "Temporal"},
        {"key": "bt_min_docs", "label": "BERTopic Minimum Documents", "value": st.session_state.get("bt_min_docs", 3), "default": 3, "group": "Semantic Topic Modeling"},
        {"key": "bt_topn", "label": "BERTopic Displayed Topics", "value": st.session_state.get("bt_topn", 10), "default": 10, "group": "Semantic Topic Modeling"},
        {"key": "bt_evo_topn", "label": "BERTopic Evolution Topics", "value": st.session_state.get("bt_evo_topn", 8), "default": 8, "group": "Semantic Topic Modeling"},
        {"key": "mat_topn", "label": "Matrix Top Keywords", "value": st.session_state.get("mat_topn", 15), "default": 15, "group": "Structure"},
        {"key": "tm_topn", "label": "Thematic Map Top Keywords", "value": st.session_state.get("tm_topn", 20), "default": 20, "group": "Structure"},
        {"key": "author_time_topn", "label": "Author Timeline Top Authors", "value": st.session_state.get("author_time_topn", 10), "default": 10, "group": "Structure"},
        {"key": "3f_auth", "label": "Three-Field Top Authors", "value": st.session_state.get("3f_auth", 10), "default": 10, "group": "Structure"},
        {"key": "3f_kw", "label": "Three-Field Top Keywords", "value": st.session_state.get("3f_kw", 15), "default": 15, "group": "Structure"},
        {"key": "3f_jn", "label": "Three-Field Top Journals", "value": st.session_state.get("3f_jn", 10), "default": 10, "group": "Structure"},
        {"key": "export_lightweight_mode", "label": "Export Lightweight Mode", "value": export_lightweight_mode_value, "default": False, "group": "Export"},
        {"key": "export_robustness_scenario_count", "label": "Robustness Scenario Count", "value": export_robustness_scenario_count, "default": 9, "group": "Export"},
        {"key": "export_robustness_sample_size", "label": "Robustness Analysis Sample Size", "value": export_robustness_sample_size, "default": len(df), "group": "Export"},
        {"key": "bundle_format", "label": "Publication Figure Format", "value": bundle_format_value, "default": PUBLICATION_EXPORT_FORMATS[0], "group": "Export"},
    ]


def render_innovation_analysis_panel(df, bundle_format_value=None, halt_if_missing=False, show_export_downloads=True):
    bundle_format_value = bundle_format_value or st.session_state.get("bundle_format", PUBLICATION_EXPORT_FORMATS[0])
    figure_mime = f"image/{bundle_format_value}"

    st.markdown("### Innovation Analysis")
    st.caption(
        "This module provides disruption, structural-hole, and robustness metrics for the current dataset."
    )
    with st.expander("Metric Notes", expanded=False):
        st.markdown(
            "- `Disruption Index (DI1)`: compares whether later records cite a focal record without also citing its references. Higher values indicate more disruptive patterns and lower values indicate more consolidating patterns.\n"
            "- `Structural Hole`: uses betweenness centrality, structural constraint, and effective size to identify nodes that connect otherwise weakly linked groups in the bibliographic coupling network.\n"
            "- `Robustness`: reruns brokerage detection under alternative bibliographic-coupling parameter settings and summarizes which high-importance bridge nodes remain stable across scenarios."
        )

    is_large_dataset = len(df) > 2000
    if is_large_dataset:
        st.warning(f"Large dataset detected ({len(df)} records). Innovation metrics and robustness analysis may take some time.")
        if "export_lightweight_mode" not in st.session_state:
            st.session_state["export_lightweight_mode"] = True
        lightweight_mode = st.checkbox(
            "Enable Lightweight Mode",
            key="export_lightweight_mode",
            help="Reduce robustness scenarios to speed up innovation analysis.",
        )
    else:
        st.session_state["export_lightweight_mode"] = False
        lightweight_mode = False

    parameter_col1, parameter_col2 = st.columns(2)
    with parameter_col1:
        bc_topn_val = integer_control(
            st,
            "Brokerage Analysis top_n",
            10,
            int(st.session_state.get("bc_topn", 30)),
            key="bc_topn",
            input_max=max(10, min(len(df), 200)),
            slider_soft_cap=80,
        )
    with parameter_col2:
        bc_min_shared_val = integer_control(
            st,
            "Brokerage Analysis min_shared_refs",
            1,
            int(st.session_state.get("bc_min_shared", 2)),
            key="bc_min_shared",
            input_max=100,
            slider_soft_cap=10,
        )
    st.caption(
        "Recommended baseline for large datasets: `top_n=30-40` and `min_shared_refs=3`. "
        "Lower `min_shared_refs` keeps more weak links; higher values make bridge detection stricter."
    )
    analysis_parameters = _collect_innovation_analysis_parameters(df, bundle_format_value)

    if lightweight_mode:
        robustness_top_n_values = [bc_topn_val]
        robustness_min_shared_values = [bc_min_shared_val]
        df_robustness = df.sample(n=3000, random_state=42) if len(df) > 5000 else df
    elif len(df) > 5000:
        robustness_top_n_values = sorted({max(10, bc_topn_val - 5), bc_topn_val})
        robustness_min_shared_values = sorted({bc_min_shared_val, bc_min_shared_val + 1})
        df_robustness = df.sample(n=5000, random_state=42)
    else:
        robustness_top_n_values = sorted({max(10, bc_topn_val - 10), bc_topn_val, bc_topn_val + 10})
        robustness_min_shared_values = sorted({max(1, bc_min_shared_val - 1), bc_min_shared_val, bc_min_shared_val + 1})
        df_robustness = df

    current_execution_policy = {
        "lightweight_mode": bool(lightweight_mode),
        "full_record_count": int(len(df)),
        "analysis_record_count": int(len(df_robustness)),
        "downsampled": bool(len(df_robustness) < len(df)),
        "scenario_count_requested": int(len(robustness_top_n_values) * len(robustness_min_shared_values)),
    }
    st.caption(f"Current robustness plan: {format_execution_policy_summary(current_execution_policy)}.")

    innovation_signature = (
        len(df),
        bc_topn_val,
        bc_min_shared_val,
        bool(lightweight_mode),
        tuple((item.get("key"), item.get("value")) for item in analysis_parameters),
    )
    innovation_action_col1, innovation_action_col2 = st.columns(2)
    with innovation_action_col1:
        generate_innovation_requested = st.button("Run Innovation Metrics", key="innovation_generate")
    with innovation_action_col2:
        refresh_innovation_requested = st.button("Refresh Innovation Status", key="innovation_refresh")

    cached_innovation_payload = st.session_state.get("innovation_cached_payload")
    if generate_innovation_requested:
        innovation_job_id = submit_innovation_background_job(
            df,
            analysis_parameters,
            bc_topn_val=bc_topn_val,
            bc_min_shared_val=bc_min_shared_val,
            lightweight_mode=lightweight_mode,
            top_k=5,
        )
        st.session_state["innovation_active_job_id"] = innovation_job_id
        st.session_state["innovation_active_signature"] = innovation_signature
        st.session_state["innovation_cached_payload"] = None
        cached_innovation_payload = None

    active_innovation_job = None
    active_innovation_job_id = st.session_state.get("innovation_active_job_id")
    if active_innovation_job_id and st.session_state.get("innovation_active_signature") == innovation_signature:
        active_innovation_job = get_innovation_background_job(active_innovation_job_id)
        if active_innovation_job["status"] == "done":
            st.session_state["innovation_cached_payload"] = {
                "signature": innovation_signature,
                "payload": active_innovation_job.get("result"),
            }
            cached_innovation_payload = st.session_state["innovation_cached_payload"]
            discard_innovation_background_job(active_innovation_job_id)
            st.session_state["innovation_active_job_id"] = None
            st.session_state["innovation_active_signature"] = None
            active_innovation_job = None
        elif active_innovation_job["status"] in {"error", "cancelled", "missing"}:
            st.session_state["innovation_active_job_id"] = None
            st.session_state["innovation_active_signature"] = None

    innovation_payload = None
    if cached_innovation_payload and cached_innovation_payload.get("signature") == innovation_signature:
        innovation_payload = cached_innovation_payload.get("payload")
    elif active_innovation_job and active_innovation_job["status"] == "running":
        st.info(
            f"Innovation analysis is running in the background: {active_innovation_job.get('record_count', len(df))} records, "
            f"top_n={active_innovation_job.get('bc_topn_val', bc_topn_val)}, min_shared={active_innovation_job.get('bc_min_shared_val', bc_min_shared_val)}."
        )
        innovation_status_col1, innovation_status_col2 = st.columns(2)
        with innovation_status_col1:
            st.caption("Use `Refresh Innovation Status` to check whether the task is ready.")
        with innovation_status_col2:
            if st.button("Cancel Innovation Task", key="innovation_cancel"):
                discard_innovation_background_job(active_innovation_job_id)
                st.session_state["innovation_active_job_id"] = None
                st.session_state["innovation_active_signature"] = None
                st.info("Innovation analysis task cancelled.")
        if halt_if_missing:
            st.stop()
        return None
    elif active_innovation_job and active_innovation_job["status"] == "error":
        st.error(f"Innovation analysis failed: {active_innovation_job.get('error', 'Unknown error')}")
        if halt_if_missing:
            st.stop()
        return None
    elif innovation_payload is None:
        if refresh_innovation_requested:
            st.info("Innovation analysis is not running. Click `Run Innovation Metrics` to start.")
        else:
            st.info("Click `Run Innovation Metrics` to compute disruption, structural-hole, and robustness outputs.")
        if halt_if_missing:
            st.stop()
        return None

    df_di_report = innovation_payload["df_di_report"]
    structural_hole_frame_report = innovation_payload["structural_hole_frame_report"]
    structural_hole_summary_report = innovation_payload["structural_hole_summary_report"]
    baseline_comparison_snapshot = innovation_payload["baseline_comparison_snapshot"]
    robustness_snapshot = innovation_payload["robustness_snapshot"]
    innovation_report = innovation_payload["innovation_report"]
    robustness_report = innovation_payload["robustness_report"]
    baseline_comparison_report = innovation_payload["baseline_comparison_report"]
    innovation_snapshot = innovation_payload["innovation_snapshot"]

    def _local_download(profile, *args, **kwargs):
        return _render_export_download_button_base(
            profile,
            *args,
            record_count=len(df),
            lightweight_mode=lightweight_mode,
            scenario_count=len(robustness_top_n_values) * len(robustness_min_shared_values),
            sample_size=len(df_robustness),
            **kwargs,
        )

    innovation_metric_col1, innovation_metric_col2, innovation_metric_col3, innovation_metric_col4, innovation_metric_col5 = st.columns(5)
    with innovation_metric_col1:
        st.metric("Top Broker", structural_hole_summary_report.get("top_broker", "N/A") or "N/A")
    with innovation_metric_col2:
        st.metric("Top Brokerage", f"{structural_hole_summary_report.get('top_score', 0.0):.4f}")
    with innovation_metric_col3:
        st.metric("Core Brokers", structural_hole_summary_report.get("core_brokers", 0))
    with innovation_metric_col4:
        st.metric("Stable Brokers", robustness_snapshot.get("summary", {}).get("stable_broker_count", 0))
    with innovation_metric_col5:
        st.metric("Best Baseline", baseline_comparison_snapshot.get("summary", {}).get("best_aligned_baseline", "N/A") or "N/A")

    if df_di_report is not None and not df_di_report.empty and "Disruption_Index" in df_di_report.columns:
        st.markdown("#### Disruption Index")
        st.caption("This panel shows the distribution of DI scores and the highest and lowest values in the internal citation network.")
        di_values = pd.to_numeric(df_di_report["Disruption_Index"], errors="coerce").dropna()
        if not di_values.empty:
            di_style = render_plot_style_controls(
                "innovation_di",
                default_primary=SCIENTIFIC_COLORWAY[4],
                default_height=500,
            )
            fig_di = px.histogram(
                df_di_report,
                x="Disruption_Index",
                nbins=30,
                title="Disruption Index Distribution",
                labels={"Disruption_Index": "DI Score", "count": "Number of Papers"},
            )
            fig_di = apply_publication_style_with_overrides(fig_di, di_style)
            render_plotly_chart(fig_di, width="stretch")
            if show_export_downloads and get_plotly_static_export_status().get("available", False):
                _local_download(
                    "innovation",
                    label="Download Disruption Index Distribution",
                    data=plotly_figure_to_bytes(fig_di, bundle_format_value, width=2100, height=1485),
                    file_name=f"Biblio-HUB_disruption_index_distribution.{bundle_format_value}",
                    mime=figure_mime,
                )
        disruptive_cols = [c for c in ["Title", "Year", "Disruption_Index", "Support", "Internal_Citers", "DI_nd", "DI_nc", "DI_na"] if c in df_di_report.columns or c == "Support"]
        top_disruptive = rank_disruption_extremes(df_di_report, kind="disruptive", top_n=10)[disruptive_cols]
        top_consolidating = rank_disruption_extremes(df_di_report, kind="consolidating", top_n=10)[disruptive_cols]
        di_table_col1, di_table_col2 = st.columns(2)
        with di_table_col1:
            st.markdown("Highest DI Scores")
            st.dataframe(top_disruptive, width="stretch", hide_index=True)
        with di_table_col2:
            st.markdown("Lowest DI Scores")
            st.dataframe(top_consolidating, width="stretch", hide_index=True)
        if show_export_downloads:
            di_download_col1, di_download_col2 = st.columns(2)
            with di_download_col1:
                _local_download(
                    "innovation",
                    label="Download Highest DI Scores (CSV)",
                    data=top_disruptive.to_csv(index=False).encode("utf-8-sig"),
                    file_name=_csv_filename_from_title("Highest DI Scores"),
                    mime="text/csv",
                )
            with di_download_col2:
                _local_download(
                    "innovation",
                    label="Download Lowest DI Scores (CSV)",
                    data=top_consolidating.to_csv(index=False).encode("utf-8-sig"),
                    file_name=_csv_filename_from_title("Lowest DI Scores"),
                    mime="text/csv",
                )

    brokerage_profile_fig = render_structural_hole_brokerage_profile(
        structural_hole_frame_report,
        top_n=20,
        label_top_n=8,
    )
    if brokerage_profile_fig is not None:
        st.markdown("#### Structural-Hole Brokerage Profile")
        st.caption("This view highlights nodes that occupy open bridge positions between clusters, combining high betweenness, low structural constraint, and larger effective neighborhood reach.")
        sh_style = render_plot_style_controls(
            "innovation_structural_hole",
            default_primary=SCIENTIFIC_COLORWAY[2],
            default_secondary=SCIENTIFIC_COLORWAY[3],
            default_height=700,
            allow_color_controls=False,
            preserve_original_colors=True,
        )
        brokerage_profile_fig = apply_publication_style_with_overrides(brokerage_profile_fig, sh_style)
        render_plotly_chart(brokerage_profile_fig, width="stretch")
        if show_export_downloads:
            _render_plotly_multiformat_downloads(
                "innovation",
                brokerage_profile_fig,
                "Biblio-HUB_structural_hole_brokerage_profile",
                width=1800,
                height=1200,
                key_prefix="innovation_structural_hole_profile",
            )

    robustness_summary_fig = render_brokerage_robustness_summary(robustness_snapshot, top_stable_count=8)
    if robustness_summary_fig is not None:
        st.markdown("#### Brokerage Robustness Summary")
        st.caption("This summary tests whether the same bridge-like nodes remain visible when the bibliographic coupling network is rebuilt under alternative thresholds.")
        robustness_style = render_plot_style_controls(
            "innovation_robustness",
            default_primary=SCIENTIFIC_COLORWAY[0],
            default_secondary=SCIENTIFIC_COLORWAY[1],
            default_height=700,
            show_legend_default=True,
            allow_color_controls=False,
            preserve_original_colors=True,
        )
        robustness_summary_fig = apply_publication_style_with_overrides(robustness_summary_fig, robustness_style)
        render_plotly_chart(robustness_summary_fig, width="stretch")
        if show_export_downloads:
            _render_plotly_multiformat_downloads(
                "innovation",
                robustness_summary_fig,
                "Biblio-HUB_brokerage_robustness_summary",
                width=2000,
                height=1200,
                key_prefix="innovation_brokerage_robustness",
            )

    with st.expander("Preview Innovation Report", expanded=False):
        st.code(innovation_report, language="markdown")
    with st.expander("Preview Brokerage Robustness Report", expanded=False):
        st.code(robustness_report, language="markdown")
    with st.expander("Preview Brokerage Baseline Comparison", expanded=False):
        st.code(baseline_comparison_report, language="markdown")

    if show_export_downloads:
        st.markdown("#### Innovation Exports")
        export_col1, export_col2, export_col3, export_col4 = st.columns(4)
        with export_col1:
            _render_named_export_download(
                "Download Innovation Report (MD)",
                innovation_report,
                "Biblio-HUB_innovation_metrics_report.md",
                "text/markdown",
                speed_label="Innovation Report : Moderate",
                key="innovation_export_report",
            )
        with export_col2:
            _render_named_export_download(
                "Download Innovation Snapshot (JSON)",
                json.dumps(innovation_snapshot, indent=2, ensure_ascii=False),
                "Biblio-HUB_innovation_metrics_snapshot.json",
                "application/json",
                speed_label="Innovation Snapshot : Fast",
                key="innovation_export_snapshot",
            )
        with export_col3:
            _render_named_export_download(
                "Download Robustness Report (MD)",
                robustness_report,
                "Biblio-HUB_brokerage_robustness_report.md",
                "text/markdown",
                speed_label="Robustness Report : Slow",
                key="innovation_export_robustness",
            )
        with export_col4:
            _render_named_export_download(
                "Download Baseline Comparison (MD)",
                baseline_comparison_report,
                "Biblio-HUB_brokerage_baseline_comparison.md",
                "text/markdown",
                speed_label="Baseline Comparison : Moderate",
                key="innovation_export_baseline",
            )

    return innovation_payload


def render_export_center(df, keywords_list, keyword_freq, cooccurrence, dedup_report):
    st.title("Export Center")
    st.markdown("Export processed data, publication-ready figure bundles, and interoperable bibliometric files.")
    st.caption(
        "This export layer is designed as a submission-oriented workflow bridge that turns analytical outputs "
        "into manuscript-ready assets, rather than acting as a generic file downloader."
    )

    export_record_count = len(df)

    def _current_network_export_modes():
        return {
            "network_keyword_cooccurrence_static": st.session_state.get("kw_figure_mode", "Publication Figure"),
            "network_keyword_journal_static": st.session_state.get("kj_figure_mode", "Publication Figure"),
            "network_coauthorship_static": st.session_state.get("auth_figure_mode", "Publication Figure"),
            "network_institution_collaboration_static": st.session_state.get("inst_figure_mode", "Publication Figure"),
            "network_country_collaboration_static": st.session_state.get("country_figure_mode", "Publication Figure"),
            "network_bibliographic_coupling_static": st.session_state.get("bc_figure_mode", "Publication Figure"),
            "network_reference_cocitation_static": st.session_state.get("cocite_figure_mode", "Publication Figure"),
            "network_author_cocitation_static": st.session_state.get("auth_cocite_figure_mode", "Publication Figure"),
            "network_journal_cocitation_static": st.session_state.get("j_cocite_figure_mode", "Publication Figure"),
        }

    def _current_export_analysis_parameters(bundle_format_value=None):
        export_lightweight_mode_value = bool(
            st.session_state.get("export_lightweight_mode", export_record_count > 2000)
        )
        if export_lightweight_mode_value:
            export_robustness_sample_size = min(export_record_count, 3000) if export_record_count > 5000 else export_record_count
            export_robustness_scenario_count = 1
        elif export_record_count > 5000:
            export_robustness_sample_size = min(export_record_count, 5000)
            export_robustness_scenario_count = 4
        else:
            export_robustness_sample_size = export_record_count
            export_robustness_scenario_count = 9

        resolved_bundle_format = bundle_format_value or st.session_state.get(
            "publication_bundle_format",
            PUBLICATION_EXPORT_FORMATS[0].upper(),
        )
        resolved_bundle_format = str(resolved_bundle_format).lower()

        return [
            {"key": "keyword_source", "label": "Keyword Source", "value": st.session_state.get("keyword_source", "DE+ID"), "default": "DE+ID", "group": "Network"},
            {"key": "vos_topn", "label": "Keyword Network Top N", "value": st.session_state.get("vos_topn", 30), "default": 30, "group": "Network"},
            {"key": "vos_minw", "label": "Keyword Network Min Co-occurrence", "value": st.session_state.get("vos_minw", 2), "default": 2, "group": "Network"},
            {"key": "overview_top_journals_n", "label": "Overview Top Journals", "value": st.session_state.get("overview_top_journals_n", 30), "default": 30, "group": "Overview"},
            {"key": "kj_topn_kw", "label": "Keyword-Journal Top Keywords", "value": st.session_state.get("kj_topn_kw", 15), "default": 15, "group": "Network"},
            {"key": "kj_topn_jn", "label": "Keyword-Journal Top Journals", "value": st.session_state.get("kj_topn_jn", 10), "default": 10, "group": "Network"},
            {"key": "auth_min_papers", "label": "Author Collaboration Min Papers", "value": st.session_state.get("auth_min_papers", 2), "default": 2, "group": "Collaboration"},
            {"key": "auth_topn", "label": "Author Collaboration Max Authors", "value": st.session_state.get("auth_topn", 50), "default": 50, "group": "Collaboration"},
            {"key": "inst_topn", "label": "Institution Network Top N", "value": st.session_state.get("inst_topn", 25), "default": 25, "group": "Collaboration"},
            {"key": "bc_topn", "label": "Bibliographic Coupling Top Papers", "value": st.session_state.get("bc_topn", 30), "default": 30, "group": "Citation"},
            {"key": "bc_min_shared", "label": "Bibliographic Coupling Min Shared References", "value": st.session_state.get("bc_min_shared", 2), "default": 2, "group": "Citation"},
            {"key": "cocite_topn", "label": "Co-citation Top References", "value": st.session_state.get("cocite_topn", 20), "default": 20, "group": "Citation"},
            {"key": "cocite_minw", "label": "Co-citation Minimum Weight", "value": st.session_state.get("cocite_minw", 2), "default": 2, "group": "Citation"},
            {"key": "auth_cocite_topn", "label": "Author Co-citation Top Authors", "value": st.session_state.get("auth_cocite_topn", 20), "default": 20, "group": "Citation"},
            {"key": "auth_cocite_minw", "label": "Author Co-citation Minimum Weight", "value": st.session_state.get("auth_cocite_minw", 2), "default": 2, "group": "Citation"},
            {"key": "j_cocite_topn", "label": "Journal Co-citation Top Journals", "value": st.session_state.get("j_cocite_topn", 20), "default": 20, "group": "Citation"},
            {"key": "j_cocite_minw", "label": "Journal Co-citation Minimum Weight", "value": st.session_state.get("j_cocite_minw", 2), "default": 2, "group": "Citation"},
            {"key": "di_extremes_min_support", "label": "DI Extremes Min Support", "value": st.session_state.get("di_extremes_min_support", DEFAULT_DI_EXTREMES_MIN_SUPPORT), "default": DEFAULT_DI_EXTREMES_MIN_SUPPORT, "group": "Innovation"},
            {"key": "di_extremes_min_internal_citers", "label": "DI Extremes Min Internal Citers", "value": st.session_state.get("di_extremes_min_internal_citers", DEFAULT_DI_EXTREMES_MIN_INTERNAL_CITERS), "default": DEFAULT_DI_EXTREMES_MIN_INTERNAL_CITERS, "group": "Innovation"},
            {"key": "di_extremes_min_internal_references", "label": "DI Extremes Min Internal References", "value": st.session_state.get("di_extremes_min_internal_references", DEFAULT_DI_EXTREMES_MIN_INTERNAL_REFERENCES), "default": DEFAULT_DI_EXTREMES_MIN_INTERNAL_REFERENCES, "group": "Innovation"},
            {"key": "di_extremes_support_filter_mode", "label": "DI Extremes Support Rule", "value": st.session_state.get("di_extremes_support_filter_mode", "any"), "default": "any", "group": "Innovation"},
            {"key": "di_extremes_require_topic_match", "label": "DI Extremes Require Topic Match", "value": st.session_state.get("di_extremes_require_topic_match", False), "default": False, "group": "Innovation"},
            {"key": "cs_topn", "label": "Timeline Top Keywords", "value": st.session_state.get("cs_topn", 15), "default": 15, "group": "Temporal"},
            {"key": "burst_topn", "label": "Burst Detection Keyword Count", "value": st.session_state.get("burst_topn", 20), "default": 20, "group": "Temporal"},
            {"key": "publication_forecast_horizon", "label": "Publication Forecast Horizon", "value": st.session_state.get("publication_forecast_horizon", 4), "default": 4, "group": "Temporal"},
            {"key": "keyword_opportunity_topn", "label": "Keyword Opportunity Map Count", "value": st.session_state.get("keyword_opportunity_topn", 20), "default": 20, "group": "Temporal"},
            {"key": "entity_forecast_topn", "label": "Entity Forecast Leaders", "value": st.session_state.get("entity_forecast_topn", 10), "default": 10, "group": "Temporal"},
            {"key": "forward_leadership_shift_type", "label": "Leadership Shift Entity Type", "value": st.session_state.get("forward_leadership_shift_type", "Countries"), "default": "Countries", "group": "Temporal"},
            {"key": "leadership_shift_topn", "label": "Leadership Shift Entities", "value": st.session_state.get("leadership_shift_topn", 10), "default": 10, "group": "Temporal"},
            {"key": "theme_migration_slices", "label": "Theme Migration Time Slices", "value": st.session_state.get("theme_migration_slices", 4), "default": 4, "group": "Temporal"},
            {"key": "theme_migration_topn", "label": "Theme Migration Chains", "value": st.session_state.get("theme_migration_topn", 10), "default": 10, "group": "Temporal"},
            {"key": "bt_min_docs", "label": "BERTopic Minimum Documents", "value": st.session_state.get("bt_min_docs", 3), "default": 3, "group": "Semantic Topic Modeling"},
            {"key": "bt_topn", "label": "BERTopic Displayed Topics", "value": st.session_state.get("bt_topn", 10), "default": 10, "group": "Semantic Topic Modeling"},
            {"key": "bt_evo_topn", "label": "BERTopic Evolution Topics", "value": st.session_state.get("bt_evo_topn", 8), "default": 8, "group": "Semantic Topic Modeling"},
            {"key": "mat_topn", "label": "Matrix Top Keywords", "value": st.session_state.get("mat_topn", 15), "default": 15, "group": "Structure"},
            {"key": "tm_topn", "label": "Thematic Map Top Keywords", "value": st.session_state.get("tm_topn", 20), "default": 20, "group": "Structure"},
            {"key": "author_time_topn", "label": "Author Timeline Top Authors", "value": st.session_state.get("author_time_topn", 10), "default": 10, "group": "Structure"},
            {"key": "3f_auth", "label": "Three-Field Top Authors", "value": st.session_state.get("3f_auth", 10), "default": 10, "group": "Structure"},
            {"key": "3f_kw", "label": "Three-Field Top Keywords", "value": st.session_state.get("3f_kw", 15), "default": 15, "group": "Structure"},
            {"key": "3f_jn", "label": "Three-Field Top Journals", "value": st.session_state.get("3f_jn", 10), "default": 10, "group": "Structure"},
            {"key": "export_lightweight_mode", "label": "Export Lightweight Mode", "value": export_lightweight_mode_value, "default": False, "group": "Export"},
            {"key": "export_robustness_scenario_count", "label": "Robustness Scenario Count", "value": export_robustness_scenario_count, "default": 9, "group": "Export"},
            {"key": "export_robustness_sample_size", "label": "Robustness Analysis Sample Size", "value": export_robustness_sample_size, "default": len(df), "group": "Export"},
            {"key": "bundle_format", "label": "Publication Figure Format", "value": resolved_bundle_format, "default": PUBLICATION_EXPORT_FORMATS[0], "group": "Export"},
        ]

    def _build_zip_package(file_map):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_name, content in file_map.items():
                payload = content if isinstance(content, bytes) else str(content).encode("utf-8-sig")
                zf.writestr(file_name, payload)
        buffer.seek(0)
        return buffer.getvalue()

    top_country_df = build_top_country_table(df, top_n=50)
    top_institution_df = build_top_institution_table(df, top_n=50)
    top_journal_df = build_top_journal_table(df, top_n=50)
    quick_keyword_df = pd.DataFrame(
        keyword_freq.most_common(50),
        columns=["Keyword", "Frequency"],
    )
    key_tables_bundle = _build_zip_package(
        {
            "top_keywords.csv": quick_keyword_df.to_csv(index=False),
            "top_countries.csv": top_country_df.to_csv(index=False),
            "top_institutions.csv": top_institution_df.to_csv(index=False),
            "top_journals.csv": top_journal_df.to_csv(index=False),
        }
    )

    bib_entries = []
    for idx, row in df.iterrows():
        cite_key = str(row.get("UT", row.get("PMID", f"ref{idx}"))).strip()
        if cite_key == "nan" or not cite_key:
            cite_key = f"ref{idx}"
        entry = f"@article{{{cite_key},\n"
        title = str(row.get("Title", "")).strip()
        if title and title != "nan":
            entry += f'  title = {{{title}}},\n'
        authors = str(row.get("Authors", "")).strip()
        if authors and authors != "nan":
            entry += f'  author = {{{authors}}},\n'
        journal = str(row.get("Journal", "")).strip()
        if journal and journal != "nan":
            entry += f'  journal = {{{journal}}},\n'
        year = row.get("Year", "")
        if pd.notna(year):
            y = safe_year(year)
            if y is not None:
                entry += f'  year = {{{y}}},\n'
        doi = str(row.get("DOI", "")).strip()
        if doi and doi != "nan":
            entry += f'  doi = {{{doi}}},\n'
        vol = str(row.get("Volume", "")).strip()
        if vol and vol != "nan":
            entry += f'  volume = {{{vol}}},\n'
        issue = str(row.get("Issue", "")).strip()
        if issue and issue != "nan":
            entry += f'  number = {{{issue}}},\n'
        sp = str(row.get("Start_Page", "")).strip()
        ep = str(row.get("End_Page", "")).strip()
        if sp and sp != "nan":
            pages = sp if (not ep or ep == "nan") else f"{sp}-{ep}"
            entry += f'  pages = {{{pages}}},\n'
        entry += "}\n"
        bib_entries.append(entry)
    bib_content = "\n".join(bib_entries)

    ris_entries = []
    for _, row in df.iterrows():
        entry = "TY  - JOUR\n"
        title = str(row.get("Title", "")).strip()
        if title and title != "nan":
            entry += f"TI  - {title}\n"
        authors = str(row.get("Authors", "")).strip()
        if authors and authors != "nan":
            for au in authors.split(";"):
                au = au.strip()
                if au:
                    entry += f"AU  - {au}\n"
        abstract = str(row.get("Abstract", "")).strip()
        if abstract and abstract != "nan":
            entry += f"AB  - {abstract}\n"
        year = row.get("Year", "")
        if pd.notna(year):
            y = safe_year(year)
            if y is not None:
                entry += f"PY  - {y}\n"
        journal = str(row.get("Journal", "")).strip()
        if journal and journal != "nan":
            entry += f"JO  - {journal}\n"
        doi = str(row.get("DOI", "")).strip()
        if doi and doi != "nan":
            entry += f"DO  - {doi}\n"
        vol = str(row.get("Volume", "")).strip()
        if vol and vol != "nan":
            entry += f"VL  - {vol}\n"
        issue = str(row.get("Issue", "")).strip()
        if issue and issue != "nan":
            entry += f"IS  - {issue}\n"
        sp = str(row.get("Start_Page", "")).strip()
        if sp and sp != "nan":
            entry += f"BP  - {sp}\n"
        ep = str(row.get("End_Page", "")).strip()
        if ep and ep != "nan":
            entry += f"EP  - {ep}\n"
        entry += "ER  - \n"
        ris_entries.append(entry)
    ris_content = "\n".join(ris_entries)
    reference_bundle = _build_zip_package(
        {
            "references.bib": bib_content,
            "references.ris": ris_content,
        }
    )
    planned_lightweight_mode = bool(
        st.session_state.get("export_lightweight_mode", export_record_count > 2000)
    )
    if planned_lightweight_mode:
        planned_robustness_sample_size = min(export_record_count, 3000) if export_record_count > 5000 else export_record_count
        planned_robustness_scenario_count = 1
    elif export_record_count > 5000:
        planned_robustness_sample_size = min(export_record_count, 5000)
        planned_robustness_scenario_count = 4
    else:
        planned_robustness_sample_size = export_record_count
        planned_robustness_scenario_count = 9

    st.markdown("## Quick Export")
    st.caption("Export current figures, core tables, and reference files first. This layer is optimized for fast output of the current analytical state.")
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)
    with quick_col1:
        st.markdown("**Master Package**")
        st.caption("Bundle figures and tables in one ZIP.")
    with quick_col2:
        st.markdown("**All Figures**")
        st.caption("Generate a publication-ready figure bundle.")
    with quick_col3:
        st.markdown("**Key Tables**")
        st.caption("Export hotspot and entity summary tables.")
    with quick_col4:
        st.markdown("**References**")
        st.caption("Export BibTeX and RIS together.")
    quick_download_col1, quick_download_col2 = st.columns(2)
    with quick_download_col1:
        _render_export_download_button_base(
            "report",
            label="Download Key Tables (ZIP)",
            data=key_tables_bundle,
            file_name="Biblio-HUB_key_tables.zip",
            mime="application/zip",
            key="quick_key_tables_zip",
            record_count=export_record_count,
            lightweight_mode=planned_lightweight_mode,
            scenario_count=planned_robustness_scenario_count,
            sample_size=planned_robustness_sample_size,
        )
    with quick_download_col2:
        _render_export_download_button_base(
            "report",
            label="Download References (ZIP)",
            data=reference_bundle,
            file_name="Biblio-HUB_references_bundle.zip",
            mime="application/zip",
            key="quick_reference_bundle_zip",
            record_count=export_record_count,
            lightweight_mode=planned_lightweight_mode,
            scenario_count=planned_robustness_scenario_count,
            sample_size=planned_robustness_sample_size,
        )

    st.markdown("### \U0001F4E6 Master One-Click Export")
    st.info("Bundle ALL generated figures (PNG/SVG/PDF) and ALL processed data tables (CSV) into a single master ZIP file.")
    
    master_job_id = st.session_state.get("master_export_job_id")
    if not master_job_id:
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            m_format = st.selectbox("Figure Format", ["PNG", "SVG", "PDF"], key="master_fmt")
        with col_m2:
            st.write("") # Spacer
            st.write("") # Spacer
            if st.button("\U0001F680 Export Everything (All Charts + Tables)", key="btn_master_export", use_container_width=True):
                job_id = submit_export_background_job(
                    'master_export',
                    df,
                    keywords_list,
                    keyword_freq,
                    cooccurrence,
                    export_format=m_format.lower(),
                    network_export_modes=_current_network_export_modes(),
                    analysis_parameters=_current_export_analysis_parameters(m_format.lower()),
                )
                st.session_state["master_export_job_id"] = job_id
                st.rerun()
    else:
        status = get_export_job_status(master_job_id)
        if status["status"] == "not_found":
            st.warning("The previous master export task is no longer available. Please start a new export.")
            if st.button("Start New Master Export", key="reset_missing_master", use_container_width=True):
                st.session_state["master_export_job_id"] = None
                st.rerun()
        elif status["status"] == "running":
            st.warning("Master bundle is being generated in the background... This may take a moment for large datasets.")
            if st.button("\U0001F504 Refresh Status", key="refresh_master"):
                st.rerun()
        elif status["status"] == "completed":
            st.success("Master bundle generated successfully!")
            st.download_button(
                label="\U0001F4E5 Download Master ZIP Package",
                data=status["result"],
                file_name="biblio_hub_master_export.zip",
                mime="application/zip",
                key="dl_master_export",
                use_container_width=True
            )
            if st.button("\U0001F9F9 Clear / Generate New", key="clear_master"):
                discard_export_job(master_job_id)
                st.session_state["master_export_job_id"] = None
                st.rerun()
        elif status["status"] == "failed":
            st.error(f"\u274C Export failed: {status['error']}")
            if st.button("Try Again", key="fail_master"):
                discard_export_job(master_job_id)
                st.session_state["master_export_job_id"] = None
                st.rerun()

    st.markdown("---")
    def _get_export_time_hint(profile):
        return _get_export_time_hint_base(
            profile,
            record_count=export_record_count,
            lightweight_mode=planned_lightweight_mode,
            scenario_count=planned_robustness_scenario_count,
            sample_size=planned_robustness_sample_size,
        )

    def _render_export_download_button(profile, *args, **kwargs):
        return _render_export_download_button_base(
            profile,
            *args,
            record_count=export_record_count,
            lightweight_mode=planned_lightweight_mode,
            scenario_count=planned_robustness_scenario_count,
            sample_size=planned_robustness_sample_size,
            **kwargs,
        )
    
    manuscript_snapshot = build_manuscript_case_snapshot(
        df,
        keyword_freq,
        cooccurrence,
        dedup_report=dedup_report,
        top_n=10,
    )
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    with overview_col1:
        st.metric("Records", manuscript_snapshot["records"])
    with overview_col2:
        year_range_label = (
            f"{manuscript_snapshot['year_range'][0]}-{manuscript_snapshot['year_range'][1]}"
            if manuscript_snapshot["year_range"]
            else "N/A"
        )
        st.metric("Year Range", year_range_label)
    with overview_col3:
        st.metric("DOI Coverage", f"{manuscript_snapshot['doi_coverage']:.1%}")
    with overview_col4:
        st.metric("Abstract Coverage", f"{manuscript_snapshot['abstract_coverage']:.1%}")
    manuscript_report = build_manuscript_case_report(
        df,
        keyword_freq,
        cooccurrence,
        dedup_report=dedup_report,
        top_n=10,
    )
    manuscript_submission_snapshot = build_manuscript_submission_snapshot(
        df,
        keyword_freq,
        cooccurrence,
        dedup_report=dedup_report,
        top_n=10,
    )
    manuscript_submission_report = build_manuscript_submission_report(
        df,
        keyword_freq,
        cooccurrence,
        dedup_report=dedup_report,
        top_n=10,
    )
    manuscript_case_bundle = build_manuscript_case_bundle(
        manuscript_report=manuscript_report,
        manuscript_snapshot=manuscript_snapshot,
        manuscript_submission_report=manuscript_submission_report,
        manuscript_submission_snapshot=manuscript_submission_snapshot,
    )

    bundle_format_value = str(
        st.session_state.get("publication_bundle_format", PUBLICATION_EXPORT_FORMATS[0].upper())
    ).lower()
    figure_mime = {
        "png": "image/png",
        "svg": "image/svg+xml",
        "pdf": "application/pdf",
    }.get(bundle_format_value, "application/octet-stream")
    journal_main_text_options = ["balanced", "compact", "evidence_dense"]
    journal_supplement_options = ["standard", "supplement_heavy", "minimal"]
    journal_review_options = ["standard", "reviewer_friendly", "revision_ready"]
    journal_article_options = ["full_article", "short_article", "rapid_communication"]
    selected_journal_preferences = {
        "main_text_policy": st.session_state.get("journal_main_text_policy", "balanced"),
        "supplement_policy": st.session_state.get("journal_supplement_policy", "standard"),
        "review_intensity": st.session_state.get("journal_review_intensity", "standard"),
        "article_format": st.session_state.get("journal_article_format", "full_article"),
    }
    selected_journal_template = build_parameterized_journal_template(selected_journal_preferences)
    export_lightweight_default = export_record_count > 2000
    export_lightweight_mode_value = bool(
        st.session_state.get("export_lightweight_mode", export_lightweight_default)
    )
    if export_lightweight_mode_value:
        export_robustness_sample_size = min(export_record_count, 3000) if export_record_count > 5000 else export_record_count
        export_robustness_scenario_count = 1
    elif export_record_count > 5000:
        export_robustness_sample_size = min(export_record_count, 5000)
        export_robustness_scenario_count = 4
    else:
        export_robustness_sample_size = export_record_count
        export_robustness_scenario_count = 9
    planned_execution_policy = {
        "lightweight_mode": bool(export_lightweight_mode_value),
        "triggered_by_large_dataset": bool(export_record_count > 2000),
        "full_record_count": int(export_record_count),
        "analysis_record_count": int(export_robustness_sample_size),
        "downsampled": bool(export_robustness_sample_size < export_record_count),
        "downsample_threshold": 5000,
        "downsample_cap": 3000 if export_lightweight_mode_value else (5000 if export_record_count > 5000 else export_record_count),
        "scenario_count_requested": int(export_robustness_scenario_count),
    }
    analysis_parameters = _current_export_analysis_parameters(bundle_format_value)
    st.markdown("---")
    st.markdown("## Publication Support")
    st.caption("Prepare reproducibility records, figure explanation materials, and submission-ready manuscript files after the core exports are ready.")
    reproducibility_snapshot = build_reproducibility_snapshot(
        df,
        analysis_parameters=analysis_parameters,
        dedup_report=dedup_report,
        keyword_freq=keyword_freq,
        cooccurrence=cooccurrence,
        journal_preferences=selected_journal_preferences,
        recommended_template=selected_journal_template,
        execution_policy=planned_execution_policy,
    )
    reproducibility_report = build_reproducibility_report(
        df,
        analysis_parameters=analysis_parameters,
        dedup_report=dedup_report,
        keyword_freq=keyword_freq,
        cooccurrence=cooccurrence,
        journal_preferences=selected_journal_preferences,
        recommended_template=selected_journal_template,
        execution_policy=planned_execution_policy,
    )
    modified_parameter_count = sum(
        1 for item in reproducibility_snapshot["analysis_parameters"] if item.get("changed", False)
    )
    st.markdown("### Reproducibility Checklist")
    st.caption("Capture metadata coverage, active thresholds, and algorithm profile for manuscript methods, appendices, and supplementary material.")
    repro_col1, repro_col2, repro_col3, repro_col4 = st.columns(4)
    with repro_col1:
        st.metric("Tracked Fields", len(reproducibility_snapshot["field_coverage"]))
    with repro_col2:
        st.metric("Modified Parameters", modified_parameter_count)
    with repro_col3:
        st.metric("Unique Keywords", reproducibility_snapshot["keyword_statistics"]["unique_keywords"])
    with repro_col4:
        st.metric("Co-occurrence Pairs", reproducibility_snapshot["keyword_statistics"]["cooccurrence_pairs"])
    st.caption(
        "Modified parameters: "
        + (
            ", ".join(
                item["label"]
                for item in reproducibility_snapshot["analysis_parameters"]
                if item.get("changed", False)
            )
            or "None"
        )
    )
    repro_download_col1, repro_download_col2 = st.columns(2)
    with repro_download_col1:
        _render_export_download_button(
            "report",
            label="Download Reproducibility Report (MD)",
            data=reproducibility_report,
            file_name="Biblio-HUB_reproducibility_report.md",
            mime="text/markdown",
        )
    with repro_download_col2:
        _render_export_download_button(
            "report",
            label="Download Reproducibility Snapshot (JSON)",
            data=json.dumps(reproducibility_snapshot, indent=2, ensure_ascii=False),
            file_name="Biblio-HUB_reproducibility_snapshot.json",
            mime="application/json",
        )
    with st.expander("Preview Reproducibility Checklist", expanded=False):
        st.code(reproducibility_report, language="markdown")

    methods_bundle = build_methods_package_bundle(reproducibility_snapshot)
    _render_export_download_button(
        "methods",
        label="Download Methods Package (.zip)",
        data=methods_bundle,
        file_name="Biblio-HUB_methods_package.zip",
        mime="application/zip",
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### \U0001F9EA Comparative Experiment Framework")
    st.caption("Benchmark current results against industry standards and generate network science metrics for manuscript validation.")
    
    # Build a temporary graph for metrics
    from modules.experiment_framework import build_graph_from_cooccurrence, calculate_network_metrics, build_experiment_comparison_report, get_baseline_comparison_data
    exp_graph = build_graph_from_cooccurrence(cooccurrence, top_n=st.session_state.get("vos_topn", 30))
    exp_metrics = calculate_network_metrics(exp_graph)
    
    exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
    with exp_col1:
        st.metric("Density", f"{exp_metrics['density']:.4f}")
    with exp_col2:
        st.metric("Modularity (Q)", f"{exp_metrics['modularity']:.4f}")
    with exp_col3:
        st.metric("Thematic Clusters", exp_metrics["clusters"])
    with exp_col4:
        st.metric("Avg Clustering", f"{exp_metrics['avg_clustering_coeff']:.4f}")
        
    exp_report = build_experiment_comparison_report(df, exp_graph, analysis_parameters)
    
    exp_down1, exp_down2 = st.columns(2)
    with exp_down1:
        _render_export_download_button(
            "report",
            label="Download Experiment Report (MD)",
            data=exp_report,
            file_name="Biblio-HUB_experiment_report.md",
            mime="text/markdown",
        )
    with exp_down2:
        _render_export_download_button(
            "report",
            label="Download Metrics Snapshot (JSON)",
            data=json.dumps(exp_metrics, indent=2),
            file_name="Biblio-HUB_experiment_metrics.json",
            mime="application/json",
        )
    
    with st.expander("View Baseline Comparison Table", expanded=False):
        baseline_df = pd.DataFrame(get_baseline_comparison_data())
        st.table(baseline_df)
    
    with st.expander("Preview Experiment Report", expanded=False):
        st.code(exp_report, language="markdown")

    st.markdown("---")
    st.markdown("### \U0001F4A1 Innovation Metrics")
    st.caption("Run disruption, structural-hole, and robustness analysis for the current dataset.")

    is_large_dataset = len(df) > 2000
    if is_large_dataset:
        st.warning(f"Large dataset detected ({len(df)} records). Innovation metrics and robustness analysis may take some time.")
        if "export_lightweight_mode" not in st.session_state:
            st.session_state["export_lightweight_mode"] = True
        lightweight_mode = st.checkbox(
            "Enable Lightweight Mode (Recommended for large datasets)",
            key="export_lightweight_mode",
            help="Reduces the number of robustness scenarios to speed up export.",
        )
    else:
        st.session_state["export_lightweight_mode"] = False
        lightweight_mode = False

    bc_topn_val = st.session_state.get("bc_topn", 30)
    bc_min_shared_val = st.session_state.get("bc_min_shared", 2)
    
    if lightweight_mode:
        robustness_top_n_values = [bc_topn_val]
        robustness_min_shared_values = [bc_min_shared_val]
        if len(df) > 5000:
            df_robustness = df.sample(n=3000, random_state=42)
        else:
            df_robustness = df
    elif len(df) > 5000:
        robustness_top_n_values = sorted({max(10, bc_topn_val - 5), bc_topn_val})
        robustness_min_shared_values = sorted({bc_min_shared_val, bc_min_shared_val + 1})
        df_robustness = df.sample(n=5000, random_state=42)
    else:
        robustness_top_n_values = sorted({max(10, bc_topn_val - 10), bc_topn_val, bc_topn_val + 10})
        robustness_min_shared_values = sorted({max(1, bc_min_shared_val - 1), bc_min_shared_val, bc_min_shared_val + 1})
        df_robustness = df
    robustness_scenario_count = len(robustness_top_n_values) * len(robustness_min_shared_values)
    current_execution_policy = {
        "lightweight_mode": bool(lightweight_mode),
        "full_record_count": int(len(df)),
        "analysis_record_count": int(len(df_robustness)),
        "downsampled": bool(len(df_robustness) < len(df)),
        "scenario_count_requested": int(robustness_scenario_count),
    }
    st.caption(
        f"Current robustness plan: {format_execution_policy_summary(current_execution_policy)}."
    )
    innovation_signature = (
        len(df),
        bc_topn_val,
        bc_min_shared_val,
        bool(lightweight_mode),
        tuple((item.get("key"), item.get("value")) for item in analysis_parameters),
    )
    innovation_action_col1, innovation_action_col2 = st.columns(2)
    with innovation_action_col1:
        generate_innovation_requested = st.button("Run Innovation Metrics", key="innovation_generate")
    with innovation_action_col2:
        refresh_innovation_requested = st.button("Refresh Innovation Status", key="innovation_refresh")

    cached_innovation_payload = st.session_state.get("innovation_cached_payload")
    if generate_innovation_requested:
        innovation_job_id = submit_innovation_background_job(
            df,
            analysis_parameters,
            bc_topn_val=bc_topn_val,
            bc_min_shared_val=bc_min_shared_val,
            lightweight_mode=lightweight_mode,
            top_k=5,
        )
        st.session_state["innovation_active_job_id"] = innovation_job_id
        st.session_state["innovation_active_signature"] = innovation_signature
        st.session_state["innovation_cached_payload"] = None
        cached_innovation_payload = None

    active_innovation_job = None
    active_innovation_job_id = st.session_state.get("innovation_active_job_id")
    if active_innovation_job_id and st.session_state.get("innovation_active_signature") == innovation_signature:
        active_innovation_job = get_innovation_background_job(active_innovation_job_id)
        if active_innovation_job["status"] == "done":
            st.session_state["innovation_cached_payload"] = {
                "signature": innovation_signature,
                "payload": active_innovation_job.get("result"),
            }
            cached_innovation_payload = st.session_state["innovation_cached_payload"]
            discard_innovation_background_job(active_innovation_job_id)
            st.session_state["innovation_active_job_id"] = None
            st.session_state["innovation_active_signature"] = None
            active_innovation_job = None
        elif active_innovation_job["status"] in {"error", "cancelled", "missing"}:
            st.session_state["innovation_active_job_id"] = None
            st.session_state["innovation_active_signature"] = None

    innovation_payload = None
    if cached_innovation_payload and cached_innovation_payload.get("signature") == innovation_signature:
        innovation_payload = cached_innovation_payload.get("payload")
    elif active_innovation_job and active_innovation_job["status"] == "running":
        st.info(
            f"Innovation analysis is running in the background: {active_innovation_job.get('record_count', len(df))} records, "
            f"top_n={active_innovation_job.get('bc_topn_val', bc_topn_val)}, min_shared={active_innovation_job.get('bc_min_shared_val', bc_min_shared_val)}."
        )
        innovation_status_col1, innovation_status_col2 = st.columns(2)
        with innovation_status_col1:
            st.caption("Use `Refresh Innovation Status` to check whether the report is ready.")
        with innovation_status_col2:
            if st.button("Cancel Innovation Task", key="innovation_cancel"):
                discard_innovation_background_job(active_innovation_job_id)
                st.session_state["innovation_active_job_id"] = None
                st.session_state["innovation_active_signature"] = None
                st.info("Innovation analysis task cancelled.")
        st.stop()
    elif active_innovation_job and active_innovation_job["status"] == "error":
        st.error(f"Innovation analysis failed: {active_innovation_job.get('error', 'Unknown error')}")
        st.stop()
    else:
        if refresh_innovation_requested:
            st.info("Innovation analysis is not running. Click `Run Innovation Metrics` to start.")
        else:
            st.info("Click `Run Innovation Metrics` to compute the current innovation-analysis outputs.")
        st.stop()

    G_bc_report = innovation_payload["G_bc_report"]
    bc_pairs_report = innovation_payload["bc_pairs_report"]
    bc_top_papers_report = innovation_payload["bc_top_papers_report"]
    df_di_report = innovation_payload["df_di_report"]
    structural_hole_frame_report = innovation_payload["structural_hole_frame_report"]
    structural_hole_summary_report = innovation_payload["structural_hole_summary_report"]
    baseline_comparison_snapshot = innovation_payload["baseline_comparison_snapshot"]
    baseline_comparison_report = innovation_payload["baseline_comparison_report"]
    robustness_snapshot = innovation_payload["robustness_snapshot"]
    robustness_report = innovation_payload["robustness_report"]
    innovation_report = innovation_payload["innovation_report"]
    innovation_snapshot = innovation_payload["innovation_snapshot"]

    innovation_col1, innovation_col2, innovation_col3, innovation_col4 = st.columns(4)
    with innovation_col1:
        _render_export_download_button(
            "innovation",
            label="Download Innovation Report (MD)",
            data=innovation_report,
            file_name="Biblio-HUB_innovation_metrics_report.md",
            mime="text/markdown",
        )
    with innovation_col2:
        _render_export_download_button(
            "innovation",
            label="Download Innovation Snapshot (JSON)",
            data=json.dumps(innovation_snapshot, indent=2, ensure_ascii=False),
            file_name="Biblio-HUB_innovation_metrics_snapshot.json",
            mime="application/json",
        )
    with innovation_col3:
        _render_export_download_button(
            "innovation",
            label="Download Robustness Report (MD)",
            data=robustness_report,
            file_name="Biblio-HUB_brokerage_robustness_report.md",
            mime="text/markdown",
        )
    with innovation_col4:
        _render_export_download_button(
            "innovation",
            label="Download Baseline Comparison (MD)",
            data=baseline_comparison_report,
            file_name="Biblio-HUB_brokerage_baseline_comparison.md",
            mime="text/markdown",
        )
    innovation_metric_col1, innovation_metric_col2, innovation_metric_col3, innovation_metric_col4, innovation_metric_col5 = st.columns(5)
    with innovation_metric_col1:
        st.metric("Top Broker", structural_hole_summary_report.get("top_broker", "N/A") or "N/A")
    with innovation_metric_col2:
        st.metric("Top Brokerage", f"{structural_hole_summary_report.get('top_score', 0.0):.4f}")
    with innovation_metric_col3:
        st.metric("Core Brokers", structural_hole_summary_report.get("core_brokers", 0))
    with innovation_metric_col4:
        st.metric("Stable Brokers", robustness_snapshot.get("summary", {}).get("stable_broker_count", 0))
    with innovation_metric_col5:
        st.metric(
            "Best Baseline",
            baseline_comparison_snapshot.get("summary", {}).get("best_aligned_baseline", "N/A") or "N/A"
        )
    brokerage_profile_fig = render_structural_hole_brokerage_profile(
        structural_hole_frame_report,
        top_n=20,
        label_top_n=8,
    )
    robustness_summary_fig = render_brokerage_robustness_summary(
        robustness_snapshot,
        top_stable_count=8,
    )
    if brokerage_profile_fig is not None or robustness_summary_fig is not None:
        st.markdown("### Publication-Ready Innovation Figures")
        innovation_fig_col1, innovation_fig_col2 = st.columns(2)
        if brokerage_profile_fig is not None:
            with innovation_fig_col1:
                render_plotly_chart(brokerage_profile_fig, width="stretch")
                _render_plotly_multiformat_downloads(
                    "innovation",
                    brokerage_profile_fig,
                    "Biblio-HUB_structural_hole_brokerage_profile",
                    width=1800,
                    height=1200,
                    key_prefix="export_center_structural_hole_profile",
                )
        if robustness_summary_fig is not None:
            with innovation_fig_col2:
                render_plotly_chart(robustness_summary_fig, width="stretch")
                _render_plotly_multiformat_downloads(
                    "innovation",
                    robustness_summary_fig,
                    "Biblio-HUB_brokerage_robustness_summary",
                    width=2000,
                    height=1200,
                    key_prefix="export_center_brokerage_robustness",
                )
    with st.expander("Preview Innovation Report", expanded=False):
        st.code(innovation_report, language="markdown")
    with st.expander("Preview Brokerage Robustness Report", expanded=False):
        st.code(robustness_report, language="markdown")
    with st.expander("Preview Brokerage Baseline Comparison", expanded=False):
        st.code(baseline_comparison_report, language="markdown")

    st.markdown("---")
    st.markdown("### Submission Result Package")
    st.caption("Download a manuscript-oriented ZIP package containing result report, structured snapshot, reviewer tables, and caption templates for direct submission preparation.")

    submission_snapshot = build_submission_result_snapshot(
        df,
        G_bc_report,
        bc_pairs_report,
        bc_top_papers_report,
        df_di_report,
        analysis_parameters,
        robustness_snapshot=robustness_snapshot,
        baseline_comparison_snapshot=baseline_comparison_snapshot,
        journal_preferences=selected_journal_preferences,
    )
    submission_report = build_submission_result_report(submission_snapshot)
    submission_col1, submission_col2, submission_col3 = st.columns(3)
    with submission_col1:
        st.metric("BC Nodes", submission_snapshot["innovation_metrics"]["bibliographic_coupling"]["network_metrics"].get("nodes", 0))
    with submission_col2:
        st.metric("Mean DI", f"{submission_snapshot['innovation_metrics']['disruption_index']['summary'].get('mean_di', 0.0):.4f}")
    with submission_col3:
        st.metric("Changed Params", len(submission_snapshot["changed_parameters"]))

    submission_bundle = build_submission_result_bundle(
        submission_report=submission_report,
        innovation_report=innovation_report,
        robustness_report=robustness_report,
        baseline_comparison_report=baseline_comparison_report,
        submission_snapshot=submission_snapshot,
        innovation_snapshot=innovation_snapshot,
        bc_pairs_report=bc_pairs_report,
        bc_top_papers_report=bc_top_papers_report,
        robustness_snapshot=robustness_snapshot,
        baseline_comparison_snapshot=baseline_comparison_snapshot,
        df_di_report=df_di_report,
    )
    submission_download_col1, submission_download_col2 = st.columns(2)
    with submission_download_col1:
        _render_export_download_button(
            "submission",
            label="Download Submission Result Package (ZIP)",
            data=submission_bundle,
            file_name="Biblio-HUB_submission_result_package.zip",
            mime="application/zip",
        )
    with submission_download_col2:
        _render_export_download_button(
            "submission",
            label="Download Submission Report (MD)",
            data=submission_report,
            file_name="Biblio-HUB_submission_result_package.md",
            mime="text/markdown",
        )
    with st.expander("Preview Submission Result Package Report", expanded=False):
        st.code(submission_report, language="markdown")

    st.markdown("---")
    st.markdown("### Figure Explanation Package")
    st.caption("Export figure/table captions, methods notes, reviewer notes, and result-mapping sheets for manuscript assembly and revision rounds.")

    figure_package_snapshot = build_submission_figure_package_snapshot(
        submission_snapshot,
        image_format=bundle_format_value,
    )
    figure_package_report = build_submission_figure_package_report(
        submission_snapshot,
        figure_package_snapshot,
    )
    figure_pkg_col1, figure_pkg_col2, figure_pkg_col3 = st.columns(3)
    with figure_pkg_col1:
        st.metric("Figure Notes", len(figure_package_snapshot["figure_items"]))
    with figure_pkg_col2:
        st.metric("Table Notes", len(figure_package_snapshot["table_items"]))
    with figure_pkg_col3:
        st.metric("Image Format", figure_package_snapshot["image_format"].upper())

    figure_package_bundle = build_figure_explanation_bundle(
        figure_package_report=figure_package_report,
        figure_package_snapshot=figure_package_snapshot,
    )
    figure_download_col1, figure_download_col2 = st.columns(2)
    with figure_download_col1:
        _render_export_download_button(
            "figure_package",
            label="Download Figure Explanation Package (ZIP)",
            data=figure_package_bundle,
            file_name="Biblio-HUB_figure_explanation_package.zip",
            mime="application/zip",
        )
    with figure_download_col2:
        _render_export_download_button(
            "figure_package",
            label="Download Figure Explanation Report (MD)",
            data=figure_package_report,
            file_name="Biblio-HUB_figure_explanation_package.md",
            mime="text/markdown",
        )
    with st.expander("Preview Figure Explanation Package", expanded=False):
        st.code(figure_package_report, language="markdown")

    st.markdown("---")
    with st.expander("Advanced Packages", expanded=False):
        st.caption("Use this layer for manuscript drafting, reviewer preparation, journal-specific packaging, and full narrative reporting after the main export and publication-support materials are ready.")

        st.markdown("### Manuscript Case Report")
        preview_top_keywords = ", ".join(
            item["label"] for item in manuscript_snapshot["top_keywords"][:5]
        ) or "N/A"
        overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
        with overview_col1:
            st.metric("Records", manuscript_snapshot["records"])
        with overview_col2:
            year_range_label = (
                f"{manuscript_snapshot['year_range'][0]}-{manuscript_snapshot['year_range'][1]}"
                if manuscript_snapshot["year_range"]
                else "N/A"
            )
            st.metric("Year Range", year_range_label)
        with overview_col3:
            st.metric("DOI Coverage", f"{manuscript_snapshot['doi_coverage']:.1%}")
        with overview_col4:
            st.metric("Abstract Coverage", f"{manuscript_snapshot['abstract_coverage']:.1%}")
        st.caption(f"Top keywords snapshot: {preview_top_keywords}")
        report_col1, report_col2 = st.columns(2)
        with report_col1:
            _render_export_download_button(
                "report",
                label="Download Manuscript Report (MD)",
                data=manuscript_report,
                file_name="Biblio-HUB_manuscript_case_report.md",
                mime="text/markdown",
            )
        with report_col2:
            _render_export_download_button(
                "report",
                label="Download Manuscript Snapshot (JSON)",
                data=json.dumps(manuscript_snapshot, indent=2, ensure_ascii=False),
                file_name="Biblio-HUB_manuscript_case_snapshot.json",
                mime="application/json",
            )
        with st.expander("Preview Manuscript Report", expanded=False):
            st.code(manuscript_report, language="markdown")

        st.markdown("### Submission-Oriented Case Package")
        submission_case_col1, submission_case_col2, submission_case_col3 = st.columns(3)
        with submission_case_col1:
            st.metric("Submission Highlights", len(manuscript_submission_snapshot["result_highlights"]))
        with submission_case_col2:
            st.metric("Recommended Tables", len(manuscript_submission_snapshot["recommended_tables"]))
        with submission_case_col3:
            st.metric("Reusable CSV Tables", len(manuscript_submission_snapshot["export_tables"]))
        st.caption(
            "Structured abstract draft: "
            + manuscript_submission_snapshot["structured_abstract"]["results"]
        )
        case_download_col1, case_download_col2, case_download_col3 = st.columns(3)
        with case_download_col1:
            _render_export_download_button(
                "case_package",
                label="Download Submission Case Package (ZIP)",
                data=manuscript_case_bundle,
                file_name="Biblio-HUB_submission_case_package.zip",
                mime="application/zip",
            )
        with case_download_col2:
            _render_export_download_button(
                "report",
                label="Download Submission Case Report (MD)",
                data=manuscript_submission_report,
                file_name="Biblio-HUB_submission_case_package.md",
                mime="text/markdown",
            )
        with case_download_col3:
            _render_export_download_button(
                "report",
                label="Download Submission Case Snapshot (JSON)",
                data=json.dumps(manuscript_submission_snapshot, indent=2, ensure_ascii=False),
                file_name="Biblio-HUB_submission_case_snapshot.json",
                mime="application/json",
            )
        with st.expander("Preview Submission-Oriented Case Package", expanded=False):
            st.code(manuscript_submission_report, language="markdown")

        st.markdown("### Reviewer Response Package")
        st.caption("Export structured reviewer-response materials including innovation claims, anticipated questions, reproducibility notes, evidence mapping, and stated limitations.")
        reviewer_snapshot = build_reviewer_response_snapshot(
            submission_snapshot,
            figure_package_snapshot,
        )
        reviewer_report = build_reviewer_response_report(
            submission_snapshot,
            figure_package_snapshot,
            reviewer_snapshot,
        )
        reviewer_col1, reviewer_col2, reviewer_col3 = st.columns(3)
        with reviewer_col1:
            st.metric("Innovation Claims", len(reviewer_snapshot["innovation_claims"]))
        with reviewer_col2:
            st.metric("Reviewer Questions", len(reviewer_snapshot["anticipated_questions"]))
        with reviewer_col3:
            st.metric("Evidence Links", len(reviewer_snapshot["evidence_mapping"]))
        reviewer_bundle = build_reviewer_response_bundle(
            reviewer_report=reviewer_report,
            reviewer_snapshot=reviewer_snapshot,
        )
        reviewer_download_col1, reviewer_download_col2 = st.columns(2)
        with reviewer_download_col1:
            _render_export_download_button(
                "reviewer",
                label="Download Reviewer Response Package (ZIP)",
                data=reviewer_bundle,
                file_name="Biblio-HUB_reviewer_response_package.zip",
                mime="application/zip",
            )
        with reviewer_download_col2:
            _render_export_download_button(
                "reviewer",
                label="Download Reviewer Response Report (MD)",
                data=reviewer_report,
                file_name="Biblio-HUB_reviewer_response_package.md",
                mime="text/markdown",
            )
        with st.expander("Preview Reviewer Response Package", expanded=False):
            st.code(reviewer_report, language="markdown")

        st.markdown("### Journal Submission Version Package")
        st.caption("Reorganize current outputs into a journal-ready package with separate folders for main manuscript materials, supplementary files, and reviewer appendix assets.")
        journal_pref_col1, journal_pref_col2, journal_pref_col3, journal_pref_col4 = st.columns(4)
        with journal_pref_col1:
            journal_main_text_policy = st.selectbox(
                "Main Text Policy",
                options=journal_main_text_options,
                index=journal_main_text_options.index(selected_journal_preferences["main_text_policy"]),
                format_func=lambda value: value.replace("_", " ").title(),
                key="journal_main_text_policy",
            )
        with journal_pref_col2:
            journal_supplement_policy = st.selectbox(
                "Supplement Policy",
                options=journal_supplement_options,
                index=journal_supplement_options.index(selected_journal_preferences["supplement_policy"]),
                format_func=lambda value: value.replace("_", " ").title(),
                key="journal_supplement_policy",
            )
        with journal_pref_col3:
            journal_review_intensity = st.selectbox(
                "Review Intensity",
                options=journal_review_options,
                index=journal_review_options.index(selected_journal_preferences["review_intensity"]),
                format_func=lambda value: value.replace("_", " ").title(),
                key="journal_review_intensity",
            )
        with journal_pref_col4:
            journal_article_format = st.selectbox(
                "Article Format",
                options=journal_article_options,
                index=journal_article_options.index(selected_journal_preferences["article_format"]),
                format_func=lambda value: value.replace("_", " ").title(),
                key="journal_article_format",
            )
        selected_journal_preferences = {
            "main_text_policy": journal_main_text_policy,
            "supplement_policy": journal_supplement_policy,
            "review_intensity": journal_review_intensity,
            "article_format": journal_article_format,
        }
        journal_submission_snapshot = build_journal_submission_package_snapshot(
            submission_snapshot,
            figure_package_snapshot,
            reviewer_snapshot,
            journal_preferences=selected_journal_preferences,
        )
        st.session_state["selected_journal_preferences"] = selected_journal_preferences
        st.session_state["selected_journal_template"] = journal_submission_snapshot["recommended_template"]
        journal_submission_report = build_journal_submission_package_report(
            journal_submission_snapshot
        )
        journal_col1, journal_col2, journal_col3 = st.columns(3)
        with journal_col1:
            st.metric("Main Manuscript Items", journal_submission_snapshot["summary"]["main_manuscript_items"])
        with journal_col2:
            st.metric("Supplementary Items", journal_submission_snapshot["summary"]["supplementary_items"])
        with journal_col3:
            st.metric("Recommended Template", journal_submission_snapshot["recommended_template"]["template_id"])
        st.caption(
            "Current target-journal template: "
            + journal_submission_snapshot["recommended_template"]["template_name"]
        )
        journal_bundle = build_journal_submission_bundle(
            journal_submission_report=journal_submission_report,
            journal_submission_snapshot=journal_submission_snapshot,
            selected_journal_preferences=selected_journal_preferences,
            submission_report=submission_report,
            manuscript_submission_report=manuscript_submission_report,
            submission_snapshot=submission_snapshot,
            innovation_report=innovation_report,
            reproducibility_report=reproducibility_report,
            figure_package_report=figure_package_report,
            bc_pairs_report=bc_pairs_report,
            bc_top_papers_report=bc_top_papers_report,
            df_di_report=df_di_report,
            reviewer_report=reviewer_report,
            reviewer_snapshot=reviewer_snapshot,
        )
        journal_download_col1, journal_download_col2 = st.columns(2)
        with journal_download_col1:
            _render_export_download_button(
                "journal",
                label="Download Journal Submission Package (ZIP)",
                data=journal_bundle,
                file_name="Biblio-HUB_journal_submission_version_package.zip",
                mime="application/zip",
            )
        with journal_download_col2:
            _render_export_download_button(
                "journal",
                label="Download Journal Submission Guide (MD)",
                data=journal_submission_report,
                file_name="Biblio-HUB_journal_submission_version_package.md",
                mime="text/markdown",
            )
        with st.expander("Preview Journal Submission Version Package", expanded=False):
            st.code(journal_submission_report, language="markdown")

        st.markdown("### One-Click Research Report")
        st.caption("Generate a consolidated research report that aggregates dataset overview, reproducibility records, innovation metrics, submission materials, figure guidance, and reviewer-response drafts into one package.")
        research_snapshot = build_research_report_snapshot(
            manuscript_snapshot=manuscript_snapshot,
            reproducibility_snapshot=reproducibility_snapshot,
            innovation_snapshot=innovation_snapshot,
            submission_snapshot=submission_snapshot,
            figure_package_snapshot=figure_package_snapshot,
            reviewer_snapshot=reviewer_snapshot,
            journal_submission_snapshot=journal_submission_snapshot,
        )
        research_report = build_research_report(
            research_snapshot=research_snapshot,
            manuscript_report=manuscript_report,
            reproducibility_report=reproducibility_report,
            innovation_report=innovation_report,
            submission_report=submission_report,
            figure_package_report=figure_package_report,
            reviewer_report=reviewer_report,
        )
        research_col1, research_col2, research_col3 = st.columns(3)
        with research_col1:
            st.metric("Package Sections", 6)
        with research_col2:
            st.metric("Headline Findings", len(research_snapshot["headline_findings"]))
        with research_col3:
            st.metric("Non-Default Params", len(research_snapshot["non_default_parameters"]))
        research_bundle = build_one_click_research_bundle(
            research_report=research_report,
            research_snapshot=research_snapshot,
            manuscript_report=manuscript_report,
            manuscript_snapshot=manuscript_snapshot,
            manuscript_submission_report=manuscript_submission_report,
            manuscript_submission_snapshot=manuscript_submission_snapshot,
            reproducibility_report=reproducibility_report,
            reproducibility_snapshot=reproducibility_snapshot,
            innovation_report=innovation_report,
            innovation_snapshot=innovation_snapshot,
            submission_report=submission_report,
            submission_snapshot=submission_snapshot,
            figure_package_report=figure_package_report,
            figure_package_snapshot=figure_package_snapshot,
            reviewer_report=reviewer_report,
            reviewer_snapshot=reviewer_snapshot,
        )
        research_download_col1, research_download_col2 = st.columns(2)
        with research_download_col1:
            _render_export_download_button(
                "research",
                label="Download One-Click Research Package (ZIP)",
                data=research_bundle,
                file_name="Biblio-HUB_one_click_research_report.zip",
                mime="application/zip",
            )
        with research_download_col2:
            _render_export_download_button(
                "research",
                label="Download One-Click Research Report (MD)",
                data=research_report,
                file_name="Biblio-HUB_one_click_research_report.md",
                mime="text/markdown",
            )
        with st.expander("Preview One-Click Research Report", expanded=False):
            st.code(research_report, language="markdown")

    st.markdown("---")
    st.markdown("### \U0001F5BC\uFE0F Publication Figure Bundle")
    bundle_format = st.selectbox(
        "Figure Format",
        [fmt.upper() for fmt in PUBLICATION_EXPORT_FORMATS],
        index=0,
        key="publication_bundle_format",
    )
    plotly_static_export_status = get_plotly_static_export_status()
    st.caption("Publication export uses white backgrounds, unified styling, Plotly high-resolution rendering, and 300 dpi Matplotlib output.")
    
    tier, note = _get_export_time_hint("figure_bundle")
    st.caption(f"Estimated time: {tier} | {note}")
    
    st.caption("Static network figures are now included in the formal export bundle. When a network figure is exported, the ZIP package also includes the matching interactive HTML companion for inspection and revision.")
    if not plotly_static_export_status["available"]:
        st.warning(
            "Kaleido was not detected, so Plotly static images may not be written into the ZIP package. The current export will keep all renderable items and record missing outputs in `98_export_status.json` and `99_export_log.txt`."
        )
    
    figure_bundle_job_id = st.session_state.get("figure_bundle_job_id")
    if not figure_bundle_job_id:
        if st.button("Generate All Publication Figures", key="btn_all_figs"):
            job_id = submit_export_background_job(
                'figure_bundle', 
                df, keywords_list, keyword_freq, cooccurrence, 
                    export_format=bundle_format.lower(),
                    bc_topn_val=bc_topn_val,
                    bc_min_shared_val=bc_min_shared_val,
                    lightweight_mode=lightweight_mode,
                    network_export_modes=_current_network_export_modes(),
                    analysis_parameters=_current_export_analysis_parameters(bundle_format.lower()),
            )
            st.session_state["figure_bundle_job_id"] = job_id
            st.rerun()
    else:
        status = get_export_job_status(figure_bundle_job_id)
        if status["status"] == "running":
            st.info("Publication figure bundle is being rendered in the background...")
            if st.button("Refresh Status", key="refresh_fig_bundle"):
                st.rerun()
        elif status["status"] == "completed":
            st.success("Publication figure bundle generated successfully!")
            st.download_button(
                label=f"Download All Figures ({bundle_format})",
                data=status["result"],
                file_name=f"Biblio-HUB_publication_figures_{bundle_format.lower()}.zip",
                mime="application/zip",
                key="dl_fig_bundle"
            )
            if st.button("Clear / Generate New", key="clear_fig_bundle"):
                discard_export_job(figure_bundle_job_id)
                st.session_state["figure_bundle_job_id"] = None
                st.rerun()
        elif status["status"] == "failed":
            st.error(f"Figure bundle generation failed: {status['error']}")
            if st.button("Retry", key="retry_fig_bundle"):
                discard_export_job(figure_bundle_job_id)
                st.session_state["figure_bundle_job_id"] = None
                st.rerun()

    st.markdown("### Selective Figure Export")
    st.caption("Choose only the figure panels needed for manuscript assembly, supplementary files, or figure revision rounds.")
    figure_options = get_available_figure_options(df, keyword_freq)
    grouped_figure_options = group_figure_options(figure_options)
    selected_figure_ids = []

    if figure_options:
        helper_col1, helper_col2, helper_col3 = st.columns([1, 1, 2])
        with helper_col1:
            if st.button("Select All Figures"):
                for option in figure_options:
                    st.session_state[f"figure_export_{option['id']}"] = True
        with helper_col2:
            if st.button("Clear Figure Selection"):
                for option in figure_options:
                    st.session_state[f"figure_export_{option['id']}"] = False
        with helper_col3:
            st.caption(f"{len(figure_options)} publication-ready figure items are currently available for export.")

        for group_name, group_items in grouped_figure_options.items():
            with st.expander(f"{group_name} ({len(group_items)})", expanded=True):
                cols = st.columns(2)
                for idx, item in enumerate(group_items):
                    checkbox_key = f"figure_export_{item['id']}"
                    if checkbox_key not in st.session_state:
                        st.session_state[checkbox_key] = False
                    checked = cols[idx % 2].checkbox(item["item_label"], key=checkbox_key)
                    if checked:
                        selected_figure_ids.append(item["id"])

        st.caption(f"Selected items: {len(selected_figure_ids)}")
        
        selected_fig_job_id = st.session_state.get("selected_fig_job_id")
        if not selected_fig_job_id:
            if st.button("Generate Selected Figure Bundle", key="btn_selected_figs"):
                if not selected_figure_ids:
                    st.warning("Please select at least one figure before generating the export bundle.")
                else:
                    job_id = submit_export_background_job(
                        'figure_bundle',
                        df,
                        keywords_list,
                        keyword_freq,
                        cooccurrence,
                        export_format=bundle_format.lower(),
                        selected_items=selected_figure_ids,
                        bc_topn_val=bc_topn_val,
                        bc_min_shared_val=bc_min_shared_val,
                        lightweight_mode=lightweight_mode,
                        network_export_modes=_current_network_export_modes(),
                        analysis_parameters=_current_export_analysis_parameters(bundle_format.lower()),
                    )
                    st.session_state["selected_fig_job_id"] = job_id
                    st.rerun()
        else:
            status = get_export_job_status(selected_fig_job_id)
            if status["status"] == "running":
                st.info("Selected figure bundle is being rendered in the background...")
                if st.button("Refresh Status", key="refresh_selected_fig"):
                    st.rerun()
            elif status["status"] == "completed":
                st.success(f"Selected figure bundle generated successfully!")
                st.download_button(
                    label=f"Download Selected Figures ({bundle_format})",
                    data=status["result"],
                    file_name=f"Biblio-HUB_selected_figures_{bundle_format.lower()}.zip",
                    mime="application/zip",
                    key="dl_selected_fig"
                )
                if st.button("Clear / Generate New", key="clear_selected_fig"):
                    discard_export_job(selected_fig_job_id)
                    st.session_state["selected_fig_job_id"] = None
                    st.rerun()
            elif status["status"] == "failed":
                st.error(f"Selected figure bundle generation failed: {status['error']}")
                if st.button("Retry", key="retry_selected_fig"):
                    discard_export_job(selected_fig_job_id)
                    st.session_state["selected_fig_job_id"] = None
                    st.rerun()
    else:
        st.info("No publication-ready figure items are available for the current dataset.")

    st.markdown("---")
    st.markdown("### Key Tables and References")
    st.caption("Keep hotspot tables and reference files on the first screen so users can export core manuscript materials without scrolling through package-specific options.")
    quick_asset_col1, quick_asset_col2, quick_asset_col3 = st.columns(3)
    with quick_asset_col1:
        _render_export_download_button(
            "report",
            "Download Key Tables (ZIP)",
            key_tables_bundle,
            "Biblio-HUB_key_tables.zip",
            "application/zip",
        )
    with quick_asset_col2:
        _render_export_download_button(
            "report",
            "Download References (ZIP)",
            reference_bundle,
            "Biblio-HUB_references_bundle.zip",
            "application/zip",
        )
    with quick_asset_col3:
        vos_nodes = [{"id": kw, "label": kw, "weight": freq} for kw, freq in keyword_freq.most_common(50)]
        vos_edges = []
        for (k1, k2), w in cooccurrence.most_common(200):
            if k1 in [n["id"] for n in vos_nodes] and k2 in [n["id"] for n in vos_nodes]:
                vos_edges.append({"source": k1, "target": k2, "weight": w})
        vos_data = {"network": {"items": vos_nodes, "links": vos_edges}}
        vos_json = json.dumps(vos_data, indent=2, ensure_ascii=False)
        _render_export_download_button(
            "report",
            "Download VOSviewer JSON",
            vos_json,
            "Biblio-HUB_vosviewer_network.json",
            "application/json",
        )

    st.markdown("---")
    st.markdown("### \U0001F4C2 Software Registration Materials")
    st.caption("Generate a package containing technical documentation, source code samples, and a manual draft for software registration.")
    
    tier, note = _get_export_time_hint("copyright_package")
    st.caption(f"Estimated time: {tier} | {note}")
    
    copyright_job_id = st.session_state.get("copyright_job_id")
    if not copyright_job_id:
        if st.button("Generate Copyright Materials Package", key="btn_copyright"):
            job_id = submit_export_background_job('copyright_package', df)
            st.session_state["copyright_job_id"] = job_id
            st.rerun()
    else:
        status = get_export_job_status(copyright_job_id)
        if status["status"] == "running":
            st.info("Copyright materials are being prepared in the background...")
            if st.button("Refresh Status", key="refresh_copyright"):
                st.rerun()
        elif status["status"] == "completed":
            st.success("Copyright materials package generated successfully!")
            st.download_button(
                label="\U0001F4E5 Download Copyright Package (ZIP)",
                data=status["result"],
                file_name="Biblio-HUB_software_copyright_materials.zip",
                mime="application/zip",
                key="dl_copyright"
            )
            if st.button("Clear / Generate New", key="clear_copyright"):
                discard_export_job(copyright_job_id)
                st.session_state["copyright_job_id"] = None
                st.rerun()
        elif status["status"] == "failed":
            st.error(f"Copyright materials generation failed: {status['error']}")
            if st.button("Retry", key="retry_copyright"):
                discard_export_job(copyright_job_id)
                st.session_state["copyright_job_id"] = None
                st.rerun()
