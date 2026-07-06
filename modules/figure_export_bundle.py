import io
import json
import os
import re
import textwrap
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap
from wordcloud import WordCloud

from modules.data_pipeline import _count_semicolon_terms, clean_year_column
from modules.export_bundle import (
    build_publication_manifest,
    get_plotly_static_export_status,
    KEYWORD_MATRIX_SEQUENTIAL_SCALE,
    MORANDI_SEQUENTIAL_SCALE,
    SCIENTIFIC_COLORWAY,
    matplotlib_figure_to_bytes,
    plotly_figure_to_bytes,
    style_publication_figure,
)
from modules.network_builders import build_keyword_journal_cooccurrence
from modules.network_visualization import (
    node_groups_from_cluster_stats,
    render_keyword_journal_network,
    render_network_publication_figure,
    render_vosviewer_style,
)
from modules.structure_visualization import (
    render_author_production_over_time,
    render_hierarchical_cluster_heatmap,
    render_lotkas_law,
    render_thematic_map,
    render_three_field_plot,
)
from modules.temporal_analysis import (
    build_entity_leadership_shift_tables,
    build_entity_forecast_tables,
    build_keyword_burst_table,
    build_keyword_opportunity_map_frame,
    build_publication_forecast_frame,
    build_theme_migration_forecast_tables,
    render_alluvial_topic_flow,
    render_burst_detection,
    render_citespace_timeline,
    render_entity_leadership_shift_figure,
    render_entity_leadership_trajectory_figure,
    render_entity_forecast_rank_figure,
    render_entity_forecast_trajectory_figure,
    render_keyword_opportunity_map,
    render_publication_forecast_figure,
    render_theme_migration_opportunity_map,
    render_theme_migration_trajectory_figure,
    summarize_entity_leadership_shift,
    summarize_entity_forecast_signals,
    summarize_keyword_opportunity_map,
    summarize_publication_forecast,
    summarize_theme_migration_signals,
)
from modules.experiment_framework import (
    build_bibliographic_coupling_network,
    compute_brokerage_robustness_experiment,
    compute_structural_hole_frame,
    render_brokerage_robustness_summary,
    render_structural_hole_brokerage_profile,
)
from modules.citation_analysis import (
    build_author_cocitation_network,
    build_cocitation_network,
    build_journal_cocitation_network,
    build_publication_citation_trend_frame,
    build_reference_burst_table,
    extract_rpys_statistics,
    render_publication_citation_dual_axis_figure,
    render_reference_burst_figure,
    render_rpys_figure,
)
from modules.ui_relational_network import (
    _build_partitioned_node_groups,
    _build_coauthorship_network_assets,
    _build_country_collaboration_assets,
    _build_institution_collaboration_assets,
)
from modules.advanced_visualizations import (
    build_keyword_circular_cluster_figure,
    build_country_impact_quadrant_frame,
    render_country_impact_quadrant_figure,
    render_circular_cluster_chord_figure,
    render_ranked_lollipop_figure,
)

WORDCLOUD_COLORMAP = LinearSegmentedColormap.from_list(
    "wordcloud_vivid",
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
KEYWORD_MATRIX_SEQUENTIAL_SCALE = [
    [0.0, "#fbd2bc"],
    [0.25, "#feab88"],
    [0.5, "#b71c2c"],
    [0.75, "#8b0824"],
    [1.0, "#6a0624"],
]


def _most_common_items(values, limit):
    if hasattr(values, "most_common"):
        return values.most_common(limit)
    return sorted(values.items(), key=lambda item: item[1], reverse=True)[:limit]


def _resolve_parallel_export_workers(request_count):
    if request_count < 4:
        return 1
    cpu_total = os.cpu_count() or 1
    return max(1, min(request_count, max(cpu_total - 1, 1), 4))


def _render_plotly_export_request(fig, export_format, width, height, title_visible=True, transparent_background=False):
    return plotly_figure_to_bytes(
        fig,
        export_format,
        width=width,
        height=height,
        title_visible=title_visible,
        transparent_background=transparent_background,
    )


def _render_matplotlib_export_request(fig, export_format, transparent_background=False):
    try:
        return matplotlib_figure_to_bytes(fig, export_format, transparent_background=transparent_background)
    finally:
        plt.close(fig)


def _build_plotly_export_placeholder_bytes(path, export_format, error_message):
    if str(export_format).lower() != "png":
        raise RuntimeError(error_message)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    fig.patch.set_facecolor("white")
    ax.axis("off")
    title = os.path.basename(path)
    wrapped_error = "\n".join(textwrap.wrap(str(error_message), width=80)) or "Static export failed."
    body = (
        f"Placeholder image generated for:\n{title}\n\n"
        "The analytical result was computed, but static Plotly export was unavailable in this environment.\n\n"
        f"Reason:\n{wrapped_error}"
    )
    ax.text(0.5, 0.62, "Export Placeholder", ha="center", va="center", fontsize=22, fontweight="bold")
    ax.text(0.5, 0.34, body, ha="center", va="center", fontsize=12, wrap=True)
    return _render_matplotlib_export_request(fig, "png", transparent_background=False)


def generate_all_figure_bundle(
    df,
    keywords_list,
    keyword_freq,
    cooccurrence,
    export_format="png",
    selected_items=None,
    bc_topn_val=30,
    bc_min_shared_val=2,
    lightweight_mode=False,
    top_k=5,
    bundle_zip=None,
    network_export_modes=None,
    analysis_parameters=None,
):
    if bundle_zip is None:
        bundle_buffer = io.BytesIO()
        zf = zipfile.ZipFile(bundle_buffer, "w", zipfile.ZIP_DEFLATED)
        own_zip = True
    else:
        zf = bundle_zip
        own_zip = False

    export_format = export_format.lower()
    selected_items = set(selected_items or [])
    export_all = not selected_items
    network_export_modes = dict(network_export_modes or {})
    skipped_items = []
    static_export_status = get_plotly_static_export_status()
    plotly_export_attempts = 0
    plotly_export_successes = 0
    matplotlib_export_attempts = 0
    matplotlib_export_successes = 0
    export_parallel_workers_used = 1
    pending_plotly_exports = []
    pending_matplotlib_exports = []
    if isinstance(analysis_parameters, dict):
        analysis_parameter_map = dict(analysis_parameters)
    else:
        analysis_parameter_map = {
            item.get("key"): item.get("value")
            for item in (analysis_parameters or [])
            if isinstance(item, dict) and item.get("key")
        }

    def get_analysis_value(key, default):
        value = analysis_parameter_map.get(key, default)
        try:
            if isinstance(default, bool):
                return bool(value)
            if isinstance(default, int) and not isinstance(default, bool):
                return int(value)
            if isinstance(default, float):
                return float(value)
        except (TypeError, ValueError):
            return default
        return value

    keyword_top_n = max(1, min(get_analysis_value("vos_topn", 30), max(len(keyword_freq), 1)))
    keyword_min_weight = max(1, get_analysis_value("vos_minw", 2))
    keyword_journal_top_keywords = max(1, get_analysis_value("kj_topn_kw", 15))
    keyword_journal_top_journals = max(1, get_analysis_value("kj_topn_jn", 10))
    overview_top_journals_n = max(1, get_analysis_value("overview_top_journals_n", 30))
    author_collaboration_min_papers = max(1, get_analysis_value("auth_min_papers", 2))
    author_collaboration_top_n = max(1, get_analysis_value("auth_topn", 50))
    institution_collaboration_top_n = max(1, get_analysis_value("inst_topn", 25))
    reference_cocitation_top_n = max(1, get_analysis_value("cocite_topn", 20))
    reference_cocitation_min_weight = max(1, get_analysis_value("cocite_minw", 2))
    author_cocitation_top_n = max(1, get_analysis_value("auth_cocite_topn", 20))
    author_cocitation_min_weight = max(1, get_analysis_value("auth_cocite_minw", 2))
    journal_cocitation_top_n = max(1, get_analysis_value("j_cocite_topn", 20))
    journal_cocitation_min_weight = max(1, get_analysis_value("j_cocite_minw", 2))

    def include(item_id):
        return export_all or item_id in selected_items

    def flush_pending_exports():
        nonlocal plotly_export_attempts
        nonlocal plotly_export_successes
        nonlocal matplotlib_export_attempts
        nonlocal matplotlib_export_successes
        nonlocal export_parallel_workers_used

        batched_requests = []
        for request in pending_plotly_exports:
            batched_requests.append(("plotly", request))
        for request in pending_matplotlib_exports:
            batched_requests.append(("matplotlib", request))
        if not batched_requests:
            return

        workers = _resolve_parallel_export_workers(len(batched_requests))
        export_parallel_workers_used = max(export_parallel_workers_used, workers)

        if workers <= 1:
            for request_type, request in batched_requests:
                path = request["path"]
                try:
                    if request_type == "plotly":
                        plotly_export_attempts += 1
                        payload = _render_plotly_export_request(
                            request["fig"],
                            export_format,
                            request["width"],
                            request["height"],
                            request.get("title_visible", True),
                            request.get("transparent_background", False),
                        )
                        zf.writestr(path, payload)
                        plotly_export_successes += 1
                    else:
                        matplotlib_export_attempts += 1
                        payload = _render_matplotlib_export_request(
                            request["fig"],
                            export_format,
                            request.get("transparent_background", False),
                        )
                        zf.writestr(path, payload)
                        matplotlib_export_successes += 1
                except Exception as exc:
                    if request_type == "plotly":
                        try:
                            payload = _build_plotly_export_placeholder_bytes(path, export_format, exc)
                            zf.writestr(path, payload)
                            skipped_items.append(f"{path}: static export fallback placeholder generated ({exc})")
                            continue
                        except Exception:
                            pass
                    skipped_items.append(f"{path}: {exc}")
            pending_plotly_exports.clear()
            pending_matplotlib_exports.clear()
            return

        future_map = {}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="figure-export") as executor:
            for request in pending_plotly_exports:
                plotly_export_attempts += 1
                future = executor.submit(
                    _render_plotly_export_request,
                    request["fig"],
                    export_format,
                    request["width"],
                    request["height"],
                    request.get("title_visible", True),
                    request.get("transparent_background", False),
                )
                future_map[future] = ("plotly", request)
            for request in pending_matplotlib_exports:
                matplotlib_export_attempts += 1
                future = executor.submit(
                    _render_matplotlib_export_request,
                    request["fig"],
                    export_format,
                    request.get("transparent_background", False),
                )
                future_map[future] = ("matplotlib", request)

            resolved_payloads = {}
            for future in as_completed(future_map):
                request_type, request = future_map[future]
                path = request["path"]
                try:
                    resolved_payloads[path] = future.result()
                    if request_type == "plotly":
                        plotly_export_successes += 1
                    else:
                        matplotlib_export_successes += 1
                except Exception as exc:
                    if request_type == "plotly":
                        try:
                            resolved_payloads[path] = _build_plotly_export_placeholder_bytes(path, export_format, exc)
                            plotly_export_successes += 1
                            skipped_items.append(f"{path}: static export fallback placeholder generated ({exc})")
                            continue
                        except Exception:
                            pass
                    skipped_items.append(f"{path}: {exc}")

        for request in pending_plotly_exports:
            payload = resolved_payloads.get(request["path"])
            if payload is not None:
                zf.writestr(request["path"], payload)
        for request in pending_matplotlib_exports:
            payload = resolved_payloads.get(request["path"])
            if payload is not None:
                zf.writestr(request["path"], payload)

        pending_plotly_exports.clear()
        pending_matplotlib_exports.clear()

    def add_plotly(path, fig, width=1600, height=900, title_visible=False, transparent_background=None):
        if transparent_background is None:
            transparent_background = export_format in {"svg", "pdf"}
        pending_plotly_exports.append(
            {
                "path": path,
                "fig": fig,
                "width": width,
                "height": height,
                "title_visible": title_visible,
                "transparent_background": transparent_background,
            }
        )
        if len(pending_plotly_exports) + len(pending_matplotlib_exports) >= 6:
            flush_pending_exports()

    def add_matplotlib(path, fig, transparent_background=None):
        if transparent_background is None:
            transparent_background = export_format in {"svg", "pdf"}
        pending_matplotlib_exports.append(
            {
                "path": path,
                "fig": fig,
                "transparent_background": transparent_background,
            }
        )
        if len(pending_plotly_exports) + len(pending_matplotlib_exports) >= 6:
            flush_pending_exports()

    def add_text(path, content):
        try:
            zf.writestr(path, str(content).encode("utf-8"))
        except Exception as exc:
            skipped_items.append(f"{path}: {exc}")

    def add_dataframe(path, frame):
        if frame is None:
            return
        if isinstance(frame, pd.DataFrame):
            if frame.empty:
                return
            payload = frame.to_csv(index=False).encode("utf-8-sig")
        else:
            temp_df = pd.DataFrame(frame)
            if temp_df.empty:
                return
            payload = temp_df.to_csv(index=False).encode("utf-8-sig")
        try:
            zf.writestr(path, payload)
        except Exception as exc:
            skipped_items.append(f"{path}: {exc}")

    def add_network_figure(
        item_id,
        path,
        graph,
        title,
        node_groups=None,
        legend_label="Frequency",
        layout_mode="auto",
        size_range=(16, 60),
        label_max_len=40,
        max_visible_labels=18,
        width=2400,
        height=1800,
        mode="publication",
        label_font_size=50,
        edge_width_scale=0.8,
        edge_alpha_scale=0.42,
    ):
        if not include(item_id):
            return
        max_visible_labels = max(8, min(int(max_visible_labels), 18))
        fig = render_network_publication_figure(
            graph,
            node_groups=node_groups,
            title=title,
            size_range=size_range,
            label_max_len=label_max_len,
            legend_label=legend_label,
            layout_mode=layout_mode,
            max_visible_labels=max_visible_labels,
            label_font_size=label_font_size,
            mode=mode,
            edge_width_scale=edge_width_scale,
            edge_alpha_scale=edge_alpha_scale,
        )
        if fig is None:
            skipped_items.append(f"{path}: no network figure generated")
            return
        add_plotly(path, fig, width=width, height=height)

    def resolve_network_export_mode(item_id):
        figure_mode_label = str(network_export_modes.get(item_id, "Publication Figure"))
        if figure_mode_label == "Dense Map (VOS-like)":
            return "map"
        return "publication"

    network_html_items = []

    df_valid_year = None
    if "Year" in df.columns and any(
        include(item_id)
        for item_id in (
            "overview_publications_by_year",
            "structure_annual_growth_rate",
        )
    ):
        df_valid_year = clean_year_column(df)

    if "Year" in df.columns and include("overview_publications_by_year"):
        if df_valid_year is not None and not df_valid_year.empty:
            year_counts = df_valid_year["Year"].value_counts().sort_index()
            year_df = pd.DataFrame({"Year": year_counts.index, "Publications": year_counts.values})
            fig = px.area(
                year_df,
                x="Year",
                y="Publications",
                labels={"Year": "Year", "Publications": "Number of Publications"},
                title="Publications by Year",
            )
            fig.update_traces(line_color=SCIENTIFIC_COLORWAY[3], fillcolor="rgba(94, 55, 157, 0.22)")
            add_plotly(f"01_overview/publications_by_year.{export_format}", fig)
            add_dataframe(
                "07_analysis_tables/publications_by_year.csv",
                pd.DataFrame({"Year": year_counts.index, "Publications": year_counts.values}),
            )

    if "Journal" in df.columns and include("overview_top_journals"):
        journal_counts = df["Journal"].value_counts().head(overview_top_journals_n)
        if not journal_counts.empty:
            journal_df = journal_counts.rename_axis("Journal").reset_index(name="Publications")
            fig = render_ranked_lollipop_figure(
                journal_df,
                label_col="Journal",
                value_col="Publications",
                title=f"Top {len(journal_df)} Journals",
                marker_color=SCIENTIFIC_COLORWAY[0],
                line_color=SCIENTIFIC_COLORWAY[0],
                xaxis_title="Number of Publications",
                yaxis_title="Journal",
            )
            fig.update_layout(height=700)
            add_plotly(f"01_overview/top_journals.{export_format}", fig, height=900)
            add_dataframe(
                "07_analysis_tables/top_journals_overview.csv",
                journal_counts.rename_axis("Journal").reset_index(name="Publications"),
            )

    if keyword_freq and include("overview_keyword_wordcloud"):
        wc = WordCloud(
            width=1400,
            height=500,
            background_color="white",
            max_words=80,
            colormap=WORDCLOUD_COLORMAP,
            prefer_horizontal=0.7,
            min_font_size=10,
        )
        wc.generate_from_frequencies(keyword_freq)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        add_matplotlib(f"01_overview/keyword_wordcloud.{export_format}", fig)

    if keyword_freq and include("overview_top_keywords"):
        top_kw = pd.DataFrame(_most_common_items(keyword_freq, 20), columns=["Keyword", "Frequency"])
        kw_plot_df = top_kw.sort_values("Frequency", ascending=False).copy()
        fig = px.bar(
            kw_plot_df,
            x="Frequency",
            y="Keyword",
            orientation="h",
            title="Top 20 Keywords",
        )
        fig.update_traces(marker_color=SCIENTIFIC_COLORWAY[5])
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=list(reversed(kw_plot_df["Keyword"].tolist())),
        )
        fig.update_layout(height=700)
        add_plotly(f"01_overview/top_keywords.{export_format}", fig, height=900)
        add_dataframe("07_analysis_tables/top_keywords_overview.csv", top_kw)

    if keyword_freq and include("structure_keyword_circular_cluster"):
        fig_kw_chord = build_keyword_circular_cluster_figure(
            keyword_freq,
            cooccurrence,
            top_n=min(keyword_top_n, len(keyword_freq)),
            min_weight=keyword_min_weight,
        )
        if fig_kw_chord:
            add_plotly(f"04_statistics/keyword_circular_cluster_map.{export_format}", fig_kw_chord, width=1800, height=1800)

    if keyword_freq and include("network_keyword_cooccurrence_static"):
        html_content, keyword_graph, keyword_stats = render_vosviewer_style(
            keyword_freq,
            cooccurrence,
            top_n=min(keyword_top_n, len(keyword_freq)),
            min_weight=keyword_min_weight,
        )
        keyword_node_groups = node_groups_from_cluster_stats(keyword_stats)
        add_network_figure(
            "network_keyword_cooccurrence_static",
            f"02_network/keyword_cooccurrence_network.{export_format}",
            keyword_graph,
            title="Keyword Co-occurrence Network",
            node_groups=keyword_node_groups,
            legend_label="Frequency",
            layout_mode="clustered" if keyword_node_groups else "auto",
            size_range=(18, 62),
            label_max_len=34,
            max_visible_labels=14,
            label_font_size=28,
            mode=resolve_network_export_mode("network_keyword_cooccurrence_static"),
        )
        if html_content and include("network_keyword_cooccurrence_static"):
            network_html_items.append(("06_network_html/keyword_cooccurrence_network.html", html_content))
        top_pairs = pd.DataFrame(
            [
                {"Keyword A": kw1, "Keyword B": kw2, "Co-occurrence": weight}
                for (kw1, kw2), weight in _most_common_items(cooccurrence, 200)
            ]
        )
        add_dataframe("07_analysis_tables/keyword_cooccurrence_pairs.csv", top_pairs)

    if keyword_freq and "Journal" in df.columns and include("network_keyword_journal_static"):
        top_keywords_kj, top_journals_kj, kw_journal_cooccur, keyword_freq_local, journal_freq = (
            build_keyword_journal_cooccurrence(
                df,
                keywords_list,
                keyword_journal_top_keywords,
                keyword_journal_top_journals,
            )
        )
        html_content, keyword_journal_graph, keyword_journal_groups, _ = render_keyword_journal_network(
            top_keywords_kj,
            top_journals_kj,
            kw_journal_cooccur,
            keyword_freq_local,
            journal_freq,
        )
        add_network_figure(
            "network_keyword_journal_static",
            f"02_network/keyword_journal_association_network.{export_format}",
            keyword_journal_graph,
            title="Keyword-Journal Association Network",
            node_groups=keyword_journal_groups,
            legend_label="Frequency",
            layout_mode="bipartite",
            size_range=(16, 52),
            label_max_len=24,
            max_visible_labels=20,
            mode=resolve_network_export_mode("network_keyword_journal_static"),
        )
        if html_content and include("network_keyword_journal_static"):
            network_html_items.append(("06_network_html/keyword_journal_association_network.html", html_content))
        pair_df = pd.DataFrame(
            [
                {"Keyword": keyword, "Journal": journal, "Co-occurrence": weight}
                for (keyword, journal), weight in sorted(
                    kw_journal_cooccur.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:200]
            ]
        )
        add_dataframe("07_analysis_tables/keyword_journal_pairs.csv", pair_df)

    if "Authors" in df.columns and include("network_coauthorship_static"):
        graph, _, node_groups, html_content, _, top_auth_df = _build_coauthorship_network_assets(
            df,
            min_papers=author_collaboration_min_papers,
            top_n_auth=author_collaboration_top_n,
            palette=SCIENTIFIC_COLORWAY,
        )
        if graph is not None and graph.number_of_nodes() > 0:
            add_network_figure(
                "network_coauthorship_static",
                f"02_network/author_collaboration_network.{export_format}",
                graph,
                title="Author Collaboration Network",
                node_groups=node_groups,
                legend_label="Papers",
                layout_mode="clustered",
                size_range=(24, 84),
                label_max_len=36,
                max_visible_labels=24,
                mode=resolve_network_export_mode("network_coauthorship_static"),
            )
            if html_content:
                network_html_items.append(("06_network_html/author_collaboration_network.html", html_content))
            add_dataframe("07_analysis_tables/top_authors_by_publications.csv", top_auth_df)

    if "Affiliations" in df.columns and include("network_institution_collaboration_static"):
        graph, _, node_groups, html_content, _, top_inst_df = _build_institution_collaboration_assets(
            df,
            top_n_inst=institution_collaboration_top_n,
            palette=SCIENTIFIC_COLORWAY,
        )
        if graph is not None and graph.number_of_nodes() > 0:
            add_network_figure(
                "network_institution_collaboration_static",
                f"02_network/institution_collaboration_network.{export_format}",
                graph,
                title="Institution Collaboration Network",
                node_groups=node_groups,
                legend_label="Papers",
                layout_mode="clustered",
                size_range=(24, 84),
                label_max_len=40,
                max_visible_labels=24,
                mode=resolve_network_export_mode("network_institution_collaboration_static"),
            )
            if html_content:
                network_html_items.append(("06_network_html/institution_collaboration_network.html", html_content))
            add_dataframe("07_analysis_tables/top_institutions_by_publications.csv", top_inst_df)

    if "Affiliations" in df.columns and (
        include("network_country_collaboration_static")
        or include("network_country_publications_bar")
        or include("network_country_collaboration_chord")
        or include("network_country_impact_quadrant")
    ):
        graph, _, node_groups, html_content, _, country_df = _build_country_collaboration_assets(
            df,
            top_n_country=25,
            palette=SCIENTIFIC_COLORWAY,
        )
        if graph is not None and graph.number_of_nodes() > 0:
            if include("network_country_collaboration_static"):
                add_network_figure(
                    "network_country_collaboration_static",
                    f"02_network/country_collaboration_network.{export_format}",
                    graph,
                    title="Country Collaboration Network",
                    node_groups=node_groups,
                    legend_label="Papers",
                    layout_mode="clustered",
                    size_range=(24, 84),
                    label_max_len=34,
                    max_visible_labels=24,
                    mode=resolve_network_export_mode("network_country_collaboration_static"),
                )
                if html_content:
                    network_html_items.append(("06_network_html/country_collaboration_network.html", html_content))
            add_dataframe("07_analysis_tables/top_countries_by_publications.csv", country_df)
            if include("network_country_publications_bar"):
                country_plot_df = country_df.sort_values("Papers", ascending=False).copy()
                fig_country = px.bar(
                    country_plot_df,
                    x="Papers",
                    y="Country",
                    orientation="h",
                    title="Top 30 Countries by Publications",
                )
                fig_country.update_traces(marker_color=SCIENTIFIC_COLORWAY[2])
                fig_country.update_yaxes(
                    categoryorder="array",
                    categoryarray=list(reversed(country_plot_df["Country"].tolist())),
                )
                fig_country = style_publication_figure(fig_country, height=700)
                add_plotly(f"02_network/country_publications_bar.{export_format}", fig_country, height=900)

            if include("network_country_collaboration_chord"):
                fig_chord = render_circular_cluster_chord_figure(
                    graph,
                    node_groups=node_groups,
                    title="Country Collaboration Chord Map",
                    legend_title="Publications",
                )
                if fig_chord:
                    add_plotly(f"02_network/country_collaboration_chord_map.{export_format}", fig_chord, width=1800, height=1800)

            if include("network_country_impact_quadrant"):
                quadrant_df, publication_median, citation_median = build_country_impact_quadrant_frame(df)
                if not quadrant_df.empty:
                    fig_quadrant = render_country_impact_quadrant_figure(
                        quadrant_df,
                        publication_median,
                        citation_median,
                    )
                    add_plotly(f"02_network/country_impact_quadrant.{export_format}", fig_quadrant, width=2100, height=1485)

    if "Cited_References" in df.columns and include("network_bibliographic_coupling_static"):
        graph, bc_pairs, bc_top_papers = build_bibliographic_coupling_network(
            df,
            top_n=bc_topn_val,
            min_shared_refs=bc_min_shared_val,
        )
        if graph is not None and graph.number_of_nodes() > 0:
            _, node_groups = _build_partitioned_node_groups(graph, SCIENTIFIC_COLORWAY)
            add_network_figure(
                "network_bibliographic_coupling_static",
                f"02_network/bibliographic_coupling_network.{export_format}",
                graph,
                title="Bibliographic Coupling Network",
                node_groups=node_groups,
                legend_label="Shared references",
                layout_mode="clustered",
                size_range=(24, 84),
                label_max_len=40,
                max_visible_labels=24,
                mode=resolve_network_export_mode("network_bibliographic_coupling_static"),
            )
            add_dataframe("07_analysis_tables/bibliographic_coupling_pairs.csv", bc_pairs)
            add_dataframe("07_analysis_tables/bibliographic_coupling_top_papers.csv", bc_top_papers)

    if "Cited_References" in df.columns and include("network_reference_cocitation_static"):
        html_content, graph, node_groups, _, ref_freq = build_cocitation_network(
            df,
            top_n_ref=reference_cocitation_top_n,
            min_cocite=reference_cocitation_min_weight,
        )
        if graph is not None and graph.number_of_nodes() > 0:
            add_network_figure(
                "network_reference_cocitation_static",
                f"02_network/reference_cocitation_network.{export_format}",
                graph,
                title="Co-citation Network",
                node_groups=node_groups,
                legend_label="Citations",
                layout_mode="clustered",
                size_range=(24, 84),
                label_max_len=30,
                max_visible_labels=12,
                label_font_size=26,
                mode=resolve_network_export_mode("network_reference_cocitation_static"),
            )
            if html_content:
                network_html_items.append(("06_network_html/reference_cocitation_network.html", html_content))
            add_dataframe(
                "07_analysis_tables/top_cited_references.csv",
                pd.DataFrame(ref_freq.most_common(200), columns=["Reference", "Citations"]),
            )

    if "Cited_References" in df.columns and include("network_author_cocitation_static"):
        html_content, graph, node_groups, _, auth_freq = build_author_cocitation_network(
            df,
            top_n_authors=author_cocitation_top_n,
            min_cocite=author_cocitation_min_weight,
        )
        if graph is not None and graph.number_of_nodes() > 0:
            add_network_figure(
                "network_author_cocitation_static",
                f"02_network/author_cocitation_network.{export_format}",
                graph,
                title="Author Co-citation Network",
                node_groups=node_groups,
                legend_label="Co-citations",
                layout_mode="clustered",
                size_range=(24, 84),
                label_max_len=25,
                max_visible_labels=10,
                label_font_size=25,
                mode=resolve_network_export_mode("network_author_cocitation_static"),
            )
            if html_content:
                network_html_items.append(("06_network_html/author_cocitation_network.html", html_content))
            add_dataframe(
                "07_analysis_tables/top_cited_authors.csv",
                pd.DataFrame(auth_freq.most_common(200), columns=["Author", "Co-citations"]),
            )

    if "Cited_References" in df.columns and include("network_journal_cocitation_static"):
        html_content, graph, node_groups, _, journal_freq = build_journal_cocitation_network(
            df,
            top_n_journals=journal_cocitation_top_n,
            min_cocite=journal_cocitation_min_weight,
        )
        if graph is not None and graph.number_of_nodes() > 0:
            add_network_figure(
                "network_journal_cocitation_static",
                f"02_network/journal_cocitation_network.{export_format}",
                graph,
                title="Journal Co-citation Network",
                node_groups=node_groups,
                legend_label="Co-citations",
                layout_mode="clustered",
                size_range=(24, 84),
                label_max_len=25,
                max_visible_labels=10,
                label_font_size=25,
                mode=resolve_network_export_mode("network_journal_cocitation_static"),
            )
            if html_content:
                network_html_items.append(("06_network_html/journal_cocitation_network.html", html_content))
            add_dataframe(
                "07_analysis_tables/top_cited_journals.csv",
                pd.DataFrame(journal_freq.most_common(200), columns=["Journal", "Co-citations"]),
            )

    fig_timeline = render_citespace_timeline(df, keywords_list, keyword_freq, top_n=15) if include("temporal_keyword_timeline") else None
    if fig_timeline is not None:
        add_plotly(f"03_temporal/temporal_keyword_evolution.{export_format}", fig_timeline, height=850)

    fig_burst = render_burst_detection(df, keywords_list, keyword_freq, top_n=20) if include("temporal_burst_detection") else None
    if fig_burst is not None:
        add_plotly(
            f"03_temporal/burst_detection.{export_format}",
            fig_burst,
            height=max(850, fig_burst.layout.height or 850),
        )

    fig_alluvial = render_alluvial_topic_flow(df, keywords_list, keyword_freq) if include("temporal_alluvial_topic_flow") else None
    if fig_alluvial is not None:
        add_plotly(
            f"03_temporal/alluvial_topic_flow.{export_format}",
            fig_alluvial,
            height=max(850, fig_alluvial.layout.height or 850),
        )

    forecast_horizon = get_analysis_value("publication_forecast_horizon", 4)
    keyword_opportunity_topn = get_analysis_value("keyword_opportunity_topn", 20)
    entity_forecast_topn = get_analysis_value("entity_forecast_topn", 10)
    leadership_shift_type = str(get_analysis_value("forward_leadership_shift_type", "Countries"))
    leadership_shift_topn = get_analysis_value("leadership_shift_topn", 10)
    theme_migration_slices = get_analysis_value("theme_migration_slices", 4)
    theme_migration_topn = get_analysis_value("theme_migration_topn", 10)

    if "Year" in df.columns and include("temporal_publication_forecast"):
        publication_forecast_df, publication_forecast_meta = build_publication_forecast_frame(
            df,
            forecast_horizon=forecast_horizon,
        )
        if not publication_forecast_df.empty:
            fig = render_publication_forecast_figure(publication_forecast_df)
            if fig is not None:
                add_plotly(f"03_temporal/publication_forecast.{export_format}", fig, height=max(850, fig.layout.height or 850))
            add_dataframe("07_analysis_tables/publication_forecast.csv", publication_forecast_df)
            add_text(
                "07_analysis_tables/publication_forecast_interpretation.txt",
                summarize_publication_forecast(publication_forecast_df, publication_forecast_meta),
            )

    if "Year" in df.columns and keyword_freq and include("temporal_keyword_opportunity_map"):
        keyword_opportunity_df = build_keyword_opportunity_map_frame(
            df,
            keywords_list,
            keyword_freq,
            top_n_keywords=max(30, keyword_opportunity_topn * 2),
            recent_year_window=4,
            min_total_occurrences=3,
        )
        if not keyword_opportunity_df.empty:
            fig = render_keyword_opportunity_map(keyword_opportunity_df, top_n=keyword_opportunity_topn)
            if fig is not None:
                add_plotly(f"03_temporal/keyword_opportunity_map.{export_format}", fig, height=max(850, fig.layout.height or 850))
            add_dataframe("07_analysis_tables/keyword_opportunity_map.csv", keyword_opportunity_df.head(keyword_opportunity_topn))
            add_text(
                "07_analysis_tables/keyword_opportunity_interpretation.txt",
                summarize_keyword_opportunity_map(keyword_opportunity_df, top_n=keyword_opportunity_topn),
            )

    if "Year" in df.columns and "Journal" in df.columns and include("temporal_journal_growth_forecast"):
        entity_summary_df, entity_series_df, entity_label = build_entity_forecast_tables(
            df,
            entity_type="journal",
            top_n_entities=entity_forecast_topn,
            forecast_horizon=forecast_horizon,
            lookback_years=6,
            min_total_occurrences=3,
        )
        if not entity_summary_df.empty:
            fig = render_entity_forecast_rank_figure(entity_summary_df, entity_label=entity_label, top_n=entity_forecast_topn)
            if fig is not None:
                add_plotly(f"03_temporal/journal_growth_forecast_rank.{export_format}", fig, height=max(850, fig.layout.height or 850))
            fig = render_entity_forecast_trajectory_figure(entity_series_df, entity_label=entity_label, top_n=min(5, entity_forecast_topn))
            if fig is not None:
                add_plotly(f"03_temporal/journal_growth_forecast_trajectories.{export_format}", fig, height=max(850, fig.layout.height or 850))
            add_dataframe("07_analysis_tables/journal_growth_forecast.csv", entity_summary_df)
            add_dataframe("07_analysis_tables/journal_growth_forecast_series.csv", entity_series_df)
            add_text("07_analysis_tables/journal_growth_forecast_interpretation.txt", summarize_entity_forecast_signals(entity_summary_df, entity_label, top_n=min(5, entity_forecast_topn)))

    if "Year" in df.columns and "Affiliations" in df.columns and include("temporal_country_growth_forecast"):
        entity_summary_df, entity_series_df, entity_label = build_entity_forecast_tables(
            df,
            entity_type="country",
            top_n_entities=entity_forecast_topn,
            forecast_horizon=forecast_horizon,
            lookback_years=6,
            min_total_occurrences=3,
        )
        if not entity_summary_df.empty:
            fig = render_entity_forecast_rank_figure(entity_summary_df, entity_label=entity_label, top_n=entity_forecast_topn)
            if fig is not None:
                add_plotly(f"03_temporal/country_growth_forecast_rank.{export_format}", fig, height=max(850, fig.layout.height or 850))
            fig = render_entity_forecast_trajectory_figure(entity_series_df, entity_label=entity_label, top_n=min(5, entity_forecast_topn))
            if fig is not None:
                add_plotly(f"03_temporal/country_growth_forecast_trajectories.{export_format}", fig, height=max(850, fig.layout.height or 850))
            add_dataframe("07_analysis_tables/country_growth_forecast.csv", entity_summary_df)
            add_dataframe("07_analysis_tables/country_growth_forecast_series.csv", entity_series_df)
            add_text("07_analysis_tables/country_growth_forecast_interpretation.txt", summarize_entity_forecast_signals(entity_summary_df, entity_label, top_n=min(5, entity_forecast_topn)))

    if "Year" in df.columns and "Affiliations" in df.columns and include("temporal_institution_growth_forecast"):
        entity_summary_df, entity_series_df, entity_label = build_entity_forecast_tables(
            df,
            entity_type="institution",
            top_n_entities=entity_forecast_topn,
            forecast_horizon=forecast_horizon,
            lookback_years=6,
            min_total_occurrences=3,
        )
        if not entity_summary_df.empty:
            fig = render_entity_forecast_rank_figure(entity_summary_df, entity_label=entity_label, top_n=entity_forecast_topn)
            if fig is not None:
                add_plotly(f"03_temporal/institution_growth_forecast_rank.{export_format}", fig, height=max(850, fig.layout.height or 850))
            fig = render_entity_forecast_trajectory_figure(entity_series_df, entity_label=entity_label, top_n=min(5, entity_forecast_topn))
            if fig is not None:
                add_plotly(f"03_temporal/institution_growth_forecast_trajectories.{export_format}", fig, height=max(850, fig.layout.height or 850))
            add_dataframe("07_analysis_tables/institution_growth_forecast.csv", entity_summary_df)
            add_dataframe("07_analysis_tables/institution_growth_forecast_series.csv", entity_series_df)
            add_text("07_analysis_tables/institution_growth_forecast_interpretation.txt", summarize_entity_forecast_signals(entity_summary_df, entity_label, top_n=min(5, entity_forecast_topn)))

    if "Year" in df.columns and "Affiliations" in df.columns and include("temporal_entity_leadership_shift"):
        leadership_entity_type = "institution" if leadership_shift_type == "Institutions" else "country"
        leadership_summary_df, leadership_series_df, leadership_label = build_entity_leadership_shift_tables(
            df,
            entity_type=leadership_entity_type,
            top_n_entities=leadership_shift_topn,
            recent_year_window=4,
            min_total_occurrences=3,
        )
        if not leadership_summary_df.empty:
            fig = render_entity_leadership_shift_figure(
                leadership_summary_df,
                leadership_label,
                top_n=leadership_shift_topn,
            )
            if fig is not None:
                add_plotly(
                    f"03_temporal/{leadership_label.lower()}_leadership_shift.{export_format}",
                    fig,
                    height=max(850, fig.layout.height or 850),
                )
            fig = render_entity_leadership_trajectory_figure(
                leadership_series_df,
                leadership_summary_df,
                leadership_label,
                top_n=min(5, leadership_shift_topn),
            )
            if fig is not None:
                add_plotly(
                    f"03_temporal/{leadership_label.lower()}_leadership_share_over_time.{export_format}",
                    fig,
                    height=max(850, fig.layout.height or 850),
                )
            add_dataframe(
                f"07_analysis_tables/{leadership_label.lower()}_leadership_shift.csv",
                leadership_summary_df,
            )
            add_dataframe(
                f"07_analysis_tables/{leadership_label.lower()}_leadership_series.csv",
                leadership_series_df,
            )
            add_text(
                f"07_analysis_tables/{leadership_label.lower()}_leadership_interpretation.txt",
                summarize_entity_leadership_shift(
                    leadership_summary_df,
                    leadership_label,
                    top_n=min(5, leadership_shift_topn),
                ),
            )

    if "Year" in df.columns and keyword_freq and include("temporal_theme_migration_forecast"):
        theme_summary_df, theme_chain_df = build_theme_migration_forecast_tables(
            df,
            keywords_list,
            keyword_freq,
            slice_count=theme_migration_slices,
            top_n_keywords=max(30, theme_migration_topn * 4),
            max_topics_per_slice=min(6, max(3, theme_migration_topn)),
            keywords_per_topic=3,
        )
        if not theme_summary_df.empty:
            fig = render_theme_migration_trajectory_figure(theme_chain_df, theme_summary_df, top_n=min(6, theme_migration_topn))
            if fig is not None:
                add_plotly(f"03_temporal/theme_migration_trajectories.{export_format}", fig, height=max(850, fig.layout.height or 850))
            fig = render_theme_migration_opportunity_map(theme_summary_df, top_n=theme_migration_topn)
            if fig is not None:
                add_plotly(f"03_temporal/theme_migration_hotspot_map.{export_format}", fig, height=max(850, fig.layout.height or 850))
            add_dataframe("07_analysis_tables/theme_migration_forecast.csv", theme_summary_df)
            add_dataframe("07_analysis_tables/theme_migration_chain_series.csv", theme_chain_df)
            add_text("07_analysis_tables/theme_migration_interpretation.txt", summarize_theme_migration_signals(theme_summary_df, top_n=min(5, theme_migration_topn)))

    if "Year" in df.columns and include("structure_annual_growth_rate"):
        year_counts = df_valid_year["Year"].value_counts().sort_index() if df_valid_year is not None else pd.Series(dtype=int)
        growth_data = []
        for idx in range(1, len(year_counts)):
            prev = year_counts.iloc[idx - 1]
            curr = year_counts.iloc[idx]
            growth_data.append(
                {
                    "Year": year_counts.index[idx],
                    "Growth Rate (%)": round((curr - prev) / prev * 100, 1) if prev > 0 else 0,
                }
            )
        if growth_data:
            growth_df = pd.DataFrame(growth_data)
            fig = px.line(growth_df, x="Year", y="Growth Rate (%)", title="Annual Growth Rate", markers=True)
            fig.update_traces(line_color=SCIENTIFIC_COLORWAY[1], marker_color=SCIENTIFIC_COLORWAY[1])
            fig.add_hline(y=0, line_dash="dash", line_color="#666666", opacity=0.6)
            add_plotly(f"04_statistics/annual_growth_rate.{export_format}", fig)
            add_dataframe("07_analysis_tables/annual_growth_rate.csv", growth_df)

    top_n_matrix = min(15, len(keyword_freq)) if include("structure_cooccurrence_matrix") else 0
    if top_n_matrix >= 5:
        top_keywords = [kw for kw, _ in _most_common_items(keyword_freq, top_n_matrix)]
        matrix = np.full((top_n_matrix, top_n_matrix), np.nan)
        for i, keyword_a in enumerate(top_keywords):
            matrix[i][i] = 1.0
            for j in range(i):
                keyword_b = top_keywords[j]
                weight = cooccurrence.get((keyword_a, keyword_b), cooccurrence.get((keyword_b, keyword_a), 0))
                union = keyword_freq[keyword_a] + keyword_freq[keyword_b] - weight
                matrix[i][j] = weight / union if union > 0 else 0
        fig = px.imshow(
            pd.DataFrame(matrix, index=top_keywords, columns=top_keywords),
            text_auto=".2f",
            aspect="auto",
            title="Co-occurrence Matrix (Jaccard Index)",
            color_continuous_scale=KEYWORD_MATRIX_SEQUENTIAL_SCALE,
            range_color=[0, 1],
        )
        fig.update_layout(height=850)
                                
        fig.update_traces(textfont=dict(size=14))
        add_plotly(f"04_statistics/cooccurrence_matrix.{export_format}", fig, height=950)
        add_dataframe(
            "07_analysis_tables/cooccurrence_matrix_keyword_frequency.csv",
            pd.DataFrame({"Keyword": top_keywords, "Frequency": [keyword_freq[kw] for kw in top_keywords]}),
        )

    fig_thematic = render_thematic_map(keyword_freq, cooccurrence, top_n=min(20, len(keyword_freq))) if include("structure_thematic_map") else None
    if fig_thematic is not None:
        add_plotly(f"04_statistics/thematic_map.{export_format}", fig_thematic, height=900)

    if keyword_freq and include("structure_hierarchical_cluster_heatmap"):
        fig_heatmap, matrix_df = render_hierarchical_cluster_heatmap(
            keywords_list,
            keyword_freq,
            cooccurrence,
            top_n=16,
        )
        if fig_heatmap is not None:
            add_plotly(f"04_statistics/hierarchical_cluster_heatmap.{export_format}", fig_heatmap, height=950)
            add_dataframe("07_analysis_tables/hierarchical_cluster_heatmap_matrix.csv", matrix_df)

    if "Authors" in df.columns:
        fig_author_time = render_author_production_over_time(df, top_n=10) if include("structure_author_production_over_time") else None
        if fig_author_time is not None:
            add_plotly(f"04_statistics/authors_production_over_time.{export_format}", fig_author_time, height=850)
        fig_three_field = render_three_field_plot(df, keywords_list, keyword_freq, 10, 15, 10) if include("structure_three_field_plot") else None
        if fig_three_field is not None:
            add_plotly(f"04_statistics/three_field_plot.{export_format}", fig_three_field, height=950)
        lotka_result = render_lotkas_law(df) if include("structure_lotkas_law") else None
        if lotka_result:
            fig_lotka, _ = lotka_result
            add_plotly(f"04_statistics/lotkas_law.{export_format}", fig_lotka, height=850)

    if "Language" in df.columns and (include("structure_language_distribution_pie") or include("structure_language_distribution_bar")):
        lang_series = df["Language"].fillna("").astype(str).str.strip()
        lang_series = lang_series[(lang_series != "") & (lang_series != "nan")]
        lang_counts = lang_series.value_counts().head(20)
        if not lang_counts.empty:
            lang_df = lang_counts.rename_axis("Language").reset_index(name="Papers")
            if include("structure_language_distribution_pie"):
                fig = px.pie(lang_df, values="Papers", names="Language", title="Publication Language Distribution")
                add_plotly(f"04_Bibliometric_Indicators/language_distribution_pie.{export_format}", fig)
            if include("structure_language_distribution_bar"):
                lang_plot_df = lang_df.sort_values("Papers", ascending=False).copy()
                fig = px.bar(
                    lang_plot_df,
                    x="Papers",
                    y="Language",
                    orientation="h",
                    title="Papers by Language",
                )
                fig.update_traces(marker_color=SCIENTIFIC_COLORWAY[7])
                fig.update_yaxes(
                    categoryorder="array",
                    categoryarray=list(reversed(lang_plot_df["Language"].tolist())),
                )
                fig.update_layout(height=650)
                add_plotly(f"04_Bibliometric_Indicators/language_distribution_bar.{export_format}", fig, height=850)
            add_dataframe("07_analysis_tables/language_distribution.csv", lang_df)

    innovation_items_requested = any(
        include(item_id)
        for item_id in (
            "innovation_structural_hole_profile",
            "innovation_brokerage_robustness",
        )
    )
    if innovation_items_requested and "Cited_References" in df.columns:
        if include("innovation_structural_hole_profile"):
            graph_bc, _, _ = build_bibliographic_coupling_network(
                df,
                top_n=bc_topn_val,
                min_shared_refs=bc_min_shared_val,
            )
            structural_hole_frame = compute_structural_hole_frame(graph_bc)
            fig_brokerage = render_structural_hole_brokerage_profile(structural_hole_frame, top_n=20)
            if fig_brokerage is not None:
                add_plotly(
                    f"06_Innovation_Analysis/structural_hole_brokerage_profile.{export_format}",
                    fig_brokerage,
                    width=1800,
                    height=1200,
                )
            else:
                skipped_items.append("06_Innovation_Analysis/structural_hole_brokerage_profile: insufficient structural-hole data")

        if include("innovation_brokerage_robustness"):
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

            robustness_snapshot = compute_brokerage_robustness_experiment(
                df_robustness,
                top_n_values=robustness_top_n_values,
                min_shared_values=robustness_min_shared_values,
                reference_top_n=bc_topn_val,
                reference_min_shared=bc_min_shared_val,
                top_k=top_k,
            )
            fig_robustness = render_brokerage_robustness_summary(robustness_snapshot)
            if fig_robustness is not None:
                add_plotly(
                    f"06_Innovation_Analysis/brokerage_robustness_summary.{export_format}",
                    fig_robustness,
                    width=2000,
                    height=1200,
                )
            else:
                skipped_items.append("06_Innovation_Analysis/brokerage_robustness_summary: insufficient robustness data")

    if "Times_Cited" in df.columns and (include("citation_distribution") or include("citation_by_year") or include("citation_publication_citation_dual_axis")):
        df_cite = df.copy()
        df_cite["Times_Cited"] = pd.to_numeric(df_cite["Times_Cited"], errors="coerce").fillna(0).astype(int)
        if include("citation_distribution"):
            cite_values = df_cite["Times_Cited"].dropna()
            x_max = int(np.quantile(cite_values, 0.99)) if len(cite_values) > 0 else 50
            x_max = max(x_max, 20)
            if x_max <= 150:
                tick_step = 25
            elif x_max <= 300:
                tick_step = 50
            else:
                tick_step = 100
            fig = px.histogram(
                df_cite,
                x="Times_Cited",
                nbins=50,
                title="Citation Distribution",
                labels={"Times_Cited": "Times Cited", "count": "Number of Papers"},
            )
            fig.update_traces(marker_color=SCIENTIFIC_COLORWAY[11])
            fig.update_xaxes(range=[0, x_max], tick0=0, dtick=tick_step)
            add_plotly(f"05_Citation_Metrics/citation_distribution.{export_format}", fig)
            add_dataframe("07_analysis_tables/citation_distribution_raw.csv", df_cite[["Times_Cited"]])
        if "Year" in df_cite.columns and include("citation_publication_citation_dual_axis"):
            trend_df = build_publication_citation_trend_frame(df_cite)
            fig = render_publication_citation_dual_axis_figure(trend_df)
            if fig is not None:
                add_plotly(f"05_Citation_Metrics/annual_publications_avg_citations.{export_format}", fig, height=700)
                add_dataframe("07_analysis_tables/annual_publications_avg_citations.csv", trend_df)
        if "Year" in df_cite.columns and include("citation_by_year"):
            df_valid_cite = clean_year_column(df_cite)
            cite_by_year = df_valid_cite.groupby("Year")["Times_Cited"].mean().reset_index()
            cite_by_year.columns = ["Year", "Avg Citations"]
            fig = px.line(cite_by_year, x="Year", y="Avg Citations", title="Average Citations per Paper by Year", markers=True)
            fig.update_traces(marker_color="#2F74B8", line_color="#2F74B8")
            add_plotly(f"05_Citation_Metrics/citations_by_year.{export_format}", fig)
            add_dataframe("07_analysis_tables/average_citations_by_year.csv", cite_by_year)

    if ("WoS_Categories" in df.columns or "Research_Areas" in df.columns) and include("citation_subject_categories"):
        category_freq = Counter()
        if "WoS_Categories" in df.columns:
            category_freq.update(_count_semicolon_terms(df["WoS_Categories"]))
        if not category_freq and "Research_Areas" in df.columns:
            category_freq.update(_count_semicolon_terms(df["Research_Areas"]))
        if category_freq:
            cat_df = pd.DataFrame(_most_common_items(category_freq, 25), columns=["Category", "Count"])
            cat_plot_df = cat_df.sort_values("Count", ascending=False).copy()
            fig = px.bar(
                cat_plot_df,
                x="Count",
                y="Category",
                orientation="h",
                title="Top 25 Subject Categories",
            )
            fig.update_traces(marker_color=SCIENTIFIC_COLORWAY[6])
            fig.update_yaxes(
                categoryorder="array",
                categoryarray=list(reversed(cat_plot_df["Category"].tolist())),
            )
            fig.update_layout(height=850)
            add_plotly(f"05_Citation_Metrics/subject_categories.{export_format}", fig, height=950)
            add_dataframe("07_analysis_tables/subject_categories.csv", cat_df)

    if "DocType" in df.columns and include("citation_document_types"):
        type_counts = df["DocType"].value_counts()
        if not type_counts.empty:
            fig = px.pie(values=type_counts.values, names=type_counts.index, title="Document Types")
            add_plotly(f"05_Citation_Metrics/document_types.{export_format}", fig)
            add_dataframe(
                "07_analysis_tables/document_types.csv",
                type_counts.rename_axis("Document Type").reset_index(name="Count"),
            )

    if "Funding" in df.columns and include("citation_funding_agencies"):
        fund_freq = _count_semicolon_terms(df["Funding"])
                                     
        fund_freq = {k: v for k, v in fund_freq.items() if str(k).lower() not in ("none", "nan", "not available", "nil")}
        if fund_freq:
            fund_df = pd.DataFrame(_most_common_items(fund_freq, 20), columns=["Funding Agency", "Count"])
            fund_plot_df = fund_df.sort_values("Count", ascending=False).copy()
            fig = px.scatter(
                fund_plot_df,
                x="Count",
                y="Funding Agency",
                size="Count",
                size_max=24,
                title="Top 20 Funding Agencies",
            )
            fig.update_traces(marker_color=SCIENTIFIC_COLORWAY[8], line=dict(color="white", width=1))
            fig.update_yaxes(
                categoryorder="array",
                categoryarray=list(reversed(fund_plot_df["Funding Agency"].tolist())),
            )
            fig.update_layout(
                height=900,
                margin=dict(l=300, r=40, t=80, b=60)
            )
            add_plotly(f"05_Citation_Metrics/funding_agencies.{export_format}", fig, width=2100, height=1485)
            add_dataframe("07_analysis_tables/funding_agencies.csv", fund_df)

    if "Cited_References" in df.columns and (
        include("citation_reference_year_distribution")
        or include("citation_references_per_paper")
        or include("citation_rpys_spectroscopy")
        or include("citation_reference_burst_detection")
    ):
        ref_year_freq = {}
        ref_counts_per_paper = []
        for cr_value in df["Cited_References"].tolist():
            cr_str = str(cr_value)
            if pd.isna(cr_str) or cr_str.strip() in ("", "nan"):
                continue
            refs = [item.strip() for item in cr_str.split(";") if item.strip()]
            ref_counts_per_paper.append(len(refs))
            for ref in refs:
                year_match = re.search(r",\s*(\d{4})\b", ref)
                if year_match:
                    year = int(year_match.group(1))
                    if 1900 <= year <= 2030:
                        ref_year_freq[year] = ref_year_freq.get(year, 0) + 1
        if ref_year_freq and include("citation_reference_year_distribution"):
            ref_year_df = pd.DataFrame(sorted(ref_year_freq.items()), columns=["Year", "Reference Count"])
            fig = px.area(
                ref_year_df,
                x="Year",
                y="Reference Count",
                title="Distribution of Cited Reference Years",
            )
            fig.update_traces(line_color=SCIENTIFIC_COLORWAY[1], fillcolor="rgba(47, 116, 184, 0.18)")
            add_plotly(f"05_Citation_Metrics/reference_year_distribution.{export_format}", fig)
            add_dataframe("07_analysis_tables/reference_year_distribution.csv", ref_year_df)
        if include("citation_rpys_spectroscopy"):
            rpys_df, _ = extract_rpys_statistics(df)
            if not rpys_df.empty:
                fig = render_rpys_figure(rpys_df)
                add_plotly(f"05_Citation_Metrics/reference_publication_year_spectroscopy.{export_format}", fig, height=700)
                add_dataframe("07_analysis_tables/reference_publication_year_spectroscopy.csv", rpys_df)
        if include("citation_reference_burst_detection"):
            reference_burst_df = build_reference_burst_table(df, top_n=20)
            if not reference_burst_df.empty:
                fig = render_reference_burst_figure(reference_burst_df)
                add_plotly(f"05_Citation_Metrics/reference_burst_detection.{export_format}", fig, height=max(850, fig.layout.height or 850))
                add_dataframe("07_analysis_tables/reference_burst_detection.csv", reference_burst_df)
        if ref_counts_per_paper and include("citation_references_per_paper"):
            fig = px.histogram(
                x=ref_counts_per_paper,
                nbins=30,
                title="Number of References per Paper",
                labels={"x": "Number of References", "y": "Count"},
            )
            fig.update_traces(marker_color=SCIENTIFIC_COLORWAY[6])
            add_plotly(f"05_Citation_Metrics/references_per_paper.{export_format}", fig)
            add_dataframe(
                "07_analysis_tables/references_per_paper.csv",
                pd.DataFrame({"References Per Paper": ref_counts_per_paper}),
            )

    if "Publisher" in df.columns and include("citation_publisher_analysis"):
        publisher_series = df["Publisher"].fillna("").astype(str).str.strip()
        publisher_series = publisher_series[(publisher_series != "") & (publisher_series != "nan")]
        publisher_counts = publisher_series.value_counts().head(25)
        if not publisher_counts.empty:
            publisher_df = publisher_counts.rename_axis("Publisher").reset_index(name="Papers")
            pub_plot_df = publisher_df.sort_values("Papers", ascending=False).copy()
            fig = px.scatter(
                pub_plot_df,
                x="Papers",
                y="Publisher",
                size="Papers",
                size_max=24,
                title="Top 25 Publishers by Publication Count",
            )
            fig.update_traces(marker_color=SCIENTIFIC_COLORWAY[9], line=dict(color="white", width=1))
            fig.update_yaxes(
                categoryorder="array",
                categoryarray=list(reversed(pub_plot_df["Publisher"].tolist())),
            )
            fig.update_layout(height=850)
            add_plotly(f"05_Citation_Metrics/publisher_analysis.{export_format}", fig, height=950)
            add_dataframe("07_analysis_tables/publisher_analysis.csv", publisher_df)

    flush_pending_exports()

    for html_path, html_content in network_html_items:
        add_text(html_path, html_content)

    zf.writestr(
        "00_manifest.txt",
        build_publication_manifest(
            export_format=export_format,
            selected_items=selected_items,
            include_interactive_html=bool(network_html_items),
            skipped_items=skipped_items,
            static_export_status=static_export_status,
        ).encode("utf-8"),
    )
    zf.writestr(
        "98_export_status.json",
        json.dumps(
            {
                "static_plotly_export": static_export_status,
                "plotly_export_attempts": plotly_export_attempts,
                "plotly_export_successes": plotly_export_successes,
                "matplotlib_export_attempts": matplotlib_export_attempts,
                "matplotlib_export_successes": matplotlib_export_successes,
                "parallel_export_workers": export_parallel_workers_used,
                "skipped_item_count": len(skipped_items),
                "bundle_contains_interactive_html": bool(network_html_items),
            },
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8"),
    )
    if skipped_items:
        zf.writestr("99_export_log.txt", "\n".join(skipped_items).encode("utf-8"))

    if own_zip:
        zf.close()
        return bundle_buffer.getvalue()
    return None
