from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable

import community as community_louvain
import networkx as nx
import pandas as pd
import plotly.express as px
import streamlit as streamlit

from modules.advanced_visualizations import (
    build_country_impact_quadrant_frame,
    render_circular_cluster_chord_figure,
    render_country_impact_quadrant_figure,
)
from modules.export_bundle import SCIENTIFIC_COLORWAY, style_publication_figure
from modules.citation_analysis import (
    build_author_cocitation_network,
    build_cocitation_network,
    build_journal_cocitation_network,
    clean_cited_reference,
    extract_author_from_citation,
    extract_journal_from_citation,
)
from modules.data_pipeline import (
    _extract_country_from_affiliation,
    _parse_wos_authors,
)
from modules.experiment_framework import build_bibliographic_coupling_network
from modules.network_builders import build_keyword_journal_cooccurrence
from modules.network_visualization import (
    _compute_network_positions,
    node_groups_from_cluster_stats,
    render_keyword_journal_network,
    render_network_html,
    render_network_publication_figure,
    render_vosviewer_style,
)
from modules.ui_helpers import (
    apply_publication_style_with_overrides,
    download_html_button,
    download_plotly_button,
    integer_control,
    render_and_download_network_figure,
    render_html_iframe,
    render_plotly_chart,
    render_plot_style_controls,
    show_cluster_report,
)


FIGURE_MODE_UI_COPY = {
    "Standard Figure": (
        "Recommended for manuscript-ready figures. Shows only the most important labels in each cluster, "
        "emphasizes core nodes, weakens peripheral nodes and cross-cluster links, and is best suited to Top 20-50."
    ),
    "Dense Map": (
        "Recommended for dense map-style inspection. Keeps more labels with smaller nodes to preserve "
        "overall map texture and is best suited to Top 60-150 for VOSviewer-style comparison."
    ),
    "Interactive Exploration": (
        "Recommended for online exploration. Supports drag, zoom, and hover inspection, "
        "but is intended for exploration rather than final manuscript figures."
    ),
}

FIGURE_MODE_OPTIONS = [
    "Standard Figure",
    "Dense Map",
    "Interactive Exploration",
]

STATIC_VIEW_MODE = "static"
INTERACTIVE_VIEW_MODE = "interactive"


def _normalize_hex_color(value: str, fallback: str) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"#[0-9A-Fa-f]{6}", text):
        return text.upper()
    return fallback


def _render_color_control(
    st: Any,
    *,
    key_prefix: str,
    label: str,
    default_color: str,
) -> str:
    preset_options = {
        "Default": default_color,
        "Blue": "#4C78A8",
        "Red": "#C95C5C",
        "Gray": "#8796A5",
        "Custom": None,
    }
    preset = st.selectbox(
        f"{label} Preset",
        list(preset_options.keys()),
        key=f"{key_prefix}_preset",
    )
    if preset == "Custom":
        custom_value = st.text_input(
            f"{label} HEX",
            value=st.session_state.get(f"{key_prefix}_hex", default_color),
            key=f"{key_prefix}_hex",
            help="Use a 6-digit HEX color such as #4C78A8.",
        )
        color = _normalize_hex_color(custom_value, default_color)
    else:
        color = preset_options[preset]
    st.caption(f"Current color: `{color}`")
    return color


def _render_network_style_controls(
    st: Any,
    *,
    key_prefix: str,
    current_view_mode: str,
    current_figure_mode: str,
    default_edge_color: str = "#B0B0B0",
    default_label_font_size: int = 23,
    default_max_visible_labels: int = 24,
    default_node_size_min: int = 24,
    default_node_size_max: int = 84,
) -> dict[str, int | float | str]:
    section_label = "Interactive Style" if current_view_mode == INTERACTIVE_VIEW_MODE else "Figure Style"
    if current_view_mode == INTERACTIVE_VIEW_MODE:
        mode_token = "interactive"
        mode_caption = "Interactive Exploration defaults keep labels and nodes slightly larger for on-screen inspection."
        edge_width_default = 0.58
        edge_alpha_default = 0.68
    elif current_figure_mode == "map":
        mode_token = "dense_map"
        mode_caption = "Dense Map defaults keep links slightly lighter to avoid visual crowding in dense layouts."
        edge_width_default = 0.42
        edge_alpha_default = 0.52
    else:
        mode_token = "standard"
        mode_caption = "Standard Figure defaults emphasize readability for manuscript-ready static figures."
        edge_width_default = 0.56
        edge_alpha_default = 0.72

    style_key_prefix = f"{key_prefix}_{mode_token}"
    with st.expander(section_label, expanded=False):
        st.caption("Keep critical analysis parameters in the main panel; open this section only when you need visual fine-tuning.")
        st.caption(f"Active style profile: {mode_caption}")
        st.caption("Changes in this panel are applied only after you click Apply Network Style.")
        with st.form(key=f"{style_key_prefix}_network_style_form", clear_on_submit=False):
            max_visible_labels = st.slider(
                "Max Visible Labels",
                1,
                60,
                default_max_visible_labels,
                key=f"{style_key_prefix}_max_visible_labels",
            )
            label_font_size = st.slider(
                "Label Font Size",
                10,
                40,
                default_label_font_size,
                key=f"{style_key_prefix}_label_font_size",
            )
            node_size_min = st.slider(
                "Node Min Size",
                10,
                100,
                default_node_size_min,
                key=f"{style_key_prefix}_node_min_size",
            )
            node_size_max = st.slider(
                "Node Max Size",
                20,
                150,
                default_node_size_max,
                key=f"{style_key_prefix}_node_max_size",
            )
            edge_width_scale = st.slider(
                "Edge Thickness",
                0.0,
                1.0,
                edge_width_default,
                step=0.01,
                key=f"{style_key_prefix}_edge_width_scale",
                help="0 hides edges; 1 applies the strongest edge-width setting.",
            )
            edge_alpha_scale = st.slider(
                "Edge Opacity",
                0.0,
                1.0,
                edge_alpha_default,
                step=0.01,
                key=f"{style_key_prefix}_edge_alpha_scale",
                help="0 makes edges fully transparent; 1 applies the most solid line opacity.",
            )
            edge_color = _render_color_control(
                st,
                key_prefix=f"{style_key_prefix}_edge_color",
                label="Edge Color",
                default_color=default_edge_color,
            )
            st.form_submit_button("Apply Network Style", use_container_width=True)
    return {
        "max_visible_labels": max_visible_labels,
        "label_font_size": label_font_size,
        "node_size_min": node_size_min,
        "node_size_max": node_size_max,
        "edge_color": edge_color,
        "edge_width_scale": edge_width_scale,
        "edge_alpha_scale": edge_alpha_scale,
    }


def _resolve_network_figure_mode(
    st: Any,
    *,
    key_prefix: str,
    default_label: str = "Standard Figure",
) -> tuple[str, str, str]:
    default_index = FIGURE_MODE_OPTIONS.index(default_label) if default_label in FIGURE_MODE_OPTIONS else 0
    figure_mode_label = st.selectbox(
        "Display Mode",
        FIGURE_MODE_OPTIONS,
        index=default_index,
        key=f"{key_prefix}_figure_mode",
    )
    st.caption(FIGURE_MODE_UI_COPY[figure_mode_label])
    if figure_mode_label == "Interactive Exploration":
        st.caption(
            "Export strategy: uses interactive HTML as the primary exploration output, "
            "while keeping an optional static figure export as a companion."
        )
        return figure_mode_label, INTERACTIVE_VIEW_MODE, "publication"
    st.caption(
        "Export strategy: uses the current static figure as the primary manuscript output "
        "and provides matching HTML companion export when available."
    )
    return (
        figure_mode_label,
        STATIC_VIEW_MODE,
        "publication" if figure_mode_label == "Standard Figure" else "map",
    )


def _build_partitioned_node_groups(
    graph: nx.Graph,
    palette: list[str],
) -> tuple[dict[Any, int], dict[Any, dict[str, Any]]]:
    try:
        partition = community_louvain.best_partition(graph)
    except Exception:
        partition = {node: 0 for node in graph.nodes()}

    node_groups: dict[Any, dict[str, Any]] = {}
    for node in graph.nodes():
        cluster_id = partition.get(node, 0)
        node_groups[node] = {
            "color": palette[cluster_id % len(palette)],
            "group": cluster_id,
            "shape": "dot",
            "type_label": f"Cluster {cluster_id + 1}: ",
        }
    return partition, node_groups


_INSTITUTION_PRIMARY_TOKENS = (
    "univ",
    "university",
    "inst",
    "institute",
    "college",
    "hospital",
    "hosp",
    "centre",
    "center",
    "academy",
    "cdc",
    "foundation",
    "fundacao",
    "ministry",
    "ministerio",
    "minist",
    "school",
)

_INSTITUTION_SECONDARY_TOKENS = (
    "laboratory",
    "lab",
    "programme",
    "program",
    "agency",
    "service",
    "clinic",
)

_DEPARTMENT_LIKE_PREFIXES = (
    "dept",
    "department",
    "div",
    "division",
    "fac",
    "faculty",
    "sch",
    "school of",
    "college of",
    "lab",
    "laboratory",
    "unit",
    "program",
    "programme",
    "center for",
    "centre for",
)


def _clean_affiliation_piece(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    return cleaned.strip(" ,.;")


def _institution_segment_score(segment: str, country_hint: set[str]) -> int:
    cleaned = _clean_affiliation_piece(segment)
    if not cleaned:
        return -100

    lowered = cleaned.lower()
    if cleaned in country_hint or lowered in {value.lower() for value in country_hint}:
        return -100
    if re.fullmatch(r"[A-Z]{1,3}\s*\d[\w\- ]*", cleaned):
        return -50
    if re.fullmatch(r"\d[\d\- ]*", cleaned):
        return -50

    score = 0
    if any(token in lowered for token in _INSTITUTION_PRIMARY_TOKENS):
        score += 8
    if any(token in lowered for token in _INSTITUTION_SECONDARY_TOKENS):
        score += 3
    if any(lowered.startswith(prefix) for prefix in _DEPARTMENT_LIKE_PREFIXES):
        score -= 5
    if re.search(r"\bdept\b|\bdepartment\b|\bdivision\b|\bfaculty\b|\bunit\b", lowered):
        score -= 2
    if re.search(r"\buniv\b|\buniversity\b|\binst\b|\binstitute\b|\bhospital\b|\bhosp\b", lowered):
        score += 2
    if len(cleaned) < 4:
        score -= 10
    return score


def _extract_institutions_from_affiliation(affiliation_text: str) -> list[str]:
    if pd.isna(affiliation_text) or str(affiliation_text).strip() in ("", "nan"):
        return []

    institutions: set[str] = set()
    address_blocks = re.split(r";\s*(?=\[)", str(affiliation_text))
    if len(address_blocks) <= 1:
        address_blocks = [str(affiliation_text)]

    for block in address_blocks:
        block = block.strip()
        if not block:
            continue

        address_body = block.split("]", 1)[1].strip() if "]" in block else block
        address_body = address_body.rstrip(".")
        if not address_body:
            continue

        country_hint = {
            item.strip()
            for item in _extract_country_from_affiliation(block).split(";")
            if item.strip()
        }
        pieces = [
            _clean_affiliation_piece(piece)
            for piece in address_body.split(",")
            if _clean_affiliation_piece(piece)
        ]
        if not pieces:
            continue

        scored_pieces = sorted(
            (
                (_institution_segment_score(piece, country_hint), index, piece)
                for index, piece in enumerate(pieces[:6])
            ),
            key=lambda item: (item[0], -item[1], len(item[2])),
            reverse=True,
        )
        best_score, _, best_piece = scored_pieces[0]
        if best_score >= 0:
            institutions.add(best_piece)
            continue

        fallback_candidates = [
            piece
            for piece in pieces[:3]
            if not any(piece.lower().startswith(prefix) for prefix in _DEPARTMENT_LIKE_PREFIXES)
        ]
        if fallback_candidates:
            institutions.add(fallback_candidates[0])
        else:
            institutions.add(pieces[0])

    return sorted(institutions)


def _annotate_association_strength(graph: nx.Graph) -> nx.Graph:
    if graph is None or graph.number_of_nodes() == 0:
        return graph

    for u, v, data in graph.edges(data=True):
        raw_weight = float(data.get("weight", 1.0) or 0.0)
        node_u_weight = float(graph.nodes[u].get("weight", 1.0) or 1.0)
        node_v_weight = float(graph.nodes[v].get("weight", 1.0) or 1.0)
        denominator = max(node_u_weight * node_v_weight, 1e-12)
        association_strength = raw_weight / denominator
        data["raw_weight"] = raw_weight
        data["association_strength"] = association_strength
    return graph


def _render_html_and_exports(
    st: Any,
    view_mode: str,
    html_content: str | None,
    graph: nx.Graph,
    node_groups: dict[Any, dict[str, Any]],
    html_filename: str,
    figure_key: str,
    figure_title: str,
    legend_label: str,
    *,
    layout_mode: str = "clustered",
    size_range: tuple[int, int] = (24, 84),
    label_max_len: int = 34,
    max_visible_labels: int = 24,
    label_font_size: int = 23,
    figure_mode: str = "publication",
    edge_color_override: str | None = None,
    edge_weight_attr: str = "weight",
    edge_width_scale: float = 1.0,
    edge_alpha_scale: float = 1.0,
) -> None:
    if graph is None or graph.number_of_nodes() == 0:
        return

    effective_html_content = html_content
    if view_mode == INTERACTIVE_VIEW_MODE and (
        edge_color_override
        or abs(float(edge_width_scale) - 1.0) > 1e-9
        or abs(float(edge_alpha_scale) - 1.0) > 1e-9
    ):
        effective_html_content, _ = render_network_html(
            graph,
            node_groups=node_groups,
            size_range=size_range,
            label_max_len=label_max_len,
            legend_label=legend_label,
            layout_mode=layout_mode,
            max_visible_labels=max_visible_labels,
            label_font_size=label_font_size,
            mode=figure_mode,
            edge_color_override=edge_color_override,
            edge_weight_attr=edge_weight_attr,
            edge_width_scale=edge_width_scale,
            edge_alpha_scale=edge_alpha_scale,
        )

    fig = render_network_publication_figure(
        graph,
        node_groups=node_groups,
        title=figure_title,
        size_range=size_range,
        label_max_len=label_max_len,
        legend_label=legend_label,
        layout_mode=layout_mode,
        max_visible_labels=max_visible_labels,
        label_font_size=label_font_size,
        mode=figure_mode,
        edge_color_override=edge_color_override,
        edge_weight_attr=edge_weight_attr,
        edge_width_scale=edge_width_scale,
        edge_alpha_scale=edge_alpha_scale,
    )
    if view_mode == STATIC_VIEW_MODE:
        if fig is not None:
            render_plotly_chart(
                fig,
                width="stretch",
                config={"displayModeBar": False},
            )
        else:
            st.warning("No network data to display. Try adjusting parameters.")
    elif effective_html_content:
        render_html_iframe(effective_html_content, height=840)
    else:
        st.warning("Interactive view is unavailable for the current network.")

    if view_mode == INTERACTIVE_VIEW_MODE:
        if effective_html_content:
            st.markdown("---")
            st.markdown(f"#### Interactive HTML Export for {figure_title}")
            download_html_button(effective_html_content, html_filename, "Download Interactive HTML")
        with st.expander(f"Static Exports for {figure_title}", expanded=False):
            render_and_download_network_figure(
                graph,
                figure_key,
                figure_title,
                node_groups=node_groups,
                legend_label=legend_label,
                layout_mode=layout_mode,
                size_range=size_range,
                label_max_len=label_max_len,
                max_visible_labels=max_visible_labels,
                label_font_size=label_font_size,
                mode=figure_mode,
                edge_color_override=edge_color_override,
                edge_weight_attr=edge_weight_attr,
                edge_width_scale=edge_width_scale,
                edge_alpha_scale=edge_alpha_scale,
                precomputed_fig=fig,
            )
    else:
        with st.expander(f"Static Exports for {figure_title}", expanded=False):
            render_and_download_network_figure(
                graph,
                figure_key,
                figure_title,
                node_groups=node_groups,
                legend_label=legend_label,
                layout_mode=layout_mode,
                size_range=size_range,
                label_max_len=label_max_len,
                max_visible_labels=max_visible_labels,
                label_font_size=label_font_size,
                mode=figure_mode,
                edge_color_override=edge_color_override,
                edge_weight_attr=edge_weight_attr,
                edge_width_scale=edge_width_scale,
                edge_alpha_scale=edge_alpha_scale,
                precomputed_fig=fig,
            )
        if effective_html_content:
            with st.expander(f"Interactive HTML Companion for {figure_title}", expanded=False):
                download_html_button(effective_html_content, html_filename, "Download Network HTML")
    st.markdown("### Standard Network Tables")
    map_df, network_df = _build_standard_network_export_tables(
        graph,
        node_groups=node_groups,
        layout_mode=layout_mode,
        edge_weight_attr=edge_weight_attr,
    )
    base_name = html_filename.rsplit(".", 1)[0]
    export_col1, export_col2 = st.columns(2)
    with export_col1:
        st.download_button(
            "Download MAP CSV",
            map_df.to_csv(index=False).encode("utf-8-sig"),
            f"Biblio-HUB_{base_name}_map.csv",
            "text/csv",
            key=f"{figure_key}_map_csv",
            use_container_width=True,
        )
    with export_col2:
        st.download_button(
            "Download NETWORK CSV",
            network_df.to_csv(index=False).encode("utf-8-sig"),
            f"Biblio-HUB_{base_name}_network.csv",
            "text/csv",
            key=f"{figure_key}_network_csv",
            use_container_width=True,
        )


def _build_standard_network_export_tables(
    graph: nx.Graph,
    *,
    node_groups: dict[Any, dict[str, Any]] | None = None,
    layout_mode: str = "clustered",
    edge_weight_attr: str = "weight",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if graph is None or graph.number_of_nodes() == 0:
        return pd.DataFrame(), pd.DataFrame()

    positions = _compute_network_positions(
        graph,
        node_groups=node_groups,
        layout_mode=layout_mode,
        edge_weight_attr=edge_weight_attr,
    )
    node_ids = {node: idx + 1 for idx, node in enumerate(graph.nodes())}
    weighted_degree = dict(graph.degree(weight="weight"))
    assoc_weighted_degree = {
        node: round(
            float(
                sum(
                    graph[node][neighbor].get("association_strength", graph[node][neighbor].get("weight", 0.0))
                    for neighbor in graph.neighbors(node)
                )
            ),
            12,
        )
        for node in graph.nodes()
    }
    degree = dict(graph.degree())

    map_rows = []
    for node in graph.nodes():
        x_coord, y_coord = positions.get(node, (0.0, 0.0))
        node_info = (node_groups or {}).get(node, {})
        map_rows.append(
            {
                "id": node_ids[node],
                "label": str(node),
                "x": round(float(x_coord), 6),
                "y": round(float(y_coord), 6),
                "cluster": int(node_info.get("group", 0)) + 1,
                "weight": graph.nodes[node].get("weight", 1),
                "weight<Links>": degree.get(node, 0),
                "weight<Total link strength>": round(float(weighted_degree.get(node, 0.0)), 6),
                "weight<Association strength>": assoc_weighted_degree.get(node, 0.0),
            }
        )

    edge_rows = []
    for source, target, data in graph.edges(data=True):
        edge_rows.append(
            {
                "source": node_ids[source],
                "target": node_ids[target],
                "weight": data.get("weight", 1),
                "association_strength": round(float(data.get("association_strength", data.get("weight", 1.0))), 12),
            }
        )

    map_df = pd.DataFrame(map_rows).sort_values(["cluster", "weight", "label"], ascending=[True, False, True])
    network_df = pd.DataFrame(edge_rows).sort_values(["source", "target"]).reset_index(drop=True)
    return map_df.reset_index(drop=True), network_df


@streamlit.cache_data(show_spinner=False)
def _count_available_keyword_journal_inputs(
    df: pd.DataFrame,
    keywords_list: list[list[str]],
) -> tuple[int, int]:
    journal_count = int(df["Journal"].dropna().nunique()) if "Journal" in df.columns else 0
    keyword_count = len({keyword for keyword_row in keywords_list for keyword in keyword_row})
    return keyword_count, journal_count


@streamlit.cache_data(show_spinner=False)
def _count_available_coauthors(df: pd.DataFrame) -> int:
    authors = set()
    for _, row in df.iterrows():
        authors.update(_parse_wos_authors(row.get("Authors", "")))
    return len(authors)


@streamlit.cache_data(show_spinner=False)
def _count_available_institutions(df: pd.DataFrame) -> int:
    institutions = set()
    for _, row in df.iterrows():
        institutions.update(_extract_institutions_from_affiliation(str(row.get("Affiliations", ""))))
    return len(institutions)


@streamlit.cache_data(show_spinner=False)
def _count_available_cited_items(df: pd.DataFrame) -> tuple[int, int, int]:
    reference_labels = set()
    cited_authors = set()
    cited_journals = set()
    for _, row in df.iterrows():
        cr_value = row.get("Cited_References", "")
        cr_str = str(cr_value)
        if pd.isna(cr_value) or cr_str.strip() in ("", "nan"):
            continue
        refs = [reference.strip() for reference in cr_str.split(";") if reference.strip()]
        for reference in refs:
            clean_ref = clean_cited_reference(reference)
            if clean_ref:
                reference_labels.add(clean_ref)
            author = extract_author_from_citation(reference)
            if author:
                cited_authors.add(re.sub(r"\s+", " ", author).strip().title())
            journal = extract_journal_from_citation(reference)
            if journal:
                cited_journals.add(re.sub(r"\s+", " ", journal).strip().title())
    return len(reference_labels), len(cited_authors), len(cited_journals)


@streamlit.cache_data(show_spinner=False)
def _build_coauthorship_network_assets(
    df: pd.DataFrame,
    min_papers: int,
    top_n_auth: int,
    palette: list[str],
    counting_mode: str = "full",
    min_links: int = 1,
) -> tuple[nx.Graph, dict[Any, int], dict[Any, dict[str, Any]], str | None, dict[str, Any] | None, pd.DataFrame]:
    author_pairs: Counter[tuple[str, str]] = Counter()
    author_freq: Counter[str] = Counter()
    for _, row in df.iterrows():
        authors = _parse_wos_authors(row.get("Authors", ""))
        n_authors = len(authors)
        if counting_mode == "fractional" and n_authors > 0:
            frac_node_weight = 1.0 / n_authors
            n_pairs = n_authors * (n_authors - 1) / 2 if n_authors >= 2 else 0
            frac_edge_weight = 1.0 / n_pairs if n_pairs > 0 else 1.0
            for author in authors:
                author_freq[author] += frac_node_weight
            for index in range(n_authors):
                for next_index in range(index + 1, n_authors):
                    pair = tuple(sorted([authors[index], authors[next_index]]))
                    author_pairs[pair] += frac_edge_weight
        else:
            for author in authors:
                author_freq[author] += 1
            for index in range(n_authors):
                for next_index in range(index + 1, n_authors):
                    pair = tuple(sorted([authors[index], authors[next_index]]))
                    author_pairs[pair] += 1

    eligible_authors = {
        author
        for author, weight in author_freq.items()
        if (counting_mode == "full" and weight >= min_papers) or (counting_mode == "fractional" and weight >= 1)
    }
    filtered_pairs: list[tuple[str, str, float]] = []
    collaboration_strength: Counter[str] = Counter()
    collaboration_degree: Counter[str] = Counter()
    connected_authors: set[str] = set()
    for (author_1, author_2), weight in author_pairs.items():
        if weight < min_links or author_1 not in eligible_authors or author_2 not in eligible_authors:
            continue
        filtered_pairs.append((author_1, author_2, weight))
        collaboration_strength[author_1] += weight
        collaboration_strength[author_2] += weight
        collaboration_degree[author_1] += 1
        collaboration_degree[author_2] += 1
        connected_authors.update((author_1, author_2))

    ranked_connected_edges = sorted(
        filtered_pairs,
        key=lambda item: (
            -float(item[2]),
            -min(float(author_freq[item[0]]), float(author_freq[item[1]])),
            -max(float(author_freq[item[0]]), float(author_freq[item[1]])),
            item[0].lower(),
            item[1].lower(),
        ),
    )
    ranked_connected_authors = sorted(
        connected_authors,
        key=lambda author: (
            -float(collaboration_strength[author]),
            -int(collaboration_degree[author]),
            -float(author_freq[author]),
            author.lower(),
        ),
    )
    ranked_eligible_authors = sorted(
        eligible_authors,
        key=lambda author: (
            -float(author_freq[author]),
            -float(collaboration_strength[author]),
            -int(collaboration_degree[author]),
            author.lower(),
        ),
    )

    selected_authors: list[str] = []
    selected_author_set: set[str] = set()
    for author_1, author_2, _ in ranked_connected_edges:
        for author in (author_1, author_2):
            if author not in selected_author_set:
                selected_authors.append(author)
                selected_author_set.add(author)
                if len(selected_authors) >= top_n_auth:
                    break
        if len(selected_authors) >= top_n_auth:
            break
    for author in ranked_connected_authors:
        if len(selected_authors) >= top_n_auth:
            break
        if author not in selected_author_set:
            selected_authors.append(author)
            selected_author_set.add(author)
    for author in ranked_eligible_authors:
        if len(selected_authors) >= top_n_auth:
            break
        if author not in selected_author_set:
            selected_authors.append(author)
            selected_author_set.add(author)

    graph = nx.Graph()
    for author in selected_authors:
        graph.add_node(author, weight=author_freq[author])
    for author_1, author_2, weight in filtered_pairs:
        if author_1 in graph.nodes and author_2 in graph.nodes:
            graph.add_edge(author_1, author_2, weight=weight)

    if graph.number_of_edges() > 0:
        connected_node_count = sum(1 for _, degree in graph.degree() if degree > 0)
        isolates = [node for node, degree in graph.degree() if degree == 0]
        if connected_node_count >= min(8, top_n_auth) and isolates:
            graph.remove_nodes_from(isolates)

    if graph.number_of_nodes() == 0:
        return graph, {}, {}, None, None, pd.DataFrame(columns=["Author", "Papers"])

    graph = _annotate_association_strength(graph)
    partition, node_groups = _build_partitioned_node_groups(graph, palette)
    html_content, net_stats = render_network_html(
        graph,
        node_groups=node_groups,
        legend_label="Papers",
    )
    top_auth_rows = sorted(
        (
            {
                "Author": author,
                "Papers": round(float(author_freq[author]), 4),
                "Collaborations": int(collaboration_degree[author]),
                "Total Link Strength": round(float(collaboration_strength[author]), 4),
            }
            for author in graph.nodes()
        ),
        key=lambda row: (
            -float(row["Total Link Strength"]),
            -int(row["Collaborations"]),
            -float(row["Papers"]),
            str(row["Author"]).lower(),
        ),
    )[:20]
    top_auth_df = pd.DataFrame(
        top_auth_rows,
        columns=["Author", "Papers", "Collaborations", "Total Link Strength"],
    )
    return graph, partition, node_groups, html_content, net_stats, top_auth_df


@streamlit.cache_data(show_spinner=False)
def _build_institution_collaboration_assets(
    df: pd.DataFrame,
    top_n_inst: int,
    palette: list[str],
    counting_mode: str = "full",
    min_links: int = 1,
) -> tuple[nx.Graph, dict[Any, int], dict[Any, dict[str, Any]], str | None, dict[str, Any] | None, pd.DataFrame]:
    inst_pairs: Counter[tuple[str, str]] = Counter()
    inst_freq: Counter[str] = Counter()
    for _, row in df.iterrows():
        institutions = set(_extract_institutions_from_affiliation(str(row.get("Affiliations", ""))))
        n_inst = len(institutions)

        if counting_mode == "fractional" and n_inst > 0:
            frac_node_weight = 1.0 / n_inst
            n_pairs = n_inst * (n_inst - 1) / 2 if n_inst >= 2 else 0
            frac_edge_weight = 1.0 / n_pairs if n_pairs > 0 else 1.0
            for institution in institutions:
                inst_freq[institution] += frac_node_weight
            institution_list = sorted(institutions)
            for index in range(len(institution_list)):
                for next_index in range(index + 1, len(institution_list)):
                    pair = tuple(sorted([institution_list[index], institution_list[next_index]]))
                    inst_pairs[pair] += frac_edge_weight
        else:
            for institution in institutions:
                inst_freq[institution] += 1
            institution_list = sorted(institutions)
            for index in range(len(institution_list)):
                for next_index in range(index + 1, len(institution_list)):
                    pair = tuple(sorted([institution_list[index], institution_list[next_index]]))
                    inst_pairs[pair] += 1

    top_institutions = {name for name, _ in inst_freq.most_common(top_n_inst)}
    graph = nx.Graph()
    for institution in top_institutions:
        graph.add_node(institution, weight=inst_freq[institution])
    for (institution_1, institution_2), weight in inst_pairs.most_common(300):
        if institution_1 in graph.nodes and institution_2 in graph.nodes and weight >= min_links:
            graph.add_edge(institution_1, institution_2, weight=weight)

    if graph.number_of_nodes() == 0:
        return graph, {}, {}, None, None, pd.DataFrame(columns=["Institution", "Papers"])

    graph = _annotate_association_strength(graph)
    partition, node_groups = _build_partitioned_node_groups(graph, palette)
    html_content, net_stats = render_network_html(
        graph,
        node_groups=node_groups,
        label_max_len=40,
        legend_label="Papers",
    )
    top_inst_df = pd.DataFrame(
        inst_freq.most_common(20),
        columns=["Institution", "Papers"],
    )
    return graph, partition, node_groups, html_content, net_stats, top_inst_df


@streamlit.cache_data(show_spinner=False)
def _build_country_collaboration_assets(
    df: pd.DataFrame,
    top_n_country: int,
    palette: list[str],
    counting_mode: str = "full",
    min_links: int = 1,
) -> tuple[nx.Graph, dict[Any, int], dict[Any, dict[str, Any]], str | None, dict[str, Any] | None, pd.DataFrame]:
    country_pairs: Counter[tuple[str, str]] = Counter()
    country_freq: Counter[str] = Counter()
    for _, row in df.iterrows():
        countries_str = _extract_country_from_affiliation(str(row.get("Affiliations", "")))
        if not countries_str:
            continue
        countries = [country.strip() for country in countries_str.split(";") if country.strip()]
        n_countries = len(countries)

        if counting_mode == "fractional" and n_countries > 0:
            frac_node_weight = 1.0 / n_countries
            n_pairs = n_countries * (n_countries - 1) / 2 if n_countries >= 2 else 0
            frac_edge_weight = 1.0 / n_pairs if n_pairs > 0 else 1.0
            for country in countries:
                country_freq[country] += frac_node_weight
            for index in range(n_countries):
                for next_index in range(index + 1, n_countries):
                    pair = tuple(sorted([countries[index], countries[next_index]]))
                    country_pairs[pair] += frac_edge_weight
        else:
            for country in countries:
                country_freq[country] += 1
            for index in range(n_countries):
                for next_index in range(index + 1, n_countries):
                    pair = tuple(sorted([countries[index], countries[next_index]]))
                    country_pairs[pair] += 1

    top_countries = {country for country, _ in country_freq.most_common(top_n_country)}
    graph = nx.Graph()
    for country, freq in country_freq.items():
        if country not in top_countries:
            continue
        graph.add_node(country, weight=freq)
    for (country_1, country_2), weight in country_pairs.items():
        if country_1 in graph.nodes and country_2 in graph.nodes and weight >= min_links:
            graph.add_edge(country_1, country_2, weight=weight)

    if graph.number_of_nodes() == 0:
        return graph, {}, {}, None, None, pd.DataFrame(columns=["Country", "Papers"])

    graph = _annotate_association_strength(graph)
    partition, node_groups = _build_partitioned_node_groups(graph, palette)
    html_content, net_stats = render_network_html(
        graph,
        node_groups=node_groups,
        legend_label="Papers",
    )
    country_df = pd.DataFrame(
        country_freq.most_common(max(20, top_n_country)),
        columns=["Country", "Papers"],
    )
    return graph, partition, node_groups, html_content, net_stats, country_df


def _render_keyword_cooccurrence(
    st: Any,
    view_mode: str,
    keyword_freq: dict[str, int],
    cooccurrence: dict[tuple[str, str], int],
    log_exception: Callable[[str, Exception], None],
) -> None:
    st.markdown(
        "Keyword co-occurrence network. Use `Standard Figure` for manuscript-style viewing and "
        "`Interactive Exploration` when you need drag-and-drop inspection."
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Network Parameters**")
        st.markdown("**Visualization Mode**")
        _, effective_view_mode, mode_key = _resolve_network_figure_mode(st, key_prefix="kw", default_label="Standard Figure")
        top_n = integer_control(
            st,
            "Number of Keywords",
            10,
            30,
            key="vos_topn",
            input_max=max(10, len(keyword_freq)),
            slider_soft_cap=60,
        )
        max_cooccurrence_weight = max((int(weight) for weight in cooccurrence.values()), default=10)
        min_weight = integer_control(
            st,
            "Min Co-occurrence",
            1,
            int(st.session_state.get("vos_minw", 2)),
            key="vos_minw",
            input_max=max(10, min(200, max_cooccurrence_weight)),
            slider_soft_cap=10,
        )
        style_controls = _render_network_style_controls(
            st,
            key_prefix="kw",
            current_view_mode=effective_view_mode,
            current_figure_mode=mode_key,
            default_max_visible_labels=26 if effective_view_mode == STATIC_VIEW_MODE else 28,
            default_label_font_size=23 if effective_view_mode == STATIC_VIEW_MODE else 25,
            default_node_size_min=24 if effective_view_mode == STATIC_VIEW_MODE else 30,
            default_node_size_max=84 if effective_view_mode == STATIC_VIEW_MODE else 98,
        )

    with col2:
        try:
            html_content, graph, net_stats = render_vosviewer_style(
                keyword_freq,
                cooccurrence,
                top_n,
                min_weight,
                size_range=(style_controls["node_size_min"], style_controls["node_size_max"]),
                edge_width_factor=0.26,
                max_visible_labels=style_controls["max_visible_labels"],
                label_font_size=style_controls["label_font_size"],
                edge_color_override=style_controls["edge_color"],
            )
        except Exception as exc:
            log_exception("render_vosviewer_style", exc)
            st.error(f"Error generating network: {exc}")
            html_content, graph, net_stats = None, None, None
        if graph and net_stats:
            keyword_node_groups = node_groups_from_cluster_stats(net_stats)
            _render_html_and_exports(
                st,
                effective_view_mode,
                html_content,
                graph,
                keyword_node_groups,
                "keyword_cooccurrence_network.html",
                "keyword_cooccurrence_figure",
                "Keyword Co-occurrence Network",
                "Frequency",
                layout_mode="clustered",
                size_range=(style_controls["node_size_min"], style_controls["node_size_max"]),
                label_max_len=34,
                max_visible_labels=style_controls["max_visible_labels"],
                label_font_size=style_controls["label_font_size"],
                figure_mode=mode_key,
                edge_color_override=style_controls["edge_color"],
                edge_width_scale=style_controls["edge_width_scale"],
                edge_alpha_scale=style_controls["edge_alpha_scale"],
            )
            st.markdown("---")
            st.markdown("### Network Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Nodes", net_stats["nodes"])
            with col2:
                st.metric("Edges", net_stats["edges"])
            with col3:
                st.metric("Density", f"{net_stats['density']:.4f}")
            with col4:
                st.metric("Avg Degree", f"{net_stats['avg_degree']:.2f}")
            if "clusters" in net_stats:
                show_cluster_report(net_stats, "keywords")


def _render_keyword_journal_association(
    st: Any,
    view_mode: str,
    df: pd.DataFrame,
    keywords_list: list[list[str]],
    log_exception: Callable[[str, Exception], None],
) -> None:
    st.markdown(
        "Bipartite network showing relationships between keywords and journals. "
        "Use `Standard Figure` for the main display; switch to `Interactive Exploration` for exploration. "
        "Keywords appear as circles, journals as squares."
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Visualization Mode**")
        _, effective_view_mode, mode_key = _resolve_network_figure_mode(st, key_prefix="kj", default_label="Standard Figure")
        available_keywords, available_journals = _count_available_keyword_journal_inputs(df, keywords_list)
        top_n_kw = integer_control(
            st,
            "Number of Keywords",
            5,
            15,
            key="kj_topn_kw",
            input_max=max(5, available_keywords),
            slider_soft_cap=40,
        )
        top_n_jn = integer_control(
            st,
            "Number of Journals",
            5,
            10,
            key="kj_topn_jn",
            input_max=max(5, available_journals),
            slider_soft_cap=30,
        )

        style_controls = _render_network_style_controls(
            st,
            key_prefix="kj",
            current_view_mode=effective_view_mode,
            current_figure_mode=mode_key,
            default_max_visible_labels=24,
            default_label_font_size=23,
            default_node_size_min=22,
            default_node_size_max=72,
        )

    (
        top_keywords,
        top_journals,
        kw_journal_cooccur,
        keyword_freq_local,
        journal_freq,
    ) = build_keyword_journal_cooccurrence(df, keywords_list, top_n_kw, top_n_jn)

    with col2:
        try:
            html_content, graph, node_groups, stats = render_keyword_journal_network(
                top_keywords,
                top_journals,
                kw_journal_cooccur,
                keyword_freq_local,
                journal_freq,
                size_range=(style_controls["node_size_min"], style_controls["node_size_max"]),
                max_visible_labels=style_controls["max_visible_labels"],
                label_font_size=style_controls["label_font_size"],
                edge_color_override=style_controls["edge_color"],
            )
        except Exception as exc:
            log_exception("render_keyword_journal_network", exc)
            st.error(f"Error generating network: {exc}")
            html_content, graph, node_groups, stats = None, None, None, None
        if stats and graph is not None and node_groups is not None:
            _render_html_and_exports(
                st,
                effective_view_mode,
                html_content,
                graph,
                node_groups,
                "keyword_journal_association_network.html",
                "keyword_journal_figure",
                "Keyword-Journal Association Network",
                "Frequency",
                layout_mode="bipartite",
                size_range=(style_controls["node_size_min"], style_controls["node_size_max"]),
                label_max_len=24,
                max_visible_labels=style_controls["max_visible_labels"],
                label_font_size=style_controls["label_font_size"],
                figure_mode=mode_key,
                edge_color_override=style_controls["edge_color"],
                edge_width_scale=style_controls["edge_width_scale"],
                edge_alpha_scale=style_controls["edge_alpha_scale"],
            )
            st.markdown("---")
            st.markdown("### Network Statistics")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Total Nodes", stats["nodes"])
            with col_s2:
                st.metric("Edges", stats["edges"])
            with col_s3:
                st.metric("Keyword Nodes", stats["keyword_nodes"])
            with col_s4:
                st.metric("Journal Nodes", stats["journal_nodes"])
            show_cluster_report(stats, "nodes")
            if kw_journal_cooccur:
                with st.expander("Top Keyword-Journal Pairs", expanded=False):
                    pair_list = sorted(
                        kw_journal_cooccur.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:20]
                    pair_df = pd.DataFrame(
                        [(keyword, journal, weight) for (keyword, journal), weight in pair_list],
                        columns=["Keyword", "Journal", "Co-occurrence"],
                    )
                    st.dataframe(pair_df, width="stretch", hide_index=True)


def _render_collaboration_graph(
    st: Any,
    view_mode: str,
    graph: nx.Graph,
    node_groups: dict[Any, dict[str, Any]],
    html_content: str | None,
    net_stats: dict[str, Any] | None,
    partition: dict[Any, int],
    *,
    html_filename: str,
    figure_key: str,
    figure_title: str,
    metric_labels: tuple[str, str],
    summary_label: str,
    top_table: pd.DataFrame | None = None,
    top_table_title: str | None = None,
    plotly_size_range: tuple[int, int] = (24, 84),
    plotly_label_max_len: int = 34,
    plotly_max_visible_labels: int = 24,
    plotly_label_font_size: int = 23,
    pyvis_size_range: tuple[int, int] = (24, 84),
    pyvis_label_max_len: int = 34,
    pyvis_max_visible_labels: int = 24,
    pyvis_label_font_size: int = 23,
    figure_mode: str = "publication",
    edge_color_override: str | None = None,
    edge_width_scale: float = 1.0,
    edge_alpha_scale: float = 1.0,
) -> None:
    layout_mode = "clustered" if node_groups else "auto"
    edge_weight_attr = "association_strength" if figure_mode == "map" else "weight"
    effective_html_content = html_content
    effective_net_stats = net_stats
    if graph.number_of_nodes() > 0 and (view_mode == INTERACTIVE_VIEW_MODE or edge_color_override):
        effective_html_content, effective_net_stats = render_network_html(
            graph,
            node_groups=node_groups,
            size_range=pyvis_size_range,
            label_max_len=pyvis_label_max_len,
            legend_label="Papers",
            layout_mode=layout_mode,
            max_visible_labels=pyvis_max_visible_labels,
            label_font_size=pyvis_label_font_size,
            mode=figure_mode,
            edge_color_override=edge_color_override,
            edge_weight_attr=edge_weight_attr,
            edge_width_scale=edge_width_scale,
            edge_alpha_scale=edge_alpha_scale,
        )

    if effective_html_content or graph.number_of_nodes() > 0:
        _render_html_and_exports(
            st,
            view_mode,
            effective_html_content,
            graph,
            node_groups,
            html_filename,
            figure_key,
            figure_title,
            "Papers",
            layout_mode=layout_mode,
            size_range=plotly_size_range,
            label_max_len=plotly_label_max_len,
            max_visible_labels=plotly_max_visible_labels,
            label_font_size=plotly_label_font_size,
            figure_mode=figure_mode,
            edge_color_override=edge_color_override,
            edge_weight_attr=edge_weight_attr,
            edge_width_scale=edge_width_scale,
            edge_alpha_scale=edge_alpha_scale,
        )
    if effective_net_stats:
        st.markdown("---")
        st.markdown("### Network Statistics")
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric(metric_labels[0], effective_net_stats["nodes"])
        with col_s2:
            st.metric(metric_labels[1], effective_net_stats["edges"])
        with col_s3:
            st.metric("Density", f"{effective_net_stats['density']:.4f}")
        with col_s4:
            st.metric("Clusters", len(set(partition.values())))
        show_cluster_report(effective_net_stats, summary_label)
    if top_table is not None and top_table_title:
        with st.expander(f"{top_table_title}", expanded=False):
            st.dataframe(top_table, width="stretch", hide_index=True)


def _render_coauthorship_network(
    st: Any,
    view_mode: str,
    df: pd.DataFrame,
    palette: list[str],
) -> None:
    st.markdown(
        "Co-authorship network visualization. Node size = number of publications; "
        "edge width = collaboration strength. `Standard Figure` is optimized for presentation, "
        "`Dense Map` is recommended for VOSviewer-style comparison, and `Interactive Exploration` "
        "is kept for exploration."
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Visualization Mode**")
        _, effective_view_mode, mode_key = _resolve_network_figure_mode(st, key_prefix="auth", default_label="Standard Figure")
        min_papers = integer_control(
            st,
            "Min Papers per Author",
            1,
            int(st.session_state.get("auth_min_papers", 2)),
            key="auth_min_papers",
            input_max=max(10, min(100, len(df))),
            slider_soft_cap=10,
        )
        top_n_auth = integer_control(
            st,
            "Max Authors",
            20,
            50,
            key="auth_topn",
            input_max=max(20, _count_available_coauthors(df)),
            slider_soft_cap=100,
        )
        counting_mode = st.selectbox("Counting Mode", ["full", "fractional"], 0, key="auth_counting_mode")
        min_links = integer_control(
            st,
            "Min Collaboration Links",
            1,
            int(st.session_state.get("auth_min_links", 1)),
            key="auth_min_links",
            input_max=max(10, min(100, top_n_auth)),
            slider_soft_cap=10,
        )
        st.caption(
            "Author selection now prioritizes researchers who participate in actual collaboration links, "
            "which reduces isolated high-output nodes and keeps the co-authorship backbone more visible."
        )

        style_controls = _render_network_style_controls(
            st,
            key_prefix="auth",
            current_view_mode=effective_view_mode,
            current_figure_mode=mode_key,
        )

    with col2:
        graph, partition, node_groups, html_content, net_stats, top_auth_df = (
            _build_coauthorship_network_assets(df, min_papers, top_n_auth, palette, counting_mode=counting_mode, min_links=min_links)
        )
        if graph.number_of_nodes() == 0:
            st.warning("Not enough author data to build network. Adjust parameters or check data.")
            return

        _render_collaboration_graph(
            st,
            effective_view_mode,
            graph,
            node_groups,
            html_content,
            net_stats,
            partition,
            html_filename="author_collaboration_network.html",
            figure_key="author_collaboration_figure",
            figure_title="Author Collaboration Network",
            metric_labels=("Authors", "Collaborations"),
            summary_label="authors",
            top_table=top_auth_df,
            top_table_title="Selected Authors in Current Network",
            plotly_size_range=(style_controls["node_size_min"], style_controls["node_size_max"]),
            plotly_label_max_len=36,
            plotly_max_visible_labels=style_controls["max_visible_labels"],
            plotly_label_font_size=style_controls["label_font_size"],
            pyvis_size_range=(style_controls["node_size_min"], style_controls["node_size_max"]),
            pyvis_label_max_len=36,
            pyvis_max_visible_labels=style_controls["max_visible_labels"],
            pyvis_label_font_size=style_controls["label_font_size"],
            figure_mode=mode_key,
            edge_color_override=style_controls["edge_color"],
            edge_width_scale=style_controls["edge_width_scale"],
            edge_alpha_scale=style_controls["edge_alpha_scale"],
        )


def _render_institutional_collaboration(
    st: Any,
    view_mode: str,
    df: pd.DataFrame,
    palette: list[str],
) -> None:
    st.markdown(
        "Institution co-occurrence network based on shared affiliations. "
        "Use `Standard Figure` for presentation, `Dense Map` for VOSviewer-style comparison, "
        "and `Interactive Exploration` for exploration."
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Visualization Mode**")
        _, effective_view_mode, mode_key = _resolve_network_figure_mode(st, key_prefix="inst", default_label="Standard Figure")
        top_n_inst = integer_control(
            st,
            "Number of Institutions",
            10,
            25,
            key="inst_topn",
            input_max=max(10, _count_available_institutions(df)),
            slider_soft_cap=60,
        )
        counting_mode = st.selectbox("Counting Mode", ["full", "fractional"], 0, key="inst_counting_mode")
        min_links = integer_control(
            st,
            "Min Collaboration Links",
            1,
            int(st.session_state.get("inst_min_links", 1)),
            key="inst_min_links",
            input_max=max(10, min(100, top_n_inst)),
            slider_soft_cap=10,
        )

        style_controls = _render_network_style_controls(
            st,
            key_prefix="inst",
            current_view_mode=effective_view_mode,
            current_figure_mode=mode_key,
        )

    with col2:
        graph, partition, node_groups, html_content, net_stats, top_inst_df = (
            _build_institution_collaboration_assets(df, top_n_inst, palette, counting_mode=counting_mode, min_links=min_links)
        )
        if graph.number_of_nodes() == 0:
            st.warning("No institution data available in this dataset.")
            return

        _render_collaboration_graph(
            st,
            effective_view_mode,
            graph,
            node_groups,
            html_content,
            net_stats,
            partition,
            html_filename="institution_collaboration_network.html",
            figure_key="institution_collaboration_figure",
            figure_title="Institution Collaboration Network",
            metric_labels=("Institutions", "Co-occurrences"),
            summary_label="institutions",
            top_table=top_inst_df,
            top_table_title="Top Institutions",
            plotly_size_range=(style_controls["node_size_min"], style_controls["node_size_max"]),
            plotly_label_max_len=40,
            plotly_max_visible_labels=style_controls["max_visible_labels"],
            plotly_label_font_size=style_controls["label_font_size"],
            pyvis_size_range=(style_controls["node_size_min"], style_controls["node_size_max"]),
            pyvis_label_max_len=40,
            pyvis_max_visible_labels=style_controls["max_visible_labels"],
            pyvis_label_font_size=style_controls["label_font_size"],
            figure_mode=mode_key,
            edge_color_override=style_controls["edge_color"],
            edge_width_scale=style_controls["edge_width_scale"],
            edge_alpha_scale=style_controls["edge_alpha_scale"],
        )


def _render_international_collaboration(
    st: Any,
    view_mode: str,
    df: pd.DataFrame,
    palette: list[str],
) -> None:
    st.markdown(
        "Country collaboration network based on co-authored affiliations. "
        "Use `Standard Figure` for presentation, `Dense Map` for VOSviewer-style comparison, "
        "and `Interactive Exploration` for exploration."
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Visualization Mode**")
        _, effective_view_mode, mode_key = _resolve_network_figure_mode(st, key_prefix="country", default_label="Standard Figure")
        available_countries = int(df["Affiliations"].fillna("").astype(str).apply(_extract_country_from_affiliation).replace("", pd.NA).dropna().str.split(";").explode().str.strip().replace("", pd.NA).dropna().nunique()) if "Affiliations" in df.columns else 0
        top_n_country = integer_control(
            st,
            "Number of Countries",
            10,
            25,
            key="country_topn",
            input_max=max(10, available_countries),
            slider_soft_cap=60,
        )
        counting_mode = st.selectbox("Counting Mode", ["full", "fractional"], 0, key="country_counting_mode")
        min_links = integer_control(
            st,
            "Min Collaboration Links",
            1,
            int(st.session_state.get("country_min_links", 1)),
            key="country_min_links",
            input_max=max(10, min(100, top_n_country)),
            slider_soft_cap=10,
        )
        if mode_key == "publication":
            st.caption("For manuscript-ready country collaboration figures, Top 15-30 countries is usually the cleanest range; above ~35 the central hubs often collapse into a dense core.")
        else:
            st.caption("Dense Map can support a broader country set, but Top 30-50 is usually the upper range before regional structure starts to blur.")

        style_controls = _render_network_style_controls(
            st,
            key_prefix="country",
            current_view_mode=effective_view_mode,
            current_figure_mode=mode_key,
        )

    with col2:
        graph, partition, node_groups, html_content, net_stats, country_df = (
            _build_country_collaboration_assets(df, top_n_country, palette, counting_mode=counting_mode, min_links=min_links)
        )
        if graph.number_of_nodes() == 0:
            st.warning("No country data available in this dataset.")
            return

        _render_collaboration_graph(
            st,
            effective_view_mode,
            graph,
            node_groups,
            html_content,
            net_stats,
            partition,
            html_filename="country_collaboration_network.html",
            figure_key="country_collaboration_figure",
            figure_title="Country Collaboration Network",
            metric_labels=("Countries", "Collaborations"),
            summary_label="countries",
            plotly_size_range=(style_controls["node_size_min"], style_controls["node_size_max"]),
            plotly_label_max_len=34,
            plotly_max_visible_labels=style_controls["max_visible_labels"],
            plotly_label_font_size=style_controls["label_font_size"],
            pyvis_size_range=(style_controls["node_size_min"], style_controls["node_size_max"]),
            pyvis_label_max_len=34,
            pyvis_max_visible_labels=style_controls["max_visible_labels"],
            pyvis_label_font_size=style_controls["label_font_size"],
            figure_mode=mode_key,
            edge_color_override=style_controls["edge_color"],
            edge_width_scale=style_controls["edge_width_scale"],
            edge_alpha_scale=style_controls["edge_alpha_scale"],
        )
        if "Times_Cited" in df.columns:
            st.info(
                "Country performance is available as a separate entry in this module: "
                "`Country Publication and Citation Impact Quadrant`."
            )

    if not country_df.empty:
        with st.expander("Country Collaboration Chord-Style Map", expanded=True):
            st.caption(
                "This circular collaboration map is a presentation-oriented companion to the main "
                "country network and is useful for manuscript comparison against chord-style figures."
            )
            country_chord_topn = integer_control(
                st,
                "Countries in Circular Map",
                8,
                min(20, max(8, graph.number_of_nodes())),
                key="country_chord_topn",
                input_max=max(8, graph.number_of_nodes()),
                slider_soft_cap=30,
            )
            chord_style = render_plot_style_controls(
                "country_chord_map",
                default_primary=SCIENTIFIC_COLORWAY[0],
                default_height=920,
                show_legend_default=True,
                allow_color_controls=False,
                preserve_original_colors=True,
            )
            fig_country_chord = render_circular_cluster_chord_figure(
                graph,
                node_groups,
                title="Country Collaboration Chord-Style Map",
                legend_title="Publications",
                top_n=country_chord_topn,
                size_range=(10, 36),
                edge_alpha=0.20,
            )
            if fig_country_chord is not None:
                fig_country_chord = apply_publication_style_with_overrides(fig_country_chord, chord_style)
                render_plotly_chart(fig_country_chord, width="stretch")
                download_plotly_button(
                    fig_country_chord,
                    "country_collaboration_chord_map.png",
                    "Download Country Chord Map",
                )
            else:
                st.info("Not enough linked countries to build the circular collaboration map.")

        with st.expander("Country Publication Statistics", expanded=False):
            st.caption(
                "This descriptive bar chart is best treated as supporting context for the collaboration network "
                "rather than as the main network figure."
            )
            fig_country = px.bar(
                country_df,
                x="Papers",
                y="Country",
                orientation="h",
                title="Top 30 Countries by Publications",
            )
            fig_country.update_traces(marker_color=SCIENTIFIC_COLORWAY[2])
            fig_country = style_publication_figure(fig_country, height=600)
            render_plotly_chart(fig_country, width="stretch")
            download_plotly_button(
                fig_country,
                "country_publications.png",
                "Download Country Chart",
            )


def _render_country_impact_quadrant(
    st: Any,
    df: pd.DataFrame,
) -> None:
    st.markdown(
        "Country-level publication and citation-impact companion view. "
        "This panel stays in `Relational Network Analysis` because it extends the country-level collaboration "
        "story with a performance-oriented comparison rather than a separate descriptive module."
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        available_countries = (
            int(
                df["Affiliations"]
                .fillna("")
                .astype(str)
                .apply(_extract_country_from_affiliation)
                .replace("", pd.NA)
                .dropna()
                .str.split(";")
                .explode()
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .nunique()
            )
            if "Affiliations" in df.columns
            else 0
        )
        counting_mode = st.selectbox(
            "Counting Mode",
            ["full", "fractional"],
            0,
            key="country_quadrant_counting_mode",
        )
        quadrant_topn = integer_control(
            st,
            "Countries in Quadrant",
            6,
            10,
            key="country_quadrant_topn",
            input_max=max(6, available_countries),
            slider_soft_cap=20,
        )
        st.caption(
            "Use this view when you want to compare publication output against average citation impact "
            "without opening the country collaboration network first."
        )
        quadrant_style = render_plot_style_controls(
            "country_impact_quadrant",
            default_primary=SCIENTIFIC_COLORWAY[3],
            default_height=760,
            show_legend_default=True,
            allow_color_controls=False,
            preserve_original_colors=True,
        )

    with col2:
        quadrant_df, pub_median, cite_median = build_country_impact_quadrant_frame(
            df,
            top_n=quadrant_topn,
            counting_mode=counting_mode,
        )
        fig_country_quadrant = render_country_impact_quadrant_figure(
            quadrant_df,
            pub_median,
            cite_median,
        )
        if fig_country_quadrant is not None:
            fig_country_quadrant = apply_publication_style_with_overrides(
                fig_country_quadrant,
                quadrant_style,
            )
            render_plotly_chart(fig_country_quadrant, width="stretch")
            download_plotly_button(
                fig_country_quadrant,
                "country_publication_citation_quadrant.png",
                "Download Country Impact Quadrant",
            )
        else:
            st.info("Country citation data is insufficient for quadrant analysis.")
            return

        if not quadrant_df.empty:
            with st.expander("Country Impact Table", expanded=False):
                display_df = quadrant_df[
                    ["Country", "Publications", "Avg Citations", "Total Citations", "Quadrant"]
                ].copy()
                display_df["Avg Citations"] = display_df["Avg Citations"].round(2)
                display_df["Total Citations"] = display_df["Total Citations"].round(2)
                st.dataframe(display_df, width="stretch", hide_index=True)


def _render_bibliographic_coupling(
    st: Any,
    view_mode: str,
    df: pd.DataFrame,
    palette: list[str],
) -> None:
    st.markdown(
        "Bibliographic coupling links focal papers that cite the same references. "
        "Stronger edges indicate greater overlap in intellectual foundations. "
        "`Standard Figure` is recommended for the main display."
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Visualization Mode**")
        _, effective_view_mode, mode_key = _resolve_network_figure_mode(st, key_prefix="bc", default_label="Standard Figure")
        top_n_bc = integer_control(
            st,
            "Number of Papers",
            10,
            30,
            key="bc_topn",
            input_max=max(10, len(df)),
            slider_soft_cap=100,
        )
        min_shared_bc = integer_control(
            st,
            "Minimum Shared References",
            1,
            int(st.session_state.get("bc_min_shared", 2)),
            key="bc_min_shared",
            input_max=100,
            slider_soft_cap=10,
        )

        style_controls = _render_network_style_controls(
            st,
            key_prefix="bc",
            current_view_mode=effective_view_mode,
            current_figure_mode=mode_key,
        )

    with col2:
        graph, bc_pairs, bc_top_papers = build_bibliographic_coupling_network(
            df,
            top_n=top_n_bc,
            min_shared_refs=min_shared_bc,
        )
        if graph.number_of_nodes() == 0:
            st.warning(
                "No bibliographic coupling structure could be built. Try lowering "
                "the shared-reference threshold."
            )
            return

        partition, node_groups = _build_partitioned_node_groups(graph, palette)
        html_content = None
        bc_stats = None
        if effective_view_mode == INTERACTIVE_VIEW_MODE:
            html_content, bc_stats = render_network_html(
                graph,
                node_groups=node_groups,
                legend_label="Shared references",
                size_range=(style_controls["node_size_min"], style_controls["node_size_max"]),
                label_max_len=40,
                layout_mode="clustered",
                max_visible_labels=style_controls["max_visible_labels"],
                label_font_size=style_controls["label_font_size"],
                mode=mode_key,
                edge_color_override=style_controls["edge_color"],
                edge_width_scale=style_controls["edge_width_scale"],
                edge_alpha_scale=style_controls["edge_alpha_scale"],
            )
        if graph.number_of_nodes() > 0:
            _render_html_and_exports(
                st,
                effective_view_mode,
                html_content,
                graph,
                node_groups,
                "bibliographic_coupling_network.html",
                "bibliographic_coupling_figure",
                "Bibliographic Coupling Network",
                "Shared references",
                size_range=(style_controls["node_size_min"], style_controls["node_size_max"]),
                label_max_len=40,
                max_visible_labels=style_controls["max_visible_labels"],
                label_font_size=style_controls["label_font_size"],
                figure_mode=mode_key,
                edge_color_override=style_controls["edge_color"],
                edge_width_scale=style_controls["edge_width_scale"],
                edge_alpha_scale=style_controls["edge_alpha_scale"],
            )
        if bc_stats:
            st.markdown("---")
            st.markdown("### Network Statistics")
            bc_col1, bc_col2, bc_col3, bc_col4 = st.columns(4)
            with bc_col1:
                st.metric("Papers", bc_stats["nodes"])
            with bc_col2:
                st.metric("Coupling Links", bc_stats["edges"])
            with bc_col3:
                st.metric("Density", f"{bc_stats['density']:.4f}")
            with bc_col4:
                st.metric("Clusters", len(set(partition.values())))
            show_cluster_report(bc_stats, "papers")
        with st.expander("Strongest Coupled Paper Pairs", expanded=False):
            st.dataframe(pd.DataFrame(bc_pairs), width="stretch", hide_index=True)
        with st.expander("Top Papers by Coupling Strength", expanded=False):
            st.dataframe(pd.DataFrame(bc_top_papers), width="stretch", hide_index=True)


def _render_co_citation_network(st: Any, view_mode: str, df: pd.DataFrame) -> None:
    st.markdown(
        "Co-citation network: two references are connected if they are frequently "
        "cited together by the same papers. `Standard Figure` is recommended for the main display."
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Network Parameters**")
        available_references, _, _ = _count_available_cited_items(df)
        st.markdown("**Visualization Mode**")
        figure_mode, effective_view_mode, mode_key = _resolve_network_figure_mode(st, key_prefix="cocite", default_label="Standard Figure")
        max_references_for_render = max(10, available_references)
        top_n_ref = integer_control(
            st,
            "Number of References",
            10,
            20,
            key="cocite_topn",
            input_max=max_references_for_render,
            slider_soft_cap=min(100, max_references_for_render),
        )
        min_cocite = integer_control(
            st,
            "Min Co-citation",
            1,
            int(st.session_state.get("cocite_minw", 2)),
            key="cocite_minw",
            input_max=100,
            slider_soft_cap=10,
        )
        if mode_key == "publication":
            st.caption(
                "Standard Figure is optimized for roughly Top 20-50 references so the main clusters "
                "and labels stay clean, but the input limit now follows the actual dataset maximum."
            )
        elif available_references > 150:
            st.caption(
                f"The current dataset provides up to {available_references} unique references. "
                "Use `Standard Figure` for Top 20-50 and switch to `Dense Map` for larger "
                "reference sets."
            )
        style_controls = _render_network_style_controls(
            st,
            key_prefix="cocite",
            current_view_mode=effective_view_mode,
            current_figure_mode=mode_key,
        )

    with col2:
        html_content, graph, node_groups, net_stats, ref_freq = build_cocitation_network(
            df,
            top_n_ref=top_n_ref,
            min_cocite=min_cocite,
            size_range=(style_controls["node_size_min"], style_controls["node_size_max"]),
            max_visible_labels=style_controls["max_visible_labels"],
            label_font_size=style_controls["label_font_size"],
            generate_html=effective_view_mode == INTERACTIVE_VIEW_MODE,
        )
        if graph is None or graph.number_of_nodes() == 0:
            st.warning("No co-citation data to display. Try lowering the min co-citation threshold.")
            return

        if graph.number_of_nodes() > 0:
            _render_html_and_exports(
                st,
                effective_view_mode,
                html_content,
                graph,
                node_groups,
                "cocitation_network.html",
                "reference_cocitation_figure",
                "Co-citation Network",
                "Citations",
                size_range=(style_controls["node_size_min"], style_controls["node_size_max"]),
                label_max_len=30,
                max_visible_labels=style_controls["max_visible_labels"],
                label_font_size=style_controls["label_font_size"],
                figure_mode=mode_key,
                edge_color_override=style_controls["edge_color"],
                edge_width_scale=style_controls["edge_width_scale"],
                edge_alpha_scale=style_controls["edge_alpha_scale"],
            )
        if net_stats:
            st.markdown("---")
            st.markdown("### Network Statistics")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("References", net_stats["nodes"])
            with col_s2:
                st.metric("Co-citations", net_stats["edges"])
            with col_s3:
                st.metric("Density", f"{net_stats['density']:.4f}")
            with col_s4:
                st.metric("Clusters", len({info["group"] for info in node_groups.values()}))
            show_cluster_report(net_stats, "references")
        with st.expander("Top Cited References", expanded=False):
            top_ref_df = pd.DataFrame(
                ref_freq.most_common(20),
                columns=["Reference", "Citations"],
            )
            st.dataframe(top_ref_df, width="stretch", hide_index=True)


def _render_author_co_citation_network(st: Any, view_mode: str, df: pd.DataFrame) -> None:
    st.markdown(
        "Author co-citation network: two authors are connected if they are "
        "frequently cited together by the same papers. `Standard Figure` is recommended for the main display."
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Visualization Mode**")
        _, effective_view_mode, mode_key = _resolve_network_figure_mode(st, key_prefix="auth_cocite", default_label="Standard Figure")
        _, available_cited_authors, _ = _count_available_cited_items(df)
        top_n_authors = integer_control(
            st,
            "Number of Authors",
            10,
            20,
            key="auth_cocite_topn",
            input_max=max(10, available_cited_authors),
            slider_soft_cap=100,
        )
        min_cocite = integer_control(
            st,
            "Min Co-citation",
            1,
            int(st.session_state.get("auth_cocite_minw", 2)),
            key="auth_cocite_minw",
            input_max=100,
            slider_soft_cap=10,
        )

        style_controls = _render_network_style_controls(
            st,
            key_prefix="auth_cocite",
            current_view_mode=effective_view_mode,
            current_figure_mode=mode_key,
            default_max_visible_labels=20,
        )

    with col2:
        html_content, graph, node_groups, net_stats, auth_freq = build_author_cocitation_network(
            df,
            top_n_authors=top_n_authors,
            min_cocite=min_cocite,
            size_range=(style_controls["node_size_min"], style_controls["node_size_max"]),
            max_visible_labels=style_controls["max_visible_labels"],
            label_font_size=style_controls["label_font_size"],
            generate_html=effective_view_mode == INTERACTIVE_VIEW_MODE,
        )
        if graph is None or graph.number_of_nodes() == 0:
            st.warning(
                "No author co-citation data to display. Try lowering the min co-citation threshold."
            )
            return

        if graph.number_of_nodes() > 0:
            _render_html_and_exports(
                st,
                effective_view_mode,
                html_content,
                graph,
                node_groups,
                "author_cocitation_network.html",
                "author_cocitation_figure",
                "Author Co-citation Network",
                "Co-citations",
                size_range=(style_controls["node_size_min"], style_controls["node_size_max"]),
                label_max_len=25,
                max_visible_labels=style_controls["max_visible_labels"],
                label_font_size=style_controls["label_font_size"],
                figure_mode=mode_key,
                edge_color_override=style_controls["edge_color"],
                edge_width_scale=style_controls["edge_width_scale"],
                edge_alpha_scale=style_controls["edge_alpha_scale"],
            )
        if net_stats:
            st.markdown("---")
            st.markdown("### Network Statistics")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Authors", net_stats["nodes"])
            with col_s2:
                st.metric("Co-citations", net_stats["edges"])
            with col_s3:
                st.metric("Density", f"{net_stats['density']:.4f}")
            with col_s4:
                st.metric("Clusters", len({info["group"] for info in node_groups.values()}))
            show_cluster_report(net_stats, "authors")
        with st.expander("Top Cited Authors", expanded=False):
            top_auth_df = pd.DataFrame(
                auth_freq.most_common(20),
                columns=["Author", "Co-citations"],
            )
            st.dataframe(top_auth_df, width="stretch", hide_index=True)


def _render_journal_co_citation_network(st: Any, view_mode: str, df: pd.DataFrame) -> None:
    st.markdown(
        "Journal co-citation network: two journals are connected if they are "
        "frequently cited together by the same papers. `Standard Figure` is recommended for the main display."
    )
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("**Visualization Mode**")
        _, effective_view_mode, mode_key = _resolve_network_figure_mode(st, key_prefix="j_cocite", default_label="Standard Figure")
        _, _, available_cited_journals = _count_available_cited_items(df)
        top_n_journals = integer_control(
            st,
            "Number of Journals",
            10,
            20,
            key="j_cocite_topn",
            input_max=max(10, available_cited_journals),
            slider_soft_cap=100,
        )
        min_cocite = integer_control(
            st,
            "Min Co-citation",
            1,
            int(st.session_state.get("j_cocite_minw", 2)),
            key="j_cocite_minw",
            input_max=100,
            slider_soft_cap=10,
        )

        style_controls = _render_network_style_controls(
            st,
            key_prefix="j_cocite",
            current_view_mode=effective_view_mode,
            current_figure_mode=mode_key,
            default_max_visible_labels=20,
        )

    with col2:
        html_content, graph, node_groups, net_stats, journal_freq = build_journal_cocitation_network(
            df,
            top_n_journals=top_n_journals,
            min_cocite=min_cocite,
            size_range=(style_controls["node_size_min"], style_controls["node_size_max"]),
            max_visible_labels=style_controls["max_visible_labels"],
            label_font_size=style_controls["label_font_size"],
            generate_html=effective_view_mode == INTERACTIVE_VIEW_MODE,
        )
        if graph is None or graph.number_of_nodes() == 0:
            st.warning(
                "No journal co-citation data to display. Try lowering the min co-citation threshold."
            )
            return

        if graph.number_of_nodes() > 0:
            _render_html_and_exports(
                st,
                effective_view_mode,
                html_content,
                graph,
                node_groups,
                "journal_cocitation_network.html",
                "journal_cocitation_figure",
                "Journal Co-citation Network",
                "Co-citations",
                size_range=(style_controls["node_size_min"], style_controls["node_size_max"]),
                label_max_len=25,
                max_visible_labels=style_controls["max_visible_labels"],
                label_font_size=style_controls["label_font_size"],
                figure_mode=mode_key,
                edge_color_override=style_controls["edge_color"],
                edge_width_scale=style_controls["edge_width_scale"],
                edge_alpha_scale=style_controls["edge_alpha_scale"],
            )
        if net_stats:
            st.markdown("---")
            st.markdown("### Network Statistics")
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Journals", net_stats["nodes"])
            with col_s2:
                st.metric("Co-citations", net_stats["edges"])
            with col_s3:
                st.metric("Density", f"{net_stats['density']:.4f}")
            with col_s4:
                st.metric("Clusters", len({info["group"] for info in node_groups.values()}))
            show_cluster_report(net_stats, "journals")
        with st.expander("Top Cited Journals", expanded=False):
            top_journal_df = pd.DataFrame(
                journal_freq.most_common(20),
                columns=["Journal", "Co-citations"],
            )
            st.dataframe(top_journal_df, width="stretch", hide_index=True)


def render_relational_network_page(
    st: Any,
    df: pd.DataFrame,
    keywords_list: list[list[str]],
    keyword_freq: dict[str, int],
    cooccurrence: dict[tuple[str, str], int],
    has_authors: bool,
    has_affiliations: bool,
    has_cited_refs: bool,
    log_exception: Callable[[str, Exception], None],
    network_cluster_palette: list[str],
) -> None:
    st.title("Relational Network Analysis")
    st.caption(
        "This module combines standard bibliometric network views commonly used in "
        "mainstream mapping workflows (e.g., VOSviewer-style keyword co-occurrence and "
        "CiteSpace-style co-citation structures), while bibliographic coupling further supports "
        "downstream innovation-oriented analysis."
    )

    network_options = ["Keyword Co-occurrence", "Keyword-Journal Association"]
    if has_authors:
        network_options.append("Co-authorship Network")
    if has_affiliations:
        network_options.extend(
            ["Institutional Collaboration", "International Collaboration"]
        )
        if "Times_Cited" in df.columns:
            network_options.append("Country Publication and Citation Impact Quadrant")
    if has_cited_refs:
        network_options.extend(
            [
                "Bibliographic Coupling",
                "Co-citation Network",
                "Author Co-citation Network",
                "Journal Co-citation Network",
            ]
        )
    net_type = st.selectbox("Select Network Type", network_options)
    st.caption(
        "Each network now uses a unified three-mode workflow: `Standard Figure` for manuscript-ready output, "
        "`Dense Map` for dense structural comparison, and `Interactive Exploration` for exploratory inspection."
    )
    view_mode = STATIC_VIEW_MODE

    if net_type == "Keyword Co-occurrence":
        _render_keyword_cooccurrence(st, view_mode, keyword_freq, cooccurrence, log_exception)
    elif net_type == "Keyword-Journal Association":
        _render_keyword_journal_association(st, view_mode, df, keywords_list, log_exception)
    elif net_type == "Co-authorship Network":
        _render_coauthorship_network(st, view_mode, df, network_cluster_palette)
    elif net_type == "Institutional Collaboration":
        _render_institutional_collaboration(st, view_mode, df, network_cluster_palette)
    elif net_type == "International Collaboration":
        _render_international_collaboration(st, view_mode, df, network_cluster_palette)
    elif net_type == "Country Publication and Citation Impact Quadrant":
        _render_country_impact_quadrant(st, df)
    elif net_type == "Bibliographic Coupling":
        _render_bibliographic_coupling(st, view_mode, df, network_cluster_palette)
    elif net_type == "Co-citation Network":
        _render_co_citation_network(st, view_mode, df)
    elif net_type == "Author Co-citation Network":
        _render_author_co_citation_network(st, view_mode, df)
    elif net_type == "Journal Co-citation Network":
        _render_journal_co_citation_network(st, view_mode, df)
