from __future__ import annotations

from collections import Counter
import math

import community as community_louvain
import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from modules.data_pipeline import _extract_country_from_affiliation
from modules.export_bundle import SCIENTIFIC_COLORWAY

CHORD_CLUSTER_PALETTE = [
    "#5DA5DA",
    "#F17CB0",
    "#60BD68",
    "#F5C04A",
    "#B291D4",
    "#F28E2B",
    "#4E79A7",
    "#E15759",
]

COUNTRY_QUADRANT_COLORS = {
    "High-quality": "#2CA581",
    "High output & high impact": "#3A7FBF",
    "High-output": "#E67E22",
    "Low output & low impact": "#7A7A7A",
}


def _most_common_items(values, limit: int):
    if hasattr(values, "most_common"):
        return values.most_common(limit)
    return sorted(values.items(), key=lambda item: item[1], reverse=True)[:limit]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = str(hex_color or "").strip().lstrip("#")
    if len(color) != 6:
        return f"rgba(120,120,120,{alpha})"
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"


def _map_range(value: float, low: float, high: float, out_low: float, out_high: float) -> float:
    if high <= low:
        return (out_low + out_high) / 2.0
    ratio = (value - low) / (high - low)
    return out_low + ratio * (out_high - out_low)


def render_ranked_lollipop_figure(
    frame: pd.DataFrame,
    *,
    label_col: str,
    value_col: str,
    title: str,
    marker_color: str | None = None,
    line_color: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
) -> go.Figure | None:
    if frame is None or frame.empty or label_col not in frame.columns or value_col not in frame.columns:
        return None

    plot_df = frame[[label_col, value_col]].copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df[label_col] = plot_df[label_col].astype(str)
    plot_df = plot_df.dropna(subset=[value_col])
    if plot_df.empty:
        return None

    plot_df = plot_df.sort_values(value_col, ascending=False)
    category_order = list(reversed(plot_df[label_col].tolist()))
    marker_color = marker_color or SCIENTIFIC_COLORWAY[0]
    line_color = line_color or marker_color

    fig = go.Figure()
    for _, row in plot_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[0, row[value_col]],
                y=[row[label_col], row[label_col]],
                mode="lines",
                line=dict(color=_hex_to_rgba(line_color, 0.45), width=3),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=plot_df[value_col],
            y=plot_df[label_col],
            mode="markers",
            marker=dict(
                color=marker_color,
                size=12,
                line=dict(color="white", width=1.2),
            ),
            hovertemplate=f"{label_col}: %{{y}}<br>{value_col}: %{{x}}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title=xaxis_title or value_col,
        yaxis_title=yaxis_title or label_col,
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray", rangemode="tozero")
    fig.update_yaxes(
        showgrid=False,
        categoryorder="array",
        categoryarray=category_order,
    )
    return fig


def _bezier_curve_points(
    start: tuple[float, float],
    end: tuple[float, float],
    control: tuple[float, float],
    steps: int = 36,
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for step in range(steps + 1):
        t = step / steps
        one_minus_t = 1.0 - t
        x_value = (
            (one_minus_t ** 2) * start[0]
            + 2 * one_minus_t * t * control[0]
            + (t ** 2) * end[0]
        )
        y_value = (
            (one_minus_t ** 2) * start[1]
            + 2 * one_minus_t * t * control[1]
            + (t ** 2) * end[1]
        )
        xs.append(x_value)
        ys.append(y_value)
    return xs, ys


def _subgraph_top_nodes(
    graph: nx.Graph,
    node_groups: dict[str, dict[str, int | str]],
    top_n: int,
) -> tuple[nx.Graph, dict[str, dict[str, int | str]]]:
    if graph is None or graph.number_of_nodes() == 0:
        return nx.Graph(), {}

    selected_nodes = [
        node
        for node, _ in sorted(
            graph.nodes(data="weight"),
            key=lambda item: (-float(item[1] or 0.0), str(item[0])),
        )[: max(int(top_n), 1)]
    ]
    subgraph = graph.subgraph(selected_nodes).copy()
    isolated = [node for node in subgraph.nodes() if subgraph.degree(node) == 0]
    subgraph.remove_nodes_from(isolated)
    subgroups = {node: node_groups[node] for node in subgraph.nodes() if node in node_groups}
    return subgraph, subgroups


def _ordered_cluster_nodes(
    graph: nx.Graph,
    node_groups: dict[str, dict[str, int | str]],
) -> tuple[list[str], dict[int, list[str]]]:
    cluster_nodes: dict[int, list[str]] = {}
    for node in graph.nodes():
        cluster_id = int((node_groups or {}).get(node, {}).get("group", 0))
        cluster_nodes.setdefault(cluster_id, []).append(node)

    ordered_clusters = sorted(cluster_nodes)
    ordered_nodes: list[str] = []
    for cluster_id in ordered_clusters:
        cluster_nodes[cluster_id] = sorted(
            cluster_nodes[cluster_id],
            key=lambda node: (-float(graph.nodes[node].get("weight", 0.0)), str(node)),
        )
        ordered_nodes.extend(cluster_nodes[cluster_id])
    return ordered_nodes, cluster_nodes


def render_circular_cluster_chord_figure(
    graph: nx.Graph,
    node_groups: dict[str, dict[str, int | str]] | None,
    *,
    title: str,
    legend_title: str,
    top_n: int = 24,
    size_range: tuple[int, int] = (8, 34),
    label_radius: float = 1.18,
    edge_alpha: float = 0.22,
    edge_weight_attr: str = "weight",
) -> go.Figure | None:
    if graph is None or graph.number_of_nodes() == 0:
        return None

    subgraph, subgroups = _subgraph_top_nodes(graph, node_groups or {}, top_n=top_n)
    if subgraph.number_of_nodes() < 3 or subgraph.number_of_edges() == 0:
        return None

    ordered_nodes, cluster_nodes = _ordered_cluster_nodes(subgraph, subgroups)
    cluster_ids = sorted(cluster_nodes)
    gap_slots = 2
    total_slots = len(ordered_nodes) + max(len(cluster_ids), 1) * gap_slots
    angle_step = (2 * math.pi) / max(total_slots, 1)

    node_positions: dict[str, tuple[float, float, float]] = {}
    slot_index = 0
    for cluster_id in cluster_ids:
        for node in cluster_nodes[cluster_id]:
            theta = (math.pi / 2) - slot_index * angle_step
            node_positions[node] = (math.cos(theta), math.sin(theta), theta)
            slot_index += 1
        slot_index += gap_slots

    weights = [float(subgraph.nodes[node].get("weight", 1.0)) for node in ordered_nodes]
    min_weight = min(weights)
    max_weight = max(weights)
    edge_weights = [
        float(data.get(edge_weight_attr, data.get("weight", 1.0)))
        for _, _, data in subgraph.edges(data=True)
    ]
    min_edge = min(edge_weights)
    max_edge = max(edge_weights)

    fig = go.Figure()

    edge_items = sorted(
        subgraph.edges(data=True),
        key=lambda item: float(item[2].get(edge_weight_attr, item[2].get("weight", 1.0))),
    )
    for left, right, data in edge_items:
        x0, y0, theta0 = node_positions[left]
        x1, y1, theta1 = node_positions[right]
        midpoint_angle = (theta0 + theta1) / 2.0
        control_radius = 0.10 if abs(theta0 - theta1) < (math.pi / 3) else 0.02
        control = (
            control_radius * math.cos(midpoint_angle),
            control_radius * math.sin(midpoint_angle),
        )
        edge_x, edge_y = _bezier_curve_points((x0, y0), (x1, y1), control)
        weight = float(data.get(edge_weight_attr, data.get("weight", 1.0)))
        width = _map_range(weight, min_edge, max_edge, 0.6, 5.5)
        color = subgroups.get(left, {}).get("color", SCIENTIFIC_COLORWAY[0])
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color=_hex_to_rgba(str(color), edge_alpha), width=width),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    for cluster_id in cluster_ids:
        cluster_label = f"Cluster {cluster_id + 1}"
        cluster_color = str(
            subgroups.get(cluster_nodes[cluster_id][0], {}).get(
                "color",
                CHORD_CLUSTER_PALETTE[cluster_id % len(CHORD_CLUSTER_PALETTE)],
            )
        )
        cluster_x = [node_positions[node][0] for node in cluster_nodes[cluster_id]]
        cluster_y = [node_positions[node][1] for node in cluster_nodes[cluster_id]]
        cluster_sizes = [
            _map_range(
                float(subgraph.nodes[node].get("weight", 1.0)),
                min_weight,
                max_weight,
                size_range[0],
                size_range[1],
            )
            for node in cluster_nodes[cluster_id]
        ]
        fig.add_trace(
            go.Scatter(
                x=cluster_x,
                y=cluster_y,
                mode="markers",
                name=cluster_label,
                marker=dict(
                    size=cluster_sizes,
                    color=cluster_color,
                    line=dict(color="white", width=1.2),
                    opacity=0.96,
                ),
                customdata=[
                    [
                        node,
                        subgraph.nodes[node].get("weight", 1.0),
                        subgraph.degree(node),
                    ]
                    for node in cluster_nodes[cluster_id]
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    + f"{legend_title}: "
                    + "%{customdata[1]:.2f}<br>"
                    + "Links: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    for node in ordered_nodes:
        x_value, y_value, theta = node_positions[node]
        radial_x = label_radius * math.cos(theta)
        radial_y = label_radius * math.sin(theta)
        vertical_offset = 0.075 if radial_y >= 0 else -0.075
        fig.add_annotation(
            x=radial_x,
            y=radial_y + vertical_offset,
            text=str(node),
            showarrow=False,
            textangle=0,
            font=dict(
                size=10 + int(
                    round(
                        _map_range(
                            float(subgraph.nodes[node].get("weight", 1.0)),
                            min_weight,
                            max_weight,
                            0,
                            10,
                        )
                    )
                ),
                color=subgroups.get(node, {}).get("color", "#333333"),
            ),
            xanchor="center",
            yanchor="bottom" if radial_y >= 0 else "top",
        )

    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        showlegend=True,
        legend=dict(
            title="Clusters",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#D9D9D9",
            borderwidth=1,
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=980,
        margin=dict(l=70, r=120, t=70, b=70),
    )
    fig.update_xaxes(visible=False, range=[-1.45, 1.45], showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, range=[-1.35, 1.35], showgrid=False, zeroline=False)
    return fig


def build_keyword_circular_cluster_figure(
    keyword_freq,
    cooccurrence,
    *,
    top_n: int = 36,
    min_weight: int = 2,
) -> go.Figure | None:
    if not keyword_freq:
        return None

    graph = nx.Graph()
    top_keywords = [keyword for keyword, _ in _most_common_items(keyword_freq, max(int(top_n), 3))]
    keyword_set = set(top_keywords)
    for keyword in top_keywords:
        graph.add_node(keyword, weight=keyword_freq[keyword])
    for (left, right), weight in cooccurrence.items():
        if left in keyword_set and right in keyword_set and weight >= min_weight:
            graph.add_edge(left, right, weight=weight)

    isolated = [node for node in graph.nodes() if graph.degree(node) == 0]
    graph.remove_nodes_from(isolated)
    if graph.number_of_nodes() < 3 or graph.number_of_edges() == 0:
        return None

    try:
        partition = community_louvain.best_partition(graph, weight="weight")
    except Exception:
        partition = {node: 0 for node in graph.nodes()}

    node_groups = {
        node: {
            "group": partition.get(node, 0),
            "color": CHORD_CLUSTER_PALETTE[partition.get(node, 0) % len(CHORD_CLUSTER_PALETTE)],
        }
        for node in graph.nodes()
    }
    return render_circular_cluster_chord_figure(
        graph,
        node_groups,
        title="Keyword Circular Cluster Map",
        legend_title="Frequency",
        top_n=top_n,
        size_range=(10, 36),
        edge_alpha=0.18,
        edge_weight_attr="weight",
    )


def build_country_impact_quadrant_frame(
    df: pd.DataFrame,
    *,
    top_n: int = 10,
    counting_mode: str = "full",
) -> tuple[pd.DataFrame, float, float]:
    if (
        df is None
        or df.empty
        or "Affiliations" not in df.columns
        or "Times_Cited" not in df.columns
    ):
        return pd.DataFrame(), 0.0, 0.0

    publication_counter: Counter[str] = Counter()
    citation_counter: Counter[str] = Counter()

    for _, row in df.iterrows():
        countries_text = _extract_country_from_affiliation(str(row.get("Affiliations", "")))
        if not countries_text:
            continue
        countries = sorted({country.strip() for country in countries_text.split(";") if country.strip()})
        if not countries:
            continue
        try:
            citations = float(row.get("Times_Cited", 0) or 0)
        except (TypeError, ValueError):
            citations = 0.0
        divisor = len(countries) if counting_mode == "fractional" and countries else 1
        for country in countries:
            publication_counter[country] += 1.0 / divisor
            citation_counter[country] += citations / divisor

    if not publication_counter:
        return pd.DataFrame(), 0.0, 0.0

    rows = []
    for country, publications in publication_counter.items():
        total_citations = float(citation_counter.get(country, 0.0))
        avg_citations = total_citations / publications if publications > 0 else 0.0
        rows.append(
            {
                "Country": country,
                "Publications": round(float(publications), 2),
                "Avg Citations": round(avg_citations, 2),
                "Total Citations": round(total_citations, 2),
            }
        )

    frame = (
        pd.DataFrame(rows)
        .sort_values(["Publications", "Total Citations", "Country"], ascending=[False, False, True])
        .head(max(int(top_n), 4))
        .reset_index(drop=True)
    )
    if frame.empty:
        return frame, 0.0, 0.0

    publication_median = float(frame["Publications"].median())
    citation_median = float(frame["Avg Citations"].median())

    def classify_row(row: pd.Series) -> str:
        high_output = float(row["Publications"]) >= publication_median
        high_impact = float(row["Avg Citations"]) >= citation_median
        if high_output and high_impact:
            return "High output & high impact"
        if high_output:
            return "High-output"
        if high_impact:
            return "High-quality"
        return "Low output & low impact"

    frame["Quadrant"] = frame.apply(classify_row, axis=1)
    return frame, publication_median, citation_median


def render_country_impact_quadrant_figure(
    country_frame: pd.DataFrame,
    publication_median: float,
    citation_median: float,
) -> go.Figure | None:
    if country_frame is None or country_frame.empty:
        return None

    x_max = float(country_frame["Publications"].max()) * 1.18
    y_max = float(country_frame["Avg Citations"].max()) * 1.15
    bubble_values = country_frame["Total Citations"].astype(float)
    bubble_min = float(bubble_values.min())
    bubble_max = float(bubble_values.max())

    fig = go.Figure()
    fig.add_shape(
        type="rect",
        x0=0,
        x1=publication_median,
        y0=citation_median,
        y1=y_max,
        fillcolor="rgba(44,165,129,0.10)",
        line_width=0,
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=publication_median,
        x1=x_max,
        y0=citation_median,
        y1=y_max,
        fillcolor="rgba(58,127,191,0.10)",
        line_width=0,
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=publication_median,
        x1=x_max,
        y0=0,
        y1=citation_median,
        fillcolor="rgba(230,126,34,0.10)",
        line_width=0,
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=0,
        x1=publication_median,
        y0=0,
        y1=citation_median,
        fillcolor="rgba(122,122,122,0.10)",
        line_width=0,
        layer="below",
    )

    for quadrant_name, quadrant_frame in country_frame.groupby("Quadrant", sort=False):
        fig.add_trace(
            go.Scatter(
                x=quadrant_frame["Publications"],
                y=quadrant_frame["Avg Citations"],
                mode="markers+text",
                name=quadrant_name,
                text=quadrant_frame["Country"],
                textposition="top center",
                marker=dict(
                    size=[
                        _map_range(float(value), bubble_min, bubble_max, 16, 44)
                        for value in quadrant_frame["Total Citations"]
                    ],
                    color=COUNTRY_QUADRANT_COLORS[quadrant_name],
                    line=dict(color="#222222", width=0.8),
                    opacity=0.95,
                ),
                customdata=quadrant_frame[["Country", "Total Citations"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    + "Publications: %{x:.2f}<br>"
                    + "Avg citations: %{y:.2f}<br>"
                    + "Total citations: %{customdata[1]:.2f}<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=publication_median, line_dash="dash", line_color="#666666")
    fig.add_hline(y=citation_median, line_dash="dash", line_color="#666666")
    fig.add_annotation(
        x=publication_median * 0.45,
        y=max(y_max * 0.96, citation_median + 0.5),
        text="<b>High-quality</b>",
        showarrow=False,
        font=dict(color=COUNTRY_QUADRANT_COLORS["High-quality"], size=14),
    )
    fig.add_annotation(
        x=min(x_max * 0.78, publication_median + (x_max - publication_median) * 0.55),
        y=max(y_max * 0.96, citation_median + 0.5),
        text="<b>High output & high impact</b>",
        showarrow=False,
        font=dict(color=COUNTRY_QUADRANT_COLORS["High output & high impact"], size=14),
    )
    fig.add_annotation(
        x=min(x_max * 0.78, publication_median + (x_max - publication_median) * 0.55),
        y=max(citation_median * 0.30, 1.0),
        text="<b>High-output</b>",
        showarrow=False,
        font=dict(color=COUNTRY_QUADRANT_COLORS["High-output"], size=14),
    )
    fig.add_annotation(
        x=publication_median * 0.45,
        y=max(citation_median * 0.30, 1.0),
        text="<b>Low output & low impact</b>",
        showarrow=False,
        font=dict(color=COUNTRY_QUADRANT_COLORS["Low output & low impact"], size=14),
    )
    fig.update_layout(
        title="Country Publications and Citation Impact Quadrant Analysis",
        xaxis_title="Publications",
        yaxis_title="Avg. Citations",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=760,
        legend_title="Quadrants",
    )
    fig.update_xaxes(range=[0, x_max], gridcolor="#D9D9D9")
    fig.update_yaxes(range=[0, y_max], gridcolor="#D9D9D9")
    return fig
