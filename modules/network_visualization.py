import community as community_louvain
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
from pyvis.network import Network

from modules.export_bundle import SCIENTIFIC_COLORWAY, style_publication_figure
from modules.network_layouts import (
    compute_bipartite_layout,
    compute_cluster_layout,
    compute_force_layout,
)

NETWORK_EDGE_COLOR = "rgba(148, 143, 136, 0.4)"
NETWORK_EDGE_ALPHA = 0.32
KEYWORD_CLUSTER_PALETTE = [
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


def _most_common_items(values, limit):
    if hasattr(values, "most_common"):
        return values.most_common(limit)
    return sorted(values.items(), key=lambda item: item[1], reverse=True)[:limit]


def node_groups_from_cluster_stats(stats, shape="dot", type_label_prefix="Cluster"):
    cluster_report = (stats or {}).get("cluster_report", {})
    node_groups = {}
    for group_key, group_info in cluster_report.items():
        for member in group_info.get("members", []):
            node_groups[member] = {
                "color": group_info.get("color", "#888888"),
                "group": group_key,
                "shape": group_info.get("shape", shape),
                "type_label": f"{type_label_prefix} {int(group_key) + 1}: ",
                "border_color": group_info.get("border_color", "rgba(0,0,0,0)"),
                "border_width": group_info.get("border_width", 0),
            }
    return node_groups or None


def _map_node_size(freq, min_freq, max_freq, size_min=10, size_max=40, gamma=1.0):
    if max_freq == min_freq:
        return (size_min + size_max) / 2
    ratio = (freq - min_freq) / (max_freq - min_freq)
    ratio = max(0.0, min(1.0, ratio)) ** max(0.35, float(gamma))
    return size_min + ratio * (size_max - size_min)


def _map_edge_width(weight, min_weight, max_weight, width_min=2.2, width_max=9.2):
    if max_weight == min_weight:
        return (width_min + width_max) / 2
    ratio = (weight - min_weight) / (max_weight - min_weight)
    return width_min + ratio * (width_max - width_min)


def _resolve_edge_width_multiplier(edge_width_scale, mode="publication"):
    scale = max(0.0, min(1.0, float(edge_width_scale)))
    max_multiplier = 2.6 if mode == "map" else 2.0
    return scale * max_multiplier


def _hex_to_rgba(color, alpha=0.4):
    color = str(color or "").strip()
    if color.startswith("rgba") or color.startswith("rgb"):
        return color
    if color.startswith("#") and len(color) == 7:
        red = int(color[1:3], 16)
        green = int(color[3:5], 16)
        blue = int(color[5:7], 16)
        return f"rgba({red}, {green}, {blue}, {alpha})"
    return f"rgba(148, 143, 136, {alpha})"


def _edge_color_for_nodes(
    u,
    v,
    node_groups=None,
    alpha=NETWORK_EDGE_ALPHA,
    mode="publication",
    edge_color_override=None,
):
    if edge_color_override:
        return _hex_to_rgba(edge_color_override, alpha)
    effective_alpha = max(0.0, min(1.0, float(alpha)))
    if node_groups:
        color_u = node_groups.get(u, {}).get("color", "#888888")
        color_v = node_groups.get(v, {}).get("color", "#888888")
        if color_u == color_v:
            if mode == "map":
                return _hex_to_rgba(color_u, min(0.92, effective_alpha * 0.92))
            return _hex_to_rgba(color_u, min(0.96, effective_alpha))
        if mode == "map":
            return f"rgba(188, 188, 188, {min(0.72, effective_alpha * 0.55)})"
        return f"rgba(170, 170, 170, {min(0.82, effective_alpha * 0.62)})"
    if mode == "map":
        return f"rgba(188, 188, 188, {min(0.72, effective_alpha * 0.58)})"
    return f"rgba(160, 160, 160, {min(0.88, effective_alpha * 0.78)})"


def _visible_label_nodes(
    G,
    node_groups=None,
    max_labels=18,
    layout_mode="auto",
    mode="publication",
):
    node_count = G.number_of_nodes()
    
    if mode == "map":
        max_labels = max(max_labels, min(node_count, int(node_count * 0.86)))
    elif mode == "publication":
        if node_count > 100:
            max_labels = max(max_labels, max(16, int(node_count * 0.22)))
        elif node_count > 60:
            max_labels = max(max_labels, max(14, int(node_count * 0.28)))
        elif node_count > 35:
            max_labels = max(max_labels, max(12, int(node_count * 0.40)))
        else:
            max_labels = max(max_labels, max(8, int(node_count * 0.55)))
    
    if layout_mode == "bipartite" and node_count <= max(32, max_labels):
        return set(G.nodes())
    if node_count <= max_labels:
        return set(G.nodes())
    
    weighted_degree = dict(G.degree(weight="weight"))
    betweenness = {}
    try:
        betweenness = nx.betweenness_centrality(G, k=min(30, node_count), weight="weight", seed=42)
    except Exception:
        pass
    
    def node_score(node):
        w = G.nodes[node].get("weight", 1)
        d = weighted_degree.get(node, 0)
        b = betweenness.get(node, 0)
        return (w, d, b, str(node))
    
    if node_groups:
        groups = {}
        for node, info in node_groups.items():
            groups.setdefault(info.get("group", 0), []).append(node)
        visible = set()
        group_count = max(len(groups), 1)
        if mode == "map":
            slots_per_group = max(4, (max_labels + group_count - 1) // group_count)
        else:
            if node_count > 80:
                slots_per_group = 1
            else:
                slots_per_group = max(1, min(2, (max_labels + group_count - 1) // group_count))
        
        for nodes in groups.values():
            ranked = sorted(
                nodes,
                key=node_score,
                reverse=True,
            )
            visible.update(ranked[:slots_per_group])
        
        if len(visible) < max_labels:
            remaining = [node for node in G.nodes() if node not in visible]
            ranked = sorted(
                remaining,
                key=node_score,
                reverse=True,
            )
            visible.update(ranked[: max_labels - len(visible)])
        return visible
    
    ranked = sorted(
        G.nodes(),
        key=node_score,
        reverse=True,
    )
    return set(ranked[:max_labels])


def _node_hover_text(G, node, legend_label, type_label=""):
    node_data = G.nodes[node]
    lines = [f"{type_label}{node}" if type_label else str(node), f"{legend_label}: {node_data.get('weight', 1)}"]
    journal = str(node_data.get("journal", "") or "").strip()
    if journal:
        lines.append(f"Journal: {journal}")
    year = str(node_data.get("year", "") or "").strip()
    if year:
        lines.append(f"Year: {year}")
    return "<br>".join(lines)


def _compute_network_positions(G, node_groups=None, layout_mode="auto", edge_weight_attr="weight"):
    if layout_mode == "bipartite" and node_groups:
        left_nodes = [node for node, info in node_groups.items() if info.get("group") == 1]
        right_nodes = [node for node, info in node_groups.items() if info.get("group") == 2]
        return compute_bipartite_layout(G, left_nodes, right_nodes)
    if layout_mode in ("force", "vos", "map"):
        return compute_force_layout(G, weight_attr=edge_weight_attr)
    if node_groups and layout_mode in ("auto", "clustered"):
        group_lookup = {node: info.get("group", 0) for node, info in node_groups.items()}
        if len(set(group_lookup.values())) > 1:
            return compute_cluster_layout(G, group_lookup, weight_attr=edge_weight_attr)
    return compute_force_layout(G, weight_attr=edge_weight_attr)


def _node_text_positions(nodes, pos, *, layout_mode="auto", node_groups=None):
    if layout_mode != "bipartite":
        return "top center"
    positions = []
    for node in nodes:
        group_id = (node_groups or {}).get(node, {}).get("group")
        x_coord = pos.get(node, (0, 0))[0]
        if group_id == 1 or x_coord < 0:
            positions.append("middle right")
        else:
            positions.append("middle left")
    return positions


def _render_network_html(
    G,
    node_groups=None,
    title_prefix="",
    size_range=(24, 84),
    label_max_len=40,
    edge_width_factor=0.5,
    legend_label="Frequency",
    layout_mode="auto",
    max_visible_labels=24,
    label_font_size=23,
    mode="publication",
    edge_color_override=None,
    edge_weight_attr="weight",
    edge_width_scale=1.0,
    edge_alpha_scale=1.0,
):
    if G.number_of_nodes() == 0:
        return None, None

    pos = _compute_network_positions(
        G,
        node_groups=node_groups,
        layout_mode=layout_mode,
        edge_weight_attr=edge_weight_attr,
    )
    freqs = [G.nodes[node].get("weight", 1) for node in G.nodes()]
    min_f, max_f = min(freqs), max(freqs)
    scale = 520
    net = Network(height="820px", width="100%", bgcolor="white", font_color="#333333", directed=False)
    visible_labels = _visible_label_nodes(
        G,
        node_groups=node_groups,
        max_labels=max_visible_labels,
        layout_mode=layout_mode,
        mode=mode,
    )

    for node in G.nodes():
        freq = G.nodes[node].get("weight", 1)
        x, y = pos[node]
        node_size = _map_node_size(
            freq,
            min_f,
            max_f,
            size_min=size_range[0],
            size_max=size_range[1],
            gamma=0.95 if mode == "map" else 1.35,
        )
        if node_groups:
            group_info = node_groups.get(
                node,
                {"color": "#888888", "group": 0, "shape": "dot", "type_label": ""},
            )
            color = group_info["color"]
            border_color = group_info.get("border_color", "rgba(0,0,0,0)")
            border_width = group_info.get("border_width", 0)
            group_id = group_info["group"]
            shape = group_info.get("shape", "dot")
            type_label = group_info.get("type_label", "")
            short_label = node if len(str(node)) <= label_max_len else str(node)[: label_max_len - 3] + "..."
            short_label = short_label if node in visible_labels else ""
            tooltip = _node_hover_text(G, node, "Freq", type_label=type_label)
            net.add_node(
                node,
                label=short_label,
                size=node_size,
                color={"background": color, "border": border_color, "highlight": {"background": color, "border": border_color}},
                title=tooltip,
                x=float(x * scale),
                y=float(y * scale),
                physics=False,
                font={"size": label_font_size if short_label else 1, "color": "#333333", "face": "arial"},
                borderWidth=border_width,
                borderWidthSelected=border_width,
                shape=shape,
                group=group_id,
            )
        else:
            short_label = node if len(str(node)) <= label_max_len else str(node)[: label_max_len - 3] + "..."
            short_label = short_label if node in visible_labels else ""
            net.add_node(
                node,
                label=short_label,
                size=node_size,
                color="#4363d8",
                title=_node_hover_text(G, node, "Freq"),
                x=float(x * scale),
                y=float(y * scale),
                physics=False,
                font={"size": label_font_size if short_label else 1, "color": "#333333", "face": "arial"},
                borderWidth=0,
                borderWidthSelected=0,
                shape="dot",
            )

    edge_weight_values = [data.get(edge_weight_attr, data.get("weight", 1.0)) for _, _, data in G.edges(data=True)]
    min_edge_weight = min(edge_weight_values) if edge_weight_values else 1.0
    max_edge_weight = max(edge_weight_values) if edge_weight_values else 1.0
    edge_weight_label = str(edge_weight_attr).replace("_", " ").title()
    effective_edge_alpha = max(0.0, min(1.0, float(edge_alpha_scale)))
    edge_width_multiplier = _resolve_edge_width_multiplier(edge_width_scale, mode=mode)
    for u, v, data in G.edges(data=True):
        raw_weight = data.get("weight", 1)
        weight = data.get(edge_weight_attr, raw_weight)
        edge_color = _edge_color_for_nodes(
            u,
            v,
            node_groups=node_groups,
            alpha=effective_edge_alpha,
            mode=mode,
            edge_color_override=edge_color_override,
        )
        width = _map_edge_width(weight, min_edge_weight, max_edge_weight, width_min=1.2, width_max=7.2)
        if mode == "map":
            width = max(0.72, width * 0.62)
        width = width * edge_width_multiplier
        net.add_edge(
            u,
            v,
            width=width,
            color=edge_color,
            title=(
                f"{u} — {v}<br>"
                f"Raw links: {raw_weight}<br>"
                f"{edge_weight_label}: {weight:.6f}"
                if isinstance(weight, (int, float))
                else f"{u} — {v}<br>Raw links: {raw_weight}"
            ),
            smooth={"type": "continuous"},
        )

    net.set_options(
        """
    {
      "physics": {
        "enabled": false
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "zoomView": true,
        "dragView": true,
        "dragNodes": true,
        "navigationButtons": true
      }
    }
    """
    )
    html_content = net.generate_html()

    legend_groups = {}
    if node_groups:
        for node, info in node_groups.items():
            group_key = info.get("group", 0)
            if group_key not in legend_groups:
                legend_groups[group_key] = {
                    "color": info["color"],
                    "shape": info.get("shape", "dot"),
                    "label": info.get("type_label", f"Group {group_key}"),
                    "members": [],
                }
            legend_groups[group_key]["members"].append(str(node))

    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        "min_freq": min_f,
        "max_freq": max_f,
        "edge_weight_attr": edge_weight_attr,
    }
    if legend_groups:
        cluster_report = {}
        for group_key in sorted(legend_groups.keys()):
            group_info = legend_groups[group_key]
            cluster_report[group_key] = {
                "color": group_info["color"],
                "shape": group_info["shape"],
                "label": group_info["label"],
                "members": group_info["members"],
            }
        stats["cluster_report"] = cluster_report
    return html_content, stats


def render_network_html(
    G,
    node_groups=None,
    title_prefix="",
    size_range=(24, 84),
    label_max_len=40,
    edge_width_factor=0.5,
    legend_label="Frequency",
    layout_mode="auto",
    max_visible_labels=24,
    label_font_size=23,
    mode="publication",
    edge_color_override=None,
    edge_weight_attr="weight",
    edge_width_scale=1.0,
    edge_alpha_scale=1.0,
):
    return _render_network_html(
        G,
        node_groups=node_groups,
        title_prefix=title_prefix,
        size_range=size_range,
        label_max_len=label_max_len,
        edge_width_factor=edge_width_factor,
        legend_label=legend_label,
        layout_mode=layout_mode,
        max_visible_labels=max_visible_labels,
        label_font_size=label_font_size,
        mode=mode,
        edge_color_override=edge_color_override,
        edge_weight_attr=edge_weight_attr,
        edge_width_scale=edge_width_scale,
        edge_alpha_scale=edge_alpha_scale,
    )


def render_network_publication_figure(
    G,
    node_groups=None,
    title="Network Visualization",
    size_range=None,
    label_max_len=40,
    legend_label="Frequency",
    layout_mode="auto",
    max_visible_labels=None,
    label_font_size=None,
    mode="publication",
    edge_color_override=None,
    edge_weight_attr="weight",
    edge_width_scale=1.0,
    edge_alpha_scale=1.0,
):
    if G is None or G.number_of_nodes() == 0:
        return None

    node_count = G.number_of_nodes()

    if mode == "map":
        if size_range is None:
            if node_count > 100:
                size_range = (5, 20)
            elif node_count > 60:
                size_range = (6, 26)
            else:
                size_range = (9, 34)
        if label_font_size is None:
            label_font_size = 9 if node_count > 90 else 11
        if max_visible_labels is None:
            max_visible_labels = min(node_count, int(node_count * 0.86))
        size_gamma = 0.95
    else:
        if size_range is None:
            if node_count > 100:
                size_range = (4, 48)
            elif node_count > 80:
                size_range = (6, 52)
            elif node_count > 50:
                size_range = (8, 60)
            else:
                size_range = (12, 74)
        if max_visible_labels is None:
            if node_count > 100:
                max_visible_labels = max(10, int(node_count * 0.12))
            elif node_count > 80:
                max_visible_labels = max(10, int(node_count * 0.16))
            elif node_count > 50:
                max_visible_labels = max(10, int(node_count * 0.2))
            else:
                max_visible_labels = max(8, int(node_count * 0.3))
        if label_font_size is None:
            label_font_size = 12 if node_count > 60 else 17
        size_gamma = 1.35
    if layout_mode == "bipartite" and label_font_size is None:
        label_font_size = 18 if mode == "publication" else 16

    size_range = tuple(size_range)
    max_visible_labels = int(max_visible_labels)
    label_font_size = int(label_font_size)

    edge_alpha = max(0.0, min(1.0, float(edge_alpha_scale)))
    edge_width_multiplier = _resolve_edge_width_multiplier(edge_width_scale, mode=mode)

    pos = _compute_network_positions(
        G,
        node_groups=node_groups,
        layout_mode=layout_mode,
        edge_weight_attr=edge_weight_attr,
    )
    freqs = [G.nodes[node].get("weight", 1) for node in G.nodes()]
    min_f, max_f = min(freqs), max(freqs)
    visible_labels = _visible_label_nodes(
        G,
        node_groups=node_groups,
        max_labels=max_visible_labels,
        layout_mode=layout_mode,
        mode=mode,
    )

    fig = go.Figure()
    edge_weights = [data.get(edge_weight_attr, data.get("weight", 1.0)) for _, _, data in G.edges(data=True)]
    if edge_weights:
        min_edge_weight = min(edge_weights)
        max_edge_weight = max(edge_weights)
        edge_groups = {}
        for u, v, data in G.edges(data=True):
            weight = data.get(edge_weight_attr, data.get("weight", 1.0))
            width = round(_map_edge_width(weight, min_edge_weight, max_edge_weight), 2)
            if mode == "map":
                width = max(0.85, width * 0.7)
            else:
                width = max(1.2, width * 1.08)
            width = width * edge_width_multiplier
            edge_color = _edge_color_for_nodes(
                u,
                v,
                node_groups=node_groups,
                alpha=edge_alpha,
                mode=mode,
                edge_color_override=edge_color_override,
            )
            edge_key = (width, edge_color)
            edge_groups.setdefault(edge_key, {"x": [], "y": []})
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_groups[edge_key]["x"].extend([x0, x1, None])
            edge_groups[edge_key]["y"].extend([y0, y1, None])
        for width, edge_color in sorted(edge_groups.keys(), key=lambda item: (item[0], item[1])):
            fig.add_trace(
                go.Scatter(
                    x=edge_groups[(width, edge_color)]["x"],
                    y=edge_groups[(width, edge_color)]["y"],
                    mode="lines",
                    line=dict(color=edge_color, width=width),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    if node_groups:
        grouped_nodes = {}
        for node, info in node_groups.items():
            grouped_nodes.setdefault(info.get("group", 0), []).append(node)
        for group_key in sorted(grouped_nodes.keys()):
            nodes = grouped_nodes[group_key]
            info = node_groups[nodes[0]]
            display_label = info.get("type_label") or f"Group {group_key + 1}"
            if display_label.endswith(": "):
                display_label = f"Cluster {group_key + 1}"
            node_text = [
                node if (node in visible_labels and len(str(node)) <= label_max_len)
                else (str(node)[: label_max_len - 3] + "..." if node in visible_labels else "")
                for node in nodes
            ]
            node_positions = _node_text_positions(
                nodes,
                pos,
                layout_mode=layout_mode,
                node_groups=node_groups,
            )
            fig.add_trace(
                go.Scatter(
                    x=[pos[node][0] for node in nodes],
                    y=[pos[node][1] for node in nodes],
                    mode="markers",
                    name=display_label,
                    hovertext=[
                        _node_hover_text(
                            G,
                            node,
                            legend_label,
                            type_label=info.get("type_label", ""),
                        )
                        for node in nodes
                    ],
                    hoverinfo="text",
                    marker=dict(
                        size=[
                            _map_node_size(
                                G.nodes[node].get("weight", 1),
                                min_f,
                                max_f,
                                size_min=size_range[0],
                                size_max=size_range[1],
                                gamma=size_gamma,
                            )
                            for node in nodes
                        ],
                        color=info.get("color", "#888888"),
                        symbol="square" if info.get("shape") == "square" else "circle",
                        line=dict(
                            color=info.get("border_color", "rgba(0,0,0,0)"),
                            width=info.get("border_width", 0),
                        ),
                        opacity=0.95 if mode == "map" else 0.97,
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[pos[node][0] for node in nodes],
                    y=[pos[node][1] for node in nodes],
                    mode="text",
                    text=node_text,
                    textposition=node_positions,
                    textfont=dict(size=label_font_size, color="#333333"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    else:
        node_text = [
            node if (node in visible_labels and len(str(node)) <= label_max_len)
            else (str(node)[: label_max_len - 3] + "..." if node in visible_labels else "")
            for node in G.nodes()
        ]
        node_positions = _node_text_positions(
            list(G.nodes()),
            pos,
            layout_mode=layout_mode,
            node_groups=node_groups,
        )
        fig.add_trace(
            go.Scatter(
                x=[pos[node][0] for node in G.nodes()],
                y=[pos[node][1] for node in G.nodes()],
                mode="markers",
                hovertext=[_node_hover_text(G, node, legend_label) for node in G.nodes()],
                hoverinfo="text",
                name="Nodes",
                marker=dict(
                    size=[
                        _map_node_size(
                            G.nodes[node].get("weight", 1),
                            min_f,
                            max_f,
                            size_min=size_range[0],
                            size_max=size_range[1],
                            gamma=size_gamma,
                        )
                        for node in G.nodes()
                    ],
                    color=SCIENTIFIC_COLORWAY[0],
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    opacity=0.95 if mode == "map" else 0.97,
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[pos[node][0] for node in G.nodes()],
                y=[pos[node][1] for node in G.nodes()],
                mode="text",
                text=node_text,
                textposition=node_positions,
                textfont=dict(size=label_font_size, color="#333333"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    cluster_count = len({info.get("group", 0) for info in (node_groups or {}).values()}) if node_groups else 0
    if mode == "map":
        margin_t = 50
        margin_b = 50
        show_legend = False
    else:
        margin_t = 60
        margin_b = 80
        show_legend = bool(node_groups) and cluster_count > 1
        if layout_mode != "bipartite" and (cluster_count > 8 or node_count > 40):
            show_legend = False

    if layout_mode == "bipartite":
        margin_l = 170
        margin_r = 170
    else:
        margin_l = 10
        margin_r = 10

    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=22, color="#22313F")),
        height=940,
        showlegend=show_legend,
        legend=dict(
            title="Node Types" if layout_mode == "bipartite" else "Clusters",
            orientation="h",
            yanchor="top",
            y=-0.08, 
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.0)",
            bordercolor="rgba(210,210,210,0.8)",
            borderwidth=1.5,
            font=dict(size=18), 
            title_font=dict(size=20), 
            itemsizing="constant",
        ),
        margin=dict(l=margin_l, r=margin_r, t=margin_t, b=margin_b),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font=dict(size=12, color="#22313F")),
        dragmode="pan",
    )
    if layout_mode == "bipartite":
        fig.update_xaxes(visible=False, range=[-3.15, 3.15])
    else:
        fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    return style_publication_figure(fig, height=940)


@st.cache_data(show_spinner=False)
def render_vosviewer_style(
    keyword_freq,
    cooccurrence,
    top_n=30,
    min_weight=2,
    *,
    size_range=(24, 84),
    edge_width_factor=0.5,
    max_visible_labels=24,
    label_font_size=23,
    edge_color_override=None,
):
    graph = nx.Graph()
    top_keywords = [kw for kw, _ in _most_common_items(keyword_freq, top_n)]
    top_keyword_set = set(top_keywords)
    for keyword in top_keywords:
        graph.add_node(keyword, weight=keyword_freq[keyword])
    for (keyword_a, keyword_b), weight in _most_common_items(cooccurrence, 500):
        if keyword_a in top_keyword_set and keyword_b in top_keyword_set and weight >= min_weight:
            graph.add_edge(keyword_a, keyword_b, weight=weight)
    if len(graph.nodes) == 0:
        return None, None, None

    try:
        partition = community_louvain.best_partition(graph)
    except Exception:
        partition = {node: 0 for node in graph.nodes()}

    node_groups = {}
    for node in graph.nodes():
        cluster_id = partition.get(node, 0)
        node_groups[node] = {
            "color": KEYWORD_CLUSTER_PALETTE[cluster_id % len(KEYWORD_CLUSTER_PALETTE)],
            "group": cluster_id,
            "shape": "dot",
            "type_label": f"Cluster {cluster_id + 1}: ",
        }

    html_content, stats = _render_network_html(
        graph,
        node_groups=node_groups,
        legend_label="Frequency",
        size_range=size_range,
        edge_width_factor=edge_width_factor,
        max_visible_labels=max_visible_labels,
        label_font_size=label_font_size,
        edge_color_override=edge_color_override,
    )
    clusters = {}
    for node, cluster_id in partition.items():
        clusters.setdefault(cluster_id, []).append(node)
    stats["clusters"] = clusters
    stats["cluster_colors"] = {
        cluster_id: KEYWORD_CLUSTER_PALETTE[cluster_id % len(KEYWORD_CLUSTER_PALETTE)]
        for cluster_id in sorted(set(partition.values()))
    }
    return html_content, graph, stats


@st.cache_data(show_spinner=False)
def render_keyword_journal_network(
    top_keywords,
    top_journals,
    kw_journal_cooccur,
    keyword_freq_local,
    journal_freq,
    *,
    size_range=(22, 72),
    max_visible_labels=24,
    label_font_size=23,
    edge_color_override=None,
):
    graph = nx.Graph()
    keyword_color = "#4363d8"
    journal_color = "#e6194b"
    for keyword in top_keywords:
        graph.add_node(keyword, weight=keyword_freq_local.get(keyword, 1))
    for journal in top_journals:
        graph.add_node(journal, weight=journal_freq.get(journal, 1))
    for (keyword, journal), weight in kw_journal_cooccur.items():
        graph.add_edge(keyword, journal, weight=weight)
    if graph.number_of_nodes() == 0:
        return None, None, None, None

    node_groups = {}
    for keyword in top_keywords:
        node_groups[keyword] = {"color": keyword_color, "group": 1, "shape": "dot", "type_label": "Keyword"}
    for journal in top_journals:
        node_groups[journal] = {"color": journal_color, "group": 2, "shape": "square", "type_label": "Journal"}

    html_content, stats = _render_network_html(
        graph,
        node_groups=node_groups,
        size_range=size_range,
        label_max_len=24,
        edge_width_factor=0.3,
        legend_label="Frequency",
        layout_mode="bipartite",
        max_visible_labels=max_visible_labels,
        label_font_size=label_font_size,
        edge_color_override=edge_color_override,
    )
    stats["keyword_nodes"] = len(top_keywords)
    stats["journal_nodes"] = len(top_journals)
    stats["kw_color"] = keyword_color
    stats["journal_color"] = journal_color
    return html_content, graph, node_groups, stats
