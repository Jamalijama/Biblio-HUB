from collections import Counter
import textwrap

import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

from modules.data_pipeline import _parse_wos_authors, clean_year_column

THEMATIC_QUADRANT_COLORS = {
    "Motor themes": "#D45959",
    "Niche themes": "#5E379D",
    "Emerging/Declining themes": "#507D39",
    "Basic themes": "#2F74B8",
}

THREE_FIELD_NODE_COLORS = {
    "authors": "#2F74B8",
    "keywords": "#F2B382",
    "journals": "#D45959",
}

LOTKA_COLORS = {
    "empirical": "#2F74B8",
    "theoretical": "#D45959",
}


def _most_common_items(values, limit):
    if hasattr(values, "most_common"):
        return values.most_common(limit)
    return sorted(values.items(), key=lambda item: item[1], reverse=True)[:limit]


def _wrap_axis_label(text, width=14, max_lines=3):
    value = str(text or "").strip()
    if not value:
        return value
    wrapped = textwrap.wrap(value, width=width, break_long_words=False, break_on_hyphens=True)
    if not wrapped:
        return value
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        last_line = wrapped[-1]
        wrapped[-1] = last_line[:-1] + "..." if len(last_line) > 1 else "..."
    return "<br>".join(wrapped)


def render_thematic_map(keyword_freq, cooccurrence, top_n=20):
    top_kws = [kw for kw, _ in _most_common_items(keyword_freq, top_n)]
    if len(top_kws) < 4:
        return None

    graph = nx.Graph()
    for kw in top_kws:
        graph.add_node(kw, weight=keyword_freq[kw])
    for (keyword_a, keyword_b), weight in cooccurrence.items():
        if keyword_a in top_kws and keyword_b in top_kws:
            graph.add_edge(keyword_a, keyword_b, weight=weight)
    if graph.number_of_edges() == 0:
        return None

    try:
        partition = community_louvain.best_partition(graph, weight="weight", random_state=42)
    except Exception:
        return None

    clusters = {}
    for node, cluster_id in partition.items():
        clusters.setdefault(cluster_id, []).append(node)
    valid_clusters = {cluster_id: nodes for cluster_id, nodes in clusters.items() if len(nodes) >= 2}
    if len(valid_clusters) < 2:
        return None

    cluster_labels = {}
    for cluster_id, nodes in valid_clusters.items():
        label_words = [(node, keyword_freq.get(node, 0)) for node in nodes]
        label_words.sort(key=lambda item: -item[1])
        cluster_labels[cluster_id] = label_words[0][0] if label_words else f"Cluster {cluster_id}"

    cluster_data = []
    for cluster_id, nodes in valid_clusters.items():
        node_set = set(nodes)
        internal_weight = 0.0
        external_weight = 0.0
        for node in nodes:
            for neighbor, edge_data in graph[node].items():
                weight = float(edge_data.get("weight", 1.0))
                if neighbor in node_set:
                    if node < neighbor:
                        internal_weight += weight
                else:
                    external_weight += weight
                                                                                           
        density = (100.0 * internal_weight / len(nodes)) if nodes else 0.0
        centrality = (10.0 * external_weight / len(nodes)) if nodes else 0.0
        keyword_sum = sum(keyword_freq.get(node, 0) for node in nodes)
        if len(nodes) == 1 and density == 0:
            density = 0.05
        if len(nodes) > 1 and density == 0:
            density = 0.1
        if centrality == 0:
            centrality = 0.05
        cluster_data.append(
            {
                "cluster_id": cluster_id,
                "label": cluster_labels[cluster_id],
                "centrality": centrality,
                "density": density,
                "size": len(nodes),
                "keyword_sum": keyword_sum,
                "keywords": ", ".join(sorted(nodes)),
            }
        )
    if not cluster_data:
        return None

    centrality_values = [item["centrality"] for item in cluster_data]
    density_values = [item["density"] for item in cluster_data]
    centrality_median = np.median(centrality_values) if centrality_values else 0
    density_median = np.median(density_values) if density_values else 0

    fig = go.Figure()
    for item in cluster_data:
        if item["centrality"] >= centrality_median and item["density"] >= density_median:
            quadrant = "Motor themes"
        elif item["centrality"] >= centrality_median and item["density"] < density_median:
            quadrant = "Basic themes"
        elif item["centrality"] < centrality_median and item["density"] >= density_median:
            quadrant = "Niche themes"
        else:
            quadrant = "Emerging/Declining themes"

        fig.add_trace(
            go.Scatter(
                x=[item["centrality"]],
                y=[item["density"]],
                mode="markers+text",
                name=item["label"],
                text=[item["label"]],
                textposition="top center",
                marker=dict(
                    size=min(max(item["size"] * 10 + np.sqrt(max(item["keyword_sum"], 1)) * 2.2, 20), 64),
                    color=THEMATIC_QUADRANT_COLORS[quadrant],
                    opacity=0.88,
                    line=dict(width=1.2, color="white"),
                ),
                hovertemplate=(
                    f"<b>{item['label']}</b><br>"
                    f"Centrality: {item['centrality']:.2f}<br>"
                    f"Density: {item['density']:.3f}<br>"
                    f"Keywords: {item['keywords']}<br>"
                    f"Quadrant: {quadrant}<extra></extra>"
                ),
                legendgroup=quadrant,
                showlegend=True,
            )
        )

    fig.add_hline(y=density_median, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=centrality_median, line_dash="dash", line_color="gray", opacity=0.5)
    x_max = max(centrality_values) if centrality_values else 1
    y_max = max(density_values) if density_values else 1
    x_pad = max(x_max * 0.15, 0.5)
    y_pad = max(y_max * 0.15, 0.5)
    fig.add_annotation(
        x=x_max + x_pad * 0.1,
        y=y_max + y_pad * 0.05,
        text="<b>MOTOR THEMES</b>",                   
        font=dict(size=14, color="#b71c2c"),                      
        showarrow=False,
    )
    fig.add_annotation(
        x=x_max + x_pad * 0.1,
        y=max(density_median * 0.35, 0.05),
        text="<b>BASIC THEMES</b>",
        font=dict(size=14, color="#2F74B8"),
        showarrow=False,
    )
    fig.add_annotation(
        x=max(centrality_median * 0.35, 0.05),
        y=y_max + y_pad * 0.05,
        text="<b>NICHE THEMES</b>",
        font=dict(size=14, color="#5E379D"),
        showarrow=False,
    )
    fig.add_annotation(
        x=max(centrality_median * 0.35, 0.05),
        y=max(density_median * 0.35, 0.05),
        text="<b>EMERGING/DECLINING</b>",
        font=dict(size=14, color="#507D39"),
        showarrow=False,
    )
    fig.update_layout(
        title=dict(text="Thematic Map (Centrality vs. Density)", font=dict(size=16)),
        xaxis_title="Callon Centrality (Weighted External Links)",
        yaxis_title="Callon Density (Weighted Internal Cohesion)",
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="lightgray", range=[0, x_max + x_pad])
    fig.update_yaxes(gridcolor="lightgray", range=[0, y_max + y_pad])
    return fig


def render_three_field_plot(df, keywords_list, keyword_freq, top_n_authors=10, top_n_keywords=15, top_n_journals=10):
    if "Authors" not in df.columns or "Journal" not in df.columns:
        return None

    author_freq = Counter()
    author_kw_links = Counter()
    kw_journal_links = Counter()
    kw_freq = Counter()
    journal_freq = Counter()
    df_reset = df.reset_index(drop=True)

    for idx, row in df_reset.iterrows():
        journal = str(row.get("Journal", "")).strip()
        if not journal or journal == "nan":
            continue
        journal_freq[journal] += 1
        authors_str = str(row.get("Authors", ""))
        authors = [author.strip() for author in authors_str.split(";") if author.strip() and author.strip() != "nan"]
        keywords = keywords_list[idx] if idx < len(keywords_list) else []
        for author in authors:
            author_freq[author] += 1
            for keyword in keywords:
                author_kw_links[(author, keyword)] += 1
                kw_freq[keyword] += 1
        for keyword in keywords:
            kw_journal_links[(keyword, journal)] += 1

    top_authors = [author for author, _ in author_freq.most_common(top_n_authors)]
    top_keywords = [keyword for keyword, _ in kw_freq.most_common(top_n_keywords)]
    top_journals = [journal for journal, _ in journal_freq.most_common(top_n_journals)]
    if not top_authors or not top_keywords or not top_journals:
        return None

    sources = []
    targets = []
    values = []
    link_colors = []

    author_node_indices = {}
    keyword_node_indices = {}
    journal_node_indices = {}
    valid_labels = []
    node_colors = []

    active_authors = [
        author for author in top_authors if sum(author_kw_links.get((author, keyword), 0) for keyword in top_keywords) > 0
    ]
    active_keywords = [
        keyword
        for keyword in top_keywords
        if sum(author_kw_links.get((author, keyword), 0) for author in active_authors) > 0
        or sum(kw_journal_links.get((keyword, journal), 0) for journal in top_journals) > 0
    ]
    active_journals = [
        journal for journal in top_journals if sum(kw_journal_links.get((keyword, journal), 0) for keyword in active_keywords) > 0
    ]

    current_idx = 0
    for author in active_authors:
        author_node_indices[author] = current_idx
        valid_labels.append(author)
        node_colors.append(THREE_FIELD_NODE_COLORS["authors"])
        current_idx += 1
    for keyword in active_keywords:
        keyword_node_indices[keyword] = current_idx
        valid_labels.append(keyword)
        node_colors.append(THREE_FIELD_NODE_COLORS["keywords"])
        current_idx += 1
    for journal in active_journals:
        journal_node_indices[journal] = current_idx
        valid_labels.append(journal)
        node_colors.append(THREE_FIELD_NODE_COLORS["journals"])
        current_idx += 1

    for author in active_authors:
        for keyword in active_keywords:
            weight = author_kw_links.get((author, keyword), 0)
            if weight > 0:
                sources.append(author_node_indices[author])
                targets.append(keyword_node_indices[keyword])
                values.append(weight)
                link_colors.append("rgba(47, 116, 184, 0.38)")

    for keyword in active_keywords:
        for journal in active_journals:
            weight = kw_journal_links.get((keyword, journal), 0)
            if weight > 0:
                sources.append(keyword_node_indices[keyword])
                targets.append(journal_node_indices[journal])
                values.append(weight)
                link_colors.append("rgba(212, 89, 89, 0.36)")

    if not values:
        return None

    x_coords = []
    y_coords = []
    for idx in range(len(active_authors)):
        x_coords.append(0.01)
        y_coords.append(0.05 + 0.9 * (idx / max(1, len(active_authors) - 1)))
    for idx in range(len(active_keywords)):
        x_coords.append(0.5)
        y_coords.append(0.05 + 0.9 * (idx / max(1, len(active_keywords) - 1)))
    for idx in range(len(active_journals)):
        x_coords.append(0.99)
        y_coords.append(0.05 + 0.9 * (idx / max(1, len(active_journals) - 1)))

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=14,
                thickness=26,
                line=dict(color="white", width=0.5),
                label=valid_labels,
                color=node_colors,
                x=x_coords,
                y=y_coords,
            ),
            link=dict(source=sources, target=targets, value=values, color=link_colors),
        )
    )
    fig.update_layout(
        title=dict(
            text="Three-Field Plot: Authors → Keywords → Journals",
            font=dict(size=16),
            y=0.995,
            yanchor="top",
        ),
        height=max(500, len(valid_labels) * 12 + 100),
        font=dict(size=12),
        margin=dict(l=10, r=10, t=130, b=10),
    )
                        
    headers = ["Authors", "Keywords", "Journals"]
    header_x = [0.01, 0.5, 0.99]
    for text, x in zip(headers, header_x):
        fig.add_annotation(
            x=x,
            y=1.045,
            xref="paper",
            yref="paper",
            text=f"<b>{text}</b>",
            showarrow=False,
            font=dict(size=15, color="#22313F"),
            xanchor="center" if x == 0.5 else ("left" if x < 0.5 else "right"),
        )
    return fig


def build_author_production_over_time_frame(df, top_n=10):
    if "Authors" not in df.columns or "Year" not in df.columns:
        return pd.DataFrame()

    df_valid = clean_year_column(df)
    if df_valid.empty:
        return pd.DataFrame()

    author_freq = Counter()
    author_year_counts = Counter()
    for _, row in df_valid.iterrows():
        year = int(row["Year"])
        authors = _parse_wos_authors(row.get("Authors", ""))
        for author in authors:
            author_freq[author] += 1
            author_year_counts[(author, year)] += 1

    top_authors = [author for author, _ in author_freq.most_common(top_n)]
    if not top_authors:
        return pd.DataFrame()

    years = sorted(df_valid["Year"].unique())
    rows = []
    for author in top_authors:
        for year in years:
            rows.append(
                {
                    "Author": author,
                    "Year": int(year),
                    "Papers": author_year_counts.get((author, int(year)), 0),
                    "Total Papers": author_freq[author],
                }
            )
    return pd.DataFrame(rows)


def render_author_production_over_time(df, top_n=10):
    production_df = build_author_production_over_time_frame(df, top_n=top_n)
    if production_df.empty:
        return None

    visible_df = production_df[production_df["Papers"] > 0].copy()
    if visible_df.empty:
        return None

    fig = go.Figure()
    for author in production_df["Author"].drop_duplicates().tolist():
        author_df = visible_df[visible_df["Author"] == author]
        if author_df.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=author_df["Year"],
                y=author_df["Papers"],
                mode="lines+markers",
                name=author,
                line=dict(width=1.6),
                marker=dict(size=5),
                customdata=author_df[["Total Papers"]],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Year: %{x}<br>"
                    "Papers: %{y}<br>"
                    "Total Papers: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(text="Authors' Production Over Time", font=dict(size=16)),
        xaxis_title="Year",
        yaxis_title="Publications",
        hovermode="x unified",
        height=max(520, 360 + 18 * production_df["Author"].nunique()),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
    )
    fig.update_xaxes(gridcolor="lightgray", showgrid=True)
    fig.update_yaxes(gridcolor="lightgray", showgrid=True, rangemode="tozero")
    return fig


def render_lotkas_law(df):
    author_freq = Counter()
    for _, row in df.iterrows():
        authors = _parse_wos_authors(row.get("Authors", ""))
        for author in authors:
            author_freq[author] += 1

    if not author_freq:
        return None

    counts = list(author_freq.values())
    freq_of_freq = Counter(counts)
    max_papers = max(counts)
    if max_papers < 2:
        return None

    x_vals = list(range(1, max_papers + 1))
    total_authors = len(counts)
    empirical = [freq_of_freq.get(x, 0) / total_authors for x in x_vals]

    c_value = 1.0 / sum([1.0 / (x ** 2) for x in range(1, max_papers + 100)])
    theoretical = [c_value / (x ** 2) for x in x_vals]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=empirical,
            mode="lines+markers",
            name="Empirical Data",
            marker=dict(size=8, color=LOTKA_COLORS["empirical"]),
            line=dict(width=2, color=LOTKA_COLORS["empirical"]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=theoretical,
            mode="lines",
            name="Lotka's Law (Theoretical)",
            line=dict(dash="dash", color=LOTKA_COLORS["theoretical"], width=2),
        )
    )
    fig.update_layout(
        title=dict(text="Lotka's Law (Author Productivity Distribution)", font=dict(size=16)),
        xaxis_title="Number of Articles Written (log scale)",
        yaxis_title="Proportion of Authors (log scale)",
        yaxis_type="log",
        xaxis_type="log",
        hovermode="x unified",
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    fig.update_xaxes(gridcolor="lightgray", showgrid=True)
    fig.update_yaxes(gridcolor="lightgray", showgrid=True)

    authors_with_1_paper = freq_of_freq.get(1, 0)
    percent_with_1 = authors_with_1_paper / total_authors * 100
    stats = {
        "total_authors": total_authors,
        "max_papers": max_papers,
        "authors_with_1_paper": authors_with_1_paper,
        "percent_with_1": percent_with_1,
        "theoretical_c": c_value,
    }
    return fig, stats


def render_hierarchical_cluster_heatmap(keywords_list, keyword_freq, cooccurrence, top_n=16, generic_exclusions=None):
    if generic_exclusions is None:
        generic_exclusions = set()
    
    candidates = [kw for kw, _ in sorted(keyword_freq.items(), key=lambda x: (-x[1], x[0])) if kw.lower() not in generic_exclusions]
    selected_kws = candidates[:top_n]
    
    if len(selected_kws) < 2:
        return None, None
    
    keyword_to_idx = {kw: idx for idx, kw in enumerate(selected_kws)}
    matrix = np.zeros((len(selected_kws), len(selected_kws)), dtype=float)
    
    for doc_kws in keywords_list:
        filtered_kws = [kw for kw in doc_kws if kw in keyword_to_idx]
        unique_kws = sorted(set(filtered_kws), key=lambda x: keyword_to_idx[x])
        for i, kw_i in enumerate(unique_kws):
            idx_i = keyword_to_idx[kw_i]
            for kw_j in unique_kws[i + 1:]:
                idx_j = keyword_to_idx[kw_j]
                matrix[idx_i, idx_j] += 1
                matrix[idx_j, idx_i] += 1
    
    if len(selected_kws) >= 3:
        try:
            profile = matrix
            distances = pdist(profile, metric="cosine")
            distances = np.nan_to_num(distances, nan=1.0)
            tree = linkage(distances, method="average")
            order = leaves_list(tree)
            ordered_kws = [selected_kws[idx] for idx in order]
            ordered_matrix = matrix[order, :][:, order]
        except Exception:
            ordered_kws = selected_kws
            ordered_matrix = matrix
    else:
        ordered_kws = selected_kws
        ordered_matrix = matrix
    
    matrix_df = pd.DataFrame(ordered_matrix, index=ordered_kws, columns=ordered_kws)
    
    log_matrix = np.log1p(ordered_matrix)
    # Plotly renders the categorical y-axis from bottom to top in this setup,
    # so keeping the lower triangle in data space displays as the upper-left
    # triangle in the final figure.
    triangular_log_matrix = np.where(
        np.tril(np.ones_like(log_matrix, dtype=bool)),
        log_matrix,
        np.nan,
    )
    
    wrapped_kws = [_wrap_axis_label(keyword, width=14, max_lines=2) for keyword in ordered_kws]

    fig = go.Figure(data=go.Heatmap(
        z=triangular_log_matrix,
        x=wrapped_kws,
        y=wrapped_kws,
        colorscale="YlOrRd",
        hoverongaps=False,
        zmin=0,
        xgap=1,
        ygap=1,
        colorbar=dict(
            title=dict(
                text="ln(1 + co-occurrence count)",
                side="right",
                font=dict(size=16),
            ),
            tickfont=dict(size=15),
            len=0.9,
            thickness=32,
            x=1.03,
            xanchor="left",
        ),
    ))
    
    fig.update_layout(
        title="Hierarchical Clustering Heatmap of Keyword Co-occurrence",
        xaxis_title="Keywords",
        yaxis_title="Keywords",
        xaxis_tickangle=-30,
        height=800,
        font=dict(size=15),
        title_font=dict(size=20),
        margin=dict(l=190, r=140, t=90, b=140),
    )
    fig.update_xaxes(
        tickfont=dict(size=13),
        automargin=True,
        side="bottom",
    )
    fig.update_yaxes(
        tickfont=dict(size=13),
        automargin=True,
        autorange="reversed",
    )
    
    return fig, matrix_df
