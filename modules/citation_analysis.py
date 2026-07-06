import re
from collections import Counter

import community as community_louvain
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

from modules.data_pipeline import clean_year_column
from modules.network_visualization import render_network_html
from modules.temporal_analysis import (
    _compute_adjusted_burst_score,
    build_burst_table,
    render_burst_figure,
)


COCITATION_CLUSTER_PALETTE = [
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
]

JOURNAL_COCITATION_PALETTE = [
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
]


def _bridge_scores(graph):
    node_count = graph.number_of_nodes()
    if node_count == 0:
        return {}, 1
    try:
        if node_count > 120:
            betweenness = nx.betweenness_centrality(
                graph,
                k=min(18, node_count),
                weight="weight",
                seed=42,
            )
        elif node_count > 80:
            betweenness = nx.betweenness_centrality(
                graph,
                k=min(26, node_count),
                weight="weight",
                seed=42,
            )
        elif node_count > 50:
            betweenness = nx.betweenness_centrality(
                graph,
                k=min(36, node_count),
                weight="weight",
                seed=42,
            )
        else:
            betweenness = nx.betweenness_centrality(graph, weight="weight")
        threshold = np.percentile(list(betweenness.values()), 90) if betweenness else 1
        return betweenness, threshold
    except Exception:
        return {}, 1


def extract_journal_from_citation(raw_citation):
    citation = str(raw_citation).strip()
    if not citation or len(citation) < 10:
        return None
    citation = re.sub(r'\[.*?\]', '', citation)
    citation = re.sub(r'\(.*?\)', '', citation)
    year_match = re.search(r',\s*(\d{4})\b', citation)
    if not year_match:
        return None
    after_year = citation[year_match.end():].strip()
    if after_year.startswith(','):
        after_year = after_year[1:].strip()
    parts = [part.strip() for part in after_year.split(',')]
    journal_candidates = []
    for part in parts:
        part_upper = part.upper()
        if re.match(r'^V[OL\.]*\s*\d+', part_upper):
            continue
        if re.match(r'^P[AG\.]*\s*\d+', part_upper):
            continue
        if re.match(r'^\d+$', part):
            continue
        if part.lower() in ['et al', 'et al.']:
            continue
        if len(part) < 2:
            continue
        journal_candidates.append(part)
    if journal_candidates:
        journal = journal_candidates[0]
        journal = re.sub(r'\s+', ' ', journal).strip()
        return journal
    return None


@st.cache_data(show_spinner=False)
def build_journal_cocitation_network(
    df, top_n_journals=20, min_cocite=2,
    *,
    size_range=(24, 84),
    max_visible_labels=20,
    label_font_size=23,
    generate_html=True,
):
    journal_freq = Counter()
    paper_cited_journals = []
    for _, row in df.iterrows():
        cr_value = row.get("Cited_References", "")
        cr_str = str(cr_value)
        if pd.isna(cr_value) or cr_str.strip() in ("", "nan"):
            continue
        refs = [reference.strip() for reference in cr_str.split(";") if reference.strip()]
        cited_journals = []
        for reference in refs:
            journal = extract_journal_from_citation(reference)
            if journal:
                normalized_journal = re.sub(r'\s+', ' ', journal).strip().lower()
                normalized_journal = normalized_journal.title()
                cited_journals.append(normalized_journal)
                journal_freq[normalized_journal] += 1
        if cited_journals:
            paper_cited_journals.append(tuple(dict.fromkeys(cited_journals)))
    top_journals = {journal for journal, _ in journal_freq.most_common(top_n_journals)}
    journal_pairs = Counter()
    for journals_in_paper in paper_cited_journals:
        filtered_journals = [journal for journal in journals_in_paper if journal in top_journals]
        for i in range(len(filtered_journals)):
            for j in range(i + 1, len(filtered_journals)):
                pair = tuple(sorted([filtered_journals[i], filtered_journals[j]]))
                journal_pairs[pair] += 1
    graph = nx.Graph()
    for journal in top_journals:
        graph.add_node(journal, weight=journal_freq[journal])
    for (journal_a, journal_b), weight in journal_pairs.items():
        if journal_a in graph.nodes and journal_b in graph.nodes and weight >= min_cocite:
            graph.add_edge(journal_a, journal_b, weight=weight)
    isolated_nodes = [node for node in graph.nodes if graph.degree(node) == 0]
    graph.remove_nodes_from(isolated_nodes)
    if graph.number_of_nodes() == 0:
        return None, None, None, None, journal_freq
    try:
        partition = community_louvain.best_partition(graph)
    except Exception:
        partition = {node: 0 for node in graph.nodes()}
    betweenness, threshold = _bridge_scores(graph)
    node_groups = {}
    for node in graph.nodes():
        cluster_id = partition.get(node, 0)
        base_color = JOURNAL_COCITATION_PALETTE[cluster_id % len(JOURNAL_COCITATION_PALETTE)]
        score = betweenness.get(node, 0)
        is_bridge = score >= threshold and score > 0
        node_groups[node] = {
            "color": base_color,
            "border_color": "rgba(0,0,0,0)",
            "border_width": 0,
            "group": cluster_id,
            "shape": "square",
            "type_label": f"Cluster {cluster_id + 1}: ",
        }
    
    html_content = None
    stats = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "avg_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
        "min_freq": min(graph.nodes[node].get("weight", 1) for node in graph.nodes()),
        "max_freq": max(graph.nodes[node].get("weight", 1) for node in graph.nodes()),
    }
    if generate_html:
        html_content, html_stats = render_network_html(
            graph,
            node_groups=node_groups,
            size_range=size_range,
            label_max_len=25,
            edge_width_factor=0.5,
            legend_label="Co-citations",
            layout_mode="clustered",
            max_visible_labels=max_visible_labels,
            label_font_size=label_font_size,
        )
        if html_stats:
            stats = html_stats
    return html_content, graph, node_groups, stats, journal_freq


@st.cache_data(show_spinner=False)
def build_author_cocitation_network(
    df, top_n_authors=20, min_cocite=2,
    *,
    size_range=(24, 84),
    max_visible_labels=20,
    label_font_size=23,
    generate_html=True,
):
    author_freq = Counter()
    paper_cited_authors = []

    for _, row in df.iterrows():
        cr_value = row.get("Cited_References", "")
        cr_str = str(cr_value)
        if pd.isna(cr_value) or cr_str.strip() in ("", "nan"):
            continue
        refs = [reference.strip() for reference in cr_str.split(";") if reference.strip()]
        cited_authors = []
        for reference in refs:
            author = extract_author_from_citation(reference)
            if author:
                normalized_author = re.sub(r'\s+', ' ', author).strip().title()
                cited_authors.append(normalized_author)
                author_freq[normalized_author] += 1
        if cited_authors:
            paper_cited_authors.append(tuple(dict.fromkeys(cited_authors)))
    top_authors = {author for author, _ in author_freq.most_common(top_n_authors)}
    author_pairs = Counter()
    for authors_in_paper in paper_cited_authors:
        filtered_authors = [author for author in authors_in_paper if author in top_authors]
        for i in range(len(filtered_authors)):
            for j in range(i + 1, len(filtered_authors)):
                pair = tuple(sorted([filtered_authors[i], filtered_authors[j]]))
                author_pairs[pair] += 1
    graph = nx.Graph()
    for author in top_authors:
        graph.add_node(author, weight=author_freq[author])
    for (author_a, author_b), weight in author_pairs.items():
        if author_a in graph.nodes and author_b in graph.nodes and weight >= min_cocite:
            graph.add_edge(author_a, author_b, weight=weight)
    isolated_nodes = [node for node in graph.nodes if graph.degree(node) == 0]
    graph.remove_nodes_from(isolated_nodes)
    if graph.number_of_nodes() == 0:
        return None, None, None, None, author_freq
    try:
        partition = community_louvain.best_partition(graph)
    except Exception:
        partition = {node: 0 for node in graph.nodes()}
    betweenness, threshold = _bridge_scores(graph)
    node_groups = {}
    for node in graph.nodes():
        cluster_id = partition.get(node, 0)
        base_color = COCITATION_CLUSTER_PALETTE[cluster_id % len(COCITATION_CLUSTER_PALETTE)]
        score = betweenness.get(node, 0)
        is_bridge = score >= threshold and score > 0
        node_groups[node] = {
            "color": base_color,
            "border_color": "rgba(0,0,0,0)",
            "border_width": 0,
            "group": cluster_id,
            "shape": "dot",
            "type_label": f"Cluster {cluster_id + 1}: ",
        }
    
    html_content = None
    stats = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "avg_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
        "min_freq": min(graph.nodes[node].get("weight", 1) for node in graph.nodes()),
        "max_freq": max(graph.nodes[node].get("weight", 1) for node in graph.nodes()),
    }
    if generate_html:
        html_content, html_stats = render_network_html(
            graph,
            node_groups=node_groups,
            size_range=size_range,
            label_max_len=25,
            edge_width_factor=0.5,
            legend_label="Co-citations",
            layout_mode="clustered",
            max_visible_labels=max_visible_labels,
            label_font_size=label_font_size,
        )
        if html_stats:
            stats = html_stats
    return html_content, graph, node_groups, stats, author_freq


def extract_author_from_citation(raw_citation):
    citation = str(raw_citation).strip()
    if not citation or len(citation) < 5:
        return None
    citation = re.sub(r'\[.*?\]', '', citation)
    citation = re.sub(r'\(.*?\)', '', citation)
    parts = citation.split(',')
    if not parts:
        return None
    author_candidate = parts[0].strip()
    if not author_candidate or len(author_candidate) < 2:
        return None
    if author_candidate.lower() == "et al" or author_candidate.lower() == "et al.":
        return None
    if re.match(r'^[\d\s]+$', author_candidate):
        return None
    return author_candidate


def clean_cited_reference(raw_reference):
    reference = str(raw_reference).strip()
    if not reference or len(reference) < 5:
        return None
    if reference.lower().startswith("[anonymous]"):
        return None

    parts = reference.split(",")
    if len(parts) >= 2:
        author = parts[0].strip()
        if author.lower().startswith("[anonymous]") or len(author) < 2:
            return None
        year_part = parts[1].strip()
        if not re.match(r"\d{4}", year_part):
            for part in parts[2:]:
                if re.match(r"\d{4}", part.strip()):
                    year_part = part.strip()
                    break
            else:
                return None
        short_ref = f"{author}, {year_part}"
    else:
        if len(reference) < 8:
            return None
        short_ref = reference[:60]

    if re.match(r"^[\d\s]+$", short_ref):
        return None
    return short_ref


@st.cache_data(show_spinner=False)
def extract_cited_reference_statistics(df):
    ref_freq = Counter()
    ref_journal = {}
    ref_year = {}
    ref_year_freq = {}
    ref_counts_per_paper = []

    for _, row in df.iterrows():
        cr_value = row.get("Cited_References", "")
        cr_str = str(cr_value)
        if pd.isna(cr_value) or cr_str.strip() in ("", "nan"):
            continue

        refs = [reference.strip() for reference in cr_str.split(";") if reference.strip()]
        ref_counts_per_paper.append(len(refs))
        for reference in refs:
            clean_ref = clean_cited_reference(reference)
            if clean_ref:
                ref_freq[clean_ref] += 1
            year_match = re.search(r",\s*(\d{4})\b", reference)
            if year_match:
                year = int(year_match.group(1))
                if 1900 <= year <= 2030:
                    ref_year_freq[year] = ref_year_freq.get(year, 0) + 1

    return ref_freq, ref_year_freq, ref_counts_per_paper


@st.cache_data(show_spinner=False)
def build_cocitation_network(
    df, top_n_ref=20, min_cocite=2,
    *,
    size_range=(24, 84),
    max_visible_labels=24,
    label_font_size=23,
    generate_html=True,
):
    ref_freq = Counter()
    ref_journal = {}
    ref_year = {}
    paper_refs = []

    for _, row in df.iterrows():
        cr_value = row.get("Cited_References", "")
        cr_str = str(cr_value)
        if pd.isna(cr_value) or cr_str.strip() in ("", "nan"):
            continue

        refs = [reference.strip() for reference in cr_str.split(";") if reference.strip()]
        short_refs = []
        for reference in refs:
            clean_ref = clean_cited_reference(reference)
            if clean_ref:
                short_refs.append(clean_ref)
                ref_freq[clean_ref] += 1
                ref_journal.setdefault(clean_ref, extract_journal_from_citation(reference) or "")
                year_match = re.search(r",\s*(\d{4})\b", reference)
                if year_match:
                    ref_year.setdefault(clean_ref, year_match.group(1))
        if short_refs:
            paper_refs.append(tuple(dict.fromkeys(short_refs)))

    top_refs = {reference for reference, _ in ref_freq.most_common(top_n_ref)}
    ref_pairs = Counter()
    for short_refs in paper_refs:
        filtered_refs = [reference for reference in short_refs if reference in top_refs]
        for i in range(len(filtered_refs)):
            for j in range(i + 1, len(filtered_refs)):
                pair = tuple(sorted([filtered_refs[i], filtered_refs[j]]))
                ref_pairs[pair] += 1

    graph = nx.Graph()
    for reference in top_refs:
        graph.add_node(
            reference,
            weight=ref_freq[reference],
            journal=ref_journal.get(reference, ""),
            year=ref_year.get(reference, ""),
        )
    for (ref_a, ref_b), weight in ref_pairs.items():
        if ref_a in graph.nodes and ref_b in graph.nodes and weight >= min_cocite:
            graph.add_edge(ref_a, ref_b, weight=weight)

    isolated_nodes = [node for node in graph.nodes if graph.degree(node) == 0]
    graph.remove_nodes_from(isolated_nodes)
    if graph.number_of_nodes() == 0:
        return None, None, None, None, ref_freq

    try:
        partition = community_louvain.best_partition(graph)
    except Exception:
        partition = {node: 0 for node in graph.nodes()}

    betweenness, threshold = _bridge_scores(graph)

    node_groups = {}
    for node in graph.nodes():
        cluster_id = partition.get(node, 0)
        base_color = COCITATION_CLUSTER_PALETTE[cluster_id % len(COCITATION_CLUSTER_PALETTE)]
        score = betweenness.get(node, 0)
        is_bridge = score >= threshold and score > 0
        node_groups[node] = {
            "color": base_color,
            "border_color": "rgba(0,0,0,0)",
            "border_width": 0,
            "group": cluster_id,
            "shape": "dot",
            "type_label": f"Cluster {cluster_id + 1}: ",
        }

    html_content = None
    stats = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "avg_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
        "min_freq": min(graph.nodes[node].get("weight", 1) for node in graph.nodes()),
        "max_freq": max(graph.nodes[node].get("weight", 1) for node in graph.nodes()),
    }
    if generate_html:
        html_content, html_stats = render_network_html(
            graph,
            node_groups=node_groups,
            size_range=size_range,
            label_max_len=30,
            edge_width_factor=0.5,
            legend_label="Citations",
            layout_mode="clustered",
            max_visible_labels=max_visible_labels,
            label_font_size=label_font_size,
        )
        if html_stats:
            stats = html_stats
    return html_content, graph, node_groups, stats, ref_freq


@st.cache_data(show_spinner=False)
def extract_rpys_statistics(df):
    ref_year_freq = Counter()
    refs_by_year = {}
    for _, row in df.iterrows():
        cr_value = row.get("Cited_References", "")
        cr_str = str(cr_value)
        if pd.isna(cr_value) or cr_str.strip() in ("", "nan"):
            continue
        refs = [reference.strip() for reference in cr_str.split(";") if reference.strip()]
        for reference in refs:
            year_match = re.search(r",\s*(\d{4})\b", reference)
            if year_match:
                year = int(year_match.group(1))
                if 1900 <= year <= 2030:
                    ref_year_freq[year] += 1
                    clean_ref = clean_cited_reference(reference)
                    if clean_ref:
                        if year not in refs_by_year:
                            refs_by_year[year] = Counter()
                        refs_by_year[year][clean_ref] += 1
    if not ref_year_freq:
        return pd.DataFrame(columns=["Year", "Count", "Median_5yr", "Deviation"]), {}
    min_year = min(ref_year_freq.keys())
    max_year = max(ref_year_freq.keys())
    years = list(range(min_year, max_year + 1))
    data = []
    for year in years:
        count = ref_year_freq.get(year, 0)
        data.append({"Year": year, "Count": count})
    df_rpys = pd.DataFrame(data).sort_values("Year")
    df_rpys["Median_5yr"] = df_rpys["Count"].rolling(window=5, center=True, min_periods=1).median()
    df_rpys["Deviation"] = df_rpys["Count"] - df_rpys["Median_5yr"]
    return df_rpys, refs_by_year


def render_rpys_figure(rpys_df):
    import plotly.graph_objects as go
    import plotly.express as px
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=rpys_df["Year"],
        y=rpys_df["Count"],
        name="Reference Count",
        marker_color="#8FA7A0",
        opacity=0.7
    ))
    fig.add_trace(go.Scatter(
        x=rpys_df["Year"],
        y=rpys_df["Median_5yr"],
        name="5-Year Median",
        line=dict(color="#BFA7A8", width=2),
        mode="lines"
    ))
    positive_deviation = rpys_df[rpys_df["Deviation"] > 0]
    if not positive_deviation.empty:
        fig.add_trace(go.Bar(
            x=positive_deviation["Year"],
            y=positive_deviation["Deviation"],
            name="Peak Deviation",
            marker_color="#BFA7A8",
            opacity=0.5,
            base=positive_deviation["Median_5yr"]
        ))
    fig.update_layout(
        title="Reference Publication Year Spectroscopy (RPYS)",
        xaxis_title="Year",
        yaxis_title="Number of Cited References",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600
    )
    
    return fig


def build_rpys_peak_table(rpys_df, refs_by_year, top_n=10, refs_per_year=3):
    peaks = rpys_df[rpys_df["Deviation"] > 0].copy()
    if peaks.empty:
        return pd.DataFrame()
    peaks = peaks.sort_values("Deviation", ascending=False).head(top_n)
    result = []
    for _, row in peaks.iterrows():
        year = int(row["Year"])
        count = int(row["Count"])
        deviation = int(row["Deviation"])
        top_refs = []
        if year in refs_by_year:
            top_refs = [f"{ref} ({cnt})" for ref, cnt in refs_by_year[year].most_common(refs_per_year)]
        result.append({
            "Year": year,
            "Reference Count": count,
            "Deviation from Median": deviation,
            "Top References": "; ".join(top_refs)
        })
    
    return pd.DataFrame(result)


@st.cache_data(show_spinner=False)
def build_reference_burst_table(df, top_n=20):
    df_valid = clean_year_column(df)
    year_range = sorted(df_valid["Year"].unique())
    ref_freq = Counter()
    reference_year_freq = {}

    for row_idx, row in df_valid.iterrows():
        cr_value = row.get("Cited_References", "")
        cr_str = str(cr_value)
        if pd.isna(cr_value) or cr_str.strip() in ("", "nan"):
            continue

        refs = [reference.strip() for reference in cr_str.split(";") if reference.strip()]
        for reference in refs:
            clean_ref = clean_cited_reference(reference)
            if not clean_ref:
                continue
            ref_freq[clean_ref] += 1
            year = int(row["Year"])
            if clean_ref not in reference_year_freq:
                reference_year_freq[clean_ref] = {burst_year: 0 for burst_year in year_range}
            reference_year_freq[clean_ref][year] += 1

    top_refs = [reference for reference, _ in ref_freq.most_common(top_n)]
    top_reference_year_freq = {reference: reference_year_freq[reference] for reference in top_refs if reference in reference_year_freq}
    return build_burst_table(
        top_reference_year_freq,
        year_range,
        label_column="Reference",
        top_n=top_n,
        include_fallback=True,
    )


def render_reference_burst_figure(reference_burst_df):
    if reference_burst_df is None or reference_burst_df.empty:
        return None

    burst_df = reference_burst_df.copy()
    if "Duration" not in burst_df.columns and {"Start", "End"}.issubset(burst_df.columns):
        burst_df["Duration"] = (
            pd.to_numeric(burst_df["End"], errors="coerce")
            - pd.to_numeric(burst_df["Start"], errors="coerce")
            + 1
        ).clip(lower=1)
    if "Adjusted Burst Score" not in burst_df.columns:
        durations = (
            pd.to_numeric(burst_df["Duration"], errors="coerce").fillna(1)
            if "Duration" in burst_df.columns
            else pd.Series([1] * len(burst_df), index=burst_df.index)
        )
        raw_scores = pd.to_numeric(burst_df["Burst Strength"], errors="coerce").fillna(0)
        burst_df["Adjusted Burst Score"] = [
            round(_compute_adjusted_burst_score(raw_score, int(duration)), 4)
            for raw_score, duration in zip(raw_scores, durations)
        ]

    return render_burst_figure(
        burst_df,
        label_column="Reference",
        title="Reference Burst Detection (Kleinberg Algorithm)",
    )


@st.cache_data(show_spinner=False)
def build_publication_citation_trend_frame(df):
    if "Year" not in df.columns or "Times_Cited" not in df.columns:
        return pd.DataFrame()

    df_cite = df.copy()
    df_cite["Times_Cited"] = pd.to_numeric(df_cite["Times_Cited"], errors="coerce").fillna(0)
    df_valid = clean_year_column(df_cite)
    if df_valid.empty:
        return pd.DataFrame()

    trend_df = (
        df_valid.groupby("Year", as_index=False)
        .agg(
            Publications=("Year", "size"),
            Avg_Citations=("Times_Cited", "mean"),
            Total_Citations=("Times_Cited", "sum"),
        )
        .sort_values("Year")
    )
    return trend_df


def render_publication_citation_dual_axis_figure(trend_df):
    if trend_df is None or trend_df.empty:
        return None

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=trend_df["Year"],
            y=trend_df["Publications"],
            name="Publications",
            marker_color="#8796A5",
            opacity=0.8,
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend_df["Year"],
            y=trend_df["Avg_Citations"],
            name="Average Citations",
            mode="lines+markers",
            line=dict(color="#BFA7A8", width=1.8),
            marker=dict(size=5),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Annual Publications and Average Citations",
        xaxis_title="Year",
        yaxis=dict(title="Publications", rangemode="tozero"),
        yaxis2=dict(title="Average Citations", overlaying="y", side="right", rangemode="tozero"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="lightgray", showgrid=True)
    fig.update_yaxes(gridcolor="lightgray", showgrid=True)
    return fig
