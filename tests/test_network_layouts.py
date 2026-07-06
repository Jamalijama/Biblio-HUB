import networkx as nx
import pandas as pd

from modules.network_builders import (
    build_cooccurrence_network,
    build_journal_network,
    build_keyword_journal_cooccurrence,
)
from modules.network_layouts import (
    compute_bipartite_layout,
    compute_cluster_layout,
    compute_force_layout,
)
from modules.network_visualization import (
    node_groups_from_cluster_stats,
    render_keyword_journal_network,
    render_network_publication_figure,
    render_vosviewer_style,
)
from modules.temporal_analysis import (
    build_burst_table,
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
)


def test_compute_force_layout_returns_positions_for_all_nodes():
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=2)
    graph.add_edge("B", "C", weight=1)

    pos = compute_force_layout(graph)

    assert set(pos) == {"A", "B", "C"}
    for coords in pos.values():
        assert len(coords) == 2


def test_compute_cluster_layout_separates_clusters():
    graph = nx.Graph()
    graph.add_edge("A1", "A2", weight=2)
    graph.add_edge("B1", "B2", weight=2)
    graph.add_edge("A1", "B1", weight=1)
    groups = {"A1": 0, "A2": 0, "B1": 1, "B2": 1}

    pos = compute_cluster_layout(graph, groups)

    assert set(pos) == set(groups)
    left_cluster_center = (pos["A1"][0] + pos["A2"][0]) / 2
    right_cluster_center = (pos["B1"][0] + pos["B2"][0]) / 2
    assert left_cluster_center != right_cluster_center


def test_compute_bipartite_layout_places_nodes_on_opposite_sides():
    graph = nx.Graph()
    graph.add_edge("kw1", "journal1", weight=3)
    graph.add_edge("kw2", "journal1", weight=2)
    graph.add_edge("kw2", "journal2", weight=1)

    pos = compute_bipartite_layout(graph, ["kw1", "kw2"], ["journal1", "journal2"])

    assert pos["kw1"][0] < 0
    assert pos["kw2"][0] < 0
    assert pos["journal1"][0] > 0
    assert pos["journal2"][0] > 0


def test_build_cooccurrence_network_counts_keywords_and_pairs():
    keyword_freq, cooccurrence = build_cooccurrence_network(
        [["Network", "Citation", "Network"], ["Network", "Data"], ["Citation", "Data"]],
        min_cooccurrence=1,
    )

    assert keyword_freq["Network"] == 2
    assert keyword_freq["Citation"] == 2
    assert cooccurrence[("Citation", "Network")] == 1
    assert cooccurrence[("Data", "Network")] == 1
    assert cooccurrence[("Citation", "Data")] == 1


def test_build_journal_network_groups_counts_by_year():
    df = pd.DataFrame(
        [
            {"Journal": "Journal A", "Year": "2020"},
            {"Journal": "Journal A", "Year": "2020.0"},
            {"Journal": "Journal B", "Year": "2021"},
        ]
    )

    journal_year = build_journal_network(df)

    assert journal_year["Journal A"][2020] == 2
    assert journal_year["Journal B"][2021] == 1


def test_build_keyword_journal_cooccurrence_filters_to_top_items():
    df = pd.DataFrame(
        [
            {"Journal": "Journal A"},
            {"Journal": "Journal A"},
            {"Journal": "Journal B"},
        ]
    )
    keywords_list = [
        ["Network", "Mapping"],
        ["Network", "Data"],
        ["Data", "Visualization"],
    ]

    top_keywords, top_journals, filtered_cooccur, keyword_freq_local, journal_freq = (
        build_keyword_journal_cooccurrence(df, keywords_list, top_n_keywords=2, top_n_journals=1)
    )

    assert top_journals == ["Journal A"]
    assert set(top_keywords) == {"Network", "Data"}
    assert ("Network", "Journal A") in filtered_cooccur
    assert ("Visualization", "Journal B") not in filtered_cooccur
    assert keyword_freq_local["Network"] == 2
    assert journal_freq["Journal A"] == 2


def test_render_vosviewer_style_returns_html_graph_and_cluster_stats():
    keyword_freq = {"Network": 5, "Mapping": 4, "Data": 3}
    cooccurrence = {("Network", "Citation"): 3, ("Network", "Data"): 2, ("Data", "Citation"): 1}

    html_content, graph, stats = render_vosviewer_style(keyword_freq, cooccurrence, top_n=3, min_weight=1)

    assert html_content is not None
    assert graph.number_of_nodes() == 3
    assert "clusters" in stats
    assert "cluster_colors" in stats


def test_render_keyword_journal_network_returns_expected_stats():
    html_content, graph, node_groups, stats = render_keyword_journal_network(
        ["Network", "Data"],
        ["Journal A"],
        {("Network", "Journal A"): 2, ("Data", "Journal A"): 1},
        {"Network": 3, "Data": 2},
        {"Journal A": 4},
    )

    assert html_content is not None
    assert graph.number_of_nodes() == 3
    assert node_groups["Journal A"]["shape"] == "square"
    assert stats["keyword_nodes"] == 2
    assert stats["journal_nodes"] == 1


def test_render_network_publication_figure_returns_plotly_figure():
    graph = nx.Graph()
    graph.add_node("Network", weight=4)
    graph.add_node("Citation", weight=3)
    graph.add_edge("Network", "Citation", weight=2)

    fig = render_network_publication_figure(graph, title="Test Network")

    assert fig is not None
    assert len(fig.data) >= 2


def test_render_network_publication_figure_respects_manual_display_params():
    graph = nx.Graph()
    for index in range(81):
        node = f"N{index:02d}"
        graph.add_node(node, weight=81 - index)
        if index > 0:
            graph.add_edge(f"N{index - 1:02d}", node, weight=1)

    fig = render_network_publication_figure(
        graph,
        title="Manual Settings",
        size_range=(20, 140),
        max_visible_labels=40,
        label_font_size=22,
    )

    node_trace = fig.data[-1]
    marker_trace = next(trace for trace in fig.data if getattr(trace, "mode", "") == "markers")
    visible_label_count = sum(1 for text in node_trace.text if text)

    assert visible_label_count == 40
    assert node_trace.textfont.size == 22
    assert max(marker_trace.marker.size) > 52


def test_node_groups_from_cluster_stats_returns_expected_mapping():
    stats = {
        "cluster_report": {
            0: {"color": "#123456", "members": ["Network", "Citation"], "shape": "dot"},
            1: {"color": "#654321", "members": ["Data"], "shape": "square"},
        }
    }

    node_groups = node_groups_from_cluster_stats(stats, type_label_prefix="Cluster")

    assert node_groups["Network"]["color"] == "#123456"
    assert node_groups["Network"]["group"] == 0
    assert node_groups["Data"]["shape"] == "square"
    assert node_groups["Data"]["type_label"] == "Cluster 2: "


def test_temporal_analysis_renderers_return_figures_for_valid_inputs():
    df = pd.DataFrame(
        [
            {"Year": 2019},
            {"Year": 2020},
            {"Year": 2021},
            {"Year": 2022},
            {"Year": 2023},
            {"Year": 2023},
            {"Year": 2024},
            {"Year": 2024},
        ]
    )
    keywords_list = [
        ["Baseline"],
        ["Baseline"],
        ["Baseline"],
        ["Network"],
        ["Network"],
        ["Network"],
        ["Network", "Data"],
        ["Network", "Data"],
    ]
    keyword_freq = {"Network": 5, "Baseline": 3, "Data": 2}

    timeline_fig = render_citespace_timeline(df, keywords_list, keyword_freq, top_n=3)
    burst_fig = render_burst_detection(df, keywords_list, keyword_freq, top_n=3)

    assert timeline_fig is not None
    assert len(timeline_fig.data) >= 1
    assert burst_fig is not None
    assert len(burst_fig.data) >= 1


def test_render_alluvial_topic_flow_returns_sankey_for_temporal_keyword_communities():
    df = pd.DataFrame(
        [
            {"Year": 2020},
            {"Year": 2020},
            {"Year": 2021},
            {"Year": 2021},
            {"Year": 2022},
            {"Year": 2022},
        ]
    )
    keywords_list = [
        ["Network", "Citation", "Clusters"],
        ["Network", "Visualization"],
        ["Network", "Citation", "Data"],
        ["Data", "Mining"],
        ["Data", "Mining", "Graph"],
        ["Graph", "Network"],
    ]
    keyword_freq = {"Network": 4, "Data": 3, "Citation": 2, "Graph": 2, "Mining": 2, "Visualization": 1, "Clusters": 1}

    fig = render_alluvial_topic_flow(df, keywords_list, keyword_freq, slice_count=3, top_n_keywords=6, max_topics_per_slice=3)

    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].type == "sankey"


def test_build_keyword_burst_table_returns_ranked_keyword_events():
    rows = []
    keywords_list = []
    for year in range(2015, 2025):
        rows.append({"Year": year})
        keywords_list.append(["Graph"])
        if year >= 2021:
            rows.append({"Year": year})
            keywords_list.append(["Network"])
            rows.append({"Year": year})
            keywords_list.append(["Network"])
    df = pd.DataFrame(rows)
    keyword_freq = {"Network": 8, "Graph": 10}

    burst_df = build_keyword_burst_table(df, keywords_list, keyword_freq, top_n=4)

    assert not burst_df.empty
    assert "Keyword" in burst_df.columns
    assert "Adjusted Burst Score" in burst_df.columns
    assert "Duration" in burst_df.columns
    assert burst_df.iloc[0]["Keyword"] == "Network"


def test_build_keyword_burst_table_respects_year_window():
    rows = []
    keywords_list = []
    for year in range(2018, 2026):
        rows.append({"Year": year})
        keywords_list.append(["Legacy"])
        if year >= 2024:
            rows.append({"Year": year})
            keywords_list.append(["Current"])
            rows.append({"Year": year})
            keywords_list.append(["Current"])
    df = pd.DataFrame(rows)
    keyword_freq = {"Current": 8, "Legacy": 8}

    burst_df = build_keyword_burst_table(
        df,
        keywords_list,
        keyword_freq,
        top_n=2,
        start_year=2020,
        end_year=2025,
    )

    assert not burst_df.empty
    assert set(burst_df["Keyword"]) == {"Current"}
    assert burst_df["Start"].min() >= 2020
    assert burst_df["End"].max() <= 2025


def test_build_keyword_burst_table_prioritizes_true_bursts_over_global_frequency():
    rows = []
    keywords_list = []
    for year in range(2010, 2021):
        rows.append({"Year": year})
        keywords_list.append(["Stable"])
        if year in {2017, 2018, 2019}:
            rows.append({"Year": year})
            keywords_list.append(["Outbreak"])
            rows.append({"Year": year})
            keywords_list.append(["Outbreak"])
    df = pd.DataFrame(rows)
    keyword_freq = {"Stable": 11, "Outbreak": 6}

    burst_df = build_keyword_burst_table(df, keywords_list, keyword_freq, top_n=2)

    assert not burst_df.empty
    assert burst_df.iloc[0]["Keyword"] == "Outbreak"


def test_build_burst_table_uses_duration_penalty_for_adjusted_sorting():
    year_range = [2018, 2019, 2020, 2021, 2022, 2023]
    label_sequences = {
        "Short Burst": {2018: 0, 2019: 0, 2020: 0, 2021: 4, 2022: 0, 2023: 0},
        "Long Burst": {2018: 0, 2019: 1, 2020: 1, 2021: 1, 2022: 1, 2023: 0},
    }

    burst_df = build_burst_table(label_sequences, year_range, label_column="Keyword", top_n=4, include_fallback=True)

    assert not burst_df.empty
    assert "Adjusted Burst Score" in burst_df.columns
    assert "Duration" in burst_df.columns
    assert burst_df.iloc[0]["Keyword"] == "Short Burst"
    assert burst_df.iloc[0]["Adjusted Burst Score"] >= burst_df.iloc[1]["Adjusted Burst Score"]


def test_build_keyword_burst_table_collapses_markup_variants_into_one_keyword():
    rows = []
    keywords_list = []
    for year in range(2018, 2025):
        rows.append({"Year": year})
        keywords_list.append(["Baseline"])
        if year >= 2023:
            rows.append({"Year": year})
            keywords_list.append(["Aedes Aegypti", "<italic>aedes Aegypti< Italic>"])
            rows.append({"Year": year})
            keywords_list.append(["<italic>aedes Aegypti< Italic>"])
    df = pd.DataFrame(rows)
    keyword_freq = {"Baseline": 7, "Aedes Aegypti": 4}

    burst_df = build_keyword_burst_table(df, keywords_list, keyword_freq, top_n=3)

    assert not burst_df.empty
    assert "Aedes Aegypti" in burst_df["Keyword"].tolist()
    assert not any("<italic>" in keyword.lower() for keyword in burst_df["Keyword"].tolist())


def test_build_keyword_burst_table_suppresses_generic_and_corpus_anchor_terms():
    rows = []
    keywords_list = []
    for year in range(2010, 2025):
        rows.append({"Year": year})
        keywords_list.append(["DiseaseX", "Infection"])
        if year >= 2018:
            rows.append({"Year": year})
            keywords_list.append(["Outbreak"])
            rows.append({"Year": year})
            keywords_list.append(["Outbreak"])
    df = pd.DataFrame(rows)
    keyword_freq = {"DiseaseX": 15, "Infection": 15, "Outbreak": 14}

    burst_df = build_keyword_burst_table(df, keywords_list, keyword_freq, top_n=3)

    assert not burst_df.empty
    assert "Outbreak" in burst_df["Keyword"].tolist()
    assert "Infection" not in burst_df["Keyword"].tolist()
    assert "DiseaseX" not in burst_df["Keyword"].tolist()


def test_selected_keyword_share_table_normalizes_selected_terms_to_100_percent():
    df = pd.DataFrame(
        [
            {"Year": 2020},
            {"Year": 2020},
            {"Year": 2021},
            {"Year": 2021},
        ]
    )
    keywords_list = [
        ["Network", "Graph"],
        ["Network"],
        ["Graph"],
        ["Network", "Graph"],
    ]

    selected_keywords = parse_selected_keywords("network, graph, network")
    share_df = build_selected_keyword_share_table(df, keywords_list, selected_keywords)
    fig = render_selected_keyword_share_figure(share_df, selected_keywords)

    assert selected_keywords == ["Network", "Graph"]
    assert not share_df.empty
    assert set(share_df["Keyword"]) == {"Network", "Graph"}
    assert share_df.groupby("Year")["Share (%)"].sum().round(6).tolist() == [100.0, 100.0]
    assert fig is not None


def test_keyword_growth_and_forecast_renderers_return_expected_outputs():
    df = pd.DataFrame(
        [
            {"Year": 2020},
            {"Year": 2020},
            {"Year": 2021},
            {"Year": 2021},
            {"Year": 2021},
            {"Year": 2022},
            {"Year": 2022},
            {"Year": 2022},
            {"Year": 2022},
            {"Year": 2023},
            {"Year": 2023},
            {"Year": 2023},
            {"Year": 2023},
            {"Year": 2023},
        ]
    )
    keywords_list = [
        ["Network"],
        ["Data"],
        ["Network"],
        ["Network", "Data"],
        ["Data"],
        ["Network"],
        ["Network"],
        ["Network", "Graph"],
        ["Data"],
        ["Network"],
        ["Network"],
        ["Network", "Graph"],
        ["Network", "Graph"],
        ["Graph"],
    ]
    keyword_freq = {"Network": 9, "Data": 4, "Graph": 4}

    growth_df, yearly_df = build_keyword_growth_table(df, keywords_list, keyword_freq, candidate_top_n=3)
    growth_rank_fig = render_keyword_growth_leader_figure(growth_df, top_n=3)
    growth_line_fig = render_keyword_growth_trend_figure(yearly_df, growth_df, top_n=3)
    forecast_df, metadata = build_publication_forecast_frame(df, forecast_horizon=3)
    forecast_fig = render_publication_forecast_figure(forecast_df)

    assert not growth_df.empty
    assert "Growth Rate (%)" in growth_df.columns
    assert growth_rank_fig is not None
    assert growth_line_fig is not None
    assert not forecast_df.empty
    assert set(forecast_df["Series"]) == {"Actual", "Forecast"}
    assert metadata["forecast_horizon"] == 3
    assert forecast_fig is not None
    assert summarize_publication_forecast(forecast_df, metadata)


def test_keyword_opportunity_map_renderer_returns_figure():
    df = pd.DataFrame(
        [
            {"Year": 2020},
            {"Year": 2020},
            {"Year": 2021},
            {"Year": 2021},
            {"Year": 2022},
            {"Year": 2022},
            {"Year": 2023},
            {"Year": 2023},
            {"Year": 2024},
            {"Year": 2024},
        ]
    )
    keywords_list = [
        ["Network"],
        ["Data"],
        ["Network"],
        ["Data"],
        ["Network", "Graph"],
        ["Graph"],
        ["Network", "Graph"],
        ["Graph"],
        ["Network", "Graph"],
        ["Network", "Graph"],
    ]
    keyword_freq = {"Network": 5, "Graph": 5, "Data": 2}

    opportunity_df = build_keyword_opportunity_map_frame(df, keywords_list, keyword_freq, top_n_keywords=3)
    fig = render_keyword_opportunity_map(opportunity_df, top_n=3)

    assert not opportunity_df.empty
    assert "Signal Type" in opportunity_df.columns
    assert fig is not None
    assert summarize_keyword_opportunity_map(opportunity_df, top_n=3)


def test_entity_forecast_tables_and_renderers_return_outputs():
    df = pd.DataFrame(
        [
            {"Year": 2020, "Journal": "Journal A", "Affiliations": "Univ A, China"},
            {"Year": 2020, "Journal": "Journal B", "Affiliations": "Inst X, USA"},
            {"Year": 2021, "Journal": "Journal A", "Affiliations": "Univ A, China"},
            {"Year": 2021, "Journal": "Journal A", "Affiliations": "Univ B, China"},
            {"Year": 2022, "Journal": "Journal A", "Affiliations": "Univ A, China"},
            {"Year": 2022, "Journal": "Journal B", "Affiliations": "Inst X, USA"},
            {"Year": 2023, "Journal": "Journal A", "Affiliations": "Univ A, China"},
            {"Year": 2023, "Journal": "Journal A", "Affiliations": "Univ B, China"},
            {"Year": 2023, "Journal": "Journal C", "Affiliations": "Inst Y, UK"},
        ]
    )

    summary_df, series_df, entity_label = build_entity_forecast_tables(
        df,
        entity_type="journal",
        top_n_entities=3,
        forecast_horizon=2,
        min_total_occurrences=2,
    )
    rank_fig = render_entity_forecast_rank_figure(summary_df, entity_label, top_n=3)
    trend_fig = render_entity_forecast_trajectory_figure(series_df, entity_label, top_n=3)

    assert entity_label == "Journal"
    assert not summary_df.empty
    assert "Projected Growth (%)" in summary_df.columns
    assert not series_df.empty
    assert rank_fig is not None
    assert trend_fig is not None
    assert summarize_entity_forecast_signals(summary_df, entity_label)


def test_entity_leadership_shift_tables_and_renderers_return_outputs():
    df = pd.DataFrame(
        [
            {"Year": 2020, "Affiliations": "Dept A, Univ A, Beijing, China"},
            {"Year": 2020, "Affiliations": "Dept B, Inst X, Boston, USA"},
            {"Year": 2021, "Affiliations": "Dept A, Univ A, Beijing, China"},
            {"Year": 2021, "Affiliations": "Dept C, Univ B, Shanghai, China"},
            {"Year": 2022, "Affiliations": "Dept B, Inst X, Boston, USA"},
            {"Year": 2022, "Affiliations": "Dept C, Univ B, Shanghai, China"},
            {"Year": 2023, "Affiliations": "Dept C, Univ B, Shanghai, China"},
            {"Year": 2023, "Affiliations": "Dept D, Inst Y, London, United Kingdom"},
            {"Year": 2024, "Affiliations": "Dept C, Univ B, Shanghai, China"},
            {"Year": 2024, "Affiliations": "Dept D, Inst Y, London, United Kingdom"},
        ]
    )

    summary_df, series_df, entity_label = build_entity_leadership_shift_tables(
        df,
        entity_type="country",
        top_n_entities=3,
        recent_year_window=4,
        min_total_occurrences=2,
    )
    shift_fig = render_entity_leadership_shift_figure(summary_df, entity_label, top_n=3)
    traj_fig = render_entity_leadership_trajectory_figure(series_df, summary_df, entity_label, top_n=3)

    assert entity_label == "Country"
    assert not summary_df.empty
    assert "Share Shift (pp)" in summary_df.columns
    assert not series_df.empty
    assert shift_fig is not None
    assert traj_fig is not None
    assert summarize_entity_leadership_shift(summary_df, entity_label)


def test_theme_migration_forecast_returns_summary_and_figures():
    df = pd.DataFrame(
        [
            {"Year": 2020},
            {"Year": 2020},
            {"Year": 2021},
            {"Year": 2021},
            {"Year": 2022},
            {"Year": 2022},
            {"Year": 2023},
            {"Year": 2023},
            {"Year": 2024},
            {"Year": 2024},
        ]
    )
    keywords_list = [
        ["Network", "Visualization", "Clusters"],
        ["Network", "Visualization"],
        ["Network", "Visualization", "Themes"],
        ["Network", "Themes"],
        ["Network", "Themes", "Citation"],
        ["Themes", "Citation"],
        ["Citation", "Collaboration", "Network"],
        ["Citation", "Collaboration"],
        ["Collaboration", "Trend Mapping", "Citation"],
        ["Collaboration", "Trend Mapping"],
    ]
    keyword_freq = {
        "Network": 6,
        "Visualization": 3,
        "Clusters": 1,
        "Themes": 4,
        "Citation": 4,
        "Collaboration": 4,
        "Trend Mapping": 2,
    }

    summary_df, chain_df = build_theme_migration_forecast_tables(
        df,
        keywords_list,
        keyword_freq,
        slice_count=4,
        top_n_keywords=7,
        max_topics_per_slice=4,
    )
    traj_fig = render_theme_migration_trajectory_figure(chain_df, summary_df, top_n=4)
    map_fig = render_theme_migration_opportunity_map(summary_df, top_n=4)

    assert not summary_df.empty
    assert not chain_df.empty
    assert "Signal Type" in summary_df.columns
    assert traj_fig is not None
    assert map_fig is not None


def test_forward_signals_overview_summary_combines_sections():
    summary_text = summarize_forward_signals_overview(
        publication_summary="Publication trend remains upward.",
        keyword_summary="Emerging signals concentrate around hydrogel and exosomes.",
        entity_summary="Journal A and China keep accelerating.",
        leadership_summary="Leadership is shifting toward China and Univ B.",
        theme_summary="Regenerative biomaterials are shifting toward core hotspots.",
    )

    assert "Publication Trend" in summary_text
    assert "Keyword Hotspots" in summary_text
    assert "Entity Growth" in summary_text
    assert "Leadership Shift" in summary_text
    assert "Theme Migration" in summary_text
