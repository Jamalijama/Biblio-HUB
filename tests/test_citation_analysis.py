from collections import Counter

import pandas as pd

from modules.citation_analysis import (
    build_cocitation_network,
    build_journal_cocitation_network,
    build_author_cocitation_network,
    build_publication_citation_trend_frame,
    build_reference_burst_table,
    build_rpys_peak_table,
    clean_cited_reference,
    extract_cited_reference_statistics,
    extract_journal_from_citation,
    extract_author_from_citation,
    extract_rpys_statistics,
    render_publication_citation_dual_axis_figure,
    render_reference_burst_figure,
    render_rpys_figure,
)


def test_clean_cited_reference_extracts_author_year_and_filters_invalid_values():
    assert clean_cited_reference("SMITH J, 2020, SCIENCE") == "SMITH J, 2020"
    assert clean_cited_reference("[ANONYMOUS], 2020, REPORT") is None
    assert clean_cited_reference("12345") is None


def test_extract_cited_reference_statistics_returns_frequency_years_and_counts():
    df = pd.DataFrame(
        [
            {"Cited_References": "SMITH J, 2020, SCIENCE; LEE K, 2019, NATURE"},
            {"Cited_References": "SMITH J, 2020, SCIENCE"},
            {"Cited_References": ""},
        ]
    )

    ref_freq, ref_year_freq, ref_counts_per_paper = extract_cited_reference_statistics(df)

    assert ref_freq["SMITH J, 2020"] == 2
    assert ref_freq["LEE K, 2019"] == 1
    assert ref_year_freq[2020] == 2
    assert ref_year_freq[2019] == 1
    assert ref_counts_per_paper == [2, 1]


def test_build_cocitation_network_returns_graph_and_stats_for_valid_inputs():
    df = pd.DataFrame(
        [
            {"Cited_References": "SMITH J, 2020, SCIENCE; LEE K, 2019, NATURE; WANG H, 2018, CELL"},
            {"Cited_References": "SMITH J, 2020, SCIENCE; LEE K, 2019, NATURE"},
            {"Cited_References": "SMITH J, 2020, SCIENCE; WANG H, 2018, CELL"},
        ]
    )

    html_content, graph, node_groups, stats, ref_freq = build_cocitation_network(df, top_n_ref=5, min_cocite=1)

    assert html_content is not None
    assert graph.number_of_nodes() >= 2
    assert stats["nodes"] == graph.number_of_nodes()
    assert "SMITH J, 2020" in ref_freq
    assert node_groups["SMITH J, 2020"]["group"] >= 0


def test_extract_rpys_statistics_returns_year_counts_medians_and_reference_buckets():
    df = pd.DataFrame(
        [
            {"Cited_References": "SMITH J, 2018, SCIENCE; LEE K, 2019, NATURE"},
            {"Cited_References": "SMITH J, 2018, SCIENCE; WANG H, 2020, CELL"},
            {"Cited_References": "CHEN L, 2018, PNAS"},
        ]
    )

    rpys_df, refs_by_year = extract_rpys_statistics(df)

    year_rows = rpys_df.set_index("Year")
    assert list(rpys_df["Year"]) == [2018, 2019, 2020]
    assert year_rows.loc[2018, "Count"] == 3
    assert year_rows.loc[2019, "Count"] == 1
    assert year_rows.loc[2020, "Count"] == 1
    assert year_rows.loc[2018, "Median_5yr"] == 1
    assert year_rows.loc[2018, "Deviation"] == 2
    assert refs_by_year[2018]["SMITH J, 2018"] == 2
    assert refs_by_year[2018]["CHEN L, 2018"] == 1


def test_build_rpys_peak_table_returns_top_peak_years_with_references():
    rpys_df = pd.DataFrame(
        [
            {"Year": 2018, "Count": 5, "Median_5yr": 2, "Deviation": 3},
            {"Year": 2019, "Count": 2, "Median_5yr": 2, "Deviation": 0},
            {"Year": 2020, "Count": 6, "Median_5yr": 3, "Deviation": 3},
        ]
    )
    refs_by_year = {
        2018: Counter({"SMITH J, 2018": 3, "LEE K, 2018": 2}),
        2020: Counter({"WANG H, 2020": 4, "CHEN L, 2020": 1}),
    }

    peak_df = build_rpys_peak_table(rpys_df, refs_by_year, top_n=2, refs_per_year=2)

    assert list(peak_df["Year"]) == [2018, 2020]
    assert list(peak_df["Deviation from Median"]) == [3, 3]
    assert peak_df.iloc[0]["Top References"] == "SMITH J, 2018 (3); LEE K, 2018 (2)"
    assert peak_df.iloc[1]["Top References"] == "WANG H, 2020 (4); CHEN L, 2020 (1)"


def test_render_rpys_figure_returns_bar_and_median_line_with_peak_overlay():
    rpys_df = pd.DataFrame(
        [
            {"Year": 2018, "Count": 5, "Median_5yr": 3, "Deviation": 2},
            {"Year": 2019, "Count": 2, "Median_5yr": 3, "Deviation": -1},
            {"Year": 2020, "Count": 4, "Median_5yr": 3, "Deviation": 1},
        ]
    )

    fig = render_rpys_figure(rpys_df)

    assert fig is not None
    assert len(fig.data) == 3
    assert fig.data[0].type == "bar"
    assert fig.data[1].type == "scatter"
    assert fig.data[2].type == "bar"
    assert fig.layout.title.text == "Reference Publication Year Spectroscopy (RPYS)"


def test_build_reference_burst_table_returns_ranked_reference_events():
    df = pd.DataFrame(
        [
            {"Year": 2020, "Cited_References": "SMITH J, 2018, SCIENCE; LEE K, 2017, NATURE"},
            {"Year": 2021, "Cited_References": "SMITH J, 2018, SCIENCE"},
            {"Year": 2022, "Cited_References": "SMITH J, 2018, SCIENCE; WANG H, 2019, CELL"},
            {"Year": 2023, "Cited_References": "WANG H, 2019, CELL"},
        ]
    )

    burst_df = build_reference_burst_table(df, top_n=5)

    assert not burst_df.empty
    assert "Reference" in burst_df.columns
    assert burst_df.iloc[0]["Reference"] == "SMITH J, 2018"
    assert burst_df.iloc[0]["Max Freq"] >= 1


def test_render_reference_burst_figure_returns_burst_chart():
    reference_burst_df = pd.DataFrame(
        [
            {"Reference": "SMITH J, 2018", "Burst Strength": 2, "Burst Weight": 2.5, "Start": 2020, "End": 2022, "Max Freq": 3},
            {"Reference": "WANG H, 2019", "Burst Strength": 1, "Burst Weight": 1.6, "Start": 2022, "End": 2023, "Max Freq": 2},
        ]
    )

    fig = render_reference_burst_figure(reference_burst_df)

    assert fig is not None
    assert len(fig.data) == 2
    assert fig.data[0].type == "bar"
    assert fig.layout.title.text == "Reference Burst Detection (Kleinberg Algorithm)"


def test_publication_citation_dual_axis_returns_trend_frame_and_figure():
    df = pd.DataFrame(
        [
            {"Year": 2020, "Times_Cited": 10},
            {"Year": 2020, "Times_Cited": 6},
            {"Year": 2021, "Times_Cited": 12},
            {"Year": 2022, "Times_Cited": 3},
            {"Year": 2022, "Times_Cited": 9},
        ]
    )

    trend_df = build_publication_citation_trend_frame(df)
    fig = render_publication_citation_dual_axis_figure(trend_df)

    assert list(trend_df["Year"]) == [2020, 2021, 2022]
    assert list(trend_df["Publications"]) == [2, 1, 2]
    assert list(trend_df["Avg_Citations"]) == [8.0, 12.0, 6.0]
    assert fig is not None
    assert len(fig.data) == 2
    assert fig.data[0].type == "bar"
    assert fig.data[1].type == "scatter"
    assert fig.layout.title.text == "Annual Publications and Average Citations"


def test_extract_journal_from_citation_parses_wos_format_correctly():
    assert extract_journal_from_citation("SMITH J, 2020, SCIENCE, V368, P123") == "SCIENCE"
    assert extract_journal_from_citation("LEE K, ET AL., 2019, NATURE, V574, P456") == "NATURE"
    assert extract_journal_from_citation("WANG H, 2018, JOURNAL OF INFORMATICS, V10, P100") == "JOURNAL OF INFORMATICS"
    assert extract_journal_from_citation("CHEN L, 2021, IEEE TRANSACTIONS ON PATTERN ANALYSIS, V43, P789") == "IEEE TRANSACTIONS ON PATTERN ANALYSIS"
    assert extract_journal_from_citation("Invalid Reference") is None
    assert extract_journal_from_citation("12345") is None


def test_build_journal_cocitation_network_returns_valid_graph_and_stats():
    df = pd.DataFrame(
        [
            {"Cited_References": "SMITH J, 2020, SCIENCE; LEE K, 2019, NATURE; WANG H, 2018, CELL"},
            {"Cited_References": "SMITH J, 2020, SCIENCE; LEE K, 2019, NATURE"},
            {"Cited_References": "SMITH J, 2020, SCIENCE; WANG H, 2018, CELL"},
            {"Cited_References": "LEE K, 2019, NATURE; WANG H, 2018, CELL"},
        ]
    )

    html_content, graph, node_groups, stats, journal_freq = build_journal_cocitation_network(df, top_n_journals=10, min_cocite=1)

    assert html_content is not None
    assert graph.number_of_nodes() >= 2
    assert stats["nodes"] == graph.number_of_nodes()
    assert "Science" in journal_freq  
    assert "Nature" in journal_freq
    assert "Cell" in journal_freq
    assert node_groups["Science"]["group"] >= 0
    assert node_groups["Science"]["shape"] == "square"


def test_extract_author_from_citation_parses_wos_format_correctly():
    assert extract_author_from_citation("SMITH J, 2020, SCIENCE") == "SMITH J"
    assert extract_author_from_citation("LEE K, ET AL., 2019, NATURE") == "LEE K"
    assert extract_author_from_citation("12345") is None
    assert extract_author_from_citation("[ANONYMOUS], 2020, REPORT") is None


def test_build_author_cocitation_network_returns_valid_graph_and_stats():
    df = pd.DataFrame(
        [
            {"Cited_References": "SMITH J, 2020, SCIENCE; LEE K, 2019, NATURE; WANG H, 2018, CELL"},
            {"Cited_References": "SMITH J, 2020, SCIENCE; LEE K, 2019, NATURE"},
            {"Cited_References": "SMITH J, 2020, SCIENCE; WANG H, 2018, CELL"},
            {"Cited_References": "LEE K, 2019, NATURE; WANG H, 2018, CELL"},
        ]
    )

    html_content, graph, node_groups, stats, auth_freq = build_author_cocitation_network(df, top_n_authors=10, min_cocite=1)

    assert html_content is not None
    assert graph.number_of_nodes() >= 2
    assert stats["nodes"] == graph.number_of_nodes()
    assert "Smith J" in auth_freq  
    assert "Lee K" in auth_freq
    assert "Wang H" in auth_freq
    assert node_groups["Smith J"]["group"] >= 0
    assert node_groups["Smith J"]["shape"] == "dot"
