import io
import json
import zipfile

import networkx as nx
import pandas as pd
from collections import Counter

import modules.figure_export_bundle as figure_export_bundle
from modules.figure_export_bundle import generate_all_figure_bundle


def test_generate_all_figure_bundle_includes_manifest_and_network_html():
    df = pd.DataFrame(
        [
            {"Year": 2020, "Journal": "Journal A", "Authors": "Alice;Bob"},
            {"Year": 2021, "Journal": "Journal A", "Authors": "Alice"},
            {"Year": 2022, "Journal": "Journal B", "Authors": "Bob"},
        ]
    )
    keywords_list = [
        ["Network", "Citation"],
        ["Network", "Data"],
        ["Network", "Citation"],
    ]
    keyword_freq = {"Network": 3, "Citation": 2, "Data": 1}
    cooccurrence = {("Network", "Citation"): 2, ("Network", "Data"): 1}

    bundle_bytes = generate_all_figure_bundle(
        df,
        keywords_list,
        keyword_freq,
        cooccurrence,
        export_format="png",
        selected_items={"network_keyword_cooccurrence_static"},
    )

    with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as zip_file:
        names = set(zip_file.namelist())

    assert "00_manifest.txt" in names
    assert "06_network_html/keyword_cooccurrence_network.html" in names
    assert "98_export_status.json" in names


def test_generate_all_figure_bundle_records_static_export_failures(monkeypatch):
    df = pd.DataFrame([{"Year": 2020, "Journal": "Journal A"}])
    keywords_list = [["Network", "Citation"]]
    keyword_freq = {"Network": 1, "Citation": 1}
    cooccurrence = {("Network", "Citation"): 1}

    monkeypatch.setattr(
        figure_export_bundle,
        "plotly_figure_to_bytes",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("kaleido missing")),
    )

    bundle_bytes = generate_all_figure_bundle(
        df,
        keywords_list,
        keyword_freq,
        cooccurrence,
        export_format="png",
        selected_items={"overview_publications_by_year"},
    )

    with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as zip_file:
        names = set(zip_file.namelist())
        status = json.loads(zip_file.read("98_export_status.json").decode("utf-8"))
        export_log = zip_file.read("99_export_log.txt").decode("utf-8")

    assert "98_export_status.json" in names
    assert "99_export_log.txt" in names
    assert status["plotly_export_attempts"] == 1
    assert status["plotly_export_successes"] == 0
    assert status["skipped_item_count"] >= 1
    assert "publications_by_year.png" in export_log


def test_generate_all_figure_bundle_includes_country_impact_quadrant_export(monkeypatch):
    df = pd.DataFrame(
        [
            {"Affiliations": "University of Test, USA", "Times_Cited": 10},
            {"Affiliations": "Institute of Metrics, China", "Times_Cited": 8},
            {"Affiliations": "Lab of Signals, UK", "Times_Cited": 4},
            {"Affiliations": "Analytics Center, Germany", "Times_Cited": 7},
            {"Affiliations": "Innovation Hub, India", "Times_Cited": 5},
            {"Affiliations": "Global Research Lab, USA; Center of Analytics, China", "Times_Cited": 6},
        ]
    )

    graph = nx.Graph()
    graph.add_edge("United States", "China", weight=2)
    country_df = pd.DataFrame(
        [
            {"Country": "United States", "Papers": 2},
            {"Country": "China", "Papers": 2},
        ]
    )

    monkeypatch.setattr(
        figure_export_bundle,
        "_build_country_collaboration_assets",
        lambda *args, **kwargs: (graph, None, {}, None, None, country_df),
    )
    monkeypatch.setattr(
        figure_export_bundle,
        "plotly_figure_to_bytes",
        lambda *args, **kwargs: b"fake-image-bytes",
    )

    bundle_bytes = generate_all_figure_bundle(
        df,
        keywords_list=[],
        keyword_freq={},
        cooccurrence={},
        export_format="png",
        selected_items={"network_country_publications_bar", "network_country_impact_quadrant"},
    )

    with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as zip_file:
        names = set(zip_file.namelist())
        status = json.loads(zip_file.read("98_export_status.json").decode("utf-8"))
        export_log = zip_file.read("99_export_log.txt").decode("utf-8") if "99_export_log.txt" in names else ""

    assert "tuple' object has no attribute 'empty'" not in export_log
    assert status["skipped_item_count"] == 0
    assert "02_network/country_publications_bar.png" in names


def test_generate_all_figure_bundle_uses_network_analysis_parameters_for_cocitation_and_keywords(monkeypatch):
    df = pd.DataFrame(
        [
            {"Year": 2020, "Journal": "Journal A", "Cited_References": "REF A; REF B"},
            {"Year": 2021, "Journal": "Journal B", "Cited_References": "REF A; REF C"},
        ]
    )
    keywords_list = [["Network", "Citation"], ["Network", "Graph"]]
    keyword_freq = {"Network": 2, "Citation": 1, "Graph": 1}
    cooccurrence = {("Network", "Citation"): 1, ("Network", "Graph"): 1}
    captured = {}

    def fake_render_vosviewer_style(keyword_freq_arg, cooccurrence_arg, top_n, min_weight):
        captured["keyword"] = {"top_n": top_n, "min_weight": min_weight}
        graph = nx.Graph()
        graph.add_edge("Network", "Citation", weight=1)
        return "<html></html>", graph, {}

    def fake_build_cocitation_network(df_arg, top_n_ref, min_cocite):
        captured["cocitation"] = {"top_n_ref": top_n_ref, "min_cocite": min_cocite}
        graph = nx.Graph()
        graph.add_edge("REF A", "REF B", weight=2)
        return "<html></html>", graph, {}, None, Counter({"REF A": 2, "REF B": 1})

    monkeypatch.setattr(figure_export_bundle, "render_vosviewer_style", fake_render_vosviewer_style)
    monkeypatch.setattr(figure_export_bundle, "build_cocitation_network", fake_build_cocitation_network)

    generate_all_figure_bundle(
        df,
        keywords_list,
        keyword_freq,
        cooccurrence,
        export_format="png",
        selected_items={"network_keyword_cooccurrence_static", "network_reference_cocitation_static"},
        analysis_parameters=[
            {"key": "vos_topn", "value": 40},
            {"key": "vos_minw", "value": 5},
            {"key": "cocite_topn", "value": 60},
            {"key": "cocite_minw", "value": 7},
        ],
    )

    assert captured["keyword"] == {"top_n": 3, "min_weight": 5}
    assert captured["cocitation"] == {"top_n_ref": 60, "min_cocite": 7}


def test_generate_all_figure_bundle_includes_forward_signal_exports():
    df = pd.DataFrame(
        [
            {"Year": 2020, "Journal": "Journal A", "Affiliations": "University of Test, USA"},
            {"Year": 2020, "Journal": "Journal B", "Affiliations": "Institute of Metrics, China"},
            {"Year": 2021, "Journal": "Journal A", "Affiliations": "University of Test, USA"},
            {"Year": 2021, "Journal": "Journal A", "Affiliations": "Institute of Metrics, China"},
            {"Year": 2022, "Journal": "Journal A", "Affiliations": "University of Test, USA"},
            {"Year": 2022, "Journal": "Journal B", "Affiliations": "Institute of Metrics, China"},
            {"Year": 2023, "Journal": "Journal A", "Affiliations": "University of Test, USA"},
            {"Year": 2023, "Journal": "Journal C", "Affiliations": "Lab of Trends, UK"},
        ]
    )
    keywords_list = [
        ["Network", "Visualization"],
        ["Data", "Mining"],
        ["Network", "Themes"],
        ["Network", "Themes"],
        ["Themes", "Citation"],
        ["Graph", "Mining"],
        ["Citation", "Collaboration"],
        ["Collaboration", "Trend Mapping"],
    ]
    keyword_freq = {"Network": 3, "Themes": 3, "Citation": 2, "Collaboration": 2, "Visualization": 1, "Data": 1, "Mining": 2, "Graph": 1, "Trend Mapping": 1}
    cooccurrence = {("Network", "Themes"): 2, ("Themes", "Citation"): 1, ("Citation", "Collaboration"): 1}

    bundle_bytes = generate_all_figure_bundle(
        df,
        keywords_list,
        keyword_freq,
        cooccurrence,
        export_format="png",
        selected_items={"temporal_journal_growth_forecast", "temporal_theme_migration_forecast"},
    )

    with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as zip_file:
        names = set(zip_file.namelist())
        interpretation = zip_file.read("07_analysis_tables/theme_migration_interpretation.txt").decode("utf-8")

    assert "03_temporal/journal_growth_forecast_rank.png" in names
    assert "03_temporal/journal_growth_forecast_trajectories.png" in names
    assert "03_temporal/theme_migration_trajectories.png" in names
    assert "03_temporal/theme_migration_hotspot_map.png" in names
    assert "07_analysis_tables/journal_growth_forecast.csv" in names
    assert "07_analysis_tables/theme_migration_forecast.csv" in names
    assert "07_analysis_tables/theme_migration_interpretation.txt" in names
    assert interpretation.strip()


def test_generate_all_figure_bundle_includes_publication_and_keyword_forward_exports():
    df = pd.DataFrame(
        [
            {"Year": 2020, "Journal": "Journal A"},
            {"Year": 2021, "Journal": "Journal A"},
            {"Year": 2021, "Journal": "Journal B"},
            {"Year": 2022, "Journal": "Journal A"},
            {"Year": 2022, "Journal": "Journal B"},
            {"Year": 2023, "Journal": "Journal A"},
            {"Year": 2023, "Journal": "Journal C"},
        ]
    )
    keywords_list = [
        ["Network", "Visualization"],
        ["Network", "Visualization"],
        ["Data", "Mining"],
        ["Network", "Themes"],
        ["Themes", "Citation"],
        ["Citation", "Collaboration"],
        ["Collaboration", "Trend Mapping"],
    ]
    keyword_freq = {"Network": 3, "Visualization": 2, "Themes": 2, "Citation": 2, "Collaboration": 2, "Data": 1, "Mining": 1, "Trend Mapping": 1}
    cooccurrence = {("Network", "Visualization"): 2, ("Network", "Themes"): 1, ("Themes", "Citation"): 1, ("Citation", "Collaboration"): 1}

    bundle_bytes = generate_all_figure_bundle(
        df,
        keywords_list,
        keyword_freq,
        cooccurrence,
        export_format="png",
        selected_items={"temporal_publication_forecast", "temporal_keyword_opportunity_map"},
        analysis_parameters=[
            {"key": "publication_forecast_horizon", "value": 2},
            {"key": "keyword_opportunity_topn", "value": 3},
        ],
    )

    with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as zip_file:
        names = set(zip_file.namelist())
        publication_text = zip_file.read("07_analysis_tables/publication_forecast_interpretation.txt").decode("utf-8")
        keyword_text = zip_file.read("07_analysis_tables/keyword_opportunity_interpretation.txt").decode("utf-8")
        publication_df = pd.read_csv(io.BytesIO(zip_file.read("07_analysis_tables/publication_forecast.csv")))
        keyword_df = pd.read_csv(io.BytesIO(zip_file.read("07_analysis_tables/keyword_opportunity_map.csv")))

    assert "03_temporal/publication_forecast.png" in names
    assert "03_temporal/keyword_opportunity_map.png" in names
    assert "07_analysis_tables/publication_forecast.csv" in names
    assert "07_analysis_tables/publication_forecast_interpretation.txt" in names
    assert "07_analysis_tables/keyword_opportunity_map.csv" in names
    assert "07_analysis_tables/keyword_opportunity_interpretation.txt" in names
    assert publication_text.strip()
    assert keyword_text.strip()
    assert int(publication_df["Year"].max()) == 2025
    assert len(keyword_df) <= 3


def test_generate_all_figure_bundle_uses_forward_signal_analysis_parameters():
    df = pd.DataFrame(
        [
            {"Year": 2020, "Journal": "Journal A", "Affiliations": "University of Test, USA"},
            {"Year": 2020, "Journal": "Journal B", "Affiliations": "Institute of Metrics, China"},
            {"Year": 2021, "Journal": "Journal A", "Affiliations": "University of Test, USA"},
            {"Year": 2021, "Journal": "Journal A", "Affiliations": "Institute of Metrics, China"},
            {"Year": 2022, "Journal": "Journal A", "Affiliations": "University of Test, USA"},
            {"Year": 2022, "Journal": "Journal B", "Affiliations": "Institute of Metrics, China"},
            {"Year": 2023, "Journal": "Journal A", "Affiliations": "University of Test, USA"},
            {"Year": 2023, "Journal": "Journal C", "Affiliations": "Lab of Trends, UK"},
        ]
    )
    keywords_list = [
        ["Network", "Visualization"],
        ["Data", "Mining"],
        ["Network", "Themes"],
        ["Network", "Themes"],
        ["Themes", "Citation"],
        ["Graph", "Mining"],
        ["Citation", "Collaboration"],
        ["Collaboration", "Trend Mapping"],
    ]
    keyword_freq = {"Network": 3, "Themes": 3, "Citation": 2, "Collaboration": 2, "Visualization": 1, "Data": 1, "Mining": 2, "Graph": 1, "Trend Mapping": 1}
    cooccurrence = {("Network", "Themes"): 2, ("Themes", "Citation"): 1, ("Citation", "Collaboration"): 1}
    analysis_parameters = [
        {"key": "publication_forecast_horizon", "value": 2},
        {"key": "entity_forecast_topn", "value": 2},
        {"key": "theme_migration_slices", "value": 3},
        {"key": "theme_migration_topn", "value": 3},
    ]

    bundle_bytes = generate_all_figure_bundle(
        df,
        keywords_list,
        keyword_freq,
        cooccurrence,
        export_format="png",
        selected_items={"temporal_journal_growth_forecast", "temporal_theme_migration_forecast"},
        analysis_parameters=analysis_parameters,
    )

    with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as zip_file:
        journal_summary_df = pd.read_csv(io.BytesIO(zip_file.read("07_analysis_tables/journal_growth_forecast.csv")))
        journal_series_df = pd.read_csv(io.BytesIO(zip_file.read("07_analysis_tables/journal_growth_forecast_series.csv")))
        theme_summary_df = pd.read_csv(io.BytesIO(zip_file.read("07_analysis_tables/theme_migration_forecast.csv")))
        theme_chain_df = pd.read_csv(io.BytesIO(zip_file.read("07_analysis_tables/theme_migration_chain_series.csv")))
        journal_text = zip_file.read("07_analysis_tables/journal_growth_forecast_interpretation.txt").decode("utf-8")

    assert len(journal_summary_df) <= 2
    assert int(journal_series_df["Year"].max()) == 2025
    assert len(theme_summary_df) <= 3
    assert theme_chain_df["Period"].nunique() <= 3
    assert journal_text.strip()


def test_generate_all_figure_bundle_includes_leadership_shift_exports():
    df = pd.DataFrame(
        [
            {"Year": 2020, "Affiliations": "Dept A, Univ A, Beijing, PR CHINA"},
            {"Year": 2020, "Affiliations": "Dept B, Inst X, Boston, U.S."},
            {"Year": 2021, "Affiliations": "Dept A, Univ A, Beijing, China"},
            {"Year": 2021, "Affiliations": "Dept C, Univ B, Shanghai, China"},
            {"Year": 2022, "Affiliations": "Dept B, Inst X, Boston, USA"},
            {"Year": 2022, "Affiliations": "Dept C, Univ B, Shanghai, China"},
            {"Year": 2023, "Affiliations": "Dept C, Univ B, Shanghai, China"},
            {"Year": 2023, "Affiliations": "Dept D, Inst Y, London, England"},
            {"Year": 2024, "Affiliations": "Dept C, Univ B, Shanghai, China"},
            {"Year": 2024, "Affiliations": "Dept D, Inst Y, London, United Kingdom"},
        ]
    )

    bundle_bytes = generate_all_figure_bundle(
        df,
        keywords_list=[],
        keyword_freq={},
        cooccurrence={},
        export_format="png",
        selected_items={"temporal_entity_leadership_shift"},
        analysis_parameters=[
            {"key": "forward_leadership_shift_type", "value": "Countries"},
            {"key": "leadership_shift_topn", "value": 3},
        ],
    )

    with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as zip_file:
        names = set(zip_file.namelist())
        leadership_df = pd.read_csv(io.BytesIO(zip_file.read("07_analysis_tables/country_leadership_shift.csv")))
        leadership_text = zip_file.read("07_analysis_tables/country_leadership_interpretation.txt").decode("utf-8")

    assert "03_temporal/country_leadership_shift.png" in names
    assert "03_temporal/country_leadership_share_over_time.png" in names
    assert "07_analysis_tables/country_leadership_shift.csv" in names
    assert "07_analysis_tables/country_leadership_series.csv" in names
    assert "07_analysis_tables/country_leadership_interpretation.txt" in names
    assert len(leadership_df) <= 3
    assert leadership_text.strip()
