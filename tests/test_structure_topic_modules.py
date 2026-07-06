import time

import pandas as pd

from modules.advanced_visualizations import (
    build_country_impact_quadrant_frame,
    build_keyword_circular_cluster_figure,
    render_country_impact_quadrant_figure,
)
import modules.topic_modeling as topic_modeling
from modules.structure_visualization import (
    build_author_production_over_time_frame,
    render_author_production_over_time,
    render_lotkas_law,
    render_thematic_map,
    render_three_field_plot,
)
from modules.topic_modeling import (
    discard_bertopic_background_job,
    get_bertopic_background_job,
    get_bertopic_profile_settings,
    recommend_bertopic_doc_cap,
    render_bertopic_comparison,
    render_bertopic_evolution,
    render_bertopic_overview,
    submit_bertopic_background_job,
    should_compute_bertopic_evolution,
)


def test_render_thematic_map_returns_figure_for_clustered_keywords():
    keyword_freq = {
        "Network": 8,
        "Citation": 7,
        "Data": 6,
        "Visualization": 5,
        "Graph": 4,
        "Mapping": 4,
    }
    cooccurrence = {
        ("Network", "Citation"): 5,
        ("Network", "Data"): 4,
        ("Citation", "Data"): 4,
        ("Visualization", "Graph"): 4,
        ("Visualization", "Mapping"): 3,
        ("Graph", "Mapping"): 3,
        ("Data", "Visualization"): 1,
    }

    fig = render_thematic_map(keyword_freq, cooccurrence, top_n=6)

    assert fig is not None
    assert len(fig.data) >= 2


def test_keyword_circular_cluster_figure_returns_figure_for_connected_keywords():
    keyword_freq = {
        "colistin": 12,
        "resistance": 11,
        "infection": 8,
        "klebsiella": 7,
        "antimicrobial": 6,
        "therapy": 5,
    }
    cooccurrence = {
        ("colistin", "resistance"): 7,
        ("colistin", "infection"): 4,
        ("resistance", "klebsiella"): 4,
        ("infection", "therapy"): 3,
        ("antimicrobial", "resistance"): 5,
        ("antimicrobial", "therapy"): 2,
    }

    fig = build_keyword_circular_cluster_figure(keyword_freq, cooccurrence, top_n=6, min_weight=2)

    assert fig is not None
    assert len(fig.data) >= 2


def test_country_impact_quadrant_build_and_render():
    df = pd.DataFrame(
        [
            {
                "Affiliations": "[A] Univ A, Beijing, China; [B] Univ B, London, England",
                "Times_Cited": 60,
            },
            {
                "Affiliations": "[A] Univ C, Berlin, Germany",
                "Times_Cited": 55,
            },
            {
                "Affiliations": "[A] Univ D, Shanghai, China",
                "Times_Cited": 20,
            },
            {
                "Affiliations": "[A] Univ E, Tokyo, Japan",
                "Times_Cited": 18,
            },
            {
                "Affiliations": "[A] Univ F, Delhi, India",
                "Times_Cited": 12,
            },
        ]
    )

    quadrant_df, pub_median, cite_median = build_country_impact_quadrant_frame(df, top_n=5)
    fig = render_country_impact_quadrant_figure(quadrant_df, pub_median, cite_median)

    assert not quadrant_df.empty
    assert {"Country", "Publications", "Avg Citations", "Total Citations", "Quadrant"}.issubset(quadrant_df.columns)
    assert pub_median > 0
    assert cite_median > 0
    assert fig is not None
    assert len(fig.data) >= 1


def test_render_three_field_plot_returns_sankey_figure():
    df = pd.DataFrame(
        [
            {"Authors": "Alice;Bob", "Journal": "Journal A"},
            {"Authors": "Alice;Carol", "Journal": "Journal B"},
            {"Authors": "Bob", "Journal": "Journal A"},
        ]
    )
    keywords_list = [
        ["Network", "Mapping"],
        ["Network", "Data"],
        ["Mapping", "Visualization"],
    ]

    fig = render_three_field_plot(df, keywords_list, {"Network": 2, "Mapping": 2, "Data": 1, "Visualization": 1}, 3, 4, 2)

    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].type == "sankey"


def test_render_lotkas_law_returns_figure_and_stats():
    df = pd.DataFrame(
        [
            {"Authors": "Alice;Bob"},
            {"Authors": "Alice;Carol"},
            {"Authors": "Alice"},
        ]
    )

    result = render_lotkas_law(df)

    assert result is not None
    fig, stats = result
    assert fig is not None
    assert len(fig.data) == 2
    assert stats["total_authors"] == 3
    assert stats["max_papers"] == 3


def test_author_production_over_time_returns_frame_and_figure():
    df = pd.DataFrame(
        [
            {"Year": 2020, "Authors": "Alice;Bob"},
            {"Year": 2021, "Authors": "Alice;Carol"},
            {"Year": 2021, "Authors": "Alice"},
            {"Year": 2022, "Authors": "Bob"},
        ]
    )

    production_df = build_author_production_over_time_frame(df, top_n=2)
    fig = render_author_production_over_time(df, top_n=2)

    assert not production_df.empty
    assert set(production_df["Author"]) == {"Alice", "Bob"}
    assert production_df[production_df["Author"] == "Alice"]["Total Papers"].max() == 3
    assert fig is not None
    assert len(fig.data) == 2
    assert fig.layout.title.text == "Authors' Production Over Time"


def test_bertopic_renderers_return_figures_for_valid_inputs():
    topic_info = pd.DataFrame(
        [
            {"Topic": -1, "Count": 2, "Name": "Outlier"},
            {"Topic": 0, "Count": 6, "Name": "0_network_data"},
            {"Topic": 1, "Count": 4, "Name": "1_visualization_graph"},
        ]
    )
    topics_over_time = pd.DataFrame(
        [
            {"Topic": 0, "Timestamp": 2020, "Frequency": 2, "Name": "0_network_data"},
            {"Topic": 0, "Timestamp": 2021, "Frequency": 4, "Name": "0_network_data"},
            {"Topic": 1, "Timestamp": 2020, "Frequency": 1, "Name": "1_visualization_graph"},
            {"Topic": 1, "Timestamp": 2021, "Frequency": 3, "Name": "1_visualization_graph"},
        ]
    )
    keyword_freq = {"Network": 5, "Data": 4, "Visualization": 3, "Graph": 2}

    overview_fig = render_bertopic_overview(topic_info, top_n=5)
    evolution_fig = render_bertopic_evolution(topics_over_time, top_n=5)
    comparison_fig = render_bertopic_comparison(keyword_freq, topic_info, top_n=5)

    assert overview_fig is not None
    assert len(overview_fig.data) == 2
    assert evolution_fig is not None
    assert len(evolution_fig.data) == 2
    assert comparison_fig is not None
    assert len(comparison_fig.data) == 3


def test_bertopic_fast_profile_helpers_return_expected_thresholds():
    assert recommend_bertopic_doc_cap(1200, lightweight_mode=True) == 600
    assert recommend_bertopic_doc_cap(3500, lightweight_mode=True) == 400
    assert recommend_bertopic_doc_cap(7000, lightweight_mode=False) == 600

    assert should_compute_bertopic_evolution(500, lightweight_mode=True) is False
    assert should_compute_bertopic_evolution(800, lightweight_mode=True) is False
    assert should_compute_bertopic_evolution(500, lightweight_mode=False) is True
    assert should_compute_bertopic_evolution(1200, lightweight_mode=False) is False

    quick_profile = get_bertopic_profile_settings(7000, "quick_preview")
    full_profile = get_bertopic_profile_settings(7000, "full_analysis")

    assert quick_profile["doc_cap"] == 250
    assert quick_profile["include_topics_over_time"] is False
    assert full_profile["doc_cap"] == 600
    assert full_profile["include_topics_over_time"] is True


def test_bertopic_background_job_lifecycle(monkeypatch):
    fake_result = ("model", pd.DataFrame([{"Topic": 0, "Count": 3, "Name": "0_test_topic"}]), None)
    monkeypatch.setattr(topic_modeling, "_execute_bertopic_job", lambda *args, **kwargs: fake_result)

    job_id = submit_bertopic_background_job(
        pd.DataFrame([{"Title": "A sample title", "Abstract": "A sample abstract", "Year": 2020}]),
        min_docs=3,
        include_topics_over_time=False,
        profile_name="quick_preview",
    )

    state = None
    for _ in range(50):
        state = get_bertopic_background_job(job_id)
        if state["status"] == "done":
            break
        time.sleep(0.01)

    assert state is not None
    assert state["status"] == "done"
    assert state["profile_name"] == "quick_preview"
    assert state["analyzed_records"] == 1
    assert state["result"] == fake_result
    assert discard_bertopic_background_job(job_id) is True
