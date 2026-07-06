import time

import networkx as nx
import pandas as pd
import modules.experiment_framework as experiment_framework
from modules.experiment_framework import (
    build_brokerage_baseline_comparison_report,
    build_brokerage_robustness_report,
    build_bibliographic_coupling_network,
    calculate_network_metrics,
    build_graph_from_cooccurrence,
    build_parameterized_journal_template,
    compute_brokerage_baseline_comparison,
    compute_brokerage_robustness_experiment,
    build_experiment_comparison_report,
    build_innovation_metrics_report,
    compute_structural_hole_frame,
    summarize_structural_hole_frame,
    build_journal_submission_package_report,
    build_journal_submission_package_snapshot,
    build_research_report,
    build_research_report_snapshot,
    build_reviewer_response_report,
    build_reviewer_response_snapshot,
    build_submission_figure_package_report,
    build_submission_figure_package_snapshot,
    build_submission_result_report,
    build_submission_result_snapshot,
    compute_disruption_index_frame,
    discard_innovation_background_job,
    get_innovation_background_job,
    get_baseline_comparison_data,
    render_brokerage_robustness_summary,
    render_structural_hole_brokerage_profile,
    submit_innovation_background_job,
    summarize_disruption_index,
)
from collections import Counter

def test_build_parameterized_journal_template_reflects_selected_preferences():
    template = build_parameterized_journal_template(
        {
            "main_text_policy": "compact",
            "supplement_policy": "supplement_heavy",
            "review_intensity": "revision_ready",
            "article_format": "short_article",
        }
    )

    assert template["template_id"] == "parameterized_target_journal"
    assert template["preferences"]["main_text_policy"] == "compact"
    assert "caption_templates.md" in template["main_manuscript_priority"]
    assert "Biblio-HUB_reproducibility_report.md" in template["reviewer_appendix_priority"]


def test_build_graph_from_cooccurrence_creates_valid_networkx_graph():
    cooccurrence = Counter({("Network", "Citation"): 10, ("Network", "Data"): 5})
    G = build_graph_from_cooccurrence(cooccurrence, top_n=5)
    
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
    assert G.has_edge("Network", "Citation")
    assert G["Network"]["Citation"]["weight"] == 10

def test_calculate_network_metrics_returns_correct_shapes():
    G = nx.Graph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("A", "C")
    
    metrics = calculate_network_metrics(G)
    
    assert metrics["nodes"] == 3
    assert metrics["edges"] == 3
    assert metrics["density"] == 1.0
    assert "modularity" in metrics
    assert metrics["clusters"] >= 1
    assert metrics["avg_clustering_coeff"] == 1.0

def test_calculate_network_metrics_handles_empty_graph():
    G = nx.Graph()
    metrics = calculate_network_metrics(G)
    assert metrics["nodes"] == 0
    assert metrics["density"] == 0

def test_get_baseline_comparison_data_returns_list_of_dicts():
    data = get_baseline_comparison_data()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "Feature" in data[0]
    assert "This Tool" in data[0]

def test_build_experiment_comparison_report_contains_key_sections():
    df = pd.DataFrame([{"Title": "Paper A"}])
    G = nx.Graph()
    G.add_edge("Network", "Citation")
    params = [{"key": "test", "label": "Test", "value": 1, "default": 1, "group": "Test"}]
    
    report = build_experiment_comparison_report(df, G, params)
    
    assert "# Comparative Experiment & Results Report" in report
    assert "## 1. Network Topology Analysis" in report
    assert "## 2. Brokerage and Bridge Analysis" in report
    assert "## 3. Methodology Narrative" in report
    assert "## 4. Comparison with Baseline Tools" in report
    assert "Louvain" in report
    assert "CiteSpace" in report
    assert "Structural-hole" in report


def test_build_bibliographic_coupling_network_links_papers_with_shared_references():
    df = pd.DataFrame(
        [
            {
                "Title": "Paper A",
                "Year": 2020,
                "Cited_References": "SMITH J, 2018, SCIENCE; LEE K, 2019, NATURE",
            },
            {
                "Title": "Paper B",
                "Year": 2021,
                "Cited_References": "SMITH J, 2018, SCIENCE; CHEN Q, 2017, CELL",
            },
            {
                "Title": "Paper C",
                "Year": 2022,
                "Cited_References": "WANG L, 2016, PNAS",
            },
        ]
    )

    G, pairs, top_papers = build_bibliographic_coupling_network(df, top_n=10, min_shared_refs=1)

    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1
    assert pairs[0]["shared_references"] == 1
    assert "Paper A" in pairs[0]["source"] or "Paper A" in pairs[0]["target"]
    assert len(top_papers) == 2


def test_build_bibliographic_coupling_network_uses_neutral_duplicate_suffixes_instead_of_journal_names():
    df = pd.DataFrame(
        [
            {
                "Title": "Paper Alpha",
                "Authors": "Smith, J",
                "Year": 2020,
                "Journal": "Science",
                "Cited_References": "LEE K, 2018, NATURE; CHEN Q, 2019, CELL",
            },
            {
                "Title": "Paper Beta",
                "Authors": "Smith, J",
                "Year": 2020,
                "Journal": "Nature",
                "Cited_References": "LEE K, 2018, NATURE; WANG L, 2017, PNAS",
            },
            {
                "Title": "Paper Gamma",
                "Authors": "Garcia, M",
                "Year": 2021,
                "Journal": "Cell",
                "Cited_References": "LEE K, 2018, NATURE; CHEN Q, 2019, CELL",
            },
        ]
    )

    G, pairs, top_papers = build_bibliographic_coupling_network(df, top_n=10, min_shared_refs=1)

    labels = {row["source"] for row in pairs} | {row["target"] for row in pairs}
    labels.update(item["paper"] for item in top_papers)

    assert G.number_of_nodes() == 3
    assert "Smith, 2020a" in labels
    assert "Smith, 2020b" in labels
    assert not any("Science" in label or "Nature" in label for label in labels)


def test_build_bibliographic_coupling_network_sanitizes_legacy_precalculated_labels():
    df = pd.DataFrame(
        [
            {"Title": "Paper A", "Cited_References": "REF1; REF2"},
            {"Title": "Paper B", "Cited_References": "REF1; REF3"},
            {"Title": "Paper C", "Cited_References": "REF2; REF3"},
        ]
    )
    precalculated = {
        "reference_sets": {
            0: {"REF1", "REF2"},
            1: {"REF1", "REF3"},
            2: {"REF2", "REF3"},
        },
        "labels": {
            0: "Smith, 2020 (SCIENCE)",
            1: "Smith, 2020 (NATURE) [8358]",
            2: "Garcia, 2021 [19]",
        },
        "shared_counts": {
            (0, 1): 1,
            (0, 2): 1,
            (1, 2): 1,
        },
    }

    G, pairs, top_papers = build_bibliographic_coupling_network(
        df,
        top_n=10,
        min_shared_refs=1,
        _precalculated=precalculated,
    )

    labels = {row["source"] for row in pairs} | {row["target"] for row in pairs}
    labels.update(item["paper"] for item in top_papers)

    assert G.number_of_nodes() == 3
    assert "Garcia, 2021" in labels
    assert "Smith, 2020a" in labels
    assert "Smith, 2020b" in labels
    assert not any("SCIENCE" in label or "NATURE" in label for label in labels)
    assert not any("[8358]" in label or "[19]" in label for label in labels)


def test_compute_disruption_index_frame_returns_expected_columns_and_scores():
    df = pd.DataFrame(
        [
            {
                "Title": "Paper A",
                "Authors": "Smith, J",
                "Year": 2020,
                "Journal": "Science",
                "Cited_References": "",
            },
            {
                "Title": "Paper B",
                "Authors": "Lee, K",
                "Year": 2021,
                "Journal": "Nature",
                "Cited_References": "SMITH, 2020, SCIENCE",
            },
            {
                "Title": "Paper C",
                "Authors": "Chen, Q",
                "Year": 2021,
                "Journal": "Cell",
                "Cited_References": "SMITH, 2020, SCIENCE; LEE, 2021, NATURE",
            },
        ]
    )

    df_di = compute_disruption_index_frame(df)

    assert "Disruption_Index" in df_di.columns
    assert "DI_nd" in df_di.columns
    assert "DI_nc" in df_di.columns
    assert "Internal_Citers" in df_di.columns
    assert len(df_di) == 3
    assert df_di["Disruption_Index"].between(-1, 1).all()
    assert df_di.loc[0, "Disruption_Index"] > 0


def test_compute_disruption_index_frame_matches_internal_references_via_doi_and_normalized_year():
    df = pd.DataFrame(
        [
            {
                "Title": "Paper A",
                "Authors": "Smith, J",
                "Year": 2020.0,
                "Journal": "Science",
                "DOI": "10.1000/a",
                "Cited_References": "",
            },
            {
                "Title": "Paper B",
                "Authors": "Lee, K",
                "Year": 2021.0,
                "Journal": "Nature",
                "DOI": "10.1000/b",
                "Cited_References": "Smith J, 2020, Science, DOI 10.1000/a",
            },
            {
                "Title": "Paper C",
                "Authors": "Chen, Q",
                "Year": 2022.0,
                "Journal": "Cell",
                "DOI": "10.1000/c",
                "Cited_References": "Lee K, 2021, Nature, DOI 10.1000/b",
            },
        ]
    )

    df_di = compute_disruption_index_frame(df)

    assert df_di["Internal_References"].tolist() == [0, 1, 1]
    assert df_di["Internal_Citers"].tolist() == [1, 1, 0]
    assert df_di["Disruption_Index"].between(-1, 1).all()


def test_summarize_disruption_index_counts_positive_negative_and_neutral():
    df = pd.DataFrame(
        {
            "Disruption_Index": [0.5, -0.25, 0.0, 0.25],
        }
    )

    summary = summarize_disruption_index(df)

    assert summary["papers"] == 4
    assert summary["positive_count"] == 2
    assert summary["negative_count"] == 1
    assert summary["neutral_count"] == 1


def test_compute_structural_hole_frame_returns_ranked_brokerage_scores():
    G = nx.Graph()
    G.add_edge("A", "B", weight=1)
    G.add_edge("B", "C", weight=1)
    G.add_edge("C", "D", weight=1)
    G.add_edge("B", "D", weight=1)

    frame = compute_structural_hole_frame(G)

    assert not frame.empty
    assert "brokerage_score" in frame.columns
    assert "structural_constraint" in frame.columns
    assert frame.iloc[0]["brokerage_score"] >= frame.iloc[-1]["brokerage_score"]


def test_summarize_structural_hole_frame_reports_top_broker():
    frame = pd.DataFrame(
        [
            {"node": "Node A", "brokerage_score": 0.75, "brokerage_role": "core_broker"},
            {"node": "Node B", "brokerage_score": 0.45, "brokerage_role": "bridge_candidate"},
            {"node": "Node C", "brokerage_score": 0.20, "brokerage_role": "embedded_node"},
        ]
    )

    summary = summarize_structural_hole_frame(frame)

    assert summary["nodes"] == 3
    assert summary["top_broker"] == "Node A"
    assert summary["core_brokers"] == 1
    assert summary["bridge_candidates"] == 2


def test_render_structural_hole_brokerage_profile_returns_plotly_figure():
    frame = pd.DataFrame(
        [
            {
                "node": "Node A",
                "degree": 4,
                "weighted_degree": 10.0,
                "betweenness_centrality": 0.80,
                "structural_constraint": 0.20,
                "effective_size": 3.50,
                "brokerage_score": 0.88,
                "brokerage_role": "core_broker",
            },
            {
                "node": "Node B",
                "degree": 3,
                "weighted_degree": 8.0,
                "betweenness_centrality": 0.55,
                "structural_constraint": 0.35,
                "effective_size": 2.80,
                "brokerage_score": 0.63,
                "brokerage_role": "bridge_candidate",
            },
        ]
    )

    fig = render_structural_hole_brokerage_profile(frame, top_n=10)

    assert fig is not None
    assert fig.layout.title.text == "Structural-Hole Brokerage Profile"


def test_render_brokerage_robustness_summary_returns_plotly_figure():
    snapshot = {
        "summary": {
            "scenario_count": 4,
            "valid_scenarios": 4,
        },
        "scenarios": [
            {
                "top_n": 20,
                "min_shared_refs": 1,
                "reference_overlap_ratio": 0.75,
            },
            {
                "top_n": 30,
                "min_shared_refs": 1,
                "reference_overlap_ratio": 1.0,
            },
            {
                "top_n": 20,
                "min_shared_refs": 2,
                "reference_overlap_ratio": 0.50,
            },
            {
                "top_n": 30,
                "min_shared_refs": 2,
                "reference_overlap_ratio": 0.80,
            },
        ],
        "stable_brokers": [
            {"node": "Paper A", "occurrence_count": 4, "occurrence_ratio": 1.0},
            {"node": "Paper B", "occurrence_count": 3, "occurrence_ratio": 0.75},
        ],
    }

    fig = render_brokerage_robustness_summary(snapshot, top_stable_count=5)

    assert fig is not None
    assert fig.layout.title.text == "Brokerage Robustness Summary"


def test_compute_brokerage_robustness_experiment_returns_scenarios_and_stable_brokers():
    df = pd.DataFrame(
        [
            {
                "Title": "Paper A",
                "Year": 2020,
                "Cited_References": "SMITH J, 2018, SCIENCE; LEE K, 2019, NATURE; CHEN Q, 2017, CELL",
            },
            {
                "Title": "Paper B",
                "Year": 2021,
                "Cited_References": "SMITH J, 2018, SCIENCE; LEE K, 2019, NATURE; WANG L, 2016, PNAS",
            },
            {
                "Title": "Paper C",
                "Year": 2021,
                "Cited_References": "SMITH J, 2018, SCIENCE; CHEN Q, 2017, CELL; ZHAO P, 2015, LANCET",
            },
            {
                "Title": "Paper D",
                "Year": 2022,
                "Cited_References": "LEE K, 2019, NATURE; WANG L, 2016, PNAS; ZHAO P, 2015, LANCET",
            },
        ]
    )

    snapshot = compute_brokerage_robustness_experiment(
        df,
        top_n_values=[3, 4],
        min_shared_values=[1, 2],
        reference_top_n=4,
        reference_min_shared=1,
        top_k=3,
    )

    assert snapshot["summary"]["scenario_count"] == 4
    assert len(snapshot["scenarios"]) == 4
    assert "mean_reference_overlap" in snapshot["summary"]
    assert isinstance(snapshot["stable_brokers"], list)


def test_build_brokerage_robustness_report_contains_key_sections():
    snapshot = {
        "summary": {
            "scenario_count": 4,
            "valid_scenarios": 4,
            "reference_top_n": 30,
            "reference_min_shared_refs": 2,
            "mean_reference_overlap": 0.75,
            "stable_broker_count": 2,
            "top_stable_broker": "Paper A",
        },
        "scenarios": [
            {
                "top_n": 20,
                "min_shared_refs": 1,
                "nodes": 6,
                "mean_brokerage": 0.4123,
                "top_broker": "Paper A",
                "top_score": 0.8123,
                "reference_overlap_ratio": 0.6667,
            }
        ],
        "stable_brokers": [
            {"node": "Paper A", "occurrence_count": 4, "occurrence_ratio": 1.0}
        ],
        "execution_policy": {
            "lightweight_mode": True,
            "full_record_count": 6200,
            "analysis_record_count": 3000,
            "downsampled": True,
            "scenario_count_requested": 1,
        },
    }

    report = build_brokerage_robustness_report(snapshot)

    assert "# Brokerage Robustness Report" in report
    assert "## 1. Robustness Summary" in report
    assert "## 2. Execution Policy" in report
    assert "- Policy summary: lightweight mode, 3000/6200 records, 1 scenario(s), downsampled" in report
    assert "## 3. Scenario Comparison Table" in report
    assert "## 4. Stable Broker Candidates" in report
    assert "## 6. Manuscript Snippet" in report


def test_compute_brokerage_baseline_comparison_returns_alignment_summary():
    frame = pd.DataFrame(
        [
            {
                "node": "Node A",
                "degree": 4,
                "weighted_degree": 10.0,
                "betweenness_centrality": 0.80,
                "structural_constraint": 0.20,
                "effective_size": 3.50,
                "brokerage_score": 0.88,
                "brokerage_role": "core_broker",
            },
            {
                "node": "Node B",
                "degree": 3,
                "weighted_degree": 8.0,
                "betweenness_centrality": 0.55,
                "structural_constraint": 0.35,
                "effective_size": 2.80,
                "brokerage_score": 0.63,
                "brokerage_role": "bridge_candidate",
            },
            {
                "node": "Node C",
                "degree": 5,
                "weighted_degree": 11.0,
                "betweenness_centrality": 0.30,
                "structural_constraint": 0.50,
                "effective_size": 2.10,
                "brokerage_score": 0.40,
                "brokerage_role": "bridge_candidate",
            },
        ]
    )

    snapshot = compute_brokerage_baseline_comparison(frame, top_k=2)

    assert snapshot["summary"]["nodes"] == 3
    assert snapshot["summary"]["top_brokerage_node"] == "Node A"
    assert len(snapshot["baseline_comparisons"]) == 3
    assert len(snapshot["node_comparison_table"]) == 2


def test_build_brokerage_baseline_comparison_report_contains_key_sections():
    snapshot = {
        "summary": {
            "nodes": 3,
            "top_brokerage_node": "Node A",
            "best_aligned_baseline": "Betweenness Centrality",
            "best_alignment_overlap": 1.0,
        },
        "metric_leaders": [
            {"metric": "Brokerage Score", "top_node": "Node A", "top_value": 0.88},
        ],
        "baseline_comparisons": [
            {
                "baseline_metric": "Betweenness Centrality",
                "top_node": "Node A",
                "top_k_overlap_count": 2,
                "top_k_overlap_ratio": 1.0,
                "top_1_match": True,
                "mean_rank_shift": 0.0,
            }
        ],
        "node_comparison_table": [
            {
                "node": "Node A",
                "brokerage_rank": 1,
                "brokerage_score": 0.88,
                "betweenness_rank": 1,
                "weighted_degree_rank": 2,
                "effective_size_rank": 1,
                "brokerage_role": "core_broker",
            }
        ],
    }

    report = build_brokerage_baseline_comparison_report(snapshot)

    assert "# Brokerage Baseline Comparison Report" in report
    assert "## 1. Summary" in report
    assert "## 2. Metric Leaders" in report
    assert "## 3. Baseline Alignment" in report
    assert "## 4. Top Brokerage Nodes vs Baselines" in report
    assert "## 6. Manuscript Snippet" in report

def test_build_innovation_metrics_report_contains_key_sections():
    df = pd.DataFrame(
        [
            {
                "Title": "Paper A",
                "Authors": "Smith, J",
                "Year": 2020,
                "Journal": "Science",
                "Cited_References": "",
            },
            {
                "Title": "Paper B",
                "Authors": "Lee, K",
                "Year": 2021,
                "Journal": "Nature",
                "Cited_References": "SMITH, 2020, SCIENCE",
            },
            {
                "Title": "Paper C",
                "Authors": "Chen, Q",
                "Year": 2021,
                "Journal": "Cell",
                "Cited_References": "SMITH, 2020, SCIENCE; LEE, 2021, NATURE",
            },
        ]
    )
    
    G_bc = nx.Graph()
    G_bc.add_edge("Paper A (2020)", "Paper B (2021)", weight=1)
    bc_pairs = [{
        "source_idx": 0,
        "target_idx": 1,
        "source": "Paper A (2020)",
        "target": "Paper B (2021)",
        "shared_references": 1,
    }]
    bc_top_papers = [
        {"paper": "Paper A (2020)", "coupling_strength": 1},
        {"paper": "Paper B (2021)", "coupling_strength": 1},
    ]
    df_di = compute_disruption_index_frame(df)
    analysis_parameters = [
        {"key": "bc_topn", "label": "Bibliographic Coupling Top Papers", "value": 30, "default": 30, "group": "Citation", "changed": False},
        {"key": "bc_min_shared", "label": "Bibliographic Coupling Min Shared References", "value": 2, "default": 2, "group": "Citation", "changed": False},
        {"key": "vos_topn", "label": "Keyword Network Top N", "value": 40, "default": 30, "group": "Network", "changed": True},
    ]
    robustness_snapshot = {
        "summary": {
            "scenario_count": 4,
            "valid_scenarios": 4,
            "mean_reference_overlap": 0.75,
            "stable_broker_count": 2,
            "top_stable_broker": "Paper A (2020)",
        },
        "scenarios": [
            {"top_n": 20, "min_shared_refs": 1, "top_broker": "Paper A (2020)", "reference_overlap_ratio": 0.6667}
        ],
    }
    baseline_comparison_snapshot = {
        "summary": {
            "nodes": 2,
            "top_brokerage_node": "Paper A (2020)",
            "best_aligned_baseline": "Betweenness Centrality",
            "best_alignment_overlap": 1.0,
        },
        "baseline_comparisons": [
            {
                "baseline_metric": "Betweenness Centrality",
                "top_node": "Paper A (2020)",
                "top_k_overlap_ratio": 1.0,
                "top_1_match": True,
                "mean_rank_shift": 0.0,
            }
        ],
    }

    report = build_innovation_metrics_report(
        df,
        G_bc,
        bc_pairs,
        bc_top_papers,
        df_di,
        analysis_parameters,
        robustness_snapshot=robustness_snapshot,
        baseline_comparison_snapshot=baseline_comparison_snapshot,
    )

    assert "# Innovation Metrics Experiment Report" in report
    assert "## 1. Bibliographic Coupling Analysis" in report
    assert "## 2. Disruption Index Analysis" in report
    assert "## 3. Structural Hole Brokerage Analysis" in report
    assert "## 4. Analysis Parameters" in report
    assert "## 5. Brokerage Robustness Across Parameter Settings" in report
    assert "## 6. Brokerage Baseline Comparison" in report
    assert "## 7. Methodology Snippets for Manuscript" in report
    assert "Modularity (Q)" in report
    assert "Mean Disruption Index" in report
    assert "Bibliographic Coupling Methodology" in report
    assert "Disruption Index Methodology" in report
    assert "Structural Hole Brokerage Methodology" in report
    assert "Brokerage Baseline Comparison Methodology" in report
    assert "Top stable broker" in report
    assert "Best aligned baseline metric" in report
    assert "| Keyword Network Top N | 40 | 30 | Network |" in report


def test_build_submission_result_snapshot_collects_package_components():
    df = pd.DataFrame(
        [
            {"Title": "Paper A", "Year": 2020, "Cited_References": "SMITH J, 2018, SCIENCE"},
            {"Title": "Paper B", "Year": 2021, "Cited_References": "SMITH J, 2018, SCIENCE; LEE K, 2019, NATURE"},
        ]
    )
    G_bc, bc_pairs, bc_top_papers = build_bibliographic_coupling_network(df, top_n=10, min_shared_refs=1)
    df_di = pd.DataFrame(
        {
            "Title": ["Paper A", "Paper B"],
            "Year": [2020, 2021],
            "Disruption_Index": [0.4, -0.2],
        }
    )
    analysis_parameters = [
        {"key": "bc_topn", "label": "Bibliographic Coupling Top Papers", "value": 40, "default": 30, "group": "Citation", "changed": True},
        {"key": "vos_topn", "label": "Keyword Network Top N", "value": 30, "default": 30, "group": "Network", "changed": False},
    ]

    snapshot = build_submission_result_snapshot(
        df,
        G_bc,
        bc_pairs,
        bc_top_papers,
        df_di,
        analysis_parameters,
        robustness_snapshot={
            "summary": {
                "scenario_count": 4,
                "stable_broker_count": 2,
                "top_stable_broker": "Paper A",
            },
            "execution_policy": {
                "lightweight_mode": True,
                "full_record_count": 6200,
                "analysis_record_count": 3000,
                "downsampled": True,
                "scenario_count_requested": 1,
            },
        },
        baseline_comparison_snapshot={
            "summary": {
                "best_aligned_baseline": "Betweenness Centrality",
                "best_alignment_overlap": 1.0,
            }
        },
        journal_preferences={
            "main_text_policy": "compact",
            "supplement_policy": "supplement_heavy",
            "review_intensity": "revision_ready",
            "article_format": "short_article",
        },
    )

    assert snapshot["dataset_overview"]["records"] == 2
    assert snapshot["dataset_overview"]["year_range"] == [2020, 2021]
    assert "bibliographic_coupling" in snapshot["innovation_metrics"]
    assert "disruption_index" in snapshot["innovation_metrics"]
    assert "structural_hole" in snapshot["innovation_metrics"]
    assert len(snapshot["recommended_figures"]) >= 1
    assert len(snapshot["recommended_tables"]) >= 1
    assert len(snapshot["recommended_output_sequence"]) == 7
    assert len(snapshot["chapter_target_output_plan"]) == 4
    assert len(snapshot["main_results_table"]) >= 5
    assert len(snapshot["supplementary_table_index"]) >= 5
    assert len(snapshot["figure_table_crosswalk"]) >= 5
    assert snapshot["changed_parameters"][0]["label"] == "Bibliographic Coupling Top Papers"
    assert snapshot["submission_preferences"]["main_text_policy"] == "compact"
    assert snapshot["recommended_template"]["template_id"] == "parameterized_target_journal"
    assert snapshot["recommended_output_sequence"][0]["output_kind"] == "table"
    assert snapshot["recommended_output_sequence"][0]["priority_rank"] == 1
    assert snapshot["innovation_metrics"]["structural_hole"]["summary"]["nodes"] >= 0
    assert snapshot["innovation_metrics"]["brokerage_robustness"]["summary"]["scenario_count"] == 4
    assert snapshot["innovation_metrics"]["brokerage_baseline_comparison"]["summary"]["best_aligned_baseline"] == "Betweenness Centrality"
    assert snapshot["methods_support"]["execution_policy"]["lightweight_mode"] is True
    assert "Current setting: lightweight mode, 3000/6200 records, 1 scenario(s), downsampled." in snapshot["methods_support"]["methods_paragraph"]
    assert len(snapshot["stated_limitations"]) == 1
    assert snapshot["chapter_target_output_plan"][0]["chapter"] == "Introduction"
    assert snapshot["chapter_target_output_plan"][2]["chapter"] == "Results"
    assert snapshot["chapter_target_output_plan"][2]["recommended_items"][0]["name"] in {
        "Top Bibliographic Coupling Pairs",
        "Top Disruptive and Consolidating Papers",
        "Top Structural Hole Brokers",
        "Bibliographic Coupling Network",
        "Disruption Index Distribution",
        "Structural Hole Brokerage Profile",
    }


def test_build_submission_result_report_contains_submission_sections():
    snapshot = {
        "dataset_overview": {
            "records": 12,
            "year_range": [2018, 2024],
            "has_cited_references": True,
            "has_times_cited": True,
        },
        "innovation_metrics": {
            "bibliographic_coupling": {
                "network_metrics": {"nodes": 10, "edges": 12, "density": 0.2667, "modularity": 0.4123},
            },
            "disruption_index": {
                "summary": {"mean_di": 0.0842, "positive_count": 4, "negative_count": 3, "neutral_count": 5},
            },
            "structural_hole": {
                "summary": {"mean_brokerage": 0.4567, "top_broker": "Paper X", "core_brokers": 2},
                "top_brokers": [{"node": "Paper X", "brokerage_score": 0.8123, "brokerage_role": "core_broker"}],
            },
            "brokerage_robustness": {
                "summary": {"scenario_count": 4, "stable_broker_count": 2, "top_stable_broker": "Paper X"},
            },
            "brokerage_baseline_comparison": {
                "summary": {"best_aligned_baseline": "Betweenness Centrality", "best_alignment_overlap": 0.8},
            },
        },
        "changed_parameters": [
            {"group": "Citation", "label": "Bibliographic Coupling Top Papers", "value": 40, "default": 30},
        ],
        "recommended_figures": [
            {"name": "Bibliographic Coupling Network", "caption": "Figure caption."},
        ],
        "recommended_tables": [
            {"name": "Top Bibliographic Coupling Pairs", "caption": "Table caption."},
        ],
        "result_narrative": ["Narrative line 1.", "Narrative line 2."],
        "recommended_output_sequence": [
            {
                "priority_rank": 1,
                "output_kind": "figure",
                "name": "Bibliographic Coupling Network",
                "priority_reason": "Compact manuscript prioritizes high-yield structural visuals.",
            }
        ],
        "chapter_target_output_plan": [
            {
                "chapter": "Results",
                "chapter_note": "Lead with the strongest structural and innovation evidence.",
                "recommended_items": [
                    {
                        "chapter_rank": 1,
                        "priority_rank": 1,
                        "output_kind": "figure",
                        "name": "Bibliographic Coupling Network",
                        "chapter_reason": "Results placement prioritizes coupling evidence.",
                    }
                ],
            }
        ],
        "main_results_table": [
            {"metric": "Record Count", "value": 12, "interpretation": "Interpretation A"},
        ],
        "supplementary_table_index": [
            {"table_id": "S1", "name": "Top Bibliographic Coupling Pairs", "content": "Content A", "suggested_use": "Supplementary"},
        ],
        "figure_table_crosswalk": [
            {"figure": "Bibliographic Coupling Network", "supporting_table": "Top Bibliographic Coupling Pairs", "manuscript_role": "Results"},
        ],
        "submission_preferences": {
            "main_text_policy": "evidence_dense",
            "supplement_policy": "minimal",
            "review_intensity": "reviewer_friendly",
            "article_format": "rapid_communication",
        },
        "recommended_template": {
            "template_id": "parameterized_target_journal",
            "template_name": "Target Journal Preference Template",
            "editor_note": "Keep a compact manuscript body with a rapid-communication posture.",
        },
        "methods_support": {
            "execution_policy": {
                "lightweight_mode": True,
                "full_record_count": 6200,
                "analysis_record_count": 3000,
                "downsampled": True,
                "scenario_count_requested": 1,
            },
            "methods_paragraph": "If large-sample export acceleration is used, report the recorded execution policy together with the coupling thresholds. Current setting: lightweight mode, 3000/6200 records, 1 scenario(s), downsampled.",
        },
        "stated_limitations": [
            "Large-sample export may use a reduced robustness grid or sampled subset; see the recorded execution_policy for exact settings.",
        ],
    }

    report = build_submission_result_report(snapshot)

    assert "# Manuscript Submission Result Package" in report
    assert "## 1. Dataset Overview" in report
    assert "## 2. Innovation Metric Highlights" in report
    assert "- Mean brokerage score: 0.4567" in report
    assert "- Top broker: Paper X" in report
    assert "- Brokerage robustness scenarios: 4" in report
    assert "- Top stable broker: Paper X" in report
    assert "- Best aligned baseline: Betweenness Centrality" in report
    assert "## 4. Preferred Output Sequence" in report
    assert "## 5. Chapter-Target Output Plan" in report
    assert "## 6. Main Results Table" in report
    assert "## 7. Figure Caption Templates" in report
    assert "## 8. Table Caption Templates" in report
    assert "## 9. Supplementary Table Index" in report
    assert "## 10. Figure-Table Crosswalk" in report
    assert "## 11. Target Journal Submission Strategy" in report
    assert "- Article format: rapid_communication" in report
    assert "## 12. Methods Note" in report
    assert "Current setting: lightweight mode, 3000/6200 records, 1 scenario(s), downsampled." in report


def test_build_innovation_metrics_report_handles_parameters_without_changed_flag():
    df = pd.DataFrame(
        [
            {"Title": "Paper A", "Authors": "Smith, J", "Year": 2020, "Journal": "Science", "Cited_References": ""},
            {"Title": "Paper B", "Authors": "Lee, K", "Year": 2021, "Journal": "Nature", "Cited_References": "SMITH, 2020, SCIENCE"},
        ]
    )

    graph = nx.Graph()
    graph.add_edge("Paper A (2020)", "Paper B (2021)", weight=1)
    df_di = compute_disruption_index_frame(df)
    analysis_parameters = [
        {"key": "bc_topn", "label": "Bibliographic Coupling Top Papers", "value": 30, "default": 30, "group": "Citation"},
        {"key": "vos_topn", "label": "Keyword Network Top N", "value": 40, "default": 30, "group": "Network", "changed": True},
    ]

    report = build_innovation_metrics_report(
        df,
        graph,
        [{"source": "Paper A (2020)", "target": "Paper B (2021)", "shared_references": 1}],
        [{"paper": "Paper A (2020)", "coupling_strength": 1}],
        df_di,
        analysis_parameters,
    )

    assert "## 4. Analysis Parameters" in report
    assert "| Keyword Network Top N | 40 | 30 | Network |" in report


def test_innovation_background_job_lifecycle(monkeypatch):
    fake_result = {"innovation_report": "ready", "robustness_report": "stable"}
    monkeypatch.setattr(experiment_framework, "_execute_innovation_payload_job", lambda *args, **kwargs: fake_result)

    job_id = submit_innovation_background_job(
        pd.DataFrame([{"Title": "Paper A", "Year": 2020, "Cited_References": ""}]),
        analysis_parameters=[{"key": "bc_topn", "value": 30}],
        bc_topn_val=30,
        bc_min_shared_val=2,
        lightweight_mode=True,
        top_k=5,
    )

    state = None
    for _ in range(50):
        state = get_innovation_background_job(job_id)
        if state["status"] == "done":
            break
        time.sleep(0.01)

    assert state is not None
    assert state["status"] == "done"
    assert state["lightweight_mode"] is True
    assert state["record_count"] == 1
    assert state["result"] == fake_result
    assert discard_innovation_background_job(job_id) is True


def test_build_submission_figure_package_snapshot_generates_figure_and_table_notes():
    submission_snapshot = {
        "recommended_figures": [
            {"name": "Bibliographic Coupling Network", "caption": "Figure caption."},
            {"name": "Disruption Index Distribution", "caption": "Distribution caption."},
        ],
        "recommended_tables": [
            {"name": "Top Bibliographic Coupling Pairs", "caption": "Table caption."},
        ],
    }

    figure_snapshot = build_submission_figure_package_snapshot(submission_snapshot, image_format="svg")

    assert figure_snapshot["image_format"] == "svg"
    assert len(figure_snapshot["figure_items"]) == 2
    assert len(figure_snapshot["table_items"]) == 1
    assert figure_snapshot["figure_items"][0]["id"] == "F1"
    assert figure_snapshot["table_items"][0]["id"] == "T1"
    assert figure_snapshot["figure_items"][0]["filename_suggestion"].endswith(".svg")


def test_build_submission_figure_package_report_contains_mapping_and_notes():
    submission_snapshot = {
        "recommended_figures": [
            {"name": "Bibliographic Coupling Network", "caption": "Figure caption."},
        ],
        "recommended_tables": [
            {"name": "Top Bibliographic Coupling Pairs", "caption": "Table caption."},
        ],
    }
    figure_snapshot = build_submission_figure_package_snapshot(submission_snapshot, image_format="png")

    report = build_submission_figure_package_report(submission_snapshot, figure_snapshot)

    assert "# Manuscript Figure and Table Explanation Package" in report
    assert "## 2. Figure Guidance" in report
    assert "## 3. Table Guidance" in report
    assert "## 4. Assembly Notes" in report
    assert "## 5. Result-to-Figure Mapping Summary" in report
    assert "Bibliographic Coupling Network" in report
    assert "Top Bibliographic Coupling Pairs" in report


def test_build_reviewer_response_snapshot_collects_claims_questions_and_mappings():
    submission_snapshot = {
        "methods_support": {
            "execution_policy": {
                "lightweight_mode": True,
                "full_record_count": 6200,
                "analysis_record_count": 3000,
                "downsampled": True,
                "scenario_count_requested": 1,
            },
            "methods_paragraph": "If large-sample export acceleration is used, report the recorded execution policy together with the coupling thresholds.",
        },
        "stated_limitations": [
            "Large-sample export may use a reduced robustness grid or sampled subset; see the recorded execution_policy for exact settings.",
        ],
        "innovation_metrics": {
            "bibliographic_coupling": {
                "network_metrics": {"nodes": 10, "edges": 12, "modularity": 0.4123},
            },
            "disruption_index": {
                "summary": {"mean_di": 0.0842, "positive_count": 4, "negative_count": 3},
            },
            "structural_hole": {
                "summary": {"mean_brokerage": 0.4567, "top_broker": "Paper X", "core_brokers": 2},
            },
        },
    }
    figure_package_snapshot = {
        "figure_items": [
            {
                "id": "F1",
                "name": "Bibliographic Coupling Network",
                "result_mapping": "Supports knowledge structure interpretation.",
                "reviewer_note": "Explain the structural evidence.",
            }
        ],
        "table_items": [
            {
                "id": "T1",
                "name": "Top Bibliographic Coupling Pairs",
                "result_mapping": "Provides ranked evidence.",
                "reviewer_note": "Explain the ranked evidence.",
            }
        ],
    }

    snapshot = build_reviewer_response_snapshot(submission_snapshot, figure_package_snapshot)

    assert len(snapshot["innovation_claims"]) >= 3
    assert len(snapshot["anticipated_questions"]) >= 3
    assert len(snapshot["evidence_mapping"]) == 2
    assert len(snapshot["limitations"]) >= 3
    assert any("recorded execution_policy" in item for item in snapshot["limitations"])
    assert any("Submission-oriented methods note is preserved" in item for item in snapshot["reproducibility_responses"])
    assert any("see the recorded execution_policy" in item for item in snapshot["reproducibility_responses"])


def test_build_reviewer_response_report_contains_expected_sections():
    submission_snapshot = {}
    figure_package_snapshot = {}
    reviewer_snapshot = {
        "innovation_claims": [
            {"theme": "Methodological Breadth", "claim": "Claim A", "evidence": "Evidence A"},
        ],
        "anticipated_questions": [
            {"question": "Question A?", "response": "Response A"},
        ],
        "reproducibility_responses": [
            "Reproducibility note.",
            "Submission-oriented methods note is preserved in the structured snapshot for manuscript drafting.",
        ],
        "evidence_mapping": [
            {"item_id": "F1", "item_type": "figure", "name": "Figure A", "supports_claim": "Supports claim A", "reviewer_note": "Reviewer note A"},
        ],
        "limitations": ["Limitation A.", "Large-sample export may use a reduced robustness grid or sampled subset; see the recorded execution_policy for exact settings."],
    }

    report = build_reviewer_response_report(submission_snapshot, figure_package_snapshot, reviewer_snapshot)

    assert "# Reviewer Response Material Package" in report
    assert "## 1. Suggested Innovation Claims" in report
    assert "## 2. Anticipated Reviewer Questions" in report
    assert "## 3. Reproducibility Response Notes" in report
    assert "## 4. Evidence Mapping" in report
    assert "## 5. Stated Limitations" in report
    assert "## 6. Author Reminder" in report
    assert "Question A?" in report
    assert "structured snapshot for manuscript drafting" in report


def test_build_journal_submission_package_snapshot_groups_outputs_by_submission_role():
    submission_snapshot = {
        "supplementary_table_index": [
            {"table_id": "S1", "name": "Table A", "content": "Content A", "suggested_use": "Supplementary"},
        ],
    }
    figure_package_snapshot = {
        "figure_items": [{"id": "F1"}, {"id": "F2"}],
    }
    reviewer_snapshot = {
        "anticipated_questions": [{"question": "Q1", "response": "A1"}],
    }

    snapshot = build_journal_submission_package_snapshot(
        submission_snapshot,
        figure_package_snapshot,
        reviewer_snapshot,
    )

    assert snapshot["summary"]["main_manuscript_items"] >= 3
    assert snapshot["summary"]["supplementary_items"] >= 5
    assert snapshot["summary"]["reviewer_appendix_items"] >= 3
    assert snapshot["summary"]["template_variant_count"] >= 5
    assert snapshot["summary"]["figure_count"] == 2
    assert snapshot["summary"]["reviewer_question_count"] == 1
    assert snapshot["template_variants"][0]["template_id"] == "parameterized_target_journal"
    assert snapshot["recommended_template"]["preferences"]["main_text_policy"] == "balanced"


def test_build_journal_submission_package_snapshot_applies_target_journal_preferences():
    submission_snapshot = {"supplementary_table_index": []}
    figure_package_snapshot = {"figure_items": [{"id": "F1"}]}
    reviewer_snapshot = {"anticipated_questions": [{"question": "Q1", "response": "A1"}]}

    snapshot = build_journal_submission_package_snapshot(
        submission_snapshot,
        figure_package_snapshot,
        reviewer_snapshot,
        journal_preferences={
            "main_text_policy": "compact",
            "supplement_policy": "supplement_heavy",
            "review_intensity": "revision_ready",
            "article_format": "short_article",
        },
    )

    assert snapshot["journal_preferences"]["main_text_policy"] == "compact"
    assert snapshot["recommended_template"]["preferences"]["supplement_policy"] == "supplement_heavy"
    assert "caption_templates.md" in snapshot["recommended_template"]["main_manuscript_priority"]
    assert "manuscript_figure_explanation_package.md" in snapshot["recommended_template"]["supplementary_priority"]
    assert "Biblio-HUB_reproducibility_report.md" in snapshot["recommended_template"]["reviewer_appendix_priority"]
    assert snapshot["execution_policy_note"] == ""


def test_build_journal_submission_package_report_contains_three_submission_groups():
    snapshot = {
        "main_manuscript": [
            {"artifact": "main.md", "role": "Role A", "why": "Why A"},
        ],
        "supplementary": [
            {"artifact": "supp.md", "role": "Role B", "why": "Why B"},
        ],
        "reviewer_appendix": [
            {"artifact": "review.md", "role": "Role C", "why": "Why C"},
        ],
        "summary": {
            "main_manuscript_items": 1,
            "supplementary_items": 1,
            "reviewer_appendix_items": 1,
            "supplementary_table_count": 2,
            "figure_count": 3,
            "reviewer_question_count": 4,
            "template_variant_count": 2,
        },
        "journal_preferences": {
            "main_text_policy": "compact",
            "supplement_policy": "supplement_heavy",
            "review_intensity": "revision_ready",
            "article_format": "short_article",
        },
        "recommended_template": {
            "template_name": "Target Journal Preference Template",
            "positioning": "Positioning P",
            "main_manuscript_priority": ["main.md"],
            "supplementary_priority": ["supp.md"],
            "reviewer_appendix_priority": ["review.md"],
            "editor_note": "Editor note P",
        },
        "execution_policy_note": "Robustness export policy: lightweight mode, 3000/6200 records, 1 scenario(s), downsampled.",
        "template_variants": [
            {
                "template_name": "Strict Supplement Style",
                "positioning": "Positioning A",
                "main_manuscript_priority": ["main.md"],
                "supplementary_priority": ["supp.md"],
                "reviewer_appendix_priority": ["review.md"],
                "editor_note": "Editor note A",
            }
        ],
        "assembly_notes": ["Note A."],
    }

    report = build_journal_submission_package_report(snapshot)

    assert "# Journal Submission Version Package" in report
    assert "## 2. Main Manuscript Package" in report
    assert "## 3. Supplementary Package" in report
    assert "## 4. Reviewer Appendix Package" in report
    assert "## 5. Target Journal Preferences" in report
    assert "## 6. Recommended Parameterized Template" in report
    assert "## 7. Journal Template Variants" in report
    assert "## 8. Assembly Notes" in report
    assert "Strict Supplement Style" in report
    assert "short_article" in report
    assert "main.md" in report
    assert "- Large-sample export notice: Robustness export policy: lightweight mode, 3000/6200 records, 1 scenario(s), downsampled." in report
    assert "- Report this policy consistently in the manuscript Methods, submission letter, and supplementary files when export acceleration is used." in report
    assert report.count("Robustness export policy: lightweight mode, 3000/6200 records, 1 scenario(s), downsampled.") == 1


def test_build_research_report_snapshot_aggregates_component_counts_and_headlines():
    manuscript_snapshot = {
        "records": 12,
        "year_range": [2019, 2024],
        "unique_journals": 5,
        "unique_keywords": 18,
        "doi_coverage": 0.75,
        "abstract_coverage": 0.9,
        "author_coverage": 1.0,
    }
    reproducibility_snapshot = {
        "analysis_parameters": [
            {"group": "Network", "label": "Top N", "value": 40, "default": 30, "changed": True},
            {"group": "Export", "label": "Format", "value": "png", "default": "png", "changed": False},
        ],
        "keyword_statistics": {"cooccurrence_pairs": 33},
        "methods_evidence_map": [
            {
                "step": "Dataset Provenance",
                "algorithm_or_rule": "Metadata ingestion with coverage audit",
                "key_parameters": "Tracked columns=10; records=12",
                "evidence_output": "Dataset provenance summary",
                "manuscript_use": "Methods / Data source",
            }
        ],
        "methods_writing_pack": {
            "data_source_paragraph": "Data paragraph.",
            "pipeline_paragraph": "Pipeline paragraph.",
            "parameter_paragraph": "Parameter paragraph.",
            "reproducibility_paragraph": "Reproducibility paragraph.",
            "submission_alignment_paragraph": "Submission alignment paragraph.",
        },
        "parameter_change_summary": [
            {
                "module": "Network",
                "parameter_count": 1,
                "changed_parameters": "Top N",
                "current_settings": "Top N=40",
                "default_settings": "Top N=30",
                "methods_note": "Report network thresholds in Methods.",
            }
        ],
    }
    innovation_snapshot = {"bibliographic_coupling": {}, "disruption_index": {}}
    submission_snapshot = {
        "innovation_metrics": {
            "bibliographic_coupling": {"network_metrics": {"nodes": 8, "edges": 12}},
            "disruption_index": {"summary": {"mean_di": 0.1025}},
        },
        "methods_support": {
            "execution_policy": {
                "lightweight_mode": True,
                "full_record_count": 6200,
                "analysis_record_count": 3000,
                "downsampled": True,
                "scenario_count_requested": 1,
            },
            "methods_paragraph": "If large-sample export acceleration is used, report the recorded execution policy together with the coupling thresholds.",
        },
        "recommended_figures": [
            {"name": "Publications by Year", "caption": "Caption A"},
            {"name": "Bibliographic Coupling Network", "caption": "Caption B"},
        ],
        "recommended_tables": [{"name": "Disruption Index Distribution", "caption": "Caption C"}],
        "recommended_output_sequence": [
            {
                "priority_rank": 1,
                "output_kind": "table",
                "name": "Disruption Index Distribution",
                "caption": "Caption C",
                "priority_reason": "Review-intensive settings prioritize traceable ranked evidence.",
            },
            {
                "priority_rank": 2,
                "output_kind": "figure",
                "name": "Publications by Year",
                "caption": "Caption A",
                "priority_reason": "Compact outputs keep trend context concise.",
            },
            {
                "priority_rank": 3,
                "output_kind": "figure",
                "name": "Bibliographic Coupling Network",
                "caption": "Caption B",
                "priority_reason": "Structural evidence remains central in the main text.",
            },
        ],
    }
    figure_package_snapshot = {
        "figure_items": [{"id": "F1"}],
        "table_items": [{"id": "T1"}],
    }
    reviewer_snapshot = {
        "anticipated_questions": [{"question": "Q1", "response": "A1"}],
        "innovation_claims": [{"theme": "Theme A", "claim": "Claim A", "evidence": "Evidence A"}],
        "limitations": ["Limitation A"],
    }
    journal_submission_snapshot = {
        "journal_preferences": {
            "main_text_policy": "compact",
            "supplement_policy": "supplement_heavy",
            "review_intensity": "revision_ready",
            "article_format": "short_article",
        },
        "recommended_template": {
            "template_id": "parameterized_target_journal",
            "template_name": "Target Journal Preference Template",
            "editor_note": "Compact manuscript plus strong reviewer-facing support.",
        },
        "template_variants": [{"template_id": "parameterized_target_journal"}],
    }

    snapshot = build_research_report_snapshot(
        manuscript_snapshot,
        reproducibility_snapshot,
        innovation_snapshot,
        submission_snapshot,
        figure_package_snapshot,
        reviewer_snapshot,
        journal_submission_snapshot=journal_submission_snapshot,
    )

    assert snapshot["dataset_overview"]["records"] == 12
    assert snapshot["dataset_overview"]["year_range_label"] == "2019-2024"
    assert snapshot["dataset_overview"]["scale_profile"]["tier"] == "small"
    assert snapshot["component_counts"]["changed_parameters"] == 1
    assert snapshot["component_counts"]["reviewer_questions"] == 1
    assert snapshot["component_counts"]["journal_template_variants"] == 1
    assert snapshot["component_counts"]["methods_mapping_rows"] == 1
    assert len(snapshot["headline_findings"]) == 6
    assert len(snapshot["result_narrative_templates"]) == 5
    assert len(snapshot["figure_type_narrative_templates"]) == 3
    assert snapshot["figure_type_narrative_templates"][0]["narrative_type"] == "disruption"
    assert snapshot["figure_type_narrative_templates"][1]["narrative_type"] == "trend"
    assert snapshot["figure_type_narrative_templates"][2]["narrative_type"] == "coupling"
    assert snapshot["manuscript_blueprint"][0]["section"] == "Introduction"
    assert snapshot["output_assembly_plan"][0]["type"] == "table"
    assert snapshot["output_assembly_plan"][0]["priority_rank"] == 1
    assert "priority_reason" in snapshot["output_assembly_plan"][0]
    assert "target_section" in snapshot["output_assembly_plan"][0]
    assert "placement_reason" in snapshot["output_assembly_plan"][0]
    assert snapshot["non_default_parameters"][0]["label"] == "Top N"
    assert snapshot["submission_preferences"]["journal_preferences"]["article_format"] == "short_article"
    assert snapshot["submission_preferences"]["recommended_template"]["template_id"] == "parameterized_target_journal"
    assert snapshot["methods_support"]["methods_evidence_map"][0]["step"] == "Dataset Provenance"
    assert snapshot["methods_support"]["methods_writing_pack"]["pipeline_paragraph"] == "Pipeline paragraph."
    assert snapshot["methods_support"]["parameter_change_summary"][0]["module"] == "Network"
    assert "Current setting:" not in " ".join(snapshot["headline_findings"])
    assert snapshot["methods_support"]["submission_methods_note"].startswith("If large-sample export acceleration is used")
    assert snapshot["methods_support"]["execution_policy"]["analysis_record_count"] == 3000


def test_build_research_report_contains_overview_and_appendices():
    research_snapshot = {
        "dataset_overview": {
            "records": 10,
            "year_range_label": "2020-2024",
            "unique_journals": 4,
            "unique_keywords": 20,
            "doi_coverage": 0.8,
            "abstract_coverage": 0.9,
            "author_coverage": 1.0,
        },
        "component_counts": {
            "changed_parameters": 2,
            "recommended_figures": 3,
            "recommended_tables": 2,
            "figure_guides": 3,
            "table_guides": 2,
            "reviewer_questions": 4,
            "journal_template_variants": 5,
            "chapter_target_sections": 4,
            "methods_mapping_rows": 6,
        },
        "headline_findings": ["Finding A", "Finding B"],
        "submission_preferences": {
            "journal_preferences": {
                "main_text_policy": "evidence_dense",
                "supplement_policy": "minimal",
                "review_intensity": "reviewer_friendly",
                "article_format": "rapid_communication",
            },
            "recommended_template": {
                "template_id": "parameterized_target_journal",
                "template_name": "Target Journal Preference Template",
                "editor_note": "Use a compact core manuscript with rapid communication positioning.",
            },
        },
        "result_narrative_templates": ["Narrative A", "Narrative B"],
        "figure_type_narrative_templates": [
            {
                "output_kind": "figure",
                "name": "Publications by Year",
                "narrative_type": "trend",
                "template": "Template A",
                "focus": "Focus A",
            }
        ],
        "manuscript_blueprint": [
            {"section": "Introduction", "purpose": "Purpose A", "suggested_evidence": "Evidence A"},
        ],
        "methods_support": {
            "methods_evidence_map": [
                {
                    "step": "Dataset Provenance",
                    "algorithm_or_rule": "Metadata ingestion",
                    "key_parameters": "records=10",
                    "evidence_output": "Coverage table",
                    "manuscript_use": "Methods / Data source",
                }
            ],
            "methods_writing_pack": {
                "data_source_paragraph": "Data paragraph A.",
                "pipeline_paragraph": "Pipeline paragraph A.",
                "parameter_paragraph": "Parameter paragraph A.",
                "reproducibility_paragraph": "Reproducibility paragraph A.",
                "submission_alignment_paragraph": "Submission paragraph A.",
            },
            "parameter_change_summary": [
                {
                    "module": "Network",
                    "parameter_count": 1,
                    "changed_parameters": "Top N",
                    "current_settings": "Top N=40",
                    "default_settings": "Top N=30",
                    "methods_note": "Report network thresholds in Methods.",
                }
            ],
            "submission_methods_note": "If large-sample export acceleration is used, report the recorded execution policy together with the coupling thresholds.",
            "execution_policy": {
                "lightweight_mode": True,
                "full_record_count": 6200,
                "analysis_record_count": 3000,
                "downsampled": True,
                "scenario_count_requested": 1,
            },
        },
        "non_default_parameters": [
            {"group": "Network", "label": "Top N", "value": 40, "default": 30},
        ],
        "innovation_claims": [
            {"theme": "Theme A", "claim": "Claim A", "evidence": "Evidence A"},
        ],
        "recommended_outputs": {
            "figures": [{"name": "Figure A", "caption": "Caption A"}],
            "tables": [{"name": "Table A", "caption": "Caption B"}],
        },
        "output_assembly_plan": [
            {
                "type": "figure",
                "order": 1,
                "priority_rank": 1,
                "priority_reason": "Compact output ordering prioritizes this figure first.",
                "name": "Figure A",
                "caption": "Caption A",
                "target_section": "Results",
                "placement_reason": "Reason A"
            },
        ],
        "chapter_target_output_plan": [
            {
                "chapter": "Introduction",
                "chapter_note": "Use context-setting outputs first.",
                "recommended_items": [
                    {
                        "chapter_rank": 1,
                        "priority_rank": 2,
                        "output_kind": "figure",
                        "name": "Figure A",
                        "chapter_reason": "Introduction benefits from visual framing.",
                    }
                ],
            }
        ],
        "limitations": ["Limitation A"],
    }

    report = build_research_report(
        research_snapshot,
        manuscript_report="# Case",
        reproducibility_report="# Repro",
        innovation_report="# Innovation",
        submission_report="# Submission",
        figure_package_report="# Figure Package",
        reviewer_report="# Reviewer",
    )

    assert "# One-Click Bibliometrics Research Report" in report
    assert "## 1. Executive Overview" in report
    assert "## 2. Reporting Strategy" in report
    assert "## 4. Target Journal Submission Strategy" in report
    assert "- Article format: rapid_communication" in report
    assert "## 6. Result Narrative Templates" in report
    assert "## 7. Figure-Type Narrative Templates" in report
    assert "## 8. Manuscript Blueprint" in report
    assert "## 9. Methods Evidence Pack" in report
    assert "Data paragraph A." in report
    assert "Submission execution settings are retained in the structured snapshot; use the journal submission guide for the concise large-sample export notice." in report
    assert "| Module | Changed Parameters | Current Settings | Default Settings | Methods Note |" in report
    assert "| Network | Top N | Top N=40 | Top N=30 | Report network thresholds in Methods. |" in report
    assert "## 10. Innovation Claims Snapshot" in report
    assert "## 11. Non-Default Parameters to Report" in report
    assert "## 12. Chapter-Target Output Plan" in report
    assert "### Chapter: Introduction" in report
    assert "## 13. Recommended Output Assembly Plan" in report
    assert "### Section: Results" in report
    assert "Priority rationale: Compact output ordering prioritizes this figure first." in report
    assert "## 14. Stated Limitations" in report
    assert "## Appendix A. Manuscript Case Report" in report
    assert "## Appendix F. Reviewer Response Material Package" in report
    assert "Finding A" in report
    assert "Narrative A" in report
    assert "Narrative type: trend" in report
