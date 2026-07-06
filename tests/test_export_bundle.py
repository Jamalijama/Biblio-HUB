import io
import zipfile

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from modules.export_bundle import (
    build_parameter_change_summary_markdown,
    build_manuscript_case_report,
    build_manuscript_case_snapshot,
    build_manuscript_submission_report,
    build_manuscript_submission_snapshot,
    build_reproducibility_report,
    build_reproducibility_snapshot,
    build_publication_manifest,
    matplotlib_figure_to_bytes,
    PUBLICATION_EXPORT_FORMATS,
    sanitize_filename,
    style_publication_figure,
)
from modules.export_orchestrator import (
    build_manuscript_case_bundle,
    build_methods_package_bundle,
    generate_export_zip,
    get_available_figure_options,
)


def test_sanitize_filename_normalizes_special_characters():
    assert sanitize_filename("Burst Detection: Figure 1/2") == "Burst_Detection_Figure_1_2"
    assert sanitize_filename("...") == "export"


def test_style_publication_figure_applies_white_theme():
    fig = go.Figure(data=[go.Bar(x=["A"], y=[1])])
    styled = style_publication_figure(fig, height=720)

    assert styled.layout.template.layout.paper_bgcolor == "white"
    assert styled.layout.paper_bgcolor == "white"
    assert styled.layout.height == 720
    assert styled.layout.font.family == "Arial"


def test_matplotlib_figure_to_bytes_returns_binary_image():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    payload = matplotlib_figure_to_bytes(fig, export_format="png", dpi=120)

    assert isinstance(payload, bytes)
    assert len(payload) > 0
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"


def test_publication_manifest_records_selection_and_html_companions():
    manifest = build_publication_manifest(
        export_format="svg",
        selected_items={"network_keyword_cooccurrence_static", "overview_top_keywords"},
        include_interactive_html=True,
        skipped_items=["02_network/example.svg: missing kaleido"],
    )

    assert "Image format: svg" in manifest
    assert "selected figures only (2)" in manifest
    assert "Interactive network HTML companions: included" in manifest
    assert "- network_keyword_cooccurrence_static" in manifest
    assert "- 02_network/example.svg: missing kaleido" in manifest


def test_publication_export_formats_include_static_network_targets():
    assert PUBLICATION_EXPORT_FORMATS == ("png", "svg", "pdf")


def test_get_available_figure_options_includes_alluvial_topic_flow_when_keywords_exist():
    df = __import__("pandas").DataFrame([{"Year": 2020, "Journal": "Journal A"}])

    options = get_available_figure_options(df, {"Network": 2, "Citation": 1})

    option_ids = {option["id"] for option in options}
    assert "temporal_alluvial_topic_flow" in option_ids
    assert "temporal_theme_migration_forecast" in option_ids
    assert "temporal_publication_forecast" in option_ids
    assert "temporal_keyword_opportunity_map" in option_ids
    assert "temporal_journal_growth_forecast" in option_ids


def test_get_available_figure_options_includes_affiliation_based_growth_forecasts():
    df = __import__("pandas").DataFrame([{"Year": 2020, "Affiliations": "University of Test, USA"}])

    options = get_available_figure_options(df, {})

    option_ids = {option["id"] for option in options}
    assert "temporal_country_growth_forecast" in option_ids
    assert "temporal_institution_growth_forecast" in option_ids
    assert "temporal_entity_leadership_shift" in option_ids


def test_get_available_figure_options_includes_reference_burst_when_cited_references_exist():
    df = __import__("pandas").DataFrame([{"Year": 2020, "Cited_References": "SMITH J, 2018, SCIENCE"}])

    options = get_available_figure_options(df, {})

    option_ids = {option["id"] for option in options}
    assert "citation_reference_burst_detection" in option_ids


def test_get_available_figure_options_includes_author_production_when_authors_exist():
    df = __import__("pandas").DataFrame([{"Year": 2020, "Authors": "Alice"}])

    options = get_available_figure_options(df, {})

    option_ids = {option["id"] for option in options}
    assert "structure_author_production_over_time" in option_ids


def test_get_available_figure_options_includes_dual_axis_citation_trend():
    df = __import__("pandas").DataFrame([{"Year": 2020, "Times_Cited": 5}])

    options = get_available_figure_options(df, {})

    option_ids = {option["id"] for option in options}
    assert "citation_publication_citation_dual_axis" in option_ids


def test_generate_export_zip_includes_top_entity_tables():
    df = __import__("pandas").DataFrame(
        [
            {
                "Journal": "Journal A",
                "Affiliations": "[Alice] University of Test, City, USA; [Bob] Institute of Metrics, Beijing, PR CHINA",
            },
            {
                "Journal": "Journal A",
                "Affiliations": "[Carol] Department of Data, University of Test, City, USA",
            },
            {
                "Journal": "Journal B",
                "Affiliations": "[Dan] Institute of Metrics, Beijing, PR CHINA",
            },
        ]
    )

    payload = generate_export_zip(df, {"Network": 2}, {("Network", "Citation"): 1})

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = set(zf.namelist())
        top_countries = zf.read("08_Raw_Data_Tables/top_countries.csv").decode("utf-8-sig")
        top_institutions = zf.read("08_Raw_Data_Tables/top_institutions.csv").decode("utf-8-sig")
        top_journals = zf.read("08_Raw_Data_Tables/top_journals_table.csv").decode("utf-8-sig")

    assert "08_Raw_Data_Tables/top_countries.csv" in names
    assert "08_Raw_Data_Tables/top_institutions.csv" in names
    assert "08_Raw_Data_Tables/top_journals_table.csv" in names
    assert "United States" in top_countries
    assert "China" in top_countries
    assert "University of Test" in top_institutions
    assert "Institute of Metrics" in top_institutions
    assert "Journal A" in top_journals


def test_manuscript_case_snapshot_summarizes_core_dataset_signals():
    keyword_freq = {"Network": 4, "Bibliometrics": 3}
    cooccurrence = {("Network", "Bibliometrics"): 2}
    df = __import__("pandas").DataFrame(
        [
            {
                "Title": "Paper A",
                "Year": 2020,
                "Journal": "Journal A",
                "DOI": "10.1/a",
                "Abstract": "Abstract A",
                "Authors": "Alice; Bob",
            },
            {
                "Title": "Paper B",
                "Year": 2021,
                "Journal": "Journal A",
                "DOI": "",
                "Abstract": "Abstract B",
                "Authors": "Alice",
            },
            {
                "Title": "Paper C",
                "Year": 2021,
                "Journal": "Journal B",
                "DOI": "10.1/c",
                "Abstract": "",
                "Authors": "Carol",
            },
        ]
    )

    snapshot = build_manuscript_case_snapshot(
        df,
        keyword_freq=keyword_freq,
        cooccurrence=cooccurrence,
        dedup_report={"original": 4, "removed": 1, "final": 3},
        top_n=5,
    )

    assert snapshot["records"] == 3
    assert snapshot["year_range"] == [2020, 2021]
    assert snapshot["unique_journals"] == 2
    assert snapshot["unique_keywords"] == 2
    assert snapshot["top_keywords"][0]["label"] == "Network"
    assert snapshot["top_cooccurrence_pairs"][0]["pair"] == ["Network", "Bibliometrics"]


def test_manuscript_case_report_includes_methods_and_findings_sections():
    keyword_freq = {"Network": 4, "Bibliometrics": 3}
    cooccurrence = {("Network", "Bibliometrics"): 2}
    df = __import__("pandas").DataFrame(
        [
            {
                "Title": "Paper A",
                "Year": 2020,
                "Journal": "Journal A",
                "DOI": "10.1/a",
                "Abstract": "Abstract A",
                "Authors": "Alice; Bob",
            }
        ]
    )

    report = build_manuscript_case_report(
        df,
        keyword_freq=keyword_freq,
        cooccurrence=cooccurrence,
        dedup_report={"original": 1, "removed": 0, "final": 1},
        top_n=5,
    )

    assert "## 1. Dataset Overview" in report
    assert "## 2. Data Processing Notes" in report
    assert "Network <-> Bibliometrics: 2" in report
    assert "Duplicate merging strategy" in report


def test_manuscript_submission_snapshot_collects_abstract_highlights_and_tables():
    keyword_freq = {"Network": 4, "Bibliometrics": 3}
    cooccurrence = {("Network", "Bibliometrics"): 2}
    df = __import__("pandas").DataFrame(
        [
            {
                "Title": "Paper A",
                "Year": 2020,
                "Journal": "Journal A",
                "DOI": "10.1/a",
                "Abstract": "Abstract A",
                "Authors": "Alice; Bob",
            },
            {
                "Title": "Paper B",
                "Year": 2021,
                "Journal": "Journal B",
                "DOI": "",
                "Abstract": "Abstract B",
                "Authors": "Carol",
            },
        ]
    )

    snapshot = build_manuscript_submission_snapshot(
        df,
        keyword_freq=keyword_freq,
        cooccurrence=cooccurrence,
        dedup_report={"original": 2, "removed": 0, "final": 2},
        top_n=5,
    )

    assert snapshot["case_snapshot"]["records"] == 2
    assert "background" in snapshot["structured_abstract"]
    assert len(snapshot["result_highlights"]) == 4
    assert len(snapshot["recommended_tables"]) == 4
    assert "top_journals" in snapshot["export_tables"]


def test_manuscript_submission_report_contains_submission_ready_sections():
    keyword_freq = {"Network": 4, "Bibliometrics": 3}
    cooccurrence = {("Network", "Bibliometrics"): 2}
    df = __import__("pandas").DataFrame(
        [
            {
                "Title": "Paper A",
                "Year": 2020,
                "Journal": "Journal A",
                "DOI": "10.1/a",
                "Abstract": "Abstract A",
                "Authors": "Alice; Bob",
            }
        ]
    )

    report = build_manuscript_submission_report(
        df,
        keyword_freq=keyword_freq,
        cooccurrence=cooccurrence,
        dedup_report={"original": 1, "removed": 0, "final": 1},
        top_n=5,
    )

    assert "# Manuscript Submission Case Package" in report
    assert "## 2. Structured Abstract Draft" in report
    assert "## 3. Submission Highlights" in report
    assert "## 4. Recommended Tables" in report
    assert "## 5. Paragraph Starters" in report
    assert "## 6. Package Assembly Reminder" in report


def test_build_manuscript_case_bundle_contains_expected_core_files():
    keyword_freq = {"Network": 4, "Bibliometrics": 3}
    cooccurrence = {("Network", "Bibliometrics"): 2}
    df = __import__("pandas").DataFrame(
        [
            {
                "Title": "Paper A",
                "Year": 2020,
                "Journal": "Journal A",
                "DOI": "10.1/a",
                "Abstract": "Abstract A",
                "Authors": "Alice; Bob",
            }
        ]
    )

    manuscript_snapshot = build_manuscript_case_snapshot(
        df,
        keyword_freq=keyword_freq,
        cooccurrence=cooccurrence,
        dedup_report={"original": 1, "removed": 0, "final": 1},
        top_n=5,
    )
    manuscript_report = build_manuscript_case_report(
        df,
        keyword_freq=keyword_freq,
        cooccurrence=cooccurrence,
        dedup_report={"original": 1, "removed": 0, "final": 1},
        top_n=5,
    )
    submission_snapshot = build_manuscript_submission_snapshot(
        df,
        keyword_freq=keyword_freq,
        cooccurrence=cooccurrence,
        dedup_report={"original": 1, "removed": 0, "final": 1},
        top_n=5,
    )
    submission_report = build_manuscript_submission_report(
        df,
        keyword_freq=keyword_freq,
        cooccurrence=cooccurrence,
        dedup_report={"original": 1, "removed": 0, "final": 1},
        top_n=5,
    )

    payload = build_manuscript_case_bundle(
        manuscript_report=manuscript_report,
        manuscript_snapshot=manuscript_snapshot,
        manuscript_submission_report=submission_report,
        manuscript_submission_snapshot=submission_snapshot,
    )

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        assert "01_report/Biblio-HUB_manuscript_case_report.md" in zf.namelist()
        assert "02_snapshot/Biblio-HUB_submission_case_snapshot.json" in zf.namelist()
        assert "04_templates/structured_abstract.md" in zf.namelist()


def test_build_methods_package_bundle_contains_methods_summary_outputs():
    df = __import__("pandas").DataFrame(
        [
            {
                "Title": "Paper A",
                "Year": 2020,
                "Journal": "Journal A",
                "DOI": "10.1/a",
                "Abstract": "Abstract A",
                "Authors": "Alice; Bob",
            }
        ]
    )
    snapshot = build_reproducibility_snapshot(
        df,
        analysis_parameters=[
            {"key": "vos_topn", "label": "Keyword Network Top N", "value": 40, "default": 30, "group": "Network"},
            {"key": "export_lightweight_mode", "label": "Export Lightweight Mode", "value": True, "default": False, "group": "Export"},
        ],
        dedup_report={"original": 1, "removed": 0, "final": 1},
        keyword_freq={"Network": 4},
        cooccurrence={("Network", "Citation"): 2},
        execution_policy={
            "lightweight_mode": True,
            "full_record_count": 6200,
            "analysis_record_count": 3000,
            "downsampled": True,
            "scenario_count_requested": 1,
        },
    )

    payload = build_methods_package_bundle(snapshot)

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        assert "methods_package.md" in zf.namelist()
        assert "methods_parameter_change_summary.md" in zf.namelist()
        assert "methods_mapping.csv" in zf.namelist()


def test_reproducibility_snapshot_tracks_field_coverage_and_parameter_changes():
    df = __import__("pandas").DataFrame(
        [
            {
                "Title": "Paper A",
                "Abstract": "Abstract A",
                "Year": 2020,
                "Journal": "Journal A",
                "DOI": "10.1/a",
                "Authors": "Alice; Bob",
                "Affiliations": "Org A",
            },
            {
                "Title": "Paper B",
                "Abstract": "",
                "Year": 2021,
                "Journal": "Journal B",
                "DOI": "",
                "Authors": "Carol",
                "Affiliations": "",
            },
        ]
    )
    analysis_parameters = [
        {"key": "bc_topn", "label": "Bibliographic Coupling Top Papers", "value": 40, "default": 30, "group": "Citation"},
        {"key": "vos_topn", "label": "Keyword Network Top N", "value": 40, "default": 30, "group": "Network"},
        {"key": "burst_topn", "label": "Burst Detection Keyword Count", "value": 20, "default": 20, "group": "Temporal"},
        {"key": "bundle_format", "label": "Publication Figure Format", "value": "svg", "default": "png", "group": "Export"},
    ]

    snapshot = build_reproducibility_snapshot(
        df,
        analysis_parameters=analysis_parameters,
        dedup_report={"original": 3, "removed": 1, "final": 2},
        keyword_freq={"Network": 3, "Citation": 2},
        cooccurrence={("Network", "Citation"): 2},
        journal_preferences={
            "main_text_policy": "compact",
            "supplement_policy": "supplement_heavy",
            "review_intensity": "revision_ready",
            "article_format": "short_article",
        },
        recommended_template={
            "template_id": "parameterized_target_journal",
            "template_name": "Target Journal Preference Template",
            "editor_note": "Compact manuscript with supplementary-heavy support.",
        },
        execution_policy={
            "lightweight_mode": True,
            "full_record_count": 6200,
            "analysis_record_count": 3000,
            "downsampled": True,
            "scenario_count_requested": 1,
        },
    )

    assert snapshot["records"] == 2
    assert snapshot["deduplication"] == {"original": 3, "removed": 1, "final": 2}
    assert snapshot["field_coverage"]["Title"] == 1.0
    assert snapshot["field_coverage"]["Abstract"] == 0.5
    assert snapshot["field_coverage"]["DOI"] == 0.5
    assert snapshot["keyword_statistics"] == {"unique_keywords": 2, "cooccurrence_pairs": 1}
    assert snapshot["analysis_parameters"][0]["changed"] is True
    assert snapshot["analysis_parameters"][1]["changed"] is True
    assert snapshot["analysis_parameters"][2]["changed"] is False
    assert len(snapshot["parameter_change_summary"]) == 3
    assert snapshot["parameter_change_summary"][0]["module"] == "Citation"
    assert snapshot["parameter_change_summary"][1]["module"] == "Network"
    assert snapshot["parameter_change_summary"][2]["module"] == "Export"
    assert "Keyword Network Top N=40" in snapshot["parameter_change_summary"][1]["current_settings"]
    assert snapshot["submission_preferences"]["journal_preferences"]["main_text_policy"] == "compact"
    assert snapshot["submission_preferences"]["recommended_template"]["template_id"] == "parameterized_target_journal"
    assert snapshot["execution_policy"]["lightweight_mode"] is True
    assert len(snapshot["methods_evidence_map"]) >= 5
    assert len(snapshot["reporting_artifact_index"]) >= 7
    assert any(item["artifact"] == "methods_parameter_change_summary.csv" for item in snapshot["reporting_artifact_index"])
    assert any(item["artifact"] == "methods_parameter_change_summary.md" for item in snapshot["reporting_artifact_index"])
    assert snapshot["reporting_artifact_index"][0]["artifact"] == "Biblio-HUB_reproducibility_report.md"
    assert "data_source_paragraph" in snapshot["methods_writing_pack"]
    assert "parameter_paragraph" in snapshot["methods_writing_pack"]
    assert "lightweight mode, 3000/6200 records, 1 scenario(s), downsampled." in snapshot["methods_writing_pack"]["pipeline_paragraph"]
    assert any("Kleinberg burst detection" in item for item in snapshot["algorithm_profile"])


def test_reproducibility_report_lists_metadata_parameters_and_reporting_items():
    df = __import__("pandas").DataFrame(
        [
            {
                "Title": "Paper A",
                "Abstract": "Abstract A",
                "Year": 2020,
                "Journal": "Journal A",
                "DOI": "10.1/a",
                "Authors": "Alice; Bob",
            }
        ]
    )
    analysis_parameters = [
        {"key": "vos_topn", "label": "Keyword Network Top N", "value": 40, "default": 30, "group": "Network"},
        {"key": "bundle_format", "label": "Publication Figure Format", "value": "svg", "default": "png", "group": "Export"},
    ]

    report = build_reproducibility_report(
        df,
        analysis_parameters=analysis_parameters,
        dedup_report={"original": 1, "removed": 0, "final": 1},
        keyword_freq={"Network": 3},
        cooccurrence={("Network", "Citation"): 1},
        journal_preferences={
            "main_text_policy": "evidence_dense",
            "supplement_policy": "minimal",
            "review_intensity": "reviewer_friendly",
            "article_format": "rapid_communication",
        },
        recommended_template={
            "template_id": "parameterized_target_journal",
            "template_name": "Target Journal Preference Template",
            "editor_note": "Rapid communication with lean supplementary files.",
        },
        execution_policy={
            "lightweight_mode": True,
            "full_record_count": 6200,
            "analysis_record_count": 3000,
            "downsampled": True,
            "scenario_count_requested": 1,
        },
    )

    assert "## 1. Dataset Provenance Summary" in report
    assert "- Title: 100.0%" in report
    assert "- Network / Keyword Network Top N: 40 (default=30) [modified]" in report
    assert "- Export / Publication Figure Format: svg (default=png) [modified]" in report
    assert "## 4. Methods Parameter Change Summary" in report
    assert "| Module | Changed Parameters | Current Settings | Default Settings | Methods Note |" in report
    assert "| Network | Keyword Network Top N | Keyword Network Top N=40 | Keyword Network Top N=30 |" in report
    assert "## 5. Submission Preference Record" in report
    assert "- Main text policy: evidence_dense" in report
    assert "- Recommended template: parameterized_target_journal / Target Journal Preference Template" in report
    assert "## 6. Parameter-Algorithm-Output Mapping" in report
    assert "| Step | Algorithm or Rule | Key Parameters | Evidence Output | Manuscript Use |" in report
    assert "## 7. Methods Writing Pack" in report
    assert "- Pipeline paragraph:" in report
    assert "lightweight mode, 3000/6200 records, 1 scenario(s), downsampled." in report
    assert "## 8. Algorithm Profile" in report
    assert "## 9. Reporting Artifact Index" in report
    assert "| Artifact | Format | Purpose | Manuscript Use | Trace Source |" in report
    assert "methods_parameter_change_summary.csv" in report
    assert "methods_parameter_change_summary.md" in report
    assert "methods_artifact_index.csv" in report
    assert "## 11. Minimum Reporting Checklist" in report


def test_reproducibility_snapshot_records_lightweight_export_policy_changes():
    df = __import__("pandas").DataFrame(
        [
            {"Title": "Paper A", "Year": 2020, "Journal": "Journal A"},
            {"Title": "Paper B", "Year": 2021, "Journal": "Journal B"},
        ]
    )
    analysis_parameters = [
        {"key": "export_lightweight_mode", "label": "Export Lightweight Mode", "value": True, "default": False, "group": "Export"},
        {"key": "export_robustness_scenario_count", "label": "Robustness Scenario Count", "value": 1, "default": 9, "group": "Export"},
        {"key": "export_robustness_sample_size", "label": "Robustness Analysis Sample Size", "value": 3000, "default": 6200, "group": "Export"},
    ]

    snapshot = build_reproducibility_snapshot(
        df,
        analysis_parameters=analysis_parameters,
        dedup_report={"original": 6200, "removed": 0, "final": 6200},
        execution_policy={
            "lightweight_mode": True,
            "full_record_count": 6200,
            "analysis_record_count": 3000,
            "downsampled": True,
            "scenario_count_requested": 1,
        },
    )

    assert len(snapshot["parameter_change_summary"]) == 1
    assert snapshot["parameter_change_summary"][0]["module"] == "Export"
    assert "Export Lightweight Mode=True" in snapshot["parameter_change_summary"][0]["current_settings"]
    assert "Robustness Scenario Count=1" in snapshot["parameter_change_summary"][0]["current_settings"]
    assert "Robustness Analysis Sample Size=3000" in snapshot["parameter_change_summary"][0]["current_settings"]
    assert "lightweight mode, 3000/6200 records, 1 scenario(s), downsampled." in snapshot["methods_writing_pack"]["pipeline_paragraph"]


def test_build_parameter_change_summary_markdown_returns_paste_ready_table():
    markdown = build_parameter_change_summary_markdown(
        [
            {
                "module": "Citation",
                "parameter_count": 1,
                "changed_parameters": "Bibliographic Coupling Top Papers",
                "current_settings": "Bibliographic Coupling Top Papers=40",
                "default_settings": "Bibliographic Coupling Top Papers=30",
                "methods_note": "Report citation-related thresholds.",
            }
        ]
    )

    assert "# Methods Parameter Change Summary" in markdown
    assert "| Module | Changed Parameters | Current Settings | Default Settings | Methods Note |" in markdown
    assert "| Citation | Bibliographic Coupling Top Papers | Bibliographic Coupling Top Papers=40 | Bibliographic Coupling Top Papers=30 | Report citation-related thresholds. |" in markdown
