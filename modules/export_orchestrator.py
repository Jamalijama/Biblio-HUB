import io
import json
import zipfile

import pandas as pd

from modules.export_bundle import build_parameter_change_summary_markdown
from modules.experiment_framework import rank_disruption_extremes
from modules.entity_analysis import (
    build_top_country_table,
    build_top_institution_table,
    build_top_journal_table,
)


def _json_bytes(payload):
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def _csv_bytes(rows):
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


def _text_bytes(content):
    return str(content).encode("utf-8")


def _most_common_items(values, limit):
    if hasattr(values, "most_common"):
        return values.most_common(limit)
    return sorted(values.items(), key=lambda item: item[1], reverse=True)[:limit]


def generate_export_zip(df, keyword_freq, cooccurrence, zip_file=None):
    """
    Internal logic to add tables to a zip file.
    If zip_file is provided, it adds to it. Otherwise creates a new buffer.
    """
    if zip_file is None:
        zip_buffer = io.BytesIO()
        own_zip = True
        zf = zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False)
    else:
        zf = zip_file
        own_zip = False

    zf.writestr("08_Raw_Data_Tables/processed_dataset.csv", df.to_csv(index=False).encode("utf-8-sig"))
    kw_df = pd.DataFrame(_most_common_items(keyword_freq, 500), columns=["Keyword", "Frequency"])
    zf.writestr("08_Raw_Data_Tables/top_500_keywords.csv", kw_df.to_csv(index=False).encode("utf-8-sig"))
    top_country_df = build_top_country_table(df, top_n=50)
    if not top_country_df.empty:
        zf.writestr("08_Raw_Data_Tables/top_countries.csv", top_country_df.to_csv(index=False).encode("utf-8-sig"))
    top_institution_df = build_top_institution_table(df, top_n=50)
    if not top_institution_df.empty:
        zf.writestr("08_Raw_Data_Tables/top_institutions.csv", top_institution_df.to_csv(index=False).encode("utf-8-sig"))
    top_journal_df = build_top_journal_table(df, top_n=50)
    if not top_journal_df.empty:
        zf.writestr("08_Raw_Data_Tables/top_journals_table.csv", top_journal_df.to_csv(index=False).encode("utf-8-sig"))

    vos_nodes = [{"id": kw, "label": kw, "weight": freq} for kw, freq in _most_common_items(keyword_freq, 100)]
    vos_node_ids = {node["id"] for node in vos_nodes}
    vos_edges = []
    for (k1, k2), weight in _most_common_items(cooccurrence, 500):
        if k1 in vos_node_ids and k2 in vos_node_ids:
            vos_edges.append({"source": k1, "target": k2, "weight": weight})
    zf.writestr(
        "09_Interoperability/vosviewer_network.json",
        _json_bytes({"network": {"items": vos_nodes, "links": vos_edges}}),
    )

    if own_zip:
        zf.close()
        return zip_buffer.getvalue()
    return None


def generate_master_export(
    df,
    keywords_list,
    keyword_freq,
    cooccurrence,
    export_format="png",
    network_export_modes=None,
    analysis_parameters=None,
):
    """
    One-click export for ALL figures and ALL data tables.
    """
    from modules.figure_export_bundle import generate_all_figure_bundle
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as master_zip:
                            
        generate_all_figure_bundle(
            df, keywords_list, keyword_freq, cooccurrence,
            export_format=export_format,
            bundle_zip=master_zip,
            network_export_modes=network_export_modes,
            analysis_parameters=analysis_parameters,
        )
                                
        generate_export_zip(df, keyword_freq, cooccurrence, zip_file=master_zip)
        
    return zip_buffer.getvalue()


def get_available_figure_options(df, keyword_freq):
    options = []

    def add_option(item_id, label):
        options.append({"id": item_id, "label": label})

    if "Year" in df.columns:
        add_option("overview_publications_by_year", "Overview / Publications by Year")
    if "Journal" in df.columns:
        add_option("overview_top_journals", "Overview / Top Journals")
    if keyword_freq:
        add_option("overview_keyword_wordcloud", "Overview / Keyword Word Cloud")
        add_option("overview_top_keywords", "Overview / Top Keywords")
        add_option("network_keyword_cooccurrence_static", "Network / Keyword Co-occurrence (Static)")
        add_option("structure_keyword_circular_cluster", "Structure / Keyword Circular Cluster Map")
        add_option("temporal_keyword_timeline", "Temporal / Keyword Evolution")
        add_option("temporal_burst_detection", "Temporal / Burst Detection")
        add_option("temporal_alluvial_topic_flow", "Temporal / Alluvial Topic Flow")
        if "Year" in df.columns:
            add_option("temporal_theme_migration_forecast", "Temporal / Theme Cluster Migration Forecast")
        add_option("structure_cooccurrence_matrix", "Structure / Co-occurrence Matrix")
        add_option("structure_thematic_map", "Structure / Thematic Map")
    if "Year" in df.columns:
        add_option("structure_annual_growth_rate", "Structure / Annual Growth Rate")
        add_option("temporal_publication_forecast", "Temporal / Publication Forecast")
        add_option("temporal_journal_growth_forecast", "Temporal / Journal Growth Forecast")
    if "Year" in df.columns and keyword_freq:
        add_option("temporal_keyword_opportunity_map", "Temporal / Keyword Opportunity Map")
    if "Year" in df.columns and "Affiliations" in df.columns:
        add_option("temporal_country_growth_forecast", "Temporal / Country Growth Forecast")
        add_option("temporal_institution_growth_forecast", "Temporal / Institution Growth Forecast")
        add_option("temporal_entity_leadership_shift", "Temporal / Country / Institution Leadership Shift")
    if "Authors" in df.columns:
        add_option("structure_author_production_over_time", "Structure / Authors' Production Over Time")
        add_option("structure_three_field_plot", "Structure / Three-Field Plot")
        add_option("structure_lotkas_law", "Structure / Lotka's Law")
    if keyword_freq and "Journal" in df.columns:
        add_option("network_keyword_journal_static", "Network / Keyword-Journal Association (Static)")
    if "Authors" in df.columns:
        add_option("network_coauthorship_static", "Network / Co-authorship Network (Static)")
    if "Affiliations" in df.columns:
        add_option("network_institution_collaboration_static", "Network / Institutional Collaboration (Static)")
        add_option("network_country_collaboration_static", "Network / International Collaboration (Static)")
        add_option("network_country_collaboration_chord", "Network / Country Collaboration Chord Map")
        add_option("network_country_impact_quadrant", "Network / Country Publication-Citation Impact Quadrant")
        add_option("network_country_publications_bar", "Network / Top Countries by Publications")
    if "Language" in df.columns:
        add_option("structure_language_distribution_pie", "Structure / Language Distribution (Pie)")
        add_option("structure_language_distribution_bar", "Structure / Language Distribution (Bar)")
    if "Times_Cited" in df.columns:
        add_option("citation_distribution", "Citation / Citation Distribution")
        if "Year" in df.columns:
            add_option("citation_publication_citation_dual_axis", "Citation / Annual Publications + Average Citations")
            add_option("citation_by_year", "Citation / Average Citations by Year")
    if "WoS_Categories" in df.columns or "Research_Areas" in df.columns:
        add_option("citation_subject_categories", "Citation / Subject Categories")
    if "DocType" in df.columns:
        add_option("citation_document_types", "Citation / Document Types")
    if "Funding" in df.columns:
        add_option("citation_funding_agencies", "Citation / Funding Agencies")
    if "Cited_References" in df.columns:
        add_option("network_bibliographic_coupling_static", "Network / Bibliographic Coupling (Static)")
        add_option("network_reference_cocitation_static", "Network / Co-citation Network (Static)")
        add_option("network_author_cocitation_static", "Network / Author Co-citation Network (Static)")
        add_option("network_journal_cocitation_static", "Network / Journal Co-citation Network (Static)")
        add_option("citation_reference_year_distribution", "Citation / Reference Year Distribution")
        add_option("citation_rpys_spectroscopy", "Citation / Reference Publication Year Spectroscopy")
        add_option("citation_reference_burst_detection", "Citation / Reference Burst Detection")
        add_option("citation_references_per_paper", "Citation / References per Paper")
        add_option("innovation_structural_hole_profile", "Innovation / Structural-Hole Brokerage Profile")
        add_option("innovation_brokerage_robustness", "Innovation / Brokerage Robustness Summary")
    if "Publisher" in df.columns:
        add_option("citation_publisher_analysis", "Citation / Publisher Analysis")
    return options


def group_figure_options(options):
    grouped = {}
    for option in options:
        label = option["label"]
        if " / " in label:
            group_name, item_label = label.split(" / ", 1)
        else:
            group_name, item_label = "Other", label
        grouped.setdefault(group_name, []).append(
            {"id": option["id"], "label": label, "item_label": item_label}
        )
    return grouped


def build_manuscript_case_bundle(
    manuscript_report,
    manuscript_snapshot,
    manuscript_submission_report,
    manuscript_submission_snapshot,
):
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as case_zip:
        case_zip.writestr("01_report/Biblio-HUB_manuscript_case_report.md", _text_bytes(manuscript_report))
        case_zip.writestr("01_report/Biblio-HUB_submission_case_package.md", _text_bytes(manuscript_submission_report))
        case_zip.writestr("02_snapshot/Biblio-HUB_manuscript_case_snapshot.json", _json_bytes(manuscript_snapshot))
        case_zip.writestr("02_snapshot/Biblio-HUB_submission_case_snapshot.json", _json_bytes(manuscript_submission_snapshot))
        case_zip.writestr("03_tables/top_journals.csv", _csv_bytes(manuscript_submission_snapshot["export_tables"]["top_journals"]))
        case_zip.writestr("03_tables/top_keywords.csv", _csv_bytes(manuscript_submission_snapshot["export_tables"]["top_keywords"]))
        case_zip.writestr("03_tables/top_keyword_cooccurrence_pairs.csv", _csv_bytes(manuscript_submission_snapshot["export_tables"]["top_cooccurrence_pairs"]))

        abstract_lines = ["# Structured Abstract Draft", ""]
        abstract_lines.extend(
            [
                f"- Background: {manuscript_submission_snapshot['structured_abstract']['background']}",
                f"- Objective: {manuscript_submission_snapshot['structured_abstract']['objective']}",
                f"- Methods: {manuscript_submission_snapshot['structured_abstract']['methods']}",
                f"- Results: {manuscript_submission_snapshot['structured_abstract']['results']}",
                f"- Conclusion: {manuscript_submission_snapshot['structured_abstract']['conclusion']}",
            ]
        )
        case_zip.writestr("04_templates/structured_abstract.md", _text_bytes("\n".join(abstract_lines)))
        case_zip.writestr(
            "04_templates/submission_highlights.md",
            _text_bytes("\n".join(["# Submission Highlights", ""] + [f"- {item}" for item in manuscript_submission_snapshot["result_highlights"]])),
        )
    return bundle.getvalue()


def build_methods_package_bundle(reproducibility_snapshot):
    methods_writing_pack = reproducibility_snapshot.get("methods_writing_pack", {})
    methods_evidence_map = reproducibility_snapshot.get("methods_evidence_map", [])
    parameter_change_summary = reproducibility_snapshot.get("parameter_change_summary", [])
    reporting_artifact_index = reproducibility_snapshot.get("reporting_artifact_index", [])

    md_content = "# Methods Writing Pack\n\n"
    for key, value in methods_writing_pack.items():
        md_content += f"## {key.replace('_', ' ').title()}\n\n{value}\n\n"

    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        zip_file.writestr("methods_package.md", _text_bytes(md_content))
        if methods_evidence_map:
            zip_file.writestr("methods_mapping.csv", _csv_bytes(methods_evidence_map))
        if parameter_change_summary:
            zip_file.writestr("methods_parameter_change_summary.csv", _csv_bytes(parameter_change_summary))
        zip_file.writestr(
            "methods_parameter_change_summary.md",
            _text_bytes(build_parameter_change_summary_markdown(parameter_change_summary)),
        )
        if reporting_artifact_index:
            zip_file.writestr("methods_artifact_index.csv", _csv_bytes(reporting_artifact_index))
    return bundle.getvalue()


def build_submission_result_bundle(
    submission_report,
    innovation_report,
    robustness_report,
    baseline_comparison_report,
    submission_snapshot,
    innovation_snapshot,
    bc_pairs_report,
    bc_top_papers_report,
    robustness_snapshot,
    baseline_comparison_snapshot,
    df_di_report,
):
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as submission_zip:
        submission_zip.writestr("01_submission_report/manuscript_submission_result_package.md", _text_bytes(submission_report))
        submission_zip.writestr("01_submission_report/innovation_metrics_report.md", _text_bytes(innovation_report))
        submission_zip.writestr("01_submission_report/brokerage_robustness_report.md", _text_bytes(robustness_report))
        submission_zip.writestr("01_submission_report/brokerage_baseline_comparison_report.md", _text_bytes(baseline_comparison_report))
        submission_zip.writestr("02_snapshot/submission_result_snapshot.json", _json_bytes(submission_snapshot))
        submission_zip.writestr("02_snapshot/innovation_metrics_snapshot.json", _json_bytes(innovation_snapshot))
        submission_zip.writestr("03_tables/bibliographic_coupling_pairs.csv", _csv_bytes(bc_pairs_report))
        submission_zip.writestr("03_tables/bibliographic_coupling_top_papers.csv", _csv_bytes(bc_top_papers_report))
        submission_zip.writestr("03_tables/main_results_table.csv", _csv_bytes(submission_snapshot["main_results_table"]))
        submission_zip.writestr("03_tables/supplementary_table_index.csv", _csv_bytes(submission_snapshot["supplementary_table_index"]))
        submission_zip.writestr("03_tables/figure_table_crosswalk.csv", _csv_bytes(submission_snapshot["figure_table_crosswalk"]))
        submission_zip.writestr("03_tables/preferred_output_sequence.csv", _csv_bytes(submission_snapshot.get("recommended_output_sequence", [])))

        chapter_target_rows = []
        for chapter in submission_snapshot.get("chapter_target_output_plan", []):
            for item in chapter.get("recommended_items", []):
                chapter_target_rows.append(
                    {
                        "chapter": chapter["chapter"],
                        "chapter_note": chapter["chapter_note"],
                        "chapter_rank": item.get("chapter_rank", 0),
                        "priority_rank": item.get("priority_rank", 0),
                        "output_kind": item.get("output_kind", ""),
                        "name": item.get("name", ""),
                        "chapter_reason": item.get("chapter_reason", ""),
                    }
                )
        submission_zip.writestr("03_tables/chapter_target_output_plan.csv", _csv_bytes(chapter_target_rows))
        submission_zip.writestr(
            "03_tables/top_structural_hole_brokers.csv",
            _csv_bytes(submission_snapshot.get("innovation_metrics", {}).get("structural_hole", {}).get("top_brokers", [])),
        )
        submission_zip.writestr("03_tables/brokerage_robustness_scenarios.csv", _csv_bytes(robustness_snapshot.get("scenarios", [])))
        submission_zip.writestr("03_tables/stable_broker_candidates.csv", _csv_bytes(robustness_snapshot.get("stable_brokers", [])))
        submission_zip.writestr("03_tables/brokerage_baseline_alignment.csv", _csv_bytes(baseline_comparison_snapshot.get("baseline_comparisons", [])))
        submission_zip.writestr("03_tables/brokerage_node_comparison.csv", _csv_bytes(baseline_comparison_snapshot.get("node_comparison_table", [])))

        if df_di_report is not None and "Disruption_Index" in df_di_report.columns:
            export_cols = [
                column
                for column in [
                    "Title",
                    "Authors",
                    "Journal",
                    "Year",
                    "Disruption_Index",
                    "Support",
                    "DI_nd",
                    "DI_nc",
                    "DI_na",
                    "Internal_References",
                    "Internal_Citers",
                ]
                if column in df_di_report.columns
            ]
            submission_zip.writestr("03_tables/disruption_index_scores.csv", df_di_report[export_cols].to_csv(index=False).encode("utf-8-sig"))
            submission_zip.writestr("03_tables/top_disruptive_papers.csv", rank_disruption_extremes(df_di_report, kind="disruptive", top_n=10)[export_cols].to_csv(index=False).encode("utf-8-sig"))
            submission_zip.writestr("03_tables/top_consolidating_papers.csv", rank_disruption_extremes(df_di_report, kind="consolidating", top_n=10)[export_cols].to_csv(index=False).encode("utf-8-sig"))

        caption_lines = ["# Figure and Table Caption Templates", "", "## Figure Captions"]
        caption_lines.extend(f"- {item['name']}: {item['caption']}" for item in submission_snapshot["recommended_figures"])
        caption_lines.extend(["", "## Table Captions"])
        caption_lines.extend(f"- {item['name']}: {item['caption']}" for item in submission_snapshot["recommended_tables"])
        submission_zip.writestr("04_templates/caption_templates.md", _text_bytes("\n".join(caption_lines)))

        table_assembly_lines = ["# Submission Table Assembly Guide", ""]
        table_assembly_lines.extend(f"- {item['table_id']} {item['name']}: {item['suggested_use']}" for item in submission_snapshot["supplementary_table_index"])
        table_assembly_lines.extend(["", "# Figure-Table Crosswalk", ""])
        table_assembly_lines.extend(f"- {item['figure']} -> {item['supporting_table']} ({item['manuscript_role']})" for item in submission_snapshot["figure_table_crosswalk"])
        submission_zip.writestr("04_templates/table_assembly_guide.md", _text_bytes("\n".join(table_assembly_lines)))

        output_sequence_lines = ["# Preferred Output Sequence", ""]
        output_sequence_lines.extend(
            f"- P{item.get('priority_rank', 0)} {item['output_kind'].title()} / {item['name']}: {item.get('priority_reason', '')}"
            for item in submission_snapshot.get("recommended_output_sequence", [])
        )
        submission_zip.writestr("04_templates/preferred_output_sequence.md", _text_bytes("\n".join(output_sequence_lines)))

        chapter_target_lines = ["# Chapter-Target Output Plan", ""]
        for chapter in submission_snapshot.get("chapter_target_output_plan", []):
            chapter_target_lines.extend([f"## {chapter['chapter']}", f"- Chapter note: {chapter['chapter_note']}"])
            chapter_target_lines.extend(
                f"- C{item.get('chapter_rank', 0)} / P{item.get('priority_rank', 0)} {item['output_kind'].title()} / {item['name']}: {item.get('chapter_reason', '')}"
                for item in chapter.get("recommended_items", [])
            )
            chapter_target_lines.append("")
        submission_zip.writestr("04_templates/chapter_target_output_plan.md", _text_bytes("\n".join(chapter_target_lines)))

        submission_strategy_lines = ["# Target Journal Submission Strategy", ""]
        submission_strategy_lines.extend(
            [
                f"- Main text policy: {submission_snapshot['submission_preferences'].get('main_text_policy', 'balanced')}",
                f"- Supplement policy: {submission_snapshot['submission_preferences'].get('supplement_policy', 'standard')}",
                f"- Review intensity: {submission_snapshot['submission_preferences'].get('review_intensity', 'standard')}",
                f"- Article format: {submission_snapshot['submission_preferences'].get('article_format', 'full_article')}",
                f"- Recommended template: {submission_snapshot['recommended_template'].get('template_id', '')} / {submission_snapshot['recommended_template'].get('template_name', '')}",
                f"- Template note: {submission_snapshot['recommended_template'].get('editor_note', '')}",
            ]
        )
        submission_zip.writestr("04_templates/target_journal_submission_strategy.md", _text_bytes("\n".join(submission_strategy_lines)))

        methods_and_limitations_lines = [
            "# Methods And Limitations Notes",
            "",
            "## Methods Note",
            submission_snapshot.get("methods_support", {}).get("methods_paragraph", "No submission-specific methods note was recorded."),
            "",
            "## DI Extreme-Paper Rule",
            "- Ranked disruptive and consolidating tables retain papers meeting any of the following support conditions: `Internal Citers >= 5`, `Internal References >= 5`, or `Support (nd + nc + na) >= 5`.",
            "- Ranking prioritizes `DI1` first, then `Support`, and then `Internal Citers`.",
            "- Optional topic matching remains disabled by default unless explicitly requested for a topic-focused export.",
            "",
            "## Stated Limitations",
        ]
        if submission_snapshot.get("stated_limitations"):
            methods_and_limitations_lines.extend(f"- {item}" for item in submission_snapshot["stated_limitations"])
        else:
            methods_and_limitations_lines.append("- No additional submission-oriented limitations were recorded.")
        submission_zip.writestr("04_templates/methods_and_limitations.md", _text_bytes("\n".join(methods_and_limitations_lines)))
    return bundle.getvalue()


def build_figure_explanation_bundle(figure_package_report, figure_package_snapshot):
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as figure_zip:
        figure_zip.writestr("01_report/manuscript_figure_explanation_package.md", _text_bytes(figure_package_report))
        figure_zip.writestr("02_snapshot/figure_package_snapshot.json", _json_bytes(figure_package_snapshot))
        figure_zip.writestr("03_mapping/figure_mapping.csv", _csv_bytes(figure_package_snapshot["figure_items"]))
        figure_zip.writestr("03_mapping/table_mapping.csv", _csv_bytes(figure_package_snapshot["table_items"]))

        caption_template_lines = ["# Figure Caption Templates", ""]
        caption_template_lines.extend(f"- {item['id']} {item['name']}: {item['caption']}" for item in figure_package_snapshot["figure_items"])
        caption_template_lines.extend(["", "# Table Caption Templates", ""])
        caption_template_lines.extend(f"- {item['id']} {item['name']}: {item['caption']}" for item in figure_package_snapshot["table_items"])
        figure_zip.writestr("04_templates/caption_templates.md", _text_bytes("\n".join(caption_template_lines)))

        reviewer_note_lines = ["# Reviewer Notes", ""]
        reviewer_note_lines.extend(f"- {item['id']} {item['name']}: {item['reviewer_note']}" for item in figure_package_snapshot["figure_items"])
        reviewer_note_lines.extend(f"- {item['id']} {item['name']}: {item['reviewer_note']}" for item in figure_package_snapshot["table_items"])
        figure_zip.writestr("04_templates/reviewer_notes.md", _text_bytes("\n".join(reviewer_note_lines)))

        assembly_note_lines = ["# Assembly Notes", ""]
        assembly_note_lines.extend(f"- {item}" for item in figure_package_snapshot["assembly_notes"])
        figure_zip.writestr("04_templates/assembly_notes.md", _text_bytes("\n".join(assembly_note_lines)))
    return bundle.getvalue()


def build_reviewer_response_bundle(reviewer_report, reviewer_snapshot):
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as reviewer_zip:
        reviewer_zip.writestr("01_report/reviewer_response_material_package.md", _text_bytes(reviewer_report))
        reviewer_zip.writestr("02_snapshot/reviewer_response_snapshot.json", _json_bytes(reviewer_snapshot))
        reviewer_zip.writestr("03_mapping/evidence_mapping.csv", _csv_bytes(reviewer_snapshot["evidence_mapping"]))
        reviewer_zip.writestr("03_mapping/anticipated_questions.csv", _csv_bytes(reviewer_snapshot["anticipated_questions"]))

        innovation_claim_lines = ["# Innovation Claims", ""]
        innovation_claim_lines.extend(
            f"- {item['theme']}: {item['claim']} Evidence: {item['evidence']}"
            for item in reviewer_snapshot["innovation_claims"]
        )
        reviewer_zip.writestr("04_templates/innovation_claims.md", _text_bytes("\n".join(innovation_claim_lines)))

        reproducibility_lines = ["# Reproducibility Response Notes", ""]
        reproducibility_lines.extend(f"- {item}" for item in reviewer_snapshot["reproducibility_responses"])
        reviewer_zip.writestr("04_templates/reproducibility_notes.md", _text_bytes("\n".join(reproducibility_lines)))

        limitation_lines = ["# Stated Limitations", ""]
        limitation_lines.extend(f"- {item}" for item in reviewer_snapshot["limitations"])
        reviewer_zip.writestr("04_templates/limitations.md", _text_bytes("\n".join(limitation_lines)))
    return bundle.getvalue()


def build_journal_submission_bundle(
    journal_submission_report,
    journal_submission_snapshot,
    selected_journal_preferences,
    submission_report,
    manuscript_submission_report,
    submission_snapshot,
    innovation_report,
    reproducibility_report,
    figure_package_report,
    bc_pairs_report,
    bc_top_papers_report,
    df_di_report,
    reviewer_report,
    reviewer_snapshot,
):
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as journal_zip:
        journal_zip.writestr("00_submission_guide/journal_submission_version_package.md", _text_bytes(journal_submission_report))
        journal_zip.writestr("00_submission_guide/journal_submission_version_snapshot.json", _json_bytes(journal_submission_snapshot))

        template_index_lines = ["# Journal Template Variant Index", ""]
        execution_policy_note = journal_submission_snapshot.get("execution_policy_note", "")
        if execution_policy_note:
            template_index_lines.extend(
                [
                    "## Large-Sample Export Notice",
                    f"- {execution_policy_note}",
                    "- This note indicates that a lightweight robustness policy may have been used to reduce export waiting time for large datasets.",
                    "",
                ]
            )
        template_index_lines.extend(
            [
                f"- Selected main_text_policy: {selected_journal_preferences['main_text_policy']}",
                f"- Selected supplement_policy: {selected_journal_preferences['supplement_policy']}",
                f"- Selected review_intensity: {selected_journal_preferences['review_intensity']}",
                f"- Selected article_format: {selected_journal_preferences['article_format']}",
                "",
            ]
        )
        template_index_lines.extend(
            f"- {item['template_name']}: {item['positioning']}"
            for item in journal_submission_snapshot["template_variants"]
        )
        journal_zip.writestr("00_submission_guide/journal_template_variant_index.md", _text_bytes("\n".join(template_index_lines)))

        for item in journal_submission_snapshot["template_variants"]:
            template_lines = [
                f"# {item['template_name']}",
                "",
                f"- Template ID: {item['template_id']}",
                f"- Positioning: {item['positioning']}",
                f"- Editor note: {item['editor_note']}",
                "",
                "## Main Manuscript Priority",
            ]
            template_lines.extend(f"- {value}" for value in item["main_manuscript_priority"])
            template_lines.extend(["", "## Supplementary Priority"])
            template_lines.extend(f"- {value}" for value in item["supplementary_priority"])
            template_lines.extend(["", "## Reviewer Appendix Priority"])
            template_lines.extend(f"- {value}" for value in item["reviewer_appendix_priority"])
            journal_zip.writestr(
                f"00_submission_guide/template_variants/{item['template_id']}.md",
                _text_bytes("\n".join(template_lines)),
            )

        journal_zip.writestr("01_main_manuscript/manuscript_submission_result_package.md", _text_bytes(submission_report))
        journal_zip.writestr("01_main_manuscript/manuscript_submission_case_package.md", _text_bytes(manuscript_submission_report))
        journal_zip.writestr("01_main_manuscript/main_results_table.csv", _csv_bytes(submission_snapshot["main_results_table"]))
        journal_zip.writestr(
            "01_main_manuscript/caption_templates.md",
            _text_bytes(
                "\n".join(
                    ["# Caption Templates", "", "## Figure Captions"]
                    + [f"- {item['name']}: {item['caption']}" for item in submission_snapshot["recommended_figures"]]
                    + ["", "## Table Captions"]
                    + [f"- {item['name']}: {item['caption']}" for item in submission_snapshot["recommended_tables"]]
                )
            ),
        )
        journal_zip.writestr("02_supplementary/innovation_metrics_report.md", _text_bytes(innovation_report))
        journal_zip.writestr("02_supplementary/Biblio-HUB_reproducibility_report.md", _text_bytes(reproducibility_report))
        journal_zip.writestr("02_supplementary/manuscript_figure_explanation_package.md", _text_bytes(figure_package_report))
        journal_zip.writestr("02_supplementary/supplementary_table_index.csv", _csv_bytes(submission_snapshot["supplementary_table_index"]))
        journal_zip.writestr("02_supplementary/figure_table_crosswalk.csv", _csv_bytes(submission_snapshot["figure_table_crosswalk"]))
        journal_zip.writestr("02_supplementary/bibliographic_coupling_pairs.csv", _csv_bytes(bc_pairs_report))
        journal_zip.writestr("02_supplementary/bibliographic_coupling_top_papers.csv", _csv_bytes(bc_top_papers_report))

        if df_di_report is not None and "Disruption_Index" in df_di_report.columns:
            export_cols = [
                column
                for column in [
                    "Title",
                    "Authors",
                    "Journal",
                    "Year",
                    "Disruption_Index",
                    "Support",
                    "DI_nd",
                    "DI_nc",
                    "DI_na",
                    "Internal_References",
                    "Internal_Citers",
                ]
                if column in df_di_report.columns
            ]
            journal_zip.writestr("02_supplementary/disruption_index_scores.csv", df_di_report[export_cols].to_csv(index=False).encode("utf-8-sig"))
            journal_zip.writestr("02_supplementary/top_disruptive_papers.csv", rank_disruption_extremes(df_di_report, kind="disruptive", top_n=10)[export_cols].to_csv(index=False).encode("utf-8-sig"))
            journal_zip.writestr("02_supplementary/top_consolidating_papers.csv", rank_disruption_extremes(df_di_report, kind="consolidating", top_n=10)[export_cols].to_csv(index=False).encode("utf-8-sig"))

        journal_zip.writestr("03_reviewer_appendix/reviewer_response_material_package.md", _text_bytes(reviewer_report))
        journal_zip.writestr("03_reviewer_appendix/evidence_mapping.csv", _csv_bytes(reviewer_snapshot["evidence_mapping"]))
        journal_zip.writestr("03_reviewer_appendix/anticipated_questions.csv", _csv_bytes(reviewer_snapshot["anticipated_questions"]))
    return bundle.getvalue()


def build_one_click_research_bundle(
    research_report,
    research_snapshot,
    manuscript_report,
    manuscript_snapshot,
    manuscript_submission_report,
    manuscript_submission_snapshot,
    reproducibility_report,
    reproducibility_snapshot,
    innovation_report,
    innovation_snapshot,
    submission_report,
    submission_snapshot,
    figure_package_report,
    figure_package_snapshot,
    reviewer_report,
    reviewer_snapshot,
):
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as research_zip:
        research_zip.writestr("00_master_report/one_click_Biblio-HUB_research_report.md", _text_bytes(research_report))
        research_zip.writestr("00_master_report/research_report_snapshot.json", _json_bytes(research_snapshot))
        research_zip.writestr("01_case_report/Biblio-HUB_manuscript_case_report.md", _text_bytes(manuscript_report))
        research_zip.writestr("01_case_report/Biblio-HUB_manuscript_case_snapshot.json", _json_bytes(manuscript_snapshot))
        research_zip.writestr("01_case_report/Biblio-HUB_submission_case_package.md", _text_bytes(manuscript_submission_report))
        research_zip.writestr("01_case_report/Biblio-HUB_submission_case_snapshot.json", _json_bytes(manuscript_submission_snapshot))
        research_zip.writestr("02_reproducibility/Biblio-HUB_reproducibility_report.md", _text_bytes(reproducibility_report))
        research_zip.writestr("02_reproducibility/Biblio-HUB_reproducibility_snapshot.json", _json_bytes(reproducibility_snapshot))
        research_zip.writestr("03_innovation/Biblio-HUB_innovation_metrics_report.md", _text_bytes(innovation_report))
        research_zip.writestr("03_innovation/Biblio-HUB_innovation_metrics_snapshot.json", _json_bytes(innovation_snapshot))
        research_zip.writestr("04_submission/Biblio-HUB_submission_result_package.md", _text_bytes(submission_report))
        research_zip.writestr("04_submission/Biblio-HUB_submission_result_snapshot.json", _json_bytes(submission_snapshot))
        research_zip.writestr("05_figure_explanation/Biblio-HUB_figure_explanation_package.md", _text_bytes(figure_package_report))
        research_zip.writestr("05_figure_explanation/Biblio-HUB_figure_package_snapshot.json", _json_bytes(figure_package_snapshot))
        research_zip.writestr("06_reviewer_response/Biblio-HUB_reviewer_response_package.md", _text_bytes(reviewer_report))
        research_zip.writestr("06_reviewer_response/Biblio-HUB_reviewer_response_snapshot.json", _json_bytes(reviewer_snapshot))
    return bundle.getvalue()
