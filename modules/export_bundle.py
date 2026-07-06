import io
import importlib.util
import re
from datetime import date

import plotly.graph_objects as go

from modules.experiment_framework import format_execution_policy_summary

SCIENTIFIC_COLORWAY = [
    "#D45959",
    "#E2A479",
    "#D45959",
    "#2F74B8",
    "#5E379D",
    "#60B0F4",
    "#507D39",
    "#88AFD8",
    "#B9DFB9",
    "#FF6B78",
    "#36663E",
    "#3A79C0",
    "#2F74B8",
    "#5DCE9C",
    "#5A97D0",
]

MORANDI_SEQUENTIAL_SCALE = [
    [0.0, "#FFF8DD"],
    [0.2, "#FBE3D2"],
    [0.4, "#F8D0B0"],
    [0.6, "#F2B382"],
    [0.8, "#E2A479"],
    [1.0, "#D45959"],
]

KEYWORD_MATRIX_SEQUENTIAL_SCALE = [
    [0.0, "#fbd2bc"],
    [0.25, "#feab88"],
    [0.5, "#b71c2c"],
    [0.75, "#8b0824"],
    [1.0, "#6a0624"],
]

PUBLICATION_EXPORT_FORMATS = ("png", "svg", "pdf")
PARAMETER_GROUP_PRIORITY = {
    "Citation": 10,
    "Network": 20,
    "Temporal": 30,
    "Semantic Topic Modeling": 40,
    "Export": 50,
    "Collaboration": 60,
    "Structure": 70,
}


def sanitize_filename(name):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._")
    return cleaned or "export"


def get_plotly_static_export_status():
    available = importlib.util.find_spec("kaleido") is not None
    if available:
        return {
            "available": True,
            "backend": "kaleido",
            "message": "Plotly static image export is available.",
        }
    return {
        "available": False,
        "backend": None,
        "message": "Static Plotly image export requires the optional 'kaleido' package.",
    }


def build_publication_manifest(
    export_format,
    selected_items=None,
    include_interactive_html=False,
    skipped_items=None,
    static_export_status=None,
):
    selection_mode = "all figures" if not selected_items else f"selected figures only ({len(selected_items)})"
    status = static_export_status or get_plotly_static_export_status()
    lines = [
        "Bibliometrics publication figure bundle",
        f"Image format: {str(export_format).lower()}",
        f"Selection mode: {selection_mode}",
        f"Interactive network HTML companions: {'included' if include_interactive_html else 'not included'}",
        "Rendering profile: white background, manuscript-friendly styling, high-resolution export",
        f"Static Plotly export: {'available' if status.get('available') else 'limited'}",
        f"Static Plotly export note: {status.get('message', '')}",
    ]
    if selected_items:
        lines.append("Selected item ids:")
        lines.extend(f"- {item_id}" for item_id in sorted(selected_items))
    if skipped_items:
        lines.append("Skipped items:")
        lines.extend(f"- {item}" for item in skipped_items)
    return "\n".join(lines) + "\n"


def _safe_year_value(value):
    try:
        year = int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None
    current_year = date.today().year
    return year if 1900 <= year <= current_year + 1 else None


def _valid_text_series(df, column_name):
    if column_name not in df.columns:
        return []
    values = (
        df[column_name]
        .fillna("")
        .astype(str)
        .map(str.strip)
    )
    return [value for value in values if value and value.lower() != "nan"]


def _top_mapping_items(mapping, top_n):
    if hasattr(mapping, "most_common"):
        return mapping.most_common(top_n)
    return sorted(mapping.items(), key=lambda item: (-item[1], item[0]))[:top_n]


def _coverage_fraction(df, column_name):
    return round(len(_valid_text_series(df, column_name)) / max(len(df), 1), 4)


def _filter_parameters_by_groups(analysis_parameters, groups):
    return [
        item for item in analysis_parameters
        if item.get("group") in groups
    ]


def _format_parameter_labels(parameters):
    if not parameters:
        return "None recorded"
    formatted = []
    for item in parameters:
        status = "modified" if item.get("changed") else "default"
        formatted.append(f"{item['label']}={item['value']} ({status})")
    return "; ".join(formatted)


def _build_parameter_change_summary(snapshot):
    group_notes = {
        "Network": "Report non-default network thresholds to justify graph density, keyword coverage, and cluster structure.",
        "Citation": "Report citation-related thresholds and selection limits so coupling and co-citation analyses remain reproducible.",
        "Collaboration": "Report collaboration filters to clarify which authors, institutions, or countries were retained in the network.",
        "Temporal": "Report temporal-analysis settings to make trend and burst-detection results reproducible.",
        "Semantic Topic Modeling": "Report semantic-topic settings because topic granularity depends on embedding and clustering configuration.",
        "Structure": "Report structure-oriented top-N settings to explain matrix, thematic-map, and multi-field coverage.",
        "Export": "Report non-default export settings when describing manuscript-ready figure preparation and supplementary packaging.",
    }
    changed_parameters = [
        item for item in snapshot.get("analysis_parameters", [])
        if item.get("changed")
    ]
    grouped = {}
    for item in changed_parameters:
        group_name = item.get("group", "Other")
        grouped.setdefault(group_name, []).append(item)

    summary_rows = []
    for group_name in sorted(
        grouped,
        key=lambda name: (PARAMETER_GROUP_PRIORITY.get(name, 999), str(name)),
    ):
        items = sorted(grouped[group_name], key=lambda item: str(item.get("label", "")))
        summary_rows.append(
            {
                "module": group_name,
                "parameter_count": len(items),
                "changed_parameters": "; ".join(item["label"] for item in items),
                "current_settings": "; ".join(f"{item['label']}={item['value']}" for item in items),
                "default_settings": "; ".join(f"{item['label']}={item['default']}" for item in items),
                "methods_note": group_notes.get(
                    group_name,
                    "Report these non-default settings in the Methods section so readers can distinguish analyst choices from defaults.",
                ),
            }
        )
    return summary_rows


def build_parameter_change_summary_markdown(summary_rows):
    lines = [
        "# Methods Parameter Change Summary",
        "",
        "| Module | Changed Parameters | Current Settings | Default Settings | Methods Note |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    if summary_rows:
        for item in summary_rows:
            lines.append(
                f"| {item['module']} | {item['changed_parameters']} | {item['current_settings']} | {item['default_settings']} | {item['methods_note']} |"
            )
    else:
        lines.append("| None | None recorded | None recorded | None recorded | No non-default parameters were recorded in the current session. |")
    lines.append("")
    return "\n".join(lines)


def _build_methods_evidence_map(snapshot):
    analysis_parameters = snapshot.get("analysis_parameters", [])
    network_and_citation = _filter_parameters_by_groups(analysis_parameters, {"Network", "Citation", "Structure"})
    temporal = _filter_parameters_by_groups(analysis_parameters, {"Temporal"})
    semantic = _filter_parameters_by_groups(analysis_parameters, {"Semantic Topic Modeling"})
    export_related = _filter_parameters_by_groups(analysis_parameters, {"Export"})
    return [
        {
            "step": "Dataset Provenance",
            "algorithm_or_rule": "Metadata ingestion with coverage audit",
            "key_parameters": f"Tracked columns={len(snapshot.get('columns', []))}; records={snapshot.get('records', 0)}",
            "evidence_output": "Dataset provenance summary and metadata coverage table",
            "manuscript_use": "Methods / Data source and corpus description",
        },
        {
            "step": "Deduplication",
            "algorithm_or_rule": "Exact DOI matching plus fuzzy title-author-year matching",
            "key_parameters": (
                f"original={snapshot.get('deduplication', {}).get('original', 0)}; "
                f"removed={snapshot.get('deduplication', {}).get('removed', 0)}; "
                f"final={snapshot.get('deduplication', {}).get('final', 0)}"
            ),
            "evidence_output": "Deduplication summary and reproducibility checklist",
            "manuscript_use": "Methods / Data cleaning and record screening",
        },
        {
            "step": "Keyword and Structural Analysis",
            "algorithm_or_rule": "Metadata-first keyword extraction with co-occurrence/network analysis",
            "key_parameters": _format_parameter_labels(network_and_citation),
            "evidence_output": (
                f"Unique keywords={snapshot.get('keyword_statistics', {}).get('unique_keywords', 0)}; "
                f"co-occurrence pairs={snapshot.get('keyword_statistics', {}).get('cooccurrence_pairs', 0)}"
            ),
            "manuscript_use": "Methods / Structural analysis pipeline",
        },
        {
            "step": "Temporal Analysis",
            "algorithm_or_rule": "Yearly frequency trajectories and Kleinberg burst detection",
            "key_parameters": _format_parameter_labels(temporal),
            "evidence_output": "Timeline, burst-detection settings, and temporal reporting checklist",
            "manuscript_use": "Methods / Temporal dynamics analysis",
        },
        {
            "step": "Semantic Topic Modeling",
            "algorithm_or_rule": "Sentence-BERT + UMAP + HDBSCAN via BERTopic when enabled",
            "key_parameters": _format_parameter_labels(semantic),
            "evidence_output": "Semantic-topic configuration record and dependency-sensitive notes",
            "manuscript_use": "Methods / Optional semantic extension",
        },
        {
            "step": "Export and Submission Packaging",
            "algorithm_or_rule": "Publication-ready export profile and journal-preference packaging",
            "key_parameters": _format_parameter_labels(export_related),
            "evidence_output": "Figure format settings, reproducibility exports, and target-journal preference record",
            "manuscript_use": "Methods / Reporting transparency and supplementary material assembly",
        },
    ]


def _build_methods_writing_pack(snapshot):
    field_coverage = snapshot.get("field_coverage", {})
    analysis_parameters = snapshot.get("analysis_parameters", [])
    changed_parameters = [item for item in analysis_parameters if item.get("changed")]
    changed_parameter_labels = ", ".join(item["label"] for item in changed_parameters) or "no non-default parameters"
    summary_rows = snapshot.get("parameter_change_summary", [])
    module_summary = "; ".join(
        f"{item['module']} ({item['parameter_count']})"
        for item in summary_rows
    ) or "no module-specific changes"
    journal_preferences = snapshot.get("submission_preferences", {}).get("journal_preferences", {})
    template_note = snapshot.get("submission_preferences", {}).get("recommended_template", {}).get("editor_note", "")
    execution_policy = snapshot.get("execution_policy", {})
    if execution_policy:
        policy_sentence = (
            " For large-sample export tasks, robustness analysis followed the recorded execution policy: "
            f"{format_execution_policy_summary(execution_policy)}."
        )
    else:
        policy_sentence = ""
    return {
        "data_source_paragraph": (
            f"The analyzed corpus contained {snapshot.get('records', 0)} records after deduplication "
            f"(original={snapshot.get('deduplication', {}).get('original', 0)}, "
            f"removed={snapshot.get('deduplication', {}).get('removed', 0)}, "
            f"final={snapshot.get('deduplication', {}).get('final', 0)}). "
            f"Metadata coverage was tracked for core fields such as Title ({field_coverage.get('Title', 0.0):.1%}), "
            f"Abstract ({field_coverage.get('Abstract', 0.0):.1%}), DOI ({field_coverage.get('DOI', 0.0):.1%}), "
            f"and Authors ({field_coverage.get('Authors', 0.0):.1%}) to support transparent reporting of corpus quality."
        ),
        "pipeline_paragraph": (
            "The analytical workflow followed a reproducible sequence consisting of metadata import, "
            "deduplication, metadata-first keyword extraction, structural network analysis, temporal pattern detection, "
            "optional semantic topic modeling, and publication-oriented export packaging. "
            "Core algorithmic steps were logged through the reproducibility checklist so each output could be traced to its corresponding processing stage."
            + policy_sentence
        ),
        "parameter_paragraph": (
            f"Parameter traceability was maintained by recording all active analysis settings, with special attention to {changed_parameter_labels}. "
            f"Non-default settings were concentrated in the following modules: {module_summary}. "
            "This makes it possible to distinguish default configuration from analyst-driven adjustments when describing the Methods section or reproducing the same experiment later."
        ),
        "reproducibility_paragraph": (
            "To support replication and manuscript preparation, the workflow exports structured snapshots, markdown reports, "
            "CSV evidence tables, and figure-ready artifacts. These materials preserve dataset coverage, parameter choices, "
            "algorithm profile, and output-assembly decisions in a machine-readable form."
        ),
        "submission_alignment_paragraph": (
            "Target-journal preferences were also recorded to preserve how the same analytical results were organized for manuscript assembly. "
            f"The current settings used main_text_policy={journal_preferences.get('main_text_policy', 'balanced')}, "
            f"supplement_policy={journal_preferences.get('supplement_policy', 'standard')}, "
            f"review_intensity={journal_preferences.get('review_intensity', 'standard')}, and "
            f"article_format={journal_preferences.get('article_format', 'full_article')}. "
            f"{template_note}".strip()
        ),
    }


def _build_reporting_artifact_index(snapshot):
    changed_parameters = [item for item in snapshot.get("analysis_parameters", []) if item.get("changed")]
    changed_parameter_count = len(changed_parameters)
    parameter_note = (
        f"{changed_parameter_count} non-default parameter(s) recorded."
        if changed_parameters
        else "No non-default parameters recorded."
    )
    template_snapshot = snapshot.get("submission_preferences", {}).get("recommended_template", {})
    template_note = template_snapshot.get("template_name", "") or "No template recommendation recorded."
    return [
        {
            "artifact": "Biblio-HUB_reproducibility_report.md",
            "format": "markdown",
            "purpose": "Human-readable methods and reproducibility appendix",
            "manuscript_use": "Methods / Supplementary reproducibility note",
            "trace_source": "Coverage summary, parameter log, algorithm profile, and reporting checklist",
        },
        {
            "artifact": "Biblio-HUB_reproducibility_snapshot.json",
            "format": "json",
            "purpose": "Machine-readable audit snapshot for the full reproducibility state",
            "manuscript_use": "Archive / reviewer-facing traceability record",
            "trace_source": "Structured metadata coverage, parameter states, submission preferences, and methods evidence pack",
        },
        {
            "artifact": "methods_package.md",
            "format": "markdown",
            "purpose": "Reusable methods-writing paragraphs for manuscript drafting",
            "manuscript_use": "Methods drafting support",
            "trace_source": "Narrative-ready paragraphs derived from dataset coverage and current reporting settings",
        },
        {
            "artifact": "methods_mapping.csv",
            "format": "csv",
            "purpose": "Parameter-algorithm-output crosswalk for core analytical steps",
            "manuscript_use": "Methods table / reviewer appendix",
            "trace_source": parameter_note,
        },
        {
            "artifact": "methods_parameter_change_summary.csv",
            "format": "csv",
            "purpose": "Grouped summary table of all non-default parameters for direct Methods reporting",
            "manuscript_use": "Methods parameter summary table",
            "trace_source": parameter_note,
        },
        {
            "artifact": "methods_parameter_change_summary.md",
            "format": "markdown",
            "purpose": "Paste-ready markdown table of grouped non-default parameters for manuscript Methods sections",
            "manuscript_use": "Methods parameter summary table",
            "trace_source": parameter_note,
        },
        {
            "artifact": "methods_artifact_index.csv",
            "format": "csv",
            "purpose": "Index of reproducibility artifacts and their evidentiary role",
            "manuscript_use": "Supplementary package manifest",
            "trace_source": template_note,
        },
    ]


def build_manuscript_case_snapshot(df, keyword_freq, cooccurrence, dedup_report=None, top_n=10):
    years = []
    if "Year" in df.columns:
        for raw_year in df["Year"].tolist():
            year = _safe_year_value(raw_year)
            if year is not None:
                years.append(year)

    top_journals = []
    journal_values = _valid_text_series(df, "Journal")
    if journal_values:
        journal_counts = {}
        for journal in journal_values:
            journal_counts[journal] = journal_counts.get(journal, 0) + 1
        top_journals = [
            {"label": label, "count": count}
            for label, count in sorted(journal_counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]
        ]

    top_keywords = [
        {"label": keyword, "count": count}
        for keyword, count in _top_mapping_items(keyword_freq, top_n)
    ]
    top_cooccurrence_pairs = [
        {"pair": [left, right], "count": count}
        for (left, right), count in _top_mapping_items(cooccurrence, top_n)
    ]

    snapshot = {
        "records": int(len(df)),
        "year_range": [min(years), max(years)] if years else [],
        "unique_journals": int(len(set(journal_values))),
        "unique_keywords": int(len(keyword_freq)),
        "doi_coverage": round(len(_valid_text_series(df, "DOI")) / max(len(df), 1), 4),
        "abstract_coverage": round(len(_valid_text_series(df, "Abstract")) / max(len(df), 1), 4),
        "author_coverage": round(len(_valid_text_series(df, "Authors")) / max(len(df), 1), 4),
        "top_journals": top_journals,
        "top_keywords": top_keywords,
        "top_cooccurrence_pairs": top_cooccurrence_pairs,
        "deduplication": dedup_report or {"original": int(len(df)), "removed": 0, "final": int(len(df))},
        "recommended_figures": [
            "Publications by Year",
            "Top Journals",
            "Top Keywords",
            "Keyword Co-occurrence Network",
            "Temporal Keyword Evolution",
            "Burst Detection",
            "Thematic Map",
        ],
    }
    return snapshot


def build_manuscript_case_report(df, keyword_freq, cooccurrence, dedup_report=None, top_n=10):
    snapshot = build_manuscript_case_snapshot(
        df,
        keyword_freq=keyword_freq,
        cooccurrence=cooccurrence,
        dedup_report=dedup_report,
        top_n=top_n,
    )
    year_range = (
        f"{snapshot['year_range'][0]}-{snapshot['year_range'][1]}"
        if snapshot["year_range"]
        else "N/A"
    )
    lines = [
        "# Bibliometrics Manuscript Case Report",
        "",
        "## 1. Dataset Overview",
        f"- Records analyzed: {snapshot['records']}",
        f"- Year range: {year_range}",
        f"- Unique journals: {snapshot['unique_journals']}",
        f"- Unique keywords: {snapshot['unique_keywords']}",
        f"- DOI coverage: {snapshot['doi_coverage']:.1%}",
        f"- Abstract coverage: {snapshot['abstract_coverage']:.1%}",
        f"- Author coverage: {snapshot['author_coverage']:.1%}",
        "",
        "## 2. Data Processing Notes",
        f"- Deduplication summary: original={snapshot['deduplication']['original']}, removed={snapshot['deduplication']['removed']}, final={snapshot['deduplication']['final']}",
        "- Duplicate merging strategy: exact DOI matching plus fuzzy matching on title/author/year.",
        "- Keyword pipeline: metadata keywords first, automatic term extraction second, optional domain plugin enrichment third.",
        "- Network statistics in this report are calculated from document-level unique keywords.",
        "",
        "## 3. Descriptive Findings",
        "- Top journals:",
    ]
    if snapshot["top_journals"]:
        lines.extend(f"  - {item['label']}: {item['count']}" for item in snapshot["top_journals"])
    else:
        lines.append("  - No valid journal field available.")

    lines.append("- Top keywords:")
    if snapshot["top_keywords"]:
        lines.extend(f"  - {item['label']}: {item['count']}" for item in snapshot["top_keywords"])
    else:
        lines.append("  - No valid keyword data available.")

    lines.append("- Strongest keyword co-occurrence pairs:")
    if snapshot["top_cooccurrence_pairs"]:
        lines.extend(
            f"  - {item['pair'][0]} <-> {item['pair'][1]}: {item['count']}"
            for item in snapshot["top_cooccurrence_pairs"]
        )
    else:
        lines.append("  - No co-occurrence pairs available.")

    lines.extend(
        [
            "",
            "## 4. Suggested Manuscript Assets",
            "- Recommended core figures:",
        ]
    )
    lines.extend(f"  - {figure_name}" for figure_name in snapshot["recommended_figures"])
    lines.extend(
        [
            "- Recommended result narrative:",
            "  - Describe annual publication growth and the maturation stage of the field.",
            "  - Interpret the top journals and keywords as the field's publication and topic core.",
            "  - Use the keyword co-occurrence network and burst detection outputs to explain knowledge structure and emerging fronts.",
            "",
            "## 5. Reproducibility Reminder",
            "- Archive the exported figure bundle together with this case report in supplementary materials.",
            "- Record any manually adjusted thresholds in the manuscript methods section.",
            "",
        ]
    )
    return "\n".join(lines)


def build_manuscript_submission_snapshot(df, keyword_freq, cooccurrence, dedup_report=None, top_n=10):
    case_snapshot = build_manuscript_case_snapshot(
        df,
        keyword_freq=keyword_freq,
        cooccurrence=cooccurrence,
        dedup_report=dedup_report,
        top_n=top_n,
    )
    year_range_label = (
        f"{case_snapshot['year_range'][0]}-{case_snapshot['year_range'][1]}"
        if case_snapshot["year_range"]
        else "N/A"
    )
    top_journal_labels = ", ".join(
        item["label"] for item in case_snapshot["top_journals"][:3]
    ) or "the leading journal set"
    top_keyword_labels = ", ".join(
        item["label"] for item in case_snapshot["top_keywords"][:5]
    ) or "the dominant keyword profile"
    strongest_pair = case_snapshot["top_cooccurrence_pairs"][0] if case_snapshot["top_cooccurrence_pairs"] else None
    strongest_pair_label = (
        f"{strongest_pair['pair'][0]} <-> {strongest_pair['pair'][1]} ({strongest_pair['count']})"
        if strongest_pair
        else "no strong co-occurrence pair available"
    )

    structured_abstract = {
        "background": (
            f"This bibliometric case analyzes a corpus spanning {year_range_label} "
            f"to summarize publication concentration, topical structure, and emerging knowledge fronts."
        ),
        "objective": (
            "To generate manuscript-ready descriptive evidence and structured outputs "
            "for submission-oriented reporting based on the uploaded literature set."
        ),
        "methods": (
            "The workflow applies dataset deduplication, metadata-first keyword extraction, "
            "document-level co-occurrence analysis, and publication-oriented export preparation."
        ),
        "results": (
            f"The dataset contains {case_snapshot['records']} records, with core outlets including {top_journal_labels} "
            f"and a dominant keyword profile characterized by {top_keyword_labels}. "
            f"The strongest keyword association is {strongest_pair_label}."
        ),
        "conclusion": (
            "The exported package provides a concise evidence base that can be adapted into "
            "Results, Supplementary Materials, and submission attachments."
        ),
    }

    result_highlights = [
        f"The final dataset includes {case_snapshot['records']} records covering {year_range_label}.",
        f"Publication concentration is led by {top_journal_labels}.",
        f"Topical emphasis is concentrated around {top_keyword_labels}.",
        f"The strongest keyword co-occurrence signal is {strongest_pair_label}.",
    ]

    recommended_tables = [
        {
            "name": "Core Dataset Overview",
            "purpose": "Summarize dataset size, time span, and metadata coverage in the main manuscript.",
            "placement": "Methods or early Results",
        },
        {
            "name": "Top Journals by Publication Count",
            "purpose": "Show outlet concentration and publication core.",
            "placement": "Results",
        },
        {
            "name": "Top Keywords by Frequency",
            "purpose": "Summarize the topical focus of the dataset.",
            "placement": "Results",
        },
        {
            "name": "Strongest Keyword Co-occurrence Pairs",
            "purpose": "Support interpretation of theme linkage and knowledge structure.",
            "placement": "Results or Supplementary Materials",
        },
    ]

    paragraph_starters = [
        "The uploaded corpus provides a focused view of the field and can be described first through its temporal and outlet distribution.",
        "Results should then move from descriptive statistics to keyword prominence and keyword co-occurrence structure.",
        "Supplementary materials should archive the case snapshot, the reproducibility checklist, and the publication figure bundle.",
    ]

    return {
        "case_snapshot": case_snapshot,
        "structured_abstract": structured_abstract,
        "result_highlights": result_highlights,
        "recommended_tables": recommended_tables,
        "paragraph_starters": paragraph_starters,
        "export_tables": {
            "top_journals": case_snapshot["top_journals"],
            "top_keywords": case_snapshot["top_keywords"],
            "top_cooccurrence_pairs": case_snapshot["top_cooccurrence_pairs"],
        },
    }


def build_manuscript_submission_report(df, keyword_freq, cooccurrence, dedup_report=None, top_n=10):
    submission_snapshot = build_manuscript_submission_snapshot(
        df,
        keyword_freq=keyword_freq,
        cooccurrence=cooccurrence,
        dedup_report=dedup_report,
        top_n=top_n,
    )
    case_snapshot = submission_snapshot["case_snapshot"]
    year_range_label = (
        f"{case_snapshot['year_range'][0]}-{case_snapshot['year_range'][1]}"
        if case_snapshot["year_range"]
        else "N/A"
    )
    lines = [
        "# Manuscript Submission Case Package",
        "",
        "## 1. Submission-Oriented Overview",
        f"- Records analyzed: {case_snapshot['records']}",
        f"- Year range: {year_range_label}",
        f"- Unique journals: {case_snapshot['unique_journals']}",
        f"- Unique keywords: {case_snapshot['unique_keywords']}",
        f"- DOI coverage: {case_snapshot['doi_coverage']:.1%}",
        f"- Abstract coverage: {case_snapshot['abstract_coverage']:.1%}",
        "",
        "## 2. Structured Abstract Draft",
        f"- Background: {submission_snapshot['structured_abstract']['background']}",
        f"- Objective: {submission_snapshot['structured_abstract']['objective']}",
        f"- Methods: {submission_snapshot['structured_abstract']['methods']}",
        f"- Results: {submission_snapshot['structured_abstract']['results']}",
        f"- Conclusion: {submission_snapshot['structured_abstract']['conclusion']}",
        "",
        "## 3. Submission Highlights",
    ]
    lines.extend(f"- {item}" for item in submission_snapshot["result_highlights"])
    lines.extend(
        [
            "",
            "## 4. Recommended Tables",
        ]
    )
    for item in submission_snapshot["recommended_tables"]:
        lines.extend(
            [
                f"### {item['name']}",
                f"- Purpose: {item['purpose']}",
                f"- Suggested placement: {item['placement']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 5. Paragraph Starters",
        ]
    )
    lines.extend(f"- {item}" for item in submission_snapshot["paragraph_starters"])
    lines.extend(
        [
            "",
            "## 6. Package Assembly Reminder",
            "- Keep the manuscript case report and this submission case package together when drafting the paper.",
            "- Export the top journals, top keywords, and strongest co-occurrence pairs as reusable tables for Results or Supplementary Materials.",
            "- Align figure captions with the figure explanation package and record any non-default parameters in the reproducibility checklist.",
            "",
        ]
    )
    return "\n".join(lines)


def build_reproducibility_snapshot(
    df,
    analysis_parameters,
    dedup_report=None,
    keyword_freq=None,
    cooccurrence=None,
    journal_preferences=None,
    recommended_template=None,
    execution_policy=None,
):
    keyword_freq = keyword_freq or {}
    cooccurrence = cooccurrence or {}
    tracked_fields = [
        "Title",
        "Abstract",
        "Year",
        "Journal",
        "DOI",
        "Authors",
        "Affiliations",
        "Cited_References",
        "Times_Cited",
        "Funding",
        "Language",
        "Publisher",
    ]
    field_coverage = {
        field_name: _coverage_fraction(df, field_name)
        for field_name in tracked_fields
        if field_name in df.columns
    }
    active_parameters = []
    for item in analysis_parameters:
        record = {
            "key": item["key"],
            "label": item["label"],
            "value": item["value"],
            "default": item["default"],
            "group": item["group"],
        }
        if item["value"] != item["default"]:
            record["changed"] = True
        else:
            record["changed"] = False
        active_parameters.append(record)

    snapshot = {
        "records": int(len(df)),
        "columns": sorted(str(column_name) for column_name in df.columns.tolist()),
        "field_coverage": field_coverage,
        "deduplication": dedup_report or {"original": int(len(df)), "removed": 0, "final": int(len(df))},
        "keyword_statistics": {
            "unique_keywords": int(len(keyword_freq)),
            "cooccurrence_pairs": int(len(cooccurrence)),
        },
        "analysis_parameters": active_parameters,
        "algorithm_profile": [
            "Data import: WoS TXT and CSV input with multi-encoding fallback.",
            "Deduplication: exact DOI matching plus fuzzy matching on title/author/year.",
            "Keyword extraction: metadata-first, automatic term extraction fallback, optional domain plugin enrichment.",
            "Network clustering: Louvain community detection on co-occurrence and collaboration graphs.",
            "Timeline analysis: yearly keyword frequency trajectories.",
            "Burst detection: Kleinberg burst detection on keyword frequency sequences.",
            "Semantic topics: Sentence-BERT embeddings with UMAP + HDBSCAN via BERTopic when dependencies are available.",
            "Figure export: Plotly manuscript styling and Matplotlib 300 dpi rendering.",
        ],
        "submission_preferences": {
            "journal_preferences": dict(journal_preferences or {}),
            "recommended_template": {
                "template_id": recommended_template.get("template_id", ""),
                "template_name": recommended_template.get("template_name", ""),
                "editor_note": recommended_template.get("editor_note", ""),
            } if recommended_template else {},
        },
        "execution_policy": dict(execution_policy or {}),
    }
    snapshot["parameter_change_summary"] = _build_parameter_change_summary(snapshot)
    snapshot["methods_evidence_map"] = _build_methods_evidence_map(snapshot)
    snapshot["methods_writing_pack"] = _build_methods_writing_pack(snapshot)
    snapshot["reporting_artifact_index"] = _build_reporting_artifact_index(snapshot)
    return snapshot


def build_reproducibility_report(
    df,
    analysis_parameters,
    dedup_report=None,
    keyword_freq=None,
    cooccurrence=None,
    journal_preferences=None,
    recommended_template=None,
    execution_policy=None,
):
    snapshot = build_reproducibility_snapshot(
        df,
        analysis_parameters=analysis_parameters,
        dedup_report=dedup_report,
        keyword_freq=keyword_freq,
        cooccurrence=cooccurrence,
        journal_preferences=journal_preferences,
        recommended_template=recommended_template,
        execution_policy=execution_policy,
    )
    lines = [
        "# Bibliometrics Reproducibility Checklist",
        "",
        "## 1. Dataset Provenance Summary",
        f"- Final record count: {snapshot['records']}",
        f"- Deduplication summary: original={snapshot['deduplication']['original']}, removed={snapshot['deduplication']['removed']}, final={snapshot['deduplication']['final']}",
        f"- Available columns ({len(snapshot['columns'])}): {', '.join(snapshot['columns'])}",
        "",
        "## 2. Metadata Coverage",
    ]
    for field_name, coverage in snapshot["field_coverage"].items():
        lines.append(f"- {field_name}: {coverage:.1%}")

    lines.extend(
        [
            "",
            "## 3. Analysis Parameters",
        ]
    )
    for item in snapshot["analysis_parameters"]:
        suffix = " [modified]" if item["changed"] else " [default]"
        lines.append(
            f"- {item['group']} / {item['label']}: {item['value']} (default={item['default']}){suffix}"
        )

    lines.extend(
        [
            "",
            "## 4. Methods Parameter Change Summary",
        ]
    )
    if snapshot.get("parameter_change_summary"):
        lines.extend(
            [
                "| Module | Changed Parameters | Current Settings | Default Settings | Methods Note |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for item in snapshot["parameter_change_summary"]:
            lines.append(
                f"| {item['module']} | {item['changed_parameters']} | {item['current_settings']} | {item['default_settings']} | {item['methods_note']} |"
            )
    else:
        lines.append("- No non-default parameters were recorded in the current session.")

    lines.extend(
        [
            "",
            "## 5. Submission Preference Record",
        ]
    )
    journal_preferences_snapshot = snapshot.get("submission_preferences", {}).get("journal_preferences", {})
    if journal_preferences_snapshot:
        lines.append(
            f"- Main text policy: {journal_preferences_snapshot.get('main_text_policy', 'balanced')}"
        )
        lines.append(
            f"- Supplement policy: {journal_preferences_snapshot.get('supplement_policy', 'standard')}"
        )
        lines.append(
            f"- Review intensity: {journal_preferences_snapshot.get('review_intensity', 'standard')}"
        )
        lines.append(
            f"- Article format: {journal_preferences_snapshot.get('article_format', 'full_article')}"
        )
    else:
        lines.append("- No target-journal submission preferences were recorded.")

    recommended_template_snapshot = snapshot.get("submission_preferences", {}).get("recommended_template", {})
    if recommended_template_snapshot:
        lines.append(
            f"- Recommended template: {recommended_template_snapshot.get('template_id', '')} / {recommended_template_snapshot.get('template_name', '')}"
        )
        lines.append(
            f"- Template note: {recommended_template_snapshot.get('editor_note', '')}"
        )

    lines.extend(
        [
            "",
            "## 6. Parameter-Algorithm-Output Mapping",
            "| Step | Algorithm or Rule | Key Parameters | Evidence Output | Manuscript Use |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for item in snapshot.get("methods_evidence_map", []):
        lines.append(
            f"| {item['step']} | {item['algorithm_or_rule']} | {item['key_parameters']} | {item['evidence_output']} | {item['manuscript_use']} |"
        )

    lines.extend(
        [
            "",
            "## 7. Methods Writing Pack",
        ]
    )
    methods_writing_pack = snapshot.get("methods_writing_pack", {})
    if methods_writing_pack:
        lines.extend(
            [
                f"- Data source paragraph: {methods_writing_pack.get('data_source_paragraph', '')}",
                f"- Pipeline paragraph: {methods_writing_pack.get('pipeline_paragraph', '')}",
                f"- Parameter paragraph: {methods_writing_pack.get('parameter_paragraph', '')}",
                f"- Reproducibility paragraph: {methods_writing_pack.get('reproducibility_paragraph', '')}",
                f"- Submission alignment paragraph: {methods_writing_pack.get('submission_alignment_paragraph', '')}",
            ]
        )

    lines.extend(
        [
            "",
            "## 8. Algorithm Profile",
        ]
    )
    lines.extend(f"- {item}" for item in snapshot["algorithm_profile"])

    lines.extend(
        [
            "",
            "## 9. Reporting Artifact Index",
            "| Artifact | Format | Purpose | Manuscript Use | Trace Source |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
    )
    for item in snapshot.get("reporting_artifact_index", []):
        lines.append(
            f"| {item['artifact']} | {item['format']} | {item['purpose']} | {item['manuscript_use']} | {item['trace_source']} |"
        )

    lines.extend(
        [
            "",
            "## 10. Recommended Supplementary Material Package",
            "- Include the manuscript case report, reproducibility snapshot JSON, and selected figure bundle.",
            "- Record any non-default parameter values in the Methods section of the paper.",
            "- Reuse the grouped methods parameter change summary table when drafting the formal Methods section.",
            "- Reuse the parameter-algorithm-output mapping table when drafting the formal Methods section.",
            "- Archive the selected journal submission preferences together with the preferred template note when preparing journal-specific versions.",
            "- Archive the exact software dependency versions used for BERTopic-related experiments when semantic topic modeling is enabled.",
            "",
            "## 11. Minimum Reporting Checklist",
            "- Data source and export date",
            "- Search query or dataset construction logic",
            "- Deduplication rules",
            "- Keyword extraction rules",
            "- Network thresholds and top-N settings",
            "- Burst detection keyword count",
            "- BERTopic minimum documents per topic and displayed topic count",
            "- Figure selection and export format",
            "- Target journal submission preferences and selected template strategy",
            "- Parameter-algorithm-output mapping for core analytical steps",
            "",
        ]
    )
    return "\n".join(lines)


def style_publication_figure(fig, height=None, title_visible=True, transparent_background=False):
    export_fig = go.Figure(fig)
    background_color = "rgba(0,0,0,0)" if transparent_background else "white"
    legend_background = "rgba(255,255,255,0)" if transparent_background else "white"
    title_margin_top = 90 if title_visible else 36
    layout_updates = {
        "template": "plotly_white",
        "paper_bgcolor": background_color,
        "plot_bgcolor": background_color,
        "colorway": SCIENTIFIC_COLORWAY,
        "font": {"family": "Arial", "size": 17, "color": "black"},
        "title": {"font": {"size": 22, "color": "black"}},
        "margin": {"l": 80, "r": 40, "t": title_margin_top, "b": 70},
        "legend": {
            "bgcolor": legend_background,
            "bordercolor": "#D9D9D9",
            "borderwidth": 1,
            "font": {"size": 14},
        },
    }
    if not title_visible:
        layout_updates["title"] = {"text": "", "font": {"size": 22, "color": "black"}}
    if height is not None:
        layout_updates["height"] = height
    export_fig.update_layout(**layout_updates)
    export_fig.update_xaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        gridcolor="#D9D9D9",
        zeroline=False,
        ticks="outside",
        title_font={"size": 24, "color": "black"},
        tickfont={"size": 17, "color": "black"},
    )
    export_fig.update_yaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        gridcolor="#D9D9D9",
        zeroline=False,
        ticks="outside",
        title_font={"size": 24, "color": "black"},
        tickfont={"size": 17, "color": "black"},
    )
    if transparent_background:
        export_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
    return export_fig


A4_LANDSCAPE_WIDTH_PX = 1123
A4_LANDSCAPE_HEIGHT_PX = 794
A4_LANDSCAPE_SIZE_INCHES = (11.69, 8.27)


def _resolve_static_export_canvas(export_format: str, width: int | None = None, height: int | None = None) -> tuple[int, int]:
    fmt = str(export_format or "").lower()
    if fmt in {"svg", "pdf"}:
        return A4_LANDSCAPE_WIDTH_PX, A4_LANDSCAPE_HEIGHT_PX
    resolved_width = int(width or A4_LANDSCAPE_WIDTH_PX)
    resolved_height = int(height or A4_LANDSCAPE_HEIGHT_PX)
    return resolved_width, resolved_height


def plotly_figure_to_bytes(
    fig,
    export_format="png",
    width=A4_LANDSCAPE_WIDTH_PX,
    height=A4_LANDSCAPE_HEIGHT_PX,
    scale=2,
    title_visible=True,
    transparent_background=False,
):
    export_width, export_height = _resolve_static_export_canvas(export_format, width, height)
    export_fig = style_publication_figure(
        fig,
        height=export_height,
        title_visible=title_visible,
        transparent_background=transparent_background,
    )
    max_retries = 1
    for attempt in range(max_retries):
        try:
            fmt = export_format.lower()
            if fmt == "svg":
                actual_scale = 1
            elif fmt == "pdf":
                actual_scale = 1.5
            else:
                actual_scale = scale
            timeout = 20 if export_format.lower() != "pdf" else 30
            import queue
            import threading

            result_queue = queue.Queue(maxsize=1)

            def render_image():
                try:
                    payload = export_fig.to_image(
                        format=export_format,
                        width=export_width,
                        height=export_height,
                        scale=actual_scale,
                    )
                    result_queue.put(("ok", payload))
                except Exception as exc:
                    result_queue.put(("error", exc))

            worker = threading.Thread(target=render_image, name="plotly-static-export", daemon=True)
            worker.start()
            worker.join(timeout=timeout)
            if worker.is_alive():
                raise TimeoutError(
                    f"Plotly static export timed out after {timeout} seconds for {export_format.upper()}."
                )

            status, payload = result_queue.get_nowait()
            if status == "error":
                raise payload
            return payload
        except Exception:
            if attempt < max_retries - 1:
                import time
                time.sleep(2 * (attempt + 1))                       
                continue
            else:
                raise


def matplotlib_figure_to_bytes(fig, export_format="png", dpi=300, transparent_background=False):
    fmt = str(export_format or "").lower()
    buf = io.BytesIO()
    facecolor = "none" if transparent_background and fmt in {"svg", "pdf"} else "white"
    edgecolor = "none" if transparent_background and fmt in {"svg", "pdf"} else "white"
    if fmt in {"svg", "pdf"}:
        original_size = tuple(fig.get_size_inches())
        try:
            fig.set_size_inches(*A4_LANDSCAPE_SIZE_INCHES, forward=True)
            fig.savefig(
                buf,
                format=export_format,
                dpi=None,
                bbox_inches=None,
                facecolor=facecolor,
                edgecolor=edgecolor,
                transparent=transparent_background,
            )
        finally:
            fig.set_size_inches(*original_size, forward=True)
    else:
        fig.savefig(
            buf,
            format=export_format,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="white",
        )
    buf.seek(0)
    return buf.read()
