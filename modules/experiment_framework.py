import re
import time
import uuid
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from threading import Lock

import networkx as nx
import community as community_louvain
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


_INNOVATION_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="innovation-worker")
_INNOVATION_JOBS: dict[str, dict] = {}
_INNOVATION_JOBS_LOCK = Lock()


def _resolve_robustness_parallel_workers(scenario_count, reference_set_count):
    if scenario_count < 4 or reference_set_count < 120:
        return 1
    cpu_total = os.cpu_count() or 1
    return max(1, min(scenario_count, max(cpu_total - 1, 1), 4))


def _evaluate_brokerage_robustness_scenario(
    top_n,
    min_shared,
    threshold_payload,
    labels,
    top_k,
):
    node_strength = threshold_payload["node_strength"]
    if not node_strength:
        structural_hole_summary = summarize_structural_hole_frame(pd.DataFrame())
        top_brokers = []
    else:
        ranked_nodes = sorted(
            node_strength.items(),
            key=lambda item: (-item[1], item[0]),
        )[:max(top_n, 2)]
        selected_indexes = {item[0] for item in ranked_nodes}

        graph = nx.Graph()
        for node_idx, strength in ranked_nodes:
            graph.add_node(labels[node_idx], weight=strength)
        for left_idx, right_idx, shared_count in threshold_payload["edges"]:
            if left_idx in selected_indexes and right_idx in selected_indexes:
                graph.add_edge(labels[left_idx], labels[right_idx], weight=shared_count)

        structural_hole_frame = compute_structural_hole_frame(graph)
        structural_hole_summary = summarize_structural_hole_frame(structural_hole_frame)
        top_brokers = (
            structural_hole_frame.head(top_k)["node"].tolist()
            if not structural_hole_frame.empty
            else []
        )

    return {
        "top_n": top_n,
        "min_shared_refs": min_shared,
        "nodes": structural_hole_summary["nodes"],
        "mean_brokerage": structural_hole_summary["mean_brokerage"],
        "top_broker": structural_hole_summary["top_broker"],
        "top_score": structural_hole_summary["top_score"],
        "core_brokers": structural_hole_summary["core_brokers"],
        "bridge_candidates": structural_hole_summary["bridge_candidates"],
        "top_brokers": top_brokers,
    }


def _classify_dataset_scale(record_count):
    if record_count < 100:
        return {
            "tier": "small",
            "label": "Small-scale dataset",
            "guidance": "Prioritize descriptive summaries, transparent parameter reporting, and careful interpretation of sparse networks.",
        }
    if record_count < 1000:
        return {
            "tier": "medium",
            "label": "Medium-scale dataset",
            "guidance": "Balance descriptive reporting with structural network findings and innovation indicators in the main text.",
        }
    return {
        "tier": "large",
        "label": "Large-scale dataset",
        "guidance": "Emphasize network structure, thematic clustering, robustness of high-frequency patterns, and supplementary tables for traceability.",
    }


def format_execution_policy_summary(execution_policy):
    if not execution_policy:
        return ""
    summary = (
        f"{'lightweight mode' if execution_policy.get('lightweight_mode') else 'full mode'}, "
        f"{execution_policy.get('analysis_record_count', 0)}/{execution_policy.get('full_record_count', 0)} records, "
        f"{execution_policy.get('scenario_count_requested', 0)} scenario(s)"
    )
    if execution_policy.get("downsampled"):
        summary += ", downsampled"
    return summary


def _infer_output_narrative_type(name):
    lowered = str(name).lower()
    if "year" in lowered or "timeline" in lowered or "trend" in lowered:
        return "trend"
    if "bibliographic coupling" in lowered:
        return "coupling"
    if "disruption" in lowered:
        return "disruption"
    if "network" in lowered or "co-occurrence" in lowered or "cooccurrence" in lowered or "co-citation" in lowered:
        return "network"
    if "thematic" in lowered or "topic" in lowered or "theme" in lowered:
        return "thematic"
    if "top" in lowered or "distribution" in lowered or "table" in lowered or "journal" in lowered or "keyword" in lowered:
        return "descriptive"
    return "general"


def _classify_output_placement(name, output_kind):
    """
    Classify the suggested manuscript placement for a figure or table.
    """
    lowered = str(name).lower()
    if "publications by year" in lowered or "annual" in lowered:
        return {
            "section": "Introduction or Results",
            "reason": "Used to contextualize research growth and volume over time."
        }
    if "metadata" in lowered or "deduplication" in lowered or "checklist" in lowered or "parameters" in lowered:
        return {
            "section": "Methods or Supplementary",
            "reason": "Provides transparency on data provenance and analysis settings."
        }
    if "top keywords" in lowered or "top journals" in lowered or "co-occurrence" in lowered or "cooccurrence" in lowered:
        return {
            "section": "Results (Structural)",
            "reason": "Primary evidence for the structural configuration of the research field."
        }
    if "coupling" in lowered or "disruption" in lowered or "co-citation" in lowered:
        return {
            "section": "Results (Relational/Innovation)",
            "reason": "In-depth relational or innovation indicators that form the core findings."
        }
    if "thematic" in lowered or "burst" in lowered or "three-field" in lowered:
        return {
            "section": "Discussion",
            "reason": "Higher-level synthesis outputs that are often interpreted in a broader context."
        }
    
    if output_kind == "table":
        return {
            "section": "Results or Supplementary",
            "reason": "Detailed data records supporting the main narrative."
        }
    return {
        "section": "Results",
        "reason": "Standard placement for primary visual evidence."
    }


def _score_submission_output_priority(name, output_kind, journal_preferences=None):
    preferences = {
        "main_text_policy": "balanced",
        "supplement_policy": "standard",
        "review_intensity": "standard",
        "article_format": "full_article",
    }
    if journal_preferences:
        preferences.update(journal_preferences)

    narrative_type = _infer_output_narrative_type(name)
    base_scores = {
        "coupling": 80,
        "disruption": 78,
        "network": 65,
        "trend": 58,
        "descriptive": 48,
        "thematic": 38,
        "general": 42,
    }
    score = base_scores.get(narrative_type, 42)
    reasons = [
        f"Base priority reflects the {narrative_type} evidence value for submission-oriented reporting."
    ]

    if preferences["main_text_policy"] == "compact":
        if narrative_type in {"coupling", "disruption"}:
            score += 18
        elif narrative_type == "trend":
            score += 10
        if output_kind == "table":
            score -= 6
        reasons.append("Compact main-text policy favors concise, high-yield core outputs.")
    elif preferences["main_text_policy"] == "evidence_dense":
        if narrative_type in {"coupling", "disruption", "network"}:
            score += 14
        if output_kind == "table":
            score += 8
        reasons.append("Evidence-dense main-text policy raises the priority of structurally rich evidence.")
    else:
        reasons.append("Balanced main-text policy keeps descriptive and innovation evidence in moderate balance.")

    if preferences["supplement_policy"] == "supplement_heavy":
        if output_kind == "table":
            score += 22
        reasons.append("Supplement-heavy policy promotes ranked tables and traceable supporting evidence.")
    elif preferences["supplement_policy"] == "minimal":
        if output_kind == "table":
            score -= 12
        else:
            score += 6
        reasons.append("Minimal supplementary policy keeps the highest-value visuals in the front set.")
    else:
        reasons.append("Standard supplementary policy preserves a balanced mix of figures and tables.")

    if preferences["review_intensity"] in {"reviewer_friendly", "revision_ready"}:
        if output_kind == "table":
            score += 16
        if narrative_type in {"coupling", "disruption"}:
            score += 6
        reasons.append("Review-intensive settings increase the value of directly traceable evidence.")
    else:
        reasons.append("Standard review setting keeps reviewer-facing detail at a moderate priority.")

    if preferences["article_format"] == "short_article":
        if output_kind == "figure":
            score += 8
        else:
            score -= 6
        reasons.append("Short-article format prioritizes compact visuals over extended tables.")
    elif preferences["article_format"] == "rapid_communication":
        if output_kind == "figure":
            score += 10
        else:
            score -= 8
        if narrative_type in {"coupling", "disruption", "trend"}:
            score += 6
        reasons.append("Rapid-communication format prioritizes fast-reading headline evidence.")
    else:
        reasons.append("Full-article format allows both core visuals and supporting tables to remain visible.")

    unique_reasons = list(dict.fromkeys(reasons))
    return {
        "narrative_type": narrative_type,
        "priority_score": score,
        "priority_reason": " ".join(unique_reasons),
    }


def _rank_submission_outputs(recommended_figures, recommended_tables, journal_preferences=None):
    ranked_outputs = []
    for output_kind, items in (("figure", recommended_figures), ("table", recommended_tables)):
        for item in items:
            metadata = _score_submission_output_priority(
                item.get("name", ""),
                output_kind,
                journal_preferences=journal_preferences,
            )
            ranked_outputs.append(
                {
                    **item,
                    "output_kind": output_kind,
                    "narrative_type": metadata["narrative_type"],
                    "priority_score": metadata["priority_score"],
                    "priority_reason": metadata["priority_reason"],
                }
            )

    ranked_outputs.sort(
        key=lambda item: (
            -item["priority_score"],
            0 if item["output_kind"] == "figure" else 1,
            item.get("name", ""),
        )
    )
    for index, item in enumerate(ranked_outputs, start=1):
        item["priority_rank"] = index

    ranked_figures = [
        item for item in ranked_outputs
        if item["output_kind"] == "figure"
    ]
    ranked_tables = [
        item for item in ranked_outputs
        if item["output_kind"] == "table"
    ]
    return ranked_figures, ranked_tables, ranked_outputs


def _score_chapter_target_fit(name, output_kind, chapter_target, journal_preferences=None):
    narrative_type = _infer_output_narrative_type(name)
    chapter_base_scores = {
        "Introduction": {
            "trend": 84,
            "descriptive": 74,
            "network": 52,
            "coupling": 36,
            "disruption": 32,
            "thematic": 44,
            "general": 40,
        },
        "Methods": {
            "trend": 38,
            "descriptive": 50,
            "network": 54,
            "coupling": 48,
            "disruption": 46,
            "thematic": 34,
            "general": 42,
        },
        "Results": {
            "trend": 58,
            "descriptive": 66,
            "network": 80,
            "coupling": 92,
            "disruption": 90,
            "thematic": 56,
            "general": 60,
        },
        "Discussion": {
            "trend": 62,
            "descriptive": 58,
            "network": 64,
            "coupling": 70,
            "disruption": 74,
            "thematic": 86,
            "general": 55,
        },
    }
    preferences = {
        "main_text_policy": "balanced",
        "supplement_policy": "standard",
        "review_intensity": "standard",
        "article_format": "full_article",
    }
    if journal_preferences:
        preferences.update(journal_preferences)

    score = chapter_base_scores.get(chapter_target, {}).get(narrative_type, 40)
    reasons = [
        f"{chapter_target} placement prioritizes {narrative_type} evidence at this stage of the manuscript."
    ]

    if chapter_target in {"Introduction", "Discussion"} and output_kind == "figure":
        score += 8
        reasons.append(f"{chapter_target} usually benefits from visual framing rather than dense tabular detail.")
    if chapter_target in {"Methods", "Results"} and output_kind == "table":
        score += 10
        reasons.append(f"{chapter_target} can use table-based evidence for transparent, exact reporting.")

    if preferences["main_text_policy"] == "compact" and chapter_target in {"Introduction", "Results"}:
        if output_kind == "figure":
            score += 6
            reasons.append("Compact main-text policy slightly favors figures in chapter-leading positions.")
    elif preferences["main_text_policy"] == "evidence_dense" and chapter_target == "Results":
        score += 8
        reasons.append("Evidence-dense main-text policy reinforces direct empirical material in Results.")

    if preferences["review_intensity"] in {"reviewer_friendly", "revision_ready"} and chapter_target in {"Methods", "Results"}:
        if output_kind == "table":
            score += 8
            reasons.append("Review-intensive settings strengthen traceable table evidence in Methods/Results.")

    if preferences["article_format"] == "rapid_communication" and chapter_target == "Introduction":
        if narrative_type in {"trend", "descriptive"}:
            score += 6
            reasons.append("Rapid-communication format prefers quick field-context framing in the opening section.")
    if preferences["article_format"] == "short_article" and chapter_target == "Results":
        if narrative_type in {"coupling", "disruption"}:
            score += 6
            reasons.append("Short-article format emphasizes the strongest headline findings in Results.")

    return {
        "chapter_score": score,
        "chapter_reason": " ".join(dict.fromkeys(reasons)),
    }


def _build_chapter_target_output_plan(recommended_output_sequence, journal_preferences=None):
    chapter_targets = [
        ("Introduction", "Use field-context outputs that quickly establish scope, time span, and descriptive orientation."),
        ("Methods", "Use outputs that help justify data provenance, parameter transparency, and reproducibility."),
        ("Results", "Lead with the strongest structural and innovation evidence that directly supports the main claims."),
        ("Discussion", "Use outputs that support interpretation, synthesis, and cautious contextualization of findings."),
    ]
    chapter_plan = []
    for chapter_name, chapter_note in chapter_targets:
        ranked_items = []
        for item in recommended_output_sequence:
            metadata = _score_chapter_target_fit(
                item.get("name", ""),
                item.get("output_kind", "figure"),
                chapter_name,
                journal_preferences=journal_preferences,
            )
            ranked_items.append(
                {
                    **item,
                    "chapter_target": chapter_name,
                    "chapter_score": metadata["chapter_score"],
                    "chapter_reason": metadata["chapter_reason"],
                }
            )
        ranked_items.sort(
            key=lambda entry: (
                -entry["chapter_score"],
                entry.get("priority_rank", 999),
                entry.get("name", ""),
            )
        )
        for index, entry in enumerate(ranked_items, start=1):
            entry["chapter_rank"] = index
        chapter_plan.append(
            {
                "chapter": chapter_name,
                "chapter_note": chapter_note,
                "recommended_items": ranked_items,
            }
        )
    return chapter_plan


def _build_type_specific_narrative_templates(
    manuscript_snapshot,
    reproducibility_snapshot,
    submission_snapshot,
):
    templates = []
    seen_names = set()
    top_keywords = manuscript_snapshot.get("top_keywords", [])[:5]
    top_keyword_labels = ", ".join(item.get("label", "") for item in top_keywords if item.get("label")) or "the leading keyword set"
    top_journals = manuscript_snapshot.get("top_journals", [])[:3]
    top_journal_labels = ", ".join(item.get("label", "") for item in top_journals if item.get("label")) or "the core journal set"
    year_range = manuscript_snapshot.get("year_range", [])
    year_range_label = f"{year_range[0]}-{year_range[1]}" if len(year_range) == 2 else "the observed period"
    cooccurrence_pairs = reproducibility_snapshot.get("keyword_statistics", {}).get("cooccurrence_pairs", 0)
    bc_metrics = (
        submission_snapshot.get("innovation_metrics", {})
        .get("bibliographic_coupling", {})
        .get("network_metrics", {})
    )
    di_summary = (
        submission_snapshot.get("innovation_metrics", {})
        .get("disruption_index", {})
        .get("summary", {})
    )

    recommended_items = []
    for item in submission_snapshot.get("recommended_output_sequence", []) or submission_snapshot.get("recommended_figures", []):
        recommended_items.append(("figure", item))
    if not submission_snapshot.get("recommended_output_sequence"):
        for item in submission_snapshot.get("recommended_tables", []):
            recommended_items.append(("table", item))

    for output_kind, item in recommended_items:
        output_kind = item.get("output_kind", output_kind)
        name = item.get("name", "").strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        narrative_type = _infer_output_narrative_type(name)
        entry = {
            "name": name,
            "output_kind": output_kind,
            "narrative_type": narrative_type,
            "caption": item.get("caption", ""),
        }
        if narrative_type == "trend":
            entry["template"] = (
                f"The {name.lower()} shows how the field evolved across {year_range_label}, "
                "highlighting phases of early emergence, subsequent growth, and the most recent development stage."
            )
            entry["focus"] = "Explain temporal growth, inflection points, and whether output accumulation suggests maturation or continued expansion."
        elif narrative_type == "descriptive":
            entry["template"] = (
                f"The {name.lower()} indicates that the publication landscape is concentrated around {top_journal_labels}, "
                f"while the main thematic profile is represented by {top_keyword_labels}."
            )
            entry["focus"] = "Describe concentration, ranking order, and what the leading journals or keywords imply about the field core."
        elif narrative_type == "network":
            entry["template"] = (
                f"The {name.lower()} reveals the structural organization of the field, with {cooccurrence_pairs} tracked co-occurrence relationships "
                "that can be interpreted in terms of central themes, link density, and potential subdomains."
            )
            entry["focus"] = "Discuss central nodes, dense link regions, cluster boundaries, and what the network says about knowledge structure."
        elif narrative_type == "coupling":
            entry["template"] = (
                f"The {name.lower()} identifies shared intellectual foundations among focal studies, with "
                f"{bc_metrics.get('nodes', 0)} connected papers and {bc_metrics.get('edges', 0, )} coupling links in the current analytical graph."
            )
            entry["focus"] = "Interpret which studies share reference bases, whether clusters reflect schools of thought, and how coupling supports structural comparison."
        elif narrative_type == "disruption":
            entry["template"] = (
                f"The {name.lower()} summarizes innovation-oriented citation dynamics, where the mean disruption index is "
                f"{di_summary.get('mean_di', 0.0):.4f} and can be used to contrast disruptive and consolidating papers."
            )
            entry["focus"] = "Separate highly positive and highly negative DI cases, then explain them cautiously as citation-dynamic signals rather than causal proof."
        elif narrative_type == "thematic":
            entry["template"] = (
                f"The {name.lower()} outlines the thematic configuration of the dataset, allowing the researcher to distinguish motor themes, basic themes, "
                "and potentially emerging or declining topic areas."
            )
            entry["focus"] = "Explain quadrant or cluster meaning, then connect thematic position to topic maturity and development potential."
        else:
            entry["template"] = (
                f"The {name.lower()} provides supporting evidence for the current bibliometric interpretation and should be described in relation to the study objective."
            )
            entry["focus"] = "State what the output measures, summarize the dominant pattern, and link it back to the research question."
        templates.append(entry)
    return templates


def build_graph_from_cooccurrence(cooccurrence, top_n=50):
    """
    Build a NetworkX graph from a co-occurrence counter.
    """
    G = nx.Graph()
    for (u, v), weight in cooccurrence.most_common(top_n):
        G.add_edge(u, v, weight=weight)
    return G


def _clean_cited_reference(reference):
    reference = str(reference).strip()
    if not reference or reference.lower() == "nan" or len(reference) < 5:
        return None
    if reference.lower().startswith("[anonymous]"):
        return None
    parts = [part.strip() for part in reference.split(",") if part.strip()]
    if len(parts) >= 2:
        author = parts[0]
        if author.lower().startswith("[anonymous]") or len(author) < 2:
            return None
        year_part = None
        for part in parts[1:4]:
            if part[:4].isdigit():
                year_part = part[:4]
                break
        if year_part is None:
            return None
        short_ref = f"{author.upper()}, {year_part}"
    else:
        short_ref = reference[:60].upper()
    return None if short_ref.replace(" ", "").isdigit() else short_ref


def _extract_reference_set(reference_text):
    values = set()
    for item in str(reference_text).split(";"):
        cleaned = _clean_cited_reference(item)
        if cleaned:
            values.add(cleaned)
    return values


def _safe_text(value, fallback="Unknown"):
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text


_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s\],;]+", re.IGNORECASE)


def _normalize_year_token(value):
    text = _safe_text(value, "").strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text[:4] if text[:4].isdigit() else ""


def _normalize_author_token(value):
    text = _safe_text(value, "").upper()
    if not text:
        return "UNKNOWN"
    token = text.split(",", 1)[0].split(" ", 1)[0]
    token = re.sub(r"[^A-Z0-9-]", "", token)
    return token or "UNKNOWN"


def _normalize_journal_token(value):
    text = _safe_text(value, "").upper()
    if not text:
        return ""
    for token in re.split(r"[^A-Z0-9]+", text):
        if token:
            return token
    return ""


def _extract_doi_token(value):
    text = _safe_text(value, "")
    if not text:
        return None
    match = _DOI_PATTERN.search(text)
    if not match:
        return None
    return match.group(0).rstrip("].,;)").upper()


def _build_primary_paper_key(author, year, journal):
    if not author or not year or not journal:
        return ""
    return f"{author}, {year}, {journal}"


def _paper_display_label(row, index):
    """
    Generate a publication-grade label for a paper node.
    Format: 'FirstAuthor, Year' or 'ShortTitle, Year'
    """
    authors = str(row.get("Authors", "")).strip()
    first_author = ""
    if authors and authors.lower() != "nan":
                                                                        
        first_author = authors.split(";")[0].split(",")[0].strip()
                                                                                         
        if first_author.isupper() and len(first_author) > 1:
            first_author = first_author.capitalize()
    
    year_val = row.get("Year")
    year_str = "N/A"
    try:
        if year_val is not None and not pd.isna(year_val):
                                                              
            year_str = str(int(float(year_val)))
    except (ValueError, TypeError):
        year_str = str(year_val) if year_val is not None else "N/A"

    if first_author:
        return f"{first_author}, {year_str}"

    title = str(row.get("Title", "")).strip()
    if not title or title.lower() == "nan":
        return f"Paper {index + 1}, {year_str}"

                                                   
    short_title = title if len(title) <= 25 else f"{title[:22]}..."
    return f"{short_title}, {year_str}"


def _duplicate_label_suffix(position):
    """
    Generate a publication-style duplicate suffix such as a/b/c/aa.
    """
    try:
        index = max(1, int(position))
    except (TypeError, ValueError):
        index = 1
    chars = []
    while index > 0:
        index -= 1
        chars.append(chr(ord("a") + (index % 26)))
        index //= 26
    return "".join(reversed(chars))


def _strip_legacy_paper_label_suffixes(label):
    """
    Normalize legacy bibliographic-coupling labels produced by older exports,
    where duplicate handling appended journal names and/or numeric suffixes.
    """
    text = _safe_text(label, "").strip()
    if not text:
        return text
    while True:
        stripped = re.sub(r"\s*\[[^\]]+\]\s*$", "", text).strip()
        if stripped == text:
            break
        text = stripped
    text = re.sub(r"((?:,\s*(?:\d{4}|N/A)))\s*\([^)]*\)\s*$", r"\1", text).strip()
    return text


def _normalize_unique_paper_labels(base_labels):
    normalized = {
        idx: _strip_legacy_paper_label_suffixes(label)
        for idx, label in base_labels.items()
    }
    label_to_indexes = {}
    for idx, label in normalized.items():
        label_to_indexes.setdefault(label, []).append(idx)

    unique_labels = {}
    for label, indexes in label_to_indexes.items():
        if len(indexes) == 1:
            unique_labels[indexes[0]] = label
            continue

        used_candidates = set()
        for duplicate_pos, idx in enumerate(indexes, start=1):
            suffix = _duplicate_label_suffix(duplicate_pos)
            candidate = re.sub(
                r"((?:,\s*(?:\d{4}|N/A)))$",
                rf"\1{suffix}",
                label,
            )
            if candidate == label:
                candidate = f"{label} {suffix}"
            if candidate in used_candidates:
                candidate = f"{candidate} [{idx}]"
            used_candidates.add(candidate)
            unique_labels[idx] = candidate
    return unique_labels


def _make_unique_paper_labels(base_labels, rows_by_index):
    return _normalize_unique_paper_labels(base_labels)


@st.cache_data(show_spinner=False)
def build_bibliographic_coupling_network(df, top_n=30, min_shared_refs=2, _precalculated=None):
    """
    Build a bibliographic coupling network where two focal papers are linked
    when they share cited references.
    """
    if "Cited_References" not in df.columns or len(df) == 0:
        return nx.Graph(), [], []

    if _precalculated:
        reference_sets = _precalculated["reference_sets"]
        labels = _normalize_unique_paper_labels(_precalculated["labels"])
        shared_counts = _precalculated["shared_counts"]
    else:
        reference_sets = {}
        labels = {}
        rows_by_index = {}
        for idx, row in df.reset_index(drop=True).iterrows():
            rows_by_index[idx] = row
            refs = _extract_reference_set(row.get("Cited_References", ""))
            if refs:
                reference_sets[idx] = refs
                labels[idx] = _paper_display_label(row, idx)
        labels = _make_unique_paper_labels(labels, rows_by_index)
        shared_counts = {}
                                                      
        valid_indexes = sorted(reference_sets.keys())
        for left_pos, left_idx in enumerate(valid_indexes):
            for right_idx in valid_indexes[left_pos + 1:]:
                count = len(reference_sets[left_idx] & reference_sets[right_idx])
                if count > 0:
                    shared_counts[(left_idx, right_idx)] = count

    pair_rows = []
    node_strength = {}

    for (left_idx, right_idx), shared_count in shared_counts.items():
        if shared_count >= min_shared_refs:
            pair_rows.append(
                {
                    "source_idx": left_idx,
                    "target_idx": right_idx,
                    "source": labels[left_idx],
                    "target": labels[right_idx],
                    "shared_references": shared_count,
                }
            )
            node_strength[left_idx] = node_strength.get(left_idx, 0) + shared_count
            node_strength[right_idx] = node_strength.get(right_idx, 0) + shared_count

    if not pair_rows:
        return nx.Graph(), [], []

    ranked_nodes = sorted(
        node_strength.items(),
        key=lambda item: (-item[1], item[0]),
    )[:max(top_n, 2)]
    selected_indexes = {item[0] for item in ranked_nodes}
    filtered_pairs = [
        row for row in pair_rows
        if row["source_idx"] in selected_indexes and row["target_idx"] in selected_indexes
    ]

    G = nx.Graph()
    for node_idx, strength in ranked_nodes:
        G.add_node(labels[node_idx], weight=strength)
    for row in filtered_pairs:
        G.add_edge(row["source"], row["target"], weight=row["shared_references"])

    top_pairs = sorted(
        filtered_pairs,
        key=lambda item: (-item["shared_references"], item["source"], item["target"]),
    )[:20]
    top_papers = [
        {"paper": labels[node_idx], "coupling_strength": strength}
        for node_idx, strength in ranked_nodes
    ]
    return G, top_pairs, top_papers


def _parse_wos_authors(author_str):
    author_str = str(author_str).strip()
    if not author_str or author_str.lower() == "nan":
        return []
    return [author.strip() for author in author_str.split(";") if author.strip()]


def _paper_match_key(row):
    authors = _parse_wos_authors(row.get("Authors", ""))
    first_author = _normalize_author_token(authors[0]) if authors else "UNKNOWN"
    year = _normalize_year_token(row.get("Year", ""))
    journal = _normalize_journal_token(row.get("Journal", ""))
    doi = _extract_doi_token(row.get("DOI", ""))
    return {
        "primary": _build_primary_paper_key(first_author, year, journal),
        "doi": doi,
    }


def compute_disruption_index_frame(df):
    """
    Calculate internal-network DI1 scores for the current dataset.
    """
    if "Cited_References" not in df.columns or len(df) == 0:
        return None

    working_df = df.copy().reset_index(drop=True)
    working_df["_paper_id"] = range(len(working_df))
    match_keys = working_df.apply(_paper_match_key, axis=1)

    key_to_idx = {}
    doi_to_idx = {}
    for idx, key_info in enumerate(match_keys):
        primary_key = key_info["primary"]
        if primary_key and primary_key not in key_to_idx:
            key_to_idx[primary_key] = idx
        if key_info["doi"] and key_info["doi"] not in doi_to_idx:
            doi_to_idx[key_info["doi"]] = idx

    internal_refs = {}
    for idx, row in working_df.iterrows():
        mapped_refs = set()
        for raw_ref in str(row.get("Cited_References", "")).split(";"):
            ref = str(raw_ref).strip()
            if not ref or ref.lower() == "nan":
                continue
            parts = [part.strip().upper() for part in ref.split(",") if part.strip()]
            if len(parts) >= 3:
                author = _normalize_author_token(parts[0])
                year_idx = next(
                    (pos for pos in range(1, min(len(parts), 5)) if _normalize_year_token(parts[pos])),
                    None,
                )
                year = _normalize_year_token(parts[year_idx]) if year_idx is not None else ""
                journal = _normalize_journal_token(parts[year_idx + 1]) if year_idx is not None and year_idx + 1 < len(parts) else ""
                primary_key = _build_primary_paper_key(author, year, journal)
                if primary_key in key_to_idx:
                    mapped_refs.add(key_to_idx[primary_key])
                    continue
            doi_match = _extract_doi_token(ref)
            if doi_match and doi_match in doi_to_idx:
                mapped_refs.add(doi_to_idx[doi_match])
        internal_refs[idx] = mapped_refs

    cites_me = {idx: set() for idx in working_df.index}
    for idx, refs in internal_refs.items():
        for ref_idx in refs:
            if ref_idx in cites_me:
                cites_me[ref_idx].add(idx)

    disruption_scores = []
    nd_values = []
    nc_values = []
    na_values = []
    for idx in working_df.index:
        refs_of_paper = internal_refs[idx]
        citers_of_paper = cites_me[idx]
        if not citers_of_paper and not refs_of_paper:
            disruption_scores.append(0.0)
            nd_values.append(0)
            nc_values.append(0)
            na_values.append(0)
            continue

        nd = 0
        nc = 0
        na = 0
        for citer_idx in citers_of_paper:
            citer_refs = internal_refs[citer_idx]
            if citer_refs & refs_of_paper:
                nc += 1
            else:
                nd += 1

        for other_idx in working_df.index:
            if other_idx == idx or other_idx in citers_of_paper:
                continue
            if internal_refs[other_idx] & refs_of_paper:
                na += 1

        denominator = nd + nc + na
        score = 0.0 if denominator == 0 else (nd - nc) / denominator
        disruption_scores.append(score)
        nd_values.append(nd)
        nc_values.append(nc)
        na_values.append(na)

    working_df["Disruption_Index"] = disruption_scores
    working_df["DI_nd"] = nd_values
    working_df["DI_nc"] = nc_values
    working_df["DI_na"] = na_values
    working_df["Internal_References"] = [len(internal_refs[idx]) for idx in working_df.index]
    working_df["Internal_Citers"] = [len(cites_me[idx]) for idx in working_df.index]
    return working_df


def summarize_disruption_index(df_with_di):
    if df_with_di is None or "Disruption_Index" not in df_with_di.columns or len(df_with_di) == 0:
        return {
            "papers": 0,
            "mean_di": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
        }
    values = df_with_di["Disruption_Index"].fillna(0.0)
    return {
        "papers": int(len(values)),
        "mean_di": round(float(values.mean()), 4),
        "positive_count": int((values > 0).sum()),
        "negative_count": int((values < 0).sum()),
        "neutral_count": int((values == 0).sum()),
    }


DEFAULT_DI_EXTREMES_MIN_SUPPORT = 5
DEFAULT_DI_EXTREMES_MIN_INTERNAL_CITERS = 5
DEFAULT_DI_EXTREMES_MIN_INTERNAL_REFERENCES = 5


def _build_disruption_topic_match_mask(df_di, topic_terms):
    if df_di is None or df_di.empty or not topic_terms:
        return pd.Series(True, index=df_di.index if df_di is not None else pd.Index([]))

    search_columns = [
        column
        for column in ["Title", "Abstract", "DE", "ID", "Keywords", "Author Keywords", "Keywords Plus"]
        if column in df_di.columns
    ]
    if not search_columns:
        return pd.Series(False, index=df_di.index)

    normalized_terms = [str(term).strip().lower() for term in topic_terms if str(term).strip()]
    if not normalized_terms:
        return pd.Series(True, index=df_di.index)

    pattern = "|".join(re.escape(term) for term in normalized_terms)
    corpus = df_di[search_columns].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    return corpus.str.contains(pattern, regex=True, na=False)


def filter_disruption_extremes(
    df_di,
    min_support=DEFAULT_DI_EXTREMES_MIN_SUPPORT,
    min_internal_citers=DEFAULT_DI_EXTREMES_MIN_INTERNAL_CITERS,
    min_internal_references=DEFAULT_DI_EXTREMES_MIN_INTERNAL_REFERENCES,
    topic_terms=None,
    require_topic_match=False,
    support_filter_mode="any",
):
    if df_di is None or "Disruption_Index" not in df_di.columns or len(df_di) == 0:
        return pd.DataFrame(columns=list(df_di.columns) if isinstance(df_di, pd.DataFrame) else [])

    filtered = df_di.copy()
    for column in ["Disruption_Index", "DI_nd", "DI_nc", "DI_na", "Internal_Citers", "Internal_References"]:
        if column in filtered.columns:
            filtered[column] = pd.to_numeric(filtered[column], errors="coerce").fillna(0)

    if all(column in filtered.columns for column in ["DI_nd", "DI_nc", "DI_na"]):
        filtered["Support"] = filtered["DI_nd"] + filtered["DI_nc"] + filtered["DI_na"]
    else:
        filtered["Support"] = 0

    support_masks = []
    if min_support:
        support_masks.append(filtered["Support"] >= float(min_support))
    if min_internal_citers:
        if "Internal_Citers" in filtered.columns:
            support_masks.append(filtered["Internal_Citers"] >= float(min_internal_citers))
        else:
            support_masks.append(pd.Series(False, index=filtered.index))
    if min_internal_references:
        if "Internal_References" in filtered.columns:
            support_masks.append(filtered["Internal_References"] >= float(min_internal_references))
        else:
            support_masks.append(pd.Series(False, index=filtered.index))

    if support_masks:
        combined_mask = support_masks[0].copy()
        if str(support_filter_mode).lower() == "all":
            for mask in support_masks[1:]:
                combined_mask &= mask
        else:
            for mask in support_masks[1:]:
                combined_mask |= mask
        filtered = filtered[combined_mask]
    if require_topic_match and topic_terms:
        filtered = filtered[_build_disruption_topic_match_mask(filtered, topic_terms)]

    return filtered.copy()


def rank_disruption_extremes(
    df_di,
    kind="disruptive",
    top_n=10,
    min_support=DEFAULT_DI_EXTREMES_MIN_SUPPORT,
    min_internal_citers=DEFAULT_DI_EXTREMES_MIN_INTERNAL_CITERS,
    min_internal_references=DEFAULT_DI_EXTREMES_MIN_INTERNAL_REFERENCES,
    topic_terms=None,
    require_topic_match=False,
    support_filter_mode="any",
):
    filtered = filter_disruption_extremes(
        df_di,
        min_support=min_support,
        min_internal_citers=min_internal_citers,
        min_internal_references=min_internal_references,
        topic_terms=topic_terms,
        require_topic_match=require_topic_match,
        support_filter_mode=support_filter_mode,
    )
    if filtered.empty:
        return filtered

    sort_columns = ["Disruption_Index"]
    ascending = [kind == "consolidating"]
    for column in ["Support", "Internal_Citers"]:
        if column in filtered.columns:
            sort_columns.append(column)
            ascending.append(False)

    ranked = filtered.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    return ranked.head(top_n).copy()

def _normalize_metric_map(metric_map):
    values = [float(value) for value in metric_map.values() if pd.notna(value)]
    if not values:
        return {key: 0.0 for key in metric_map}
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return {
            key: (1.0 if pd.notna(value) and float(value) > 0 else 0.0)
            for key, value in metric_map.items()
        }
    return {
        key: round((float(value) - min_value) / (max_value - min_value), 4) if pd.notna(value) else 0.0
        for key, value in metric_map.items()
    }


def compute_structural_hole_frame(G):
    """
    Quantify structural-hole brokerage roles on a graph using betweenness,
    constraint, and effective size.
    """
    columns = [
        "node",
        "degree",
        "weighted_degree",
        "betweenness_centrality",
        "structural_constraint",
        "effective_size",
        "brokerage_score",
        "brokerage_role",
    ]
    if not G or G.number_of_nodes() == 0:
        return pd.DataFrame(columns=columns)

    U = G.to_undirected()
    if U.number_of_nodes() == 0:
        return pd.DataFrame(columns=columns)

    betweenness = nx.betweenness_centrality(U, weight="weight", normalized=True)
    try:
        constraint = nx.constraint(U, nodes=U.nodes(), weight="weight")
    except Exception:
        constraint = {node: np.nan for node in U.nodes()}
    try:
        effective_size = nx.effective_size(U, nodes=U.nodes(), weight="weight")
    except Exception:
        effective_size = {node: 0.0 for node in U.nodes()}

    normalized_betweenness = _normalize_metric_map(betweenness)
    normalized_effective_size = _normalize_metric_map(effective_size)

    rows = []
    for node in U.nodes():
        degree = int(U.degree(node))
        weighted_degree = round(float(U.degree(node, weight="weight")), 4)
        raw_constraint = constraint.get(node, np.nan)
        safe_constraint = 1.0 if pd.isna(raw_constraint) else max(0.0, min(float(raw_constraint), 1.0))
        openness = round(1.0 - safe_constraint, 4)
        raw_effective_size = 0.0 if pd.isna(effective_size.get(node, 0.0)) else float(effective_size.get(node, 0.0))
        brokerage_score = round(
            (0.45 * normalized_betweenness.get(node, 0.0))
            + (0.35 * normalized_effective_size.get(node, 0.0))
            + (0.20 * openness),
            4,
        )
        if brokerage_score >= 0.65:
            role = "core_broker"
        elif brokerage_score >= 0.4:
            role = "bridge_candidate"
        else:
            role = "embedded_node"
        rows.append(
            {
                "node": node,
                "degree": degree,
                "weighted_degree": weighted_degree,
                "betweenness_centrality": round(float(betweenness.get(node, 0.0)), 4),
                "structural_constraint": round(safe_constraint, 4),
                "effective_size": round(raw_effective_size, 4),
                "brokerage_score": brokerage_score,
                "brokerage_role": role,
            }
        )

    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(
        by=["brokerage_score", "betweenness_centrality", "weighted_degree", "node"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def summarize_structural_hole_frame(df_structural_hole):
    if df_structural_hole is None or len(df_structural_hole) == 0:
        return {
            "nodes": 0,
            "mean_brokerage": 0.0,
            "top_broker": "",
            "top_score": 0.0,
            "core_brokers": 0,
            "bridge_candidates": 0,
        }
    return {
        "nodes": int(len(df_structural_hole)),
        "mean_brokerage": round(float(df_structural_hole["brokerage_score"].fillna(0.0).mean()), 4),
        "top_broker": str(df_structural_hole.iloc[0]["node"]),
        "top_score": round(float(df_structural_hole.iloc[0]["brokerage_score"]), 4),
        "core_brokers": int((df_structural_hole["brokerage_role"] == "core_broker").sum()),
        "bridge_candidates": int((df_structural_hole["brokerage_role"].isin(["core_broker", "bridge_candidate"])).sum()),
    }


def _truncate_plot_label(value, max_len=48):
    text = str(value)
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 3]}..."


def render_structural_hole_brokerage_profile(df_structural_hole, top_n=20, label_top_n=None):
    """
    Render a publication-grade brokerage profile scatter plot.
    x-axis: betweenness centrality
    y-axis: openness (1 - structural constraint)
    marker size: effective size
    marker color: brokerage role
    """
    if df_structural_hole is None or len(df_structural_hole) == 0:
        return None

    plot_df = df_structural_hole.copy().head(max(int(top_n or 20), 5))
    if plot_df.empty:
        return None

    plot_df["openness"] = 1.0 - plot_df["structural_constraint"].fillna(1.0).clip(lower=0.0, upper=1.0)
    plot_df["display_node"] = plot_df["node"].apply(_truncate_plot_label)
    plot_df["role_label"] = plot_df["brokerage_role"].replace(
        {
            "core_broker": "Core broker",
            "bridge_candidate": "Bridge candidate",
            "embedded_node": "Embedded node",
        }
    )
    if label_top_n is None:
        label_limit = min(8, len(plot_df))
    else:
        label_limit = max(int(label_top_n), 0)
    if label_limit < len(plot_df):
        labeled_index = (
            plot_df.sort_values(
                ["brokerage_score", "betweenness_centrality", "openness"],
                ascending=[False, False, False],
            )
            .head(label_limit)
            .index
        )
        plot_df["display_text"] = plot_df["display_node"].where(plot_df.index.isin(labeled_index), "")
    else:
        plot_df["display_text"] = plot_df["display_node"]

    role_order = ["Core broker", "Bridge candidate", "Embedded node"]
    color_map = {
        "Core broker": "#C95C5C",
        "Bridge candidate": "#D9A25F",
        "Embedded node": "#8EA3B8",
    }
    fig = px.scatter(
        plot_df,
        x="betweenness_centrality",
        y="openness",
        size="effective_size",
        color="role_label",
        text="display_text",
        size_max=30,
        category_orders={"role_label": role_order},
        color_discrete_map=color_map,
        labels={
            "betweenness_centrality": "Betweenness Centrality",
            "openness": "Openness (1 - Structural Constraint)",
            "effective_size": "Effective Size",
            "role_label": "Brokerage Role",
        },
        title="Structural-Hole Brokerage Profile",
    )
    fig.update_traces(
        textposition="top center",
        textfont=dict(size=10),
        marker=dict(line=dict(width=1, color="white"), opacity=0.88),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Brokerage score: %{customdata[1]:.4f}<br>"
            "Betweenness: %{x:.4f}<br>"
            "Openness: %{y:.4f}<br>"
            "Effective size: %{marker.size:.2f}<br>"
            "Role: %{customdata[2]}<extra></extra>"
        ),
        customdata=np.stack(
            [
                plot_df["node"],
                plot_df["brokerage_score"],
                plot_df["role_label"],
            ],
            axis=-1,
        ),
    )
    fig.add_hline(
        y=float(plot_df["openness"].median()),
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
    )
    fig.add_vline(
        x=float(plot_df["betweenness_centrality"].median()),
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
    )
    fig.update_layout(
        height=720,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text="Brokerage Role",
    )
    fig.update_xaxes(gridcolor="lightgray", zeroline=False)
    fig.update_yaxes(gridcolor="lightgray", zeroline=False)
    return fig


def compute_brokerage_robustness_experiment(
    df,
    top_n_values,
    min_shared_values,
    reference_top_n=None,
    reference_min_shared=None,
    top_k=5,
):
    """
    Evaluate how stable structural-hole brokerage results remain across
    multiple bibliographic-coupling parameter settings.
    """
    normalized_top_n = sorted({max(int(value), 2) for value in top_n_values or [30]})
    normalized_min_shared = sorted({max(int(value), 1) for value in min_shared_values or [2]})
    if reference_top_n is None:
        reference_top_n = normalized_top_n[0]
    if reference_min_shared is None:
        reference_min_shared = normalized_min_shared[0]

    inverted_index = {}

    reference_sets = {}
    labels = {}
    for idx, row in df.reset_index(drop=True).iterrows():
        refs = _extract_reference_set(row.get("Cited_References", ""))
        if refs:
            reference_sets[idx] = refs
            labels[idx] = _paper_display_label(row, idx)
    labels = _normalize_unique_paper_labels(labels)
    
                                                 
    for paper_idx, refs in reference_sets.items():
        for ref in refs:
            inverted_index.setdefault(ref, []).append(paper_idx)
    
    from collections import Counter
    import itertools
    shared_counts = Counter()
    for papers in inverted_index.values():
        if len(papers) > 1:
            for p1, p2 in itertools.combinations(sorted(papers), 2):
                shared_counts[(p1, p2)] += 1
    
    threshold_cache = {}
    for min_shared in normalized_min_shared:
        edges = []
        node_strength = {}
        for (left_idx, right_idx), shared_count in shared_counts.items():
            if shared_count < min_shared:
                continue
            edges.append((left_idx, right_idx, shared_count))
            node_strength[left_idx] = node_strength.get(left_idx, 0) + shared_count
            node_strength[right_idx] = node_strength.get(right_idx, 0) + shared_count
        threshold_cache[min_shared] = {
            "edges": edges,
            "node_strength": node_strength,
        }

    scenarios = []
    stable_counter = {}
    scenario_specs = [
        (top_n, min_shared)
        for top_n in normalized_top_n
        for min_shared in normalized_min_shared
    ]
    parallel_workers = _resolve_robustness_parallel_workers(
        len(scenario_specs),
        len(reference_sets),
    )

    try:
        if parallel_workers > 1:
            with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
                future_map = {
                    executor.submit(
                        _evaluate_brokerage_robustness_scenario,
                        top_n,
                        min_shared,
                        threshold_cache[min_shared],
                        labels,
                        top_k,
                    ): (top_n, min_shared)
                    for top_n, min_shared in scenario_specs
                }
                scenario_results = []
                for future in as_completed(future_map):
                    scenario_results.append(future.result())
        else:
            scenario_results = [
                _evaluate_brokerage_robustness_scenario(
                    top_n,
                    min_shared,
                    threshold_cache[min_shared],
                    labels,
                    top_k,
                )
                for top_n, min_shared in scenario_specs
            ]
    except Exception:
        scenario_results = [
            _evaluate_brokerage_robustness_scenario(
                top_n,
                min_shared,
                threshold_cache[min_shared],
                labels,
                top_k,
            )
            for top_n, min_shared in scenario_specs
        ]
        parallel_workers = 1

    scenarios = sorted(
        scenario_results,
        key=lambda item: (item["top_n"], item["min_shared_refs"]),
    )

    for item in scenarios:
        for node in item["top_brokers"]:
            stable_counter[node] = stable_counter.get(node, 0) + 1

    reference_scenario = next(
        (
            item for item in scenarios
            if item["top_n"] == reference_top_n and item["min_shared_refs"] == reference_min_shared
        ),
        scenarios[0] if scenarios else None,
    )
    reference_top_brokers = reference_scenario.get("top_brokers", []) if reference_scenario else []
    valid_scenarios = [item for item in scenarios if item["nodes"] > 0]

    overlap_values = []
    for item in scenarios:
        overlap_count = len(set(item["top_brokers"]) & set(reference_top_brokers))
        overlap_ratio = round(overlap_count / max(len(reference_top_brokers), 1), 4)
        item["reference_overlap_count"] = overlap_count
        item["reference_overlap_ratio"] = overlap_ratio
        if item["nodes"] > 0:
            overlap_values.append(overlap_ratio)

    stable_brokers = sorted(
        [
            {
                "node": node,
                "occurrence_count": count,
                "occurrence_ratio": round(count / max(len(scenarios), 1), 4),
            }
            for node, count in stable_counter.items()
        ],
        key=lambda item: (-item["occurrence_count"], item["node"]),
    )
    summary = {
        "scenario_count": len(scenarios),
        "valid_scenarios": len(valid_scenarios),
        "reference_top_n": reference_top_n,
        "reference_min_shared_refs": reference_min_shared,
        "reference_top_brokers": reference_top_brokers,
        "mean_reference_overlap": round(sum(overlap_values) / max(len(overlap_values), 1), 4),
        "stable_broker_count": int(sum(1 for item in stable_brokers if item["occurrence_ratio"] >= 0.6)),
        "top_stable_broker": stable_brokers[0]["node"] if stable_brokers else "",
    }
    return {
        "summary": summary,
        "scenarios": scenarios,
        "stable_brokers": stable_brokers,
        "parallel_workers": int(parallel_workers),
    }


def render_brokerage_robustness_summary(robustness_snapshot, top_stable_count=8):
    """
    Render a compact robustness summary composed of:
    1) a heatmap of overlap with the reference scenario
    2) a ranked bar chart of stable brokers across scenarios
    """
    if not robustness_snapshot:
        return None

    scenarios = robustness_snapshot.get("scenarios", [])
    stable_brokers = robustness_snapshot.get("stable_brokers", [])
    if not scenarios:
        return None

    scenario_df = pd.DataFrame(scenarios)
    if scenario_df.empty:
        return None

    scenario_df = scenario_df.sort_values(["min_shared_refs", "top_n"]).reset_index(drop=True)
    heatmap_df = scenario_df.pivot(
        index="min_shared_refs",
        columns="top_n",
        values="reference_overlap_ratio",
    ).sort_index(ascending=False)

    top_stable_df = pd.DataFrame(stable_brokers[: max(int(top_stable_count or 8), 3)])
    if top_stable_df.empty:
        top_stable_df = pd.DataFrame(
            [{"node": "N/A", "occurrence_ratio": 0.0, "occurrence_count": 0}]
        )
    top_stable_df["display_node"] = top_stable_df["node"].apply(lambda item: _truncate_plot_label(item, max_len=42))

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.58, 0.42],
        horizontal_spacing=0.12,
        subplot_titles=(
            "Reference-Overlap Heatmap",
            "Stable Brokers Across Scenarios",
        ),
    )
    fig.add_trace(
        go.Heatmap(
            z=heatmap_df.values,
            x=[str(value) for value in heatmap_df.columns.tolist()],
            y=[str(value) for value in heatmap_df.index.tolist()],
            colorscale=[
                [0.0, "#FFF7EC"],
                [0.2, "#FDD49E"],
                [0.4, "#FDBB84"],
                [0.6, "#FC8D59"],
                [0.8, "#E34A33"],
                [1.0, "#B30000"],
            ],
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Overlap Ratio"),
            text=[[f"{cell:.2f}" for cell in row] for row in heatmap_df.values],
            texttemplate="%{text}",
            hovertemplate=(
                "top_n=%{x}<br>"
                "min_shared_refs=%{y}<br>"
                "overlap with reference=%{z:.4f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=top_stable_df["occurrence_ratio"],
            y=top_stable_df["display_node"],
            orientation="h",
            marker_color="#E2A479",
            text=[f"{value:.2f}" for value in top_stable_df["occurrence_ratio"]],
            textposition="outside",
            customdata=np.stack(
                [
                    top_stable_df["node"],
                    top_stable_df["occurrence_count"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Occurrence count: %{customdata[1]}<br>"
                "Occurrence ratio: %{x:.4f}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title="Brokerage Robustness Summary",
        height=640,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    if len(scenarios) <= 1:
        fig.add_annotation(
            text="Only one scenario was evaluated; all overlap ratios are expected to be 1.00.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.08,
            showarrow=False,
            font=dict(size=12, color="#7A7A7A"),
        )
    fig.update_xaxes(title_text="top_n", row=1, col=1)
    fig.update_yaxes(title_text="min_shared_refs", row=1, col=1)
    fig.update_xaxes(title_text="Occurrence Ratio", range=[0, 1.05], gridcolor="lightgray", row=1, col=2)
    fig.update_yaxes(title_text="", automargin=True, row=1, col=2)
    return fig


def compute_export_center_innovation_payload(
    df,
    analysis_parameters,
    bc_topn_val=30,
    bc_min_shared_val=2,
    lightweight_mode=False,
    top_k=5,
):
    is_large_dataset = len(df) > 2000
    if lightweight_mode:
        robustness_top_n_values = [bc_topn_val]
        robustness_min_shared_values = [bc_min_shared_val]
        df_robustness = df.sample(n=3000, random_state=42) if len(df) > 5000 else df
    elif len(df) > 5000:
        robustness_top_n_values = sorted({max(10, bc_topn_val - 5), bc_topn_val})
        robustness_min_shared_values = sorted({bc_min_shared_val, bc_min_shared_val + 1})
        df_robustness = df.sample(n=5000, random_state=42)
    else:
        robustness_top_n_values = sorted({max(10, bc_topn_val - 10), bc_topn_val, bc_topn_val + 10})
        robustness_min_shared_values = sorted({max(1, bc_min_shared_val - 1), bc_min_shared_val, bc_min_shared_val + 1})
        df_robustness = df

    robustness_scenario_count = len(robustness_top_n_values) * len(robustness_min_shared_values)
    current_execution_policy = {
        "lightweight_mode": bool(lightweight_mode),
        "full_record_count": int(len(df)),
        "analysis_record_count": int(len(df_robustness)),
        "downsampled": bool(len(df_robustness) < len(df)),
        "scenario_count_requested": int(robustness_scenario_count),
    }

    G_bc_report, bc_pairs_report, bc_top_papers_report = build_bibliographic_coupling_network(
        df,
        top_n=bc_topn_val,
        min_shared_refs=bc_min_shared_val,
    )

    df_di_report = None
    if "Cited_References" in df.columns:
        df_di_report = compute_disruption_index_frame(df)
    structural_hole_frame_report = compute_structural_hole_frame(G_bc_report)
    structural_hole_summary_report = summarize_structural_hole_frame(structural_hole_frame_report)
    baseline_comparison_snapshot = compute_brokerage_baseline_comparison(
        structural_hole_frame_report,
        top_k=top_k,
    )
    baseline_comparison_report = build_brokerage_baseline_comparison_report(
        baseline_comparison_snapshot
    )
    robustness_snapshot = compute_brokerage_robustness_experiment(
        df_robustness,
        top_n_values=robustness_top_n_values,
        min_shared_values=robustness_min_shared_values,
        reference_top_n=bc_topn_val,
        reference_min_shared=bc_min_shared_val,
        top_k=top_k,
    )
    robustness_snapshot["execution_policy"] = {
        "lightweight_mode": bool(lightweight_mode),
        "triggered_by_large_dataset": bool(is_large_dataset),
        "full_record_count": int(len(df)),
        "analysis_record_count": int(len(df_robustness)),
        "downsampled": bool(len(df_robustness) < len(df)),
        "downsample_threshold": 5000,
        "downsample_cap": 3000 if lightweight_mode else (5000 if len(df) > 5000 else len(df)),
        "scenario_count_requested": int(robustness_scenario_count),
    }
    robustness_report = build_brokerage_robustness_report(robustness_snapshot)

    innovation_report = build_innovation_metrics_report(
        df,
        G_bc_report,
        bc_pairs_report,
        bc_top_papers_report,
        df_di_report,
        analysis_parameters,
        robustness_snapshot=robustness_snapshot,
        baseline_comparison_snapshot=baseline_comparison_snapshot,
    )

    innovation_snapshot = {
        "bibliographic_coupling": {
            "parameters": {"top_n": bc_topn_val, "min_shared_refs": bc_min_shared_val},
            "network_metrics": calculate_network_metrics(G_bc_report) if G_bc_report else {},
            "top_pairs": bc_pairs_report,
            "top_papers_by_strength": bc_top_papers_report,
        },
        "disruption_index": {
            "summary": summarize_disruption_index(df_di_report) if df_di_report is not None else {},
            "top_disruptive_papers": rank_disruption_extremes(df_di_report, kind="disruptive", top_n=10)[["Title", "Year", "Disruption_Index"]].to_dict(orient="records") if df_di_report is not None and "Disruption_Index" in df_di_report.columns else [],
            "top_consolidating_papers": rank_disruption_extremes(df_di_report, kind="consolidating", top_n=10)[["Title", "Year", "Disruption_Index"]].to_dict(orient="records") if df_di_report is not None and "Disruption_Index" in df_di_report.columns else [],
        },
        "structural_hole": {
            "summary": structural_hole_summary_report,
            "top_brokers": structural_hole_frame_report.head(10).to_dict(orient="records"),
        },
        "brokerage_robustness": robustness_snapshot,
        "brokerage_baseline_comparison": baseline_comparison_snapshot,
    }
    return {
        "current_execution_policy": current_execution_policy,
        "G_bc_report": G_bc_report,
        "bc_pairs_report": bc_pairs_report,
        "bc_top_papers_report": bc_top_papers_report,
        "df_di_report": df_di_report,
        "structural_hole_frame_report": structural_hole_frame_report,
        "structural_hole_summary_report": structural_hole_summary_report,
        "baseline_comparison_snapshot": baseline_comparison_snapshot,
        "baseline_comparison_report": baseline_comparison_report,
        "robustness_snapshot": robustness_snapshot,
        "robustness_report": robustness_report,
        "innovation_report": innovation_report,
        "innovation_snapshot": innovation_snapshot,
    }


def _execute_innovation_payload_job(
    df,
    analysis_parameters,
    bc_topn_val=30,
    bc_min_shared_val=2,
    lightweight_mode=False,
    top_k=5,
):
    return compute_export_center_innovation_payload(
        df,
        analysis_parameters,
        bc_topn_val=bc_topn_val,
        bc_min_shared_val=bc_min_shared_val,
        lightweight_mode=lightweight_mode,
        top_k=top_k,
    )


def submit_innovation_background_job(
    df,
    analysis_parameters,
    bc_topn_val=30,
    bc_min_shared_val=2,
    lightweight_mode=False,
    top_k=5,
):
    job_id = str(uuid.uuid4())
    future = _INNOVATION_EXECUTOR.submit(
        _execute_innovation_payload_job,
        df.copy(),
        analysis_parameters,
        bc_topn_val,
        bc_min_shared_val,
        lightweight_mode,
        top_k,
    )
    with _INNOVATION_JOBS_LOCK:
        _INNOVATION_JOBS[job_id] = {
            "job_id": job_id,
            "future": future,
            "bc_topn_val": int(bc_topn_val),
            "bc_min_shared_val": int(bc_min_shared_val),
            "lightweight_mode": bool(lightweight_mode),
            "record_count": int(len(df)),
            "submitted_at": time.time(),
            "resolved_at": None,
            "result": None,
            "error": None,
        }
    return job_id


def get_innovation_background_job(job_id):
    with _INNOVATION_JOBS_LOCK:
        job = _INNOVATION_JOBS.get(job_id)
    if not job:
        return {"job_id": job_id, "status": "missing"}

    future = job["future"]
    status = "running"
    if future.cancelled():
        status = "cancelled"
    elif future.done():
        try:
            result = future.result()
            with _INNOVATION_JOBS_LOCK:
                job["result"] = result
                job["resolved_at"] = job["resolved_at"] or time.time()
                job["error"] = None
            status = "done"
        except Exception as exc:
            with _INNOVATION_JOBS_LOCK:
                job["error"] = str(exc)
                job["resolved_at"] = job["resolved_at"] or time.time()
            status = "error"

    with _INNOVATION_JOBS_LOCK:
        current_job = dict(_INNOVATION_JOBS.get(job_id, {}))
    current_job.pop("future", None)
    current_job["status"] = status
    return current_job


def discard_innovation_background_job(job_id):
    with _INNOVATION_JOBS_LOCK:
        job = _INNOVATION_JOBS.pop(job_id, None)
    if not job:
        return False
    future = job.get("future")
    if future and not future.done():
        future.cancel()
    return True


def build_brokerage_robustness_report(robustness_snapshot):
    """
    Build a markdown report describing brokerage robustness across coupling parameters.
    """
    summary = robustness_snapshot.get("summary", {})
    scenarios = robustness_snapshot.get("scenarios", [])
    stable_brokers = robustness_snapshot.get("stable_brokers", [])
    execution_policy = robustness_snapshot.get("execution_policy", {})
    execution_policy_summary = format_execution_policy_summary(execution_policy)
    lines = [
        "# Brokerage Robustness Report",
        "",
        "## 1. Robustness Summary",
        f"- Scenario count: {summary.get('scenario_count', 0)}",
        f"- Valid scenarios: {summary.get('valid_scenarios', 0)}",
        f"- Reference top_n: {summary.get('reference_top_n', 0)}",
        f"- Reference min_shared_refs: {summary.get('reference_min_shared_refs', 0)}",
        f"- Mean overlap with reference top brokers: {summary.get('mean_reference_overlap', 0.0):.4f}",
        f"- Stable broker count (>=60% scenarios): {summary.get('stable_broker_count', 0)}",
        f"- Top stable broker: {summary.get('top_stable_broker', '') or 'N/A'}",
        "",
        "## 2. Execution Policy",
        f"- Policy summary: {execution_policy_summary or 'No execution policy recorded.'}",
        "",
        "## 3. Scenario Comparison Table",
        "| top_n | min_shared_refs | nodes | mean_brokerage | top_broker | top_score | overlap_with_reference |",
        "| :---: | :---: | :---: | :---: | :--- | :---: | :---: |",
    ]
    for item in scenarios:
        lines.append(
            f"| {item['top_n']} | {item['min_shared_refs']} | {item['nodes']} | {item['mean_brokerage']:.4f} | "
            f"{item['top_broker'] or 'N/A'} | {item['top_score']:.4f} | {item.get('reference_overlap_ratio', 0.0):.4f} |"
        )

    lines.extend([
        "",
        "## 4. Stable Broker Candidates",
        "| Node | Occurrence Count | Occurrence Ratio |",
        "| :--- | :---: | :---: |",
    ])
    for item in stable_brokers[:15]:
        lines.append(f"| {item['node']} | {item['occurrence_count']} | {item['occurrence_ratio']:.4f} |")
    if not stable_brokers:
        lines.append("| N/A | 0 | 0.0000 |")

    lines.extend([
        "",
        "## 5. Interpretation Notes",
        "- High overlap with the reference scenario suggests that bridge-like papers remain visible under moderate parameter perturbation.",
        "- Stable brokers appearing in most scenarios provide stronger evidence for manuscript claims about cross-cluster knowledge mediation.",
        "- Large shifts across scenarios indicate that brokerage claims should be presented cautiously and paired with explicit threshold disclosure.",
        "",
        "## 6. Manuscript Snippet",
        "\"Brokerage robustness was assessed by repeating the bibliographic-coupling and structural-hole analysis across multiple parameter combinations. "
        "Bridge-like papers that remained visible across most scenarios were interpreted as more stable brokerage candidates, while strongly parameter-sensitive brokers were reported more cautiously.\"",
        "",
    ])
    return "\n".join(lines)


def _build_submission_execution_policy_support(execution_policy):
    if not execution_policy:
        return {
            "methods_paragraph": (
                "Large-sample export settings are recorded in the reproducibility package and the structured submission snapshot when acceleration rules are used."
            ),
            "limitations": [],
        }

    methods_paragraph = (
        "If large-sample export acceleration is used, report the recorded execution policy together with the coupling thresholds. "
        f"Current setting: {format_execution_policy_summary(execution_policy)}."
    )

    limitations = []
    if execution_policy.get("lightweight_mode") or execution_policy.get("downsampled"):
        limitations.append(
            "Large-sample export may use a reduced robustness grid or sampled subset; see the recorded execution_policy for exact settings."
        )
    return {
        "methods_paragraph": methods_paragraph,
        "limitations": limitations,
    }


def compute_brokerage_baseline_comparison(df_structural_hole, top_k=10):
    """
    Compare the composite brokerage score against simpler baseline metrics such
    as betweenness centrality, weighted degree, and effective size.
    """
    if df_structural_hole is None or len(df_structural_hole) == 0:
        return {
            "summary": {
                "nodes": 0,
                "top_brokerage_node": "",
                "best_aligned_baseline": "",
                "best_alignment_overlap": 0.0,
            },
            "metric_leaders": [],
            "baseline_comparisons": [],
            "node_comparison_table": [],
        }

    metric_specs = [
        ("brokerage_score", "Brokerage Score"),
        ("betweenness_centrality", "Betweenness Centrality"),
        ("weighted_degree", "Weighted Degree"),
        ("effective_size", "Effective Size"),
    ]
    ranking_tables = {}
    metric_leaders = []
    for column_name, label in metric_specs:
        ranked = df_structural_hole.sort_values(
            by=[column_name, "node"],
            ascending=[False, True],
        ).reset_index(drop=True)
        ranking_tables[column_name] = ranked
        metric_leaders.append(
            {
                "metric": label,
                "top_node": str(ranked.iloc[0]["node"]) if not ranked.empty else "",
                "top_value": round(float(ranked.iloc[0][column_name]), 4) if not ranked.empty else 0.0,
            }
        )

    brokerage_ranked = ranking_tables["brokerage_score"]
    brokerage_top_nodes = brokerage_ranked.head(top_k)["node"].tolist()
    brokerage_rank_map = {
        node: index for index, node in enumerate(brokerage_ranked["node"].tolist(), start=1)
    }
    baseline_comparisons = []
    for column_name, label in metric_specs[1:]:
        ranked = ranking_tables[column_name]
        baseline_top_nodes = ranked.head(top_k)["node"].tolist()
        overlap_nodes = sorted(set(brokerage_top_nodes) & set(baseline_top_nodes))
        overlap_ratio = round(len(overlap_nodes) / max(len(brokerage_top_nodes), 1), 4)
        rank_differences = []
        baseline_rank_map = {
            node: index for index, node in enumerate(ranked["node"].tolist(), start=1)
        }
        for node in overlap_nodes:
            rank_differences.append(abs(brokerage_rank_map[node] - baseline_rank_map[node]))
        mean_rank_shift = round(sum(rank_differences) / max(len(rank_differences), 1), 4)
        baseline_comparisons.append(
            {
                "baseline_metric": label,
                "top_node": str(ranked.iloc[0]["node"]) if not ranked.empty else "",
                "top_value": round(float(ranked.iloc[0][column_name]), 4) if not ranked.empty else 0.0,
                "top_k_overlap_count": len(overlap_nodes),
                "top_k_overlap_ratio": overlap_ratio,
                "top_1_match": bool(brokerage_top_nodes and baseline_top_nodes and brokerage_top_nodes[0] == baseline_top_nodes[0]),
                "mean_rank_shift": mean_rank_shift,
                "overlap_nodes": overlap_nodes,
            }
        )

    comparison_rank_maps = {
        column_name: {
            node: index for index, node in enumerate(ranking_tables[column_name]["node"].tolist(), start=1)
        }
        for column_name, _ in metric_specs
    }
    node_comparison_table = []
    for _, row in brokerage_ranked.head(top_k).iterrows():
        node = row["node"]
        node_comparison_table.append(
            {
                "node": node,
                "brokerage_rank": comparison_rank_maps["brokerage_score"].get(node, 0),
                "brokerage_score": round(float(row["brokerage_score"]), 4),
                "betweenness_rank": comparison_rank_maps["betweenness_centrality"].get(node, 0),
                "betweenness_centrality": round(float(row["betweenness_centrality"]), 4),
                "weighted_degree_rank": comparison_rank_maps["weighted_degree"].get(node, 0),
                "weighted_degree": round(float(row["weighted_degree"]), 4),
                "effective_size_rank": comparison_rank_maps["effective_size"].get(node, 0),
                "effective_size": round(float(row["effective_size"]), 4),
                "brokerage_role": row["brokerage_role"],
            }
        )

    best_alignment = max(
        baseline_comparisons,
        key=lambda item: (item["top_k_overlap_ratio"], -item["mean_rank_shift"], item["baseline_metric"]),
        default={"baseline_metric": "", "top_k_overlap_ratio": 0.0},
    )
    return {
        "summary": {
            "nodes": int(len(df_structural_hole)),
            "top_brokerage_node": str(brokerage_ranked.iloc[0]["node"]) if not brokerage_ranked.empty else "",
            "best_aligned_baseline": best_alignment.get("baseline_metric", ""),
            "best_alignment_overlap": best_alignment.get("top_k_overlap_ratio", 0.0),
        },
        "metric_leaders": metric_leaders,
        "baseline_comparisons": baseline_comparisons,
        "node_comparison_table": node_comparison_table,
    }


def build_brokerage_baseline_comparison_report(comparison_snapshot):
    """
    Build a markdown report comparing composite brokerage ranking with simpler baseline metrics.
    """
    summary = comparison_snapshot.get("summary", {})
    metric_leaders = comparison_snapshot.get("metric_leaders", [])
    baseline_comparisons = comparison_snapshot.get("baseline_comparisons", [])
    node_rows = comparison_snapshot.get("node_comparison_table", [])
    lines = [
        "# Brokerage Baseline Comparison Report",
        "",
        "## 1. Summary",
        f"- Nodes compared: {summary.get('nodes', 0)}",
        f"- Top brokerage node: {summary.get('top_brokerage_node', '') or 'N/A'}",
        f"- Best aligned baseline metric: {summary.get('best_aligned_baseline', '') or 'N/A'}",
        f"- Best top-k overlap with brokerage ranking: {summary.get('best_alignment_overlap', 0.0):.4f}",
        "",
        "## 2. Metric Leaders",
        "| Metric | Top Node | Top Value |",
        "| :--- | :--- | :---: |",
    ]
    for item in metric_leaders:
        lines.append(f"| {item['metric']} | {item['top_node'] or 'N/A'} | {item['top_value']:.4f} |")

    lines.extend([
        "",
        "## 3. Baseline Alignment",
        "| Baseline Metric | Top Node | Top-k Overlap | Top-k Overlap Ratio | Top-1 Match | Mean Rank Shift |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ])
    for item in baseline_comparisons:
        lines.append(
            f"| {item['baseline_metric']} | {item['top_node'] or 'N/A'} | {item['top_k_overlap_count']} | "
            f"{item['top_k_overlap_ratio']:.4f} | {'yes' if item['top_1_match'] else 'no'} | {item['mean_rank_shift']:.4f} |"
        )

    lines.extend([
        "",
        "## 4. Top Brokerage Nodes vs Baselines",
        "| Node | Brokerage Rank | Brokerage Score | Betweenness Rank | Weighted Degree Rank | Effective Size Rank | Role |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])
    for item in node_rows:
        lines.append(
            f"| {item['node']} | {item['brokerage_rank']} | {item['brokerage_score']:.4f} | "
            f"{item['betweenness_rank']} | {item['weighted_degree_rank']} | {item['effective_size_rank']} | {item['brokerage_role']} |"
        )
    if not node_rows:
        lines.append("| N/A | 0 | 0.0000 | 0 | 0 | 0 | N/A |")

    lines.extend([
        "",
        "## 5. Interpretation Notes",
        "- Brokerage score is expected to partially overlap with simpler baselines but should not collapse into any single centrality metric.",
        "- Strong overlap suggests the composite score preserves intuitive bridge signals, while rank differences show where the multi-factor design adds discrimination.",
        "- When the same node leads multiple metrics, it can be described as both central and bridge-like; when rankings diverge, the paper can motivate why brokerage captures a broader bridge role than a single metric.",
        "",
        "## 6. Manuscript Snippet",
        "\"To assess whether the composite brokerage score offered information beyond standard centrality indicators, we compared its ranking against betweenness centrality, weighted degree, and effective size. "
        "The overlap and rank-shift analysis showed where the composite indicator agreed with simpler baselines and where it provided additional discrimination for bridge-like papers.\"",
        "",
    ])
    return "\n".join(lines)


def calculate_network_metrics(G):
    """
    Calculate standard network science metrics for bibliometric analysis.
    """
    if not G or G.number_of_nodes() == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "density": 0,
            "avg_degree": 0,
            "modularity": 0,
            "clusters": 0,
            "avg_clustering_coeff": 0,
        }

                 
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    density = nx.density(G)
    
            
    degrees = [d for n, d in G.degree()]
    avg_degree = sum(degrees) / nodes if nodes > 0 else 0
    
                             
                                                          
    U = G.to_undirected()
    partition = community_louvain.best_partition(U)
    modularity = community_louvain.modularity(partition, U) if len(U.edges()) > 0 else 0
    clusters = len(set(partition.values()))
    
    avg_clustering = nx.average_clustering(U)
    
    return {
        "nodes": int(nodes),
        "edges": int(edges),
        "density": round(float(density), 4),
        "avg_degree": round(float(avg_degree), 2),
        "modularity": round(float(modularity), 4),
        "clusters": int(clusters),
        "avg_clustering_coeff": round(float(avg_clustering), 4),
    }

def get_baseline_comparison_data():
    """
    Returns a structured comparison of this tool vs industry standards.
    Useful for 'Related Work' or 'Discussion' sections in a paper.
    """
    return [
        {
            "Feature": "Core Algorithm",
            "This Tool": "Louvain / BERT-Semantic",
            "CiteSpace": "Betweenness / Burst",
            "VOSviewer": "VOS Clustering",
            "Bibliometrix": "Multi-algorithm (R)",
        },
        {
            "Feature": "Data Cleaning",
            "This Tool": "Fuzzy + DOI + Meta-First",
            "CiteSpace": "Alias Files (Manual)",
            "VOSviewer": "Thesaurus (Manual)",
            "Bibliometrix": "Built-in scripts",
        },
        {
            "Feature": "Visual Logic",
            "This Tool": "Interactive D3/Plotly + Publication-ready SVG",
            "CiteSpace": "Static Time-zone / Cluster",
            "VOSviewer": "Static Density / Overlay",
            "Bibliometrix": "GGPlot / Static",
        },
        {
            "Feature": "Brokerage Evidence",
            "This Tool": "Structural Hole + DI + Coupling",
            "CiteSpace": "Betweenness-centric",
            "VOSviewer": "Link strength / overlay",
            "Bibliometrix": "Centrality summaries",
        },
        {
            "Feature": "Semantic Analysis",
            "This Tool": "Integrated BERTopic",
            "CiteSpace": "Noun Phrase Extraction",
            "VOSviewer": "Text Mining (Co-occurrence)",
            "Bibliometrix": "Conceptual Structure Map",
        }
    ]

def build_experiment_comparison_report(df, G, analysis_params, baseline_name="Industry Standards"):
    """
    Builds a markdown report focusing on experimental results and baseline comparison.
    """
    metrics = calculate_network_metrics(G)
    structural_hole_frame = compute_structural_hole_frame(G)
    structural_hole_summary = summarize_structural_hole_frame(structural_hole_frame)
    baseline_data = get_baseline_comparison_data()
    
    lines = [
        "# Comparative Experiment & Results Report",
        "",
        "## 1. Network Topology Analysis",
        "The following metrics describe the structural properties of the generated knowledge network. "
        "These are standard indicators used to validate the robustness of the bibliometric model.",
        "",
        f"- **Node Count**: {metrics['nodes']} (Total entities analyzed)",
        f"- **Edge Count**: {metrics['edges']} (Total co-occurrence links)",
        f"- **Network Density**: {metrics['density']:.4f} (Connectivity strength)",
        f"- **Average Degree**: {metrics['avg_degree']:.2f} (Avg links per node)",
        f"- **Modularity (Q)**: {metrics['modularity']:.4f} (Clustering quality, >0.4 suggests significant structure)",
        f"- **Number of Clusters**: {metrics['clusters']} (Detected thematic communities)",
        f"- **Avg Clustering Coefficient**: {metrics['avg_clustering_coeff']:.4f} (Local cohesion)",
        "",
        "## 2. Brokerage and Bridge Analysis",
        "Structural-hole style brokerage metrics indicate whether certain nodes connect otherwise separated thematic neighborhoods.",
        "",
        f"- **Nodes evaluated for brokerage**: {structural_hole_summary['nodes']}",
        f"- **Mean brokerage score**: {structural_hole_summary['mean_brokerage']:.4f}",
        f"- **Top broker**: {structural_hole_summary['top_broker'] or 'N/A'}",
        f"- **Top brokerage score**: {structural_hole_summary['top_score']:.4f}",
        f"- **Core brokers**: {structural_hole_summary['core_brokers']}",
        "",
        "## 3. Methodology Narrative (Manuscript Template)",
        "> *Copy-paste the following into your paper's 'Methodology' or 'Results' section:*",
        "",
        f"\"We constructed a keyword co-occurrence network comprising {metrics['nodes']} nodes and {metrics['edges']} edges. "
        f"The network exhibits a modularity of Q={metrics['modularity']:.4f}, indicating a well-defined community structure. "
        f"Using the Louvain algorithm, we identified {metrics['clusters']} distinct thematic clusters. "
        f"The average clustering coefficient of {metrics['avg_clustering_coeff']:.4f} suggests significant local connectivity "
        f"within research sub-domains. Structural-hole brokerage analysis further identified {structural_hole_summary['bridge_candidates']} nodes "
        "that potentially bridge otherwise separated topical areas.\"",
        "",
        "## 4. Comparison with Baseline Tools",
        f"This section compares the current analysis pipeline with {baseline_name}.",
        "",
        "| Feature | This Tool | CiteSpace | VOSviewer | Bibliometrix |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    
    for row in baseline_data:
        lines.append(f"| {row['Feature']} | {row['This Tool']} | {row['CiteSpace']} | {row['VOSviewer']} | {row['Bibliometrix']} |")
    
    lines.extend([
        "",
        "## 5. Key Experimental Advantages",
        "- **Precision**: Hybrid deduplication (DOI + Fuzzy) ensures cleaner data compared to standard text-only matching.",
        "- **Depth**: BERTopic integration allows for semantic topic discovery beyond simple keyword frequency.",
        "- **Brokerage Insight**: Structural-hole metrics reveal bridge nodes that connect otherwise separated topical communities.",
        "- **Export**: High-resolution vector export (SVG/PDF) supports direct manuscript submission requirements.",
        "",
    ])
    
    return "\n".join(lines)

def build_innovation_metrics_report(
    df,
    G_bc,
    bc_pairs,
    bc_top_papers,
    df_di,
    analysis_parameters,
    robustness_snapshot=None,
    baseline_comparison_snapshot=None,
):
    """
    Builds a markdown report for innovation metrics (Bibliographic Coupling and Disruption Index).
    """
    lines = [
        "# Innovation Metrics Experiment Report",
        "",
        "This report summarizes advanced bibliometric metrics designed to identify \n"  
        "knowledge structure and innovative aspects within the dataset.",
        "",
        "## 1. Bibliographic Coupling Analysis",
        "Bibliographic coupling measures the similarity between two documents based on the number of common references they cite. "
        "A higher coupling strength indicates a stronger shared intellectual foundation.",
        "",
    ]

    if G_bc and G_bc.number_of_nodes() > 0:
        bc_metrics = calculate_network_metrics(G_bc)
        structural_hole_frame = compute_structural_hole_frame(G_bc)
        structural_hole_summary = summarize_structural_hole_frame(structural_hole_frame)
        lines.extend([
            f"- **Coupled Papers**: {bc_metrics['nodes']}",
            f"- **Coupling Links**: {bc_metrics['edges']}",
            f"- **Network Density**: {bc_metrics['density']:.4f}",
            f"- **Modularity (Q)**: {bc_metrics['modularity']:.4f}",
            f"- **Average Clustering Coefficient**: {bc_metrics['avg_clustering_coeff']:.4f}",
            f"- **Thematic Clusters**: {bc_metrics['clusters']}",
            "",
            "### Top 20 Strongest Bibliographic Coupling Pairs",
            "| Paper A | Paper B | Shared References |\n"  
            "| :--- | :--- | :---: |",
        ])
        for pair in bc_pairs:
            lines.append(f"| {pair['source']} | {pair['target']} | {pair['shared_references']} |")
        lines.extend([
            "",
            "### Top 20 Papers by Coupling Strength",
            "| Paper | Coupling Strength |\n"  
            "| :--- | :---: |",
        ])
        for paper in bc_top_papers:
            lines.append(f"| {paper['paper']} | {paper['coupling_strength']} |")
    else:
        structural_hole_frame = pd.DataFrame()
        structural_hole_summary = summarize_structural_hole_frame(structural_hole_frame)
        lines.append("- No significant bibliographic coupling found for the current parameters.")

    lines.extend([
        "",
        "## 2. Disruption Index Analysis",
        "The Disruption Index (DI1) quantifies the extent to which a paper changes the trajectory of science. "
        "Positive values indicate disruptive papers, while negative values suggest consolidating papers.",
        "",
    ])

    if df_di is not None and "Disruption_Index" in df_di.columns and len(df_di) > 0:
        di_summary = summarize_disruption_index(df_di)
        lines.extend([
            f"- **Papers Analyzed**: {di_summary['papers']}",
            f"- **Mean Disruption Index**: {di_summary['mean_di']:.4f}",
            f"- **Disruptive Papers (DI > 0)**: {di_summary['positive_count']}",
            f"- **Consolidating Papers (DI < 0)**: {di_summary['negative_count']}",
            f"- **Neutral Papers (DI = 0)**: {di_summary['neutral_count']}",
            "",
            "### Top 10 Most Disruptive Papers",
            "| Title | Year | Disruption Index |\n"  
            "| :--- | :---: | :---: |",
        ])
        top_disruptive = rank_disruption_extremes(df_di, kind="disruptive", top_n=10)
        for idx, row in top_disruptive.iterrows():
            lines.append(f"| {_safe_text(row['Title'])} | {_safe_text(row['Year'])} | {row['Disruption_Index']:.4f} |")
        
        lines.extend([
            "",
            "### Top 10 Most Consolidating Papers",
            "| Title | Year | Disruption Index |\n"  
            "| :--- | :---: | :---: |",
        ])
        top_consolidating = rank_disruption_extremes(df_di, kind="consolidating", top_n=10)
        for idx, row in top_consolidating.iterrows():
            lines.append(f"| {_safe_text(row['Title'])} | {_safe_text(row['Year'])} | {row['Disruption_Index']:.4f} |")
        lines.extend([
            "",
            f"- **Extreme-paper filtering rule**: retain papers meeting any of the following support conditions: Internal Citers >= {DEFAULT_DI_EXTREMES_MIN_INTERNAL_CITERS}, "
            f"Internal References >= {DEFAULT_DI_EXTREMES_MIN_INTERNAL_REFERENCES}, or Support (nd + nc + na) >= {DEFAULT_DI_EXTREMES_MIN_SUPPORT}.",
            "- **Extreme-paper ranking rule**: sort by DI1 first, then by support, and then by internal citers; topic matching remains optional and is disabled by default.",
        ])
    else:
        lines.append("- Disruption Index could not be calculated due to insufficient data (e.g., missing Cited_References).")

    lines.extend([
        "",
        "## 3. Structural Hole Brokerage Analysis",
        "Structural-hole analysis highlights nodes that bridge otherwise weakly connected substructures in the bibliographic coupling network.",
        "",
    ])
    if not structural_hole_frame.empty:
        lines.extend([
            f"- **Nodes Evaluated**: {structural_hole_summary['nodes']}",
            f"- **Mean Brokerage Score**: {structural_hole_summary['mean_brokerage']:.4f}",
            f"- **Top Broker**: {structural_hole_summary['top_broker']}",
            f"- **Top Brokerage Score**: {structural_hole_summary['top_score']:.4f}",
            f"- **Core Brokers**: {structural_hole_summary['core_brokers']}",
            "",
            "### Top 10 Structural Hole Brokers",
            "| Node | Brokerage Score | Betweenness | Constraint | Effective Size | Role |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |",
        ])
        for _, row in structural_hole_frame.head(10).iterrows():
            lines.append(
                f"| {row['node']} | {row['brokerage_score']:.4f} | {row['betweenness_centrality']:.4f} | {row['structural_constraint']:.4f} | {row['effective_size']:.4f} | {row['brokerage_role']} |"
            )
    else:
        lines.append("- Structural-hole brokerage could not be calculated because the current coupling network is empty or too sparse.")

    lines.extend([
        "",
        "## 4. Analysis Parameters",
        "The following parameters were active during the generation of these metrics:",
        "",
        "| Parameter | Value | Default | Group |\n"  
        "| :--- | :---: | :---: | :--- |",
    ])
    changed_parameter_count = 0
    for item in analysis_parameters:
        if item.get("group") in ["Citation", "Network", "Innovation", "Overview"] and item.get("changed"):
            lines.append(f"| {item['label']} | {item['value']} | {item['default']} | {item['group']} |")
            changed_parameter_count += 1
    if changed_parameter_count == 0:
        lines.append("| No changed citation/network/innovation/overview parameters | - | - | - |")
    
    next_section_number = 5
    if robustness_snapshot:
        summary = robustness_snapshot.get("summary", {})
        execution_policy = robustness_snapshot.get("execution_policy", {})
        execution_policy_summary = format_execution_policy_summary(execution_policy)
        lines.extend([
            "",
            f"## {next_section_number}. Brokerage Robustness Across Parameter Settings",
            f"- Scenario count: {summary.get('scenario_count', 0)}",
            f"- Valid scenarios: {summary.get('valid_scenarios', 0)}",
            f"- Mean overlap with reference top brokers: {summary.get('mean_reference_overlap', 0.0):.4f}",
            f"- Stable broker count (>=60% scenarios): {summary.get('stable_broker_count', 0)}",
            f"- Top stable broker: {summary.get('top_stable_broker', '') or 'N/A'}",
            f"- Policy summary: {execution_policy_summary or 'No execution policy recorded.'}",
            "",
            "### Brokerage Robustness Scenario Grid",
            "| top_n | min_shared_refs | top_broker | overlap_with_reference |",
            "| :---: | :---: | :--- | :---: |",
        ])
        for item in robustness_snapshot.get("scenarios", []):
            lines.append(
                f"| {item['top_n']} | {item['min_shared_refs']} | {item['top_broker'] or 'N/A'} | {item.get('reference_overlap_ratio', 0.0):.4f} |"
            )
        next_section_number += 1

    if baseline_comparison_snapshot:
        summary = baseline_comparison_snapshot.get("summary", {})
        lines.extend([
            "",
            f"## {next_section_number}. Brokerage Baseline Comparison",
            f"- Nodes compared: {summary.get('nodes', 0)}",
            f"- Top brokerage node: {summary.get('top_brokerage_node', '') or 'N/A'}",
            f"- Best aligned baseline metric: {summary.get('best_aligned_baseline', '') or 'N/A'}",
            f"- Best top-k overlap with brokerage ranking: {summary.get('best_alignment_overlap', 0.0):.4f}",
            "",
            "### Baseline Metric Alignment Table",
            "| Baseline Metric | Top Node | Top-k Overlap Ratio | Top-1 Match | Mean Rank Shift |",
            "| :--- | :--- | :---: | :---: | :---: |",
        ])
        for item in baseline_comparison_snapshot.get("baseline_comparisons", []):
            lines.append(
                f"| {item['baseline_metric']} | {item['top_node'] or 'N/A'} | {item['top_k_overlap_ratio']:.4f} | "
                f"{'yes' if item['top_1_match'] else 'no'} | {item['mean_rank_shift']:.4f} |"
            )
        next_section_number += 1

    lines.extend([
        "",
        f"## {next_section_number}. Methodology Snippets for Manuscript",
        "> *Integrate these descriptions into your paper's Methods or Results sections:*",
        "",
        "### Bibliographic Coupling Methodology",
        "\"Bibliographic coupling networks were constructed by linking papers that share common cited references. "
        "The strength of the coupling was determined by the number of shared references. "
        "Only papers with at least two shared references were considered for network formation. "
        "Network topological metrics, including density, modularity, and average clustering coefficient, were calculated to characterize the network structure.\"",
        "",
        "### Disruption Index Methodology",
        "\"The Disruption Index (DI1) was calculated for each paper based on its internal citation patterns within the dataset, following the methodology by Funk & Owen-Smith (2017) and Wu et al. (2019). "
        "DI1 values range from -1 (consolidating) to +1 (disruptive), quantifying the extent to which a paper breaks from or builds upon prior research. "
        "The index considers papers that cite the focal paper but none of its references (nd), those that cite both the focal paper and its references (nc), "
        "and those that cite the focal paper's references but not the focal paper itself (na). The formula used was DI = (nd - nc) / (nd + nc + na). "
        f"For ranked disruptive and consolidating tables, papers were retained when they satisfied any of the following support rules: Internal Citers >= {DEFAULT_DI_EXTREMES_MIN_INTERNAL_CITERS}, "
        f"Internal References >= {DEFAULT_DI_EXTREMES_MIN_INTERNAL_REFERENCES}, or Support (nd + nc + na) >= {DEFAULT_DI_EXTREMES_MIN_SUPPORT}; ranking then prioritized DI1, support, and internal citers. "
        "Optional topic matching based on title, abstract, and keyword fields remained disabled by default unless explicitly requested.\"",
        "",
        "### Structural Hole Brokerage Methodology",
        "\"Structural-hole brokerage analysis was conducted on the bibliographic coupling network to identify papers that connect otherwise separated knowledge communities. "
        "For each node, betweenness centrality, structural constraint, and effective size were computed and combined into a composite brokerage score. "
        "Higher brokerage scores indicate papers that are both well-positioned between clusters and less constrained by redundant local ties, providing evidence for bridge-like roles in the knowledge structure.\"",
        "",
        "### Brokerage Baseline Comparison Methodology",
        "\"To evaluate whether the composite brokerage indicator provided added value beyond simpler baselines, the brokerage ranking was compared with betweenness centrality, weighted degree, and effective size. "
        "Top-k overlap, top-1 agreement, and mean rank shift were used to describe where the composite indicator aligned with or diverged from standard centrality-based alternatives.\"",
    ])

    return "\n".join(lines)


def build_submission_result_snapshot(
    df,
    G_bc,
    bc_pairs,
    bc_top_papers,
    df_di,
    analysis_parameters,
    robustness_snapshot=None,
    baseline_comparison_snapshot=None,
    journal_preferences=None,
):
    """
    Build a structured snapshot for manuscript submission result packages.
    """
    years = []
    if "Year" in df.columns:
        for value in df["Year"].tolist():
            text = str(value).strip()
            if text and text.lower() != "nan":
                try:
                    years.append(int(float(text)))
                except (TypeError, ValueError):
                    continue

    bc_metrics = calculate_network_metrics(G_bc) if G_bc and G_bc.number_of_nodes() > 0 else {}
    di_summary = summarize_disruption_index(df_di) if df_di is not None else {}
    structural_hole_frame = compute_structural_hole_frame(G_bc)
    structural_hole_summary = summarize_structural_hole_frame(structural_hole_frame)
    changed_parameters = [
        {
            "label": item["label"],
            "value": item["value"],
            "default": item["default"],
            "group": item["group"],
        }
        for item in analysis_parameters
        if item.get("group") in ("Citation", "Network", "Innovation", "Overview") and item.get("changed")
    ]

    top_disruptive = []
    top_consolidating = []
    if df_di is not None and "Disruption_Index" in df_di.columns:
        top_disruptive = (
            rank_disruption_extremes(df_di, kind="disruptive", top_n=10)[
                [column for column in ["Title", "Year", "Disruption_Index"] if column in df_di.columns]
            ]
            .to_dict(orient="records")
        )
        top_consolidating = (
            rank_disruption_extremes(df_di, kind="consolidating", top_n=10)[
                [column for column in ["Title", "Year", "Disruption_Index"] if column in df_di.columns]
            ]
            .to_dict(orient="records")
        )

    main_results_table = [
        {
            "metric": "Record Count",
            "value": int(len(df)),
            "interpretation": "Final number of documents included in the current submission-oriented analysis package.",
        },
        {
            "metric": "Bibliographic Coupling Nodes",
            "value": bc_metrics.get("nodes", 0),
            "interpretation": "Number of focal papers retained in the bibliographic coupling network.",
        },
        {
            "metric": "Bibliographic Coupling Modularity",
            "value": round(float(bc_metrics.get("modularity", 0.0)), 4) if bc_metrics else 0.0,
            "interpretation": "Community separation strength of the coupling network; higher values indicate clearer structural segmentation.",
        },
        {
            "metric": "Mean Disruption Index",
            "value": round(float(di_summary.get("mean_di", 0.0)), 4) if di_summary else 0.0,
            "interpretation": "Average DI1 score summarizing the balance between disruptive and consolidating citation dynamics.",
        },
        {
            "metric": "Disruptive Papers",
            "value": int(di_summary.get("positive_count", 0)) if di_summary else 0,
            "interpretation": "Count of papers with positive DI1 values in the internal citation network.",
        },
        {
            "metric": "Top Brokerage Score",
            "value": round(float(structural_hole_summary.get("top_score", 0.0)), 4),
            "interpretation": "Highest structural-hole brokerage score observed in the coupling network, indicating the strongest bridge-like paper role.",
        },
    ]

    supplementary_table_index = [
        {
            "table_id": "S1",
            "name": "Top Bibliographic Coupling Pairs",
            "content": "Ranked strongest coupling links by shared references.",
            "suggested_use": "Supplementary Results or reviewer-facing appendix.",
        },
        {
            "table_id": "S2",
            "name": "Top Papers by Coupling Strength",
            "content": "Ranked focal papers by aggregate bibliographic coupling strength.",
            "suggested_use": "Supplementary Results or network appendix.",
        },
        {
            "table_id": "S3",
            "name": "Top Disruptive Papers",
            "content": "Highest DI1 scores within the internal citation network after support-based OR filtering and tie-breaking by support and internal citers.",
            "suggested_use": "Main Results or innovation-focused appendix.",
        },
        {
            "table_id": "S4",
            "name": "Top Consolidating Papers",
            "content": "Lowest DI1 scores within the internal citation network after support-based OR filtering and tie-breaking by support and internal citers.",
            "suggested_use": "Supplementary Results or reviewer appendix.",
        },
        {
            "table_id": "S5",
            "name": "Top Structural Hole Brokers",
            "content": "Ranked bridge-like papers based on brokerage score, betweenness, constraint, and effective size.",
            "suggested_use": "Main Results or structural-innovation appendix.",
        },
    ]

    figure_table_crosswalk = [
        {
            "figure": "Bibliographic Coupling Network",
            "supporting_table": "Top Bibliographic Coupling Pairs",
            "manuscript_role": "Results (knowledge structure)",
        },
        {
            "figure": "Bibliographic Coupling Network",
            "supporting_table": "Top Papers by Coupling Strength",
            "manuscript_role": "Supplementary structural evidence",
        },
        {
            "figure": "Disruption Index Distribution",
            "supporting_table": "Top Disruptive Papers",
            "manuscript_role": "Results (innovation dynamics)",
        },
        {
            "figure": "Disruption Index Distribution",
            "supporting_table": "Top Consolidating Papers",
            "manuscript_role": "Supplementary innovation evidence",
        },
        {
            "figure": "Structural Hole Brokerage Profile",
            "supporting_table": "Top Structural Hole Brokers",
            "manuscript_role": "Results (bridge and brokerage evidence)",
        },
    ]
    recommended_template = _build_parameterized_journal_template(journal_preferences)
    execution_policy = (robustness_snapshot or {}).get("execution_policy", {})
    methods_and_limitations = _build_submission_execution_policy_support(execution_policy)
    recommended_figures = [
        {
            "name": "Bibliographic Coupling Network",
            "caption": "Bibliographic coupling network of focal papers. Node size represents aggregate coupling strength and edge weight reflects the number of shared cited references.",
        },
        {
            "name": "Disruption Index Distribution",
            "caption": "Distribution of DI1 values across papers in the internal citation network, highlighting the balance between disruptive and consolidating studies.",
        },
        {
            "name": "Structural Hole Brokerage Profile",
            "caption": "Brokerage profile of papers in the bibliographic coupling network, combining betweenness, structural constraint, and effective size to reveal bridge-like knowledge positions.",
        },
        {
            "name": "Brokerage Robustness Summary",
            "caption": "Robustness summary of brokerage findings across alternative bibliographic-coupling parameter settings, highlighting overlap with the reference scenario and the most stable bridge-like papers.",
        },
    ]
    recommended_tables = [
        {
            "name": "Top Bibliographic Coupling Pairs",
            "caption": "Top paper pairs ranked by shared cited references, indicating the strongest common intellectual foundations.",
        },
        {
            "name": "Top Disruptive and Consolidating Papers",
            "caption": "Representative focal papers ranked by DI1, showing the extremes of disruptive and consolidating scientific contributions.",
        },
        {
            "name": "Top Structural Hole Brokers",
            "caption": "Top bridge-like papers ranked by composite brokerage score, indicating which studies connect otherwise weakly linked knowledge clusters.",
        },
    ]
    ranked_figures, ranked_tables, ranked_outputs = _rank_submission_outputs(
        recommended_figures,
        recommended_tables,
        journal_preferences=journal_preferences,
    )
    chapter_target_output_plan = _build_chapter_target_output_plan(
        ranked_outputs,
        journal_preferences=journal_preferences,
    )

    snapshot = {
        "dataset_overview": {
            "records": int(len(df)),
            "year_range": [min(years), max(years)] if years else [],
            "has_cited_references": bool("Cited_References" in df.columns),
            "has_times_cited": bool("Times_Cited" in df.columns),
        },
        "innovation_metrics": {
            "bibliographic_coupling": {
                "network_metrics": bc_metrics,
                "top_pairs": bc_pairs[:10],
                "top_papers_by_strength": bc_top_papers[:10],
            },
            "disruption_index": {
                "summary": di_summary,
                "top_disruptive_papers": top_disruptive,
                "top_consolidating_papers": top_consolidating,
            },
            "structural_hole": {
                "summary": structural_hole_summary,
                "top_brokers": structural_hole_frame.head(10).to_dict(orient="records"),
            },
            "brokerage_robustness": robustness_snapshot or {},
            "brokerage_baseline_comparison": baseline_comparison_snapshot or {},
        },
        "changed_parameters": changed_parameters,
        "recommended_figures": ranked_figures,
        "recommended_tables": ranked_tables,
        "recommended_output_sequence": ranked_outputs,
        "chapter_target_output_plan": chapter_target_output_plan,
        "result_narrative": [
            "Bibliographic coupling reveals the extent to which focal papers rely on shared knowledge bases and helps identify coherent thematic communities.",
            "The Disruption Index complements network structure by quantifying whether influential papers redirect or consolidate prior research trajectories.",
            "Structural-hole brokerage adds a bridge perspective by identifying papers that connect otherwise weakly linked intellectual clusters.",
            "Together, these metrics support a multi-perspective interpretation of knowledge structure, brokerage position, and innovation dynamics in the focal domain.",
        ],
        "submission_preferences": recommended_template["preferences"],
        "recommended_template": recommended_template,
        "main_results_table": main_results_table,
        "supplementary_table_index": supplementary_table_index,
        "figure_table_crosswalk": figure_table_crosswalk,
        "methods_support": {
            "execution_policy": execution_policy,
            "methods_paragraph": methods_and_limitations["methods_paragraph"],
        },
        "stated_limitations": methods_and_limitations["limitations"],
    }
    return snapshot


def build_submission_result_report(snapshot):
    """
    Build a manuscript-oriented report for submission result packages.
    """
    overview = snapshot["dataset_overview"]
    bc_metrics = snapshot["innovation_metrics"]["bibliographic_coupling"]["network_metrics"]
    di_summary = snapshot["innovation_metrics"]["disruption_index"]["summary"]
    structural_hole_summary = snapshot["innovation_metrics"].get("structural_hole", {}).get("summary", {})
    robustness_summary = snapshot["innovation_metrics"].get("brokerage_robustness", {}).get("summary", {})
    baseline_comparison_summary = snapshot["innovation_metrics"].get("brokerage_baseline_comparison", {}).get("summary", {})
    methods_support = snapshot.get("methods_support", {})
    stated_limitations = snapshot.get("stated_limitations", [])
    year_range = (
        f"{overview['year_range'][0]}-{overview['year_range'][1]}"
        if overview["year_range"]
        else "N/A"
    )

    lines = [
        "# Manuscript Submission Result Package",
        "",
        "## 1. Dataset Overview",
        f"- Records analyzed: {overview['records']}",
        f"- Year range: {year_range}",
        f"- Cited references available: {'yes' if overview['has_cited_references'] else 'no'}",
        f"- Times cited available: {'yes' if overview['has_times_cited'] else 'no'}",
        "",
        "## 2. Innovation Metric Highlights",
    ]

    if bc_metrics:
        lines.extend([
            f"- Bibliographic coupling nodes: {bc_metrics['nodes']}",
            f"- Bibliographic coupling edges: {bc_metrics['edges']}",
            f"- Bibliographic coupling density: {bc_metrics['density']:.4f}",
            f"- Bibliographic coupling modularity: {bc_metrics['modularity']:.4f}",
        ])
    else:
        lines.append("- Bibliographic coupling: no valid network generated under the current thresholds.")

    if di_summary:
        lines.extend([
            f"- Mean Disruption Index: {di_summary['mean_di']:.4f}",
            f"- Disruptive papers: {di_summary['positive_count']}",
            f"- Consolidating papers: {di_summary['negative_count']}",
            f"- Neutral papers: {di_summary['neutral_count']}",
        ])
    else:
        lines.append("- Disruption Index: insufficient internal citation data.")
    if structural_hole_summary:
        lines.extend([
            f"- Mean brokerage score: {structural_hole_summary.get('mean_brokerage', 0.0):.4f}",
            f"- Top broker: {structural_hole_summary.get('top_broker', 'N/A')}",
            f"- Core brokers: {structural_hole_summary.get('core_brokers', 0)}",
        ])
    if robustness_summary:
        lines.extend([
            f"- Brokerage robustness scenarios: {robustness_summary.get('scenario_count', 0)}",
            f"- Stable brokers (>=60% scenarios): {robustness_summary.get('stable_broker_count', 0)}",
            f"- Top stable broker: {robustness_summary.get('top_stable_broker', 'N/A') or 'N/A'}",
        ])
    if baseline_comparison_summary:
        lines.extend([
            f"- Best aligned baseline: {baseline_comparison_summary.get('best_aligned_baseline', 'N/A') or 'N/A'}",
            f"- Baseline overlap with brokerage top-k: {baseline_comparison_summary.get('best_alignment_overlap', 0.0):.4f}",
        ])

    lines.extend([
        "",
        "## 3. Result Narrative",
    ])
    lines.extend(f"- {item}" for item in snapshot["result_narrative"])

    lines.extend([
        "",
        "## 4. Preferred Output Sequence",
    ])
    for item in snapshot.get("recommended_output_sequence", []):
        lines.append(
            f"- {item.get('priority_rank', 0)}. {item['output_kind'].title()} / {item['name']}: {item.get('priority_reason', '')}"
        )

    lines.extend([
        "",
        "## 5. Chapter-Target Output Plan",
    ])
    for chapter in snapshot.get("chapter_target_output_plan", []):
        lines.extend([
            f"### {chapter['chapter']}",
            f"- Chapter note: {chapter['chapter_note']}",
        ])
        for item in chapter.get("recommended_items", []):
            lines.append(
                f"- C{item.get('chapter_rank', 0)} / P{item.get('priority_rank', 0)} {item['output_kind'].title()} {item['name']}: {item.get('chapter_reason', '')}"
            )
        lines.append("")

    lines.extend([
        "## 6. Main Results Table",
        "| Metric | Value | Interpretation |",
        "| :--- | :---: | :--- |",
    ])
    for item in snapshot.get("main_results_table", []):
        lines.append(f"| {item['metric']} | {item['value']} | {item['interpretation']} |")

    lines.extend([
        "",
        "## 7. Figure Caption Templates",
    ])
    lines.extend(
        f"- {item.get('priority_rank', '-')}. {item['name']}: {item['caption']}"
        for item in snapshot["recommended_figures"]
    )

    lines.extend([
        "",
        "## 8. Table Caption Templates",
    ])
    lines.extend(
        f"- {item.get('priority_rank', '-')}. {item['name']}: {item['caption']}"
        for item in snapshot["recommended_tables"]
    )

    lines.extend([
        "",
        "## 9. Supplementary Table Index",
        "| Table ID | Name | Content | Suggested Use |",
        "| :--- | :--- | :--- | :--- |",
    ])
    for item in snapshot.get("supplementary_table_index", []):
        lines.append(f"| {item['table_id']} | {item['name']} | {item['content']} | {item['suggested_use']} |")

    lines.extend([
        "",
        "## 10. Figure-Table Crosswalk",
        "| Figure | Supporting Table | Manuscript Role |",
        "| :--- | :--- | :--- |",
    ])
    for item in snapshot.get("figure_table_crosswalk", []):
        lines.append(f"| {item['figure']} | {item['supporting_table']} | {item['manuscript_role']} |")

    lines.extend([
        "",
        "## 11. Target Journal Submission Strategy",
    ])
    submission_preferences = snapshot.get("submission_preferences", {})
    recommended_template = snapshot.get("recommended_template", {})
    if submission_preferences:
        lines.extend([
            f"- Main text policy: {submission_preferences.get('main_text_policy', 'balanced')}",
            f"- Supplement policy: {submission_preferences.get('supplement_policy', 'standard')}",
            f"- Review intensity: {submission_preferences.get('review_intensity', 'standard')}",
            f"- Article format: {submission_preferences.get('article_format', 'full_article')}",
        ])
    else:
        lines.append("- No target-journal submission preferences were recorded.")
    if recommended_template:
        lines.extend([
            f"- Recommended template: {recommended_template.get('template_id', '')} / {recommended_template.get('template_name', '')}",
            f"- Template note: {recommended_template.get('editor_note', '')}",
        ])

    lines.extend([
        "",
        "## 12. Methods Note",
    ])
    lines.append(
        f"- {methods_support.get('methods_paragraph', 'No submission-specific methods note was recorded.')}"
    )

    lines.extend([
        "",
        "## 13. Stated Limitations",
    ])
    if stated_limitations:
        lines.extend(f"- {item}" for item in stated_limitations)
    else:
        lines.append("- No additional submission-oriented limitations were recorded.")

    lines.extend([
        "",
        "## 14. Non-default Parameters",
    ])
    if snapshot["changed_parameters"]:
        lines.extend(
            f"- {item['group']} / {item['label']}: {item['value']} (default={item['default']})"
            for item in snapshot["changed_parameters"]
        )
    else:
        lines.append("- No non-default citation or network parameters were used.")

    lines.extend([
        "",
        "## 15. Submission Checklist",
        "- Include the innovation metrics report in supplementary materials.",
        "- Cite the bibliographic coupling figure and DI1 distribution figure in the Results section.",
        "- Use the structural-hole brokerage table when arguing bridge roles between subfields.",
        "- Report all non-default thresholds in the Methods section.",
        "- Keep the selected journal submission preferences and template note aligned with the actual target journal instructions.",
        "- Archive the CSV tables for reviewers and replication purposes.",
        "",
    ])
    return "\n".join(lines)


def _deduplicate_preserve_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _build_parameterized_journal_template(journal_preferences):
    preferences = {
        "main_text_policy": "balanced",
        "supplement_policy": "standard",
        "review_intensity": "standard",
        "article_format": "full_article",
    }
    if journal_preferences:
        preferences.update(journal_preferences)

    main_priority = [
        "manuscript_submission_result_package.md",
        "main_results_table.csv",
    ]
    supplementary_priority = [
        "innovation_metrics_report.md",
        "Biblio-HUB_reproducibility_report.md",
        "supplementary_table_index.csv",
    ]
    reviewer_priority = [
        "reviewer_response_material_package.md",
        "evidence_mapping.csv",
    ]
    notes = []

    if preferences["main_text_policy"] == "compact":
        main_priority = [
            "manuscript_submission_result_package.md",
            "main_results_table.csv",
            "caption_templates.md",
        ]
        notes.append("Compact main-text policy keeps the manuscript core short and shifts technical detail outward.")
    elif preferences["main_text_policy"] == "evidence_dense":
        main_priority.extend([
            "manuscript_submission_case_package.md",
            "caption_templates.md",
            "figure_table_crosswalk.csv",
        ])
        notes.append("Evidence-dense main-text policy promotes more tables and mapping sheets into the manuscript-facing set.")
    else:
        main_priority.extend([
            "manuscript_submission_case_package.md",
            "caption_templates.md",
        ])
        notes.append("Balanced main-text policy keeps core narrative plus limited assembly aids in the manuscript package.")

    if preferences["supplement_policy"] == "supplement_heavy":
        supplementary_priority.extend([
            "manuscript_figure_explanation_package.md",
            "figure_table_crosswalk.csv",
            "bibliographic_coupling_pairs.csv",
            "bibliographic_coupling_top_papers.csv",
        ])
        notes.append("Supplement-heavy policy moves ranked evidence and explanation documents into supplementary files.")
    elif preferences["supplement_policy"] == "minimal":
        supplementary_priority = [
            "innovation_metrics_report.md",
            "supplementary_table_index.csv",
        ]
        notes.append("Minimal supplementary policy keeps appendices lean and reserves only high-value supporting material.")
    else:
        supplementary_priority.extend([
            "manuscript_figure_explanation_package.md",
            "figure_table_crosswalk.csv",
        ])
        notes.append("Standard supplementary policy keeps reproducibility and figure interpretation available without overloading appendices.")

    if preferences["review_intensity"] == "reviewer_friendly":
        reviewer_priority.extend([
            "anticipated_questions.csv",
            "manuscript_figure_explanation_package.md",
        ])
        notes.append("Reviewer-friendly setting prioritizes traceability and rebuttal convenience.")
    elif preferences["review_intensity"] == "revision_ready":
        reviewer_priority.extend([
            "anticipated_questions.csv",
            "manuscript_figure_explanation_package.md",
            "Biblio-HUB_reproducibility_report.md",
        ])
        notes.append("Revision-ready setting assumes likely reviewer follow-up and adds methods transparency to the appendix set.")
    else:
        reviewer_priority.append("anticipated_questions.csv")
        notes.append("Standard review setting prepares concise rebuttal support without overexpanding reviewer appendices.")

    if preferences["article_format"] == "short_article":
        main_priority = [
            "manuscript_submission_result_package.md",
            "main_results_table.csv",
        ] + [item for item in main_priority if item not in {"manuscript_submission_result_package.md", "main_results_table.csv"}]
        supplementary_priority.extend([
            "manuscript_submission_case_package.md",
        ])
        notes.append("Short-article mode minimizes manuscript payload and relocates contextual support to supplementary files.")
    elif preferences["article_format"] == "rapid_communication":
        main_priority = [
            "manuscript_submission_result_package.md",
            "main_results_table.csv",
            "caption_templates.md",
        ]
        supplementary_priority.extend([
            "innovation_metrics_report.md",
            "supplementary_table_index.csv",
        ])
        notes.append("Rapid-communication mode emphasizes a fast, compact manuscript package with minimal but clear support files.")
    else:
        notes.append("Full-article mode keeps enough contextual and methodological material available for standard manuscript development.")

    template_name = (
        f"Target Journal Preference Template - "
        f"{preferences['main_text_policy'].replace('_', ' ').title()} / "
        f"{preferences['supplement_policy'].replace('_', ' ').title()} / "
        f"{preferences['review_intensity'].replace('_', ' ').title()} / "
        f"{preferences['article_format'].replace('_', ' ').title()}"
    )
    positioning = (
        "Auto-generated template aligned to the selected target-journal preferences "
        "for manuscript density, supplementary emphasis, reviewer support, and article format."
    )
    editor_note = " ".join(notes)

    return {
        "template_id": "parameterized_target_journal",
        "template_name": template_name,
        "positioning": positioning,
        "main_manuscript_priority": _deduplicate_preserve_order(main_priority),
        "supplementary_priority": _deduplicate_preserve_order(supplementary_priority),
        "reviewer_appendix_priority": _deduplicate_preserve_order(reviewer_priority),
        "editor_note": editor_note,
        "preferences": preferences,
    }


def build_parameterized_journal_template(journal_preferences=None):
    """Build the current target-journal template from the selected preferences."""
    return _build_parameterized_journal_template(journal_preferences)


def build_journal_submission_package_snapshot(
    submission_snapshot,
    figure_package_snapshot,
    reviewer_snapshot,
    journal_preferences=None,
):
    """
    Build a journal-oriented package plan that reorganizes exported outputs
    into main-manuscript, supplementary, and reviewer-appendix groups.
    """
    main_manuscript = [
        {
            "artifact": "manuscript_submission_result_package.md",
            "role": "Core result narrative and metric summary",
            "why": "Serves as the main drafting basis for Results and submission-facing highlights.",
        },
        {
            "artifact": "main_results_table.csv",
            "role": "Primary metrics table",
            "why": "Provides a compact numerical summary that can be adapted into the main manuscript tables.",
        },
        {
            "artifact": "caption_templates.md",
            "role": "Figure and table caption seed file",
            "why": "Supports direct manuscript assembly for primary figures and tables.",
        },
    ]

    supplementary = [
        {
            "artifact": "innovation_metrics_report.md",
            "role": "Innovation-oriented supporting report",
            "why": "Expands on bibliographic coupling and disruption analysis beyond the main text.",
        },
        {
            "artifact": "Biblio-HUB_reproducibility_report.md",
            "role": "Methods and reproducibility appendix",
            "why": "Provides audit-ready metadata coverage and parameter traceability.",
        },
        {
            "artifact": "manuscript_figure_explanation_package.md",
            "role": "Figure and table guidance report",
            "why": "Helps explain how visual outputs map onto Results and Supplementary Materials.",
        },
        {
            "artifact": "supplementary_table_index.csv",
            "role": "Supplementary table registry",
            "why": "Helps editors and authors keep appendix tables organized.",
        },
        {
            "artifact": "figure_table_crosswalk.csv",
            "role": "Figure-table linkage sheet",
            "why": "Shows how each major figure is backed by ranked evidence tables.",
        },
    ]

    reviewer_appendix = [
        {
            "artifact": "reviewer_response_material_package.md",
            "role": "Reviewer-facing response draft",
            "why": "Collects common questions, innovation claims, and conservative response phrasing.",
        },
        {
            "artifact": "evidence_mapping.csv",
            "role": "Evidence traceability sheet",
            "why": "Lets reviewers trace narrative claims to exact figures and tables.",
        },
        {
            "artifact": "anticipated_questions.csv",
            "role": "Reviewer objection checklist",
            "why": "Supports revision planning and rebuttal drafting.",
        },
    ]

    if submission_snapshot.get("supplementary_table_index"):
        supplementary_count = len(submission_snapshot["supplementary_table_index"])
    else:
        supplementary_count = 0
    execution_policy = submission_snapshot.get("methods_support", {}).get("execution_policy", {})
    execution_policy_note = ""
    if execution_policy:
        execution_policy_note = f"Robustness export policy: {format_execution_policy_summary(execution_policy)}."

    parameterized_template = _build_parameterized_journal_template(journal_preferences)

    template_variants = [
        parameterized_template,
        {
            "template_id": "balanced_default",
            "template_name": "Balanced Default",
            "positioning": "General-purpose submission layout balancing main text readability and supplementary completeness.",
            "main_manuscript_priority": [
                "manuscript_submission_result_package.md",
                "manuscript_submission_case_package.md",
                "main_results_table.csv",
            ],
            "supplementary_priority": [
                "innovation_metrics_report.md",
                "Biblio-HUB_reproducibility_report.md",
                "supplementary_table_index.csv",
            ],
            "reviewer_appendix_priority": [
                "reviewer_response_material_package.md",
                "evidence_mapping.csv",
            ],
            "editor_note": "Recommended when the journal allows a standard main text plus supplementary appendix structure.",
        },
        {
            "template_id": "strict_supplement_style",
            "template_name": "Strict Supplement Style",
            "positioning": "Keep the main manuscript compact and push most technical evidence to supplementary files.",
            "main_manuscript_priority": [
                "manuscript_submission_result_package.md",
                "main_results_table.csv",
                "caption_templates.md",
            ],
            "supplementary_priority": [
                "innovation_metrics_report.md",
                "Biblio-HUB_reproducibility_report.md",
                "manuscript_figure_explanation_package.md",
                "supplementary_table_index.csv",
                "figure_table_crosswalk.csv",
            ],
            "reviewer_appendix_priority": [
                "reviewer_response_material_package.md",
                "anticipated_questions.csv",
            ],
            "editor_note": "Suitable for journals with tight main-text limits and strong preference for supplementary appendices.",
        },
        {
            "template_id": "results_heavy_style",
            "template_name": "Results-Heavy Style",
            "positioning": "Bring more evidence into the main package for journals that emphasize dense Results sections.",
            "main_manuscript_priority": [
                "manuscript_submission_result_package.md",
                "manuscript_submission_case_package.md",
                "main_results_table.csv",
                "caption_templates.md",
                "figure_table_crosswalk.csv",
            ],
            "supplementary_priority": [
                "innovation_metrics_report.md",
                "supplementary_table_index.csv",
                "bibliographic_coupling_pairs.csv",
            ],
            "reviewer_appendix_priority": [
                "reviewer_response_material_package.md",
                "evidence_mapping.csv",
                "anticipated_questions.csv",
            ],
            "editor_note": "Suitable when the journal expects strong empirical detail directly in the main Results narrative.",
        },
        {
            "template_id": "reviewer_friendly_style",
            "template_name": "Reviewer-Friendly Style",
            "positioning": "Maximize traceability and rebuttal convenience for review-intensive or revision-prone submissions.",
            "main_manuscript_priority": [
                "manuscript_submission_result_package.md",
                "main_results_table.csv",
            ],
            "supplementary_priority": [
                "innovation_metrics_report.md",
                "Biblio-HUB_reproducibility_report.md",
                "figure_table_crosswalk.csv",
                "supplementary_table_index.csv",
            ],
            "reviewer_appendix_priority": [
                "reviewer_response_material_package.md",
                "evidence_mapping.csv",
                "anticipated_questions.csv",
                "manuscript_figure_explanation_package.md",
            ],
            "editor_note": "Best for rounds where reviewer clarity, response mapping, and evidence traceability are especially important.",
        },
    ]

    return {
        "main_manuscript": main_manuscript,
        "supplementary": supplementary,
        "reviewer_appendix": reviewer_appendix,
        "template_variants": template_variants,
        "journal_preferences": parameterized_template["preferences"],
        "recommended_template": parameterized_template,
        "execution_policy": execution_policy,
        "execution_policy_note": execution_policy_note,
        "summary": {
            "main_manuscript_items": len(main_manuscript),
            "supplementary_items": len(supplementary),
            "reviewer_appendix_items": len(reviewer_appendix),
            "supplementary_table_count": supplementary_count,
            "figure_count": len(figure_package_snapshot.get("figure_items", [])),
            "reviewer_question_count": len(reviewer_snapshot.get("anticipated_questions", [])),
            "template_variant_count": len(template_variants),
        },
        "assembly_notes": [
            "Keep the main manuscript package compact and reserve ranked long tables for supplementary files.",
            "Place reproducibility and figure-explanation materials in supplementary files unless the journal explicitly requests them in the main package.",
            "Use the reviewer appendix only for revision rounds, reviewer queries, or editorial follow-up.",
        ],
    }


def build_journal_submission_package_report(snapshot):
    """
    Build a report describing how to assemble a journal-oriented submission package.
    """
    summary = snapshot["summary"]
    lines = [
        "# Journal Submission Version Package",
        "",
        "## 1. Package Summary",
        f"- Main manuscript items: {summary['main_manuscript_items']}",
        f"- Supplementary items: {summary['supplementary_items']}",
        f"- Reviewer appendix items: {summary['reviewer_appendix_items']}",
        f"- Supplementary tables referenced: {summary['supplementary_table_count']}",
        f"- Figures referenced: {summary['figure_count']}",
        f"- Reviewer questions prepared: {summary['reviewer_question_count']}",
        f"- Template variants available: {summary.get('template_variant_count', 0)}",
    ]
    if snapshot.get("execution_policy_note"):
        lines.extend(
            [
                f"- Large-sample export notice: {snapshot['execution_policy_note']}",
                "- Report this policy consistently in the manuscript Methods, submission letter, and supplementary files when export acceleration is used.",
            ]
        )
    lines.extend([
        "",
        "## 2. Main Manuscript Package",
    ])
    for item in snapshot["main_manuscript"]:
        lines.extend(
            [
                f"### {item['artifact']}",
                f"- Role: {item['role']}",
                f"- Why include it here: {item['why']}",
                "",
            ]
        )

    lines.extend([
        "## 3. Supplementary Package",
    ])
    for item in snapshot["supplementary"]:
        lines.extend(
            [
                f"### {item['artifact']}",
                f"- Role: {item['role']}",
                f"- Why include it here: {item['why']}",
                "",
            ]
        )

    lines.extend([
        "## 4. Reviewer Appendix Package",
    ])
    for item in snapshot["reviewer_appendix"]:
        lines.extend(
            [
                f"### {item['artifact']}",
                f"- Role: {item['role']}",
                f"- Why include it here: {item['why']}",
                "",
            ]
        )

    lines.extend([
        "## 5. Target Journal Preferences",
        f"- Main text policy: {snapshot.get('journal_preferences', {}).get('main_text_policy', 'balanced')}",
        f"- Supplement policy: {snapshot.get('journal_preferences', {}).get('supplement_policy', 'standard')}",
        f"- Review intensity: {snapshot.get('journal_preferences', {}).get('review_intensity', 'standard')}",
        f"- Article format: {snapshot.get('journal_preferences', {}).get('article_format', 'full_article')}",
    ])
    lines.extend([
        "",
        "## 6. Recommended Parameterized Template",
    ])
    recommended = snapshot.get("recommended_template", {})
    if recommended:
        lines.extend(
            [
                f"### {recommended.get('template_name', 'Target Journal Preference Template')}",
                f"- Positioning: {recommended.get('positioning', '')}",
                f"- Main manuscript priority: {', '.join(recommended.get('main_manuscript_priority', []))}",
                f"- Supplementary priority: {', '.join(recommended.get('supplementary_priority', []))}",
                f"- Reviewer appendix priority: {', '.join(recommended.get('reviewer_appendix_priority', []))}",
                f"- Editor note: {recommended.get('editor_note', '')}",
                "",
            ]
        )

    lines.extend([
        "## 7. Journal Template Variants",
    ])
    for item in snapshot.get("template_variants", []):
        lines.extend(
            [
                f"### {item['template_name']}",
                f"- Positioning: {item['positioning']}",
                f"- Main manuscript priority: {', '.join(item['main_manuscript_priority'])}",
                f"- Supplementary priority: {', '.join(item['supplementary_priority'])}",
                f"- Reviewer appendix priority: {', '.join(item['reviewer_appendix_priority'])}",
                f"- Editor note: {item['editor_note']}",
                "",
            ]
        )

    lines.extend([
        "## 8. Assembly Notes",
    ])
    lines.extend(f"- {item}" for item in snapshot["assembly_notes"])
    lines.append("")
    return "\n".join(lines)


def build_submission_figure_package_snapshot(submission_snapshot, image_format="png"):
    """
    Build a structured figure-and-table explanation package for manuscript submission.
    """
    figure_items = []
    for index, item in enumerate(submission_snapshot.get("recommended_figures", []), start=1):
        figure_items.append(
            {
                "id": f"F{index}",
                "name": item["name"],
                "filename_suggestion": f"figure_{index:02d}_{item['name'].lower().replace(' ', '_')}.{image_format}",
                "target_section": "Results",
                "caption": item["caption"],
                "methods_note": (
                    "Report the analysis thresholds, node/edge meaning, and any non-default filtering rules used to generate this figure."
                ),
                "reviewer_note": (
                    "Use this figure to explain the structural evidence behind the claimed knowledge organization or innovation pattern."
                ),
                "result_mapping": (
                    "Supports the narrative on knowledge structure, community formation, or innovation dynamics in the focal domain."
                ),
            }
        )

    table_items = []
    for index, item in enumerate(submission_snapshot.get("recommended_tables", []), start=1):
        table_items.append(
            {
                "id": f"T{index}",
                "name": item["name"],
                "filename_suggestion": f"table_{index:02d}_{item['name'].lower().replace(' ', '_')}.csv",
                "target_section": "Results",
                "caption": item["caption"],
                "methods_note": (
                    "State the ranking rule, threshold condition, and any tie-breaking logic used when assembling this table."
                ),
                "reviewer_note": (
                    "Use this table to provide exact ranked evidence for the narrative described in the corresponding figure."
                ),
                "result_mapping": (
                    "Provides reviewer-readable ranked evidence to complement the visual interpretation in the figures."
                ),
            }
        )

    return {
        "image_format": image_format.lower(),
        "figure_items": figure_items,
        "table_items": table_items,
        "assembly_notes": [
            "Keep figure numbering consistent with the manuscript body and supplementary material.",
            "Use vector formats for network figures when journal submission allows SVG or PDF.",
            "Pair each figure with the matching ranked table when explaining innovation metrics.",
        ],
    }


def build_submission_figure_package_report(submission_snapshot, figure_package_snapshot):
    """
    Build a manuscript-oriented figure/table explanation report.
    """
    lines = [
        "# Manuscript Figure and Table Explanation Package",
        "",
        "## 1. Package Scope",
        f"- Suggested image format: {figure_package_snapshot['image_format']}",
        f"- Figure items: {len(figure_package_snapshot['figure_items'])}",
        f"- Table items: {len(figure_package_snapshot['table_items'])}",
        "",
        "## 2. Figure Guidance",
    ]

    if figure_package_snapshot["figure_items"]:
        for item in figure_package_snapshot["figure_items"]:
            lines.extend(
                [
                    f"### {item['id']}. {item['name']}",
                    f"- Suggested filename: {item['filename_suggestion']}",
                    f"- Manuscript section: {item['target_section']}",
                    f"- Caption: {item['caption']}",
                    f"- Methods note: {item['methods_note']}",
                    f"- Reviewer note: {item['reviewer_note']}",
                    f"- Result mapping: {item['result_mapping']}",
                    "",
                ]
            )
    else:
        lines.append("- No figure guidance is available in the current snapshot.")

    lines.extend([
        "## 3. Table Guidance",
    ])
    if figure_package_snapshot["table_items"]:
        for item in figure_package_snapshot["table_items"]:
            lines.extend(
                [
                    f"### {item['id']}. {item['name']}",
                    f"- Suggested filename: {item['filename_suggestion']}",
                    f"- Manuscript section: {item['target_section']}",
                    f"- Caption: {item['caption']}",
                    f"- Methods note: {item['methods_note']}",
                    f"- Reviewer note: {item['reviewer_note']}",
                    f"- Result mapping: {item['result_mapping']}",
                    "",
                ]
            )
    else:
        lines.append("- No table guidance is available in the current snapshot.")

    lines.extend([
        "## 4. Assembly Notes",
    ])
    lines.extend(f"- {item}" for item in figure_package_snapshot["assembly_notes"])

    lines.extend([
        "",
        "## 5. Result-to-Figure Mapping Summary",
    ])
    lines.extend(
        f"- {item['id']} / {item['name']}: {item['result_mapping']}"
        for item in figure_package_snapshot["figure_items"]
    )
    lines.extend(
        f"- {item['id']} / {item['name']}: {item['result_mapping']}"
        for item in figure_package_snapshot["table_items"]
    )
    lines.append("")
    return "\n".join(lines)


def build_reviewer_response_snapshot(submission_snapshot, figure_package_snapshot):
    """
    Build a structured package for reviewer-response materials.
    """
    submission_methods_support = submission_snapshot.get("methods_support", {})
    submission_methods_paragraph = submission_methods_support.get("methods_paragraph", "")
    submission_execution_policy = submission_methods_support.get("execution_policy", {})
    bc_metrics = (
        submission_snapshot.get("innovation_metrics", {})
        .get("bibliographic_coupling", {})
        .get("network_metrics", {})
    )
    di_summary = (
        submission_snapshot.get("innovation_metrics", {})
        .get("disruption_index", {})
        .get("summary", {})
    )
    structural_hole_summary = (
        submission_snapshot.get("innovation_metrics", {})
        .get("structural_hole", {})
        .get("summary", {})
    )

    innovation_claims = [
        {
            "theme": "Methodological Breadth",
            "claim": "The platform integrates bibliographic coupling, structural-hole brokerage, disruption index analysis, and conventional bibliometric structure analysis within one reproducible workflow.",
            "evidence": (
                f"Bibliographic coupling network metrics: nodes={bc_metrics.get('nodes', 0)}, "
                f"edges={bc_metrics.get('edges', 0)}, modularity={bc_metrics.get('modularity', 0):.4f}."
            ),
        },
        {
            "theme": "Innovation Dynamics",
            "claim": "The study does not only map topic structure but also quantifies innovation orientation through DI1-based disruption analysis.",
            "evidence": (
                f"Disruption summary: mean DI={di_summary.get('mean_di', 0.0):.4f}, "
                f"positive papers={di_summary.get('positive_count', 0)}, "
                f"negative papers={di_summary.get('negative_count', 0)}."
            ),
        },
        {
            "theme": "Bridge Evidence",
            "claim": "The study identifies bridge-like papers that connect otherwise separated knowledge communities instead of reporting only dense clusters.",
            "evidence": (
                f"Structural-hole summary: mean brokerage={structural_hole_summary.get('mean_brokerage', 0.0):.4f}, "
                f"top broker={structural_hole_summary.get('top_broker', 'N/A')}, "
                f"core brokers={structural_hole_summary.get('core_brokers', 0)}."
            ),
        },
        {
            "theme": "Reproducible Export",
            "claim": "All major results can be exported as manuscript-ready reports, structured snapshots, and reviewer-facing tables.",
            "evidence": "The platform provides submission reports, figure explanation packages, and structured result snapshots for auditability.",
        },
    ]

    anticipated_questions = [
        {
            "question": "Why is this tool different from standard bibliometric visualization software?",
            "response": "The platform combines conventional mapping outputs with innovation-oriented indicators such as bibliographic coupling and DI1, while also exporting manuscript-ready structured reports and reviewer-facing evidence packages.",
        },
        {
            "question": "How do you ensure the reported findings are reproducible?",
            "response": "All non-default thresholds, dataset coverage summaries, and algorithm profiles are exported through the reproducibility checklist and submission result package, allowing external reviewers to trace parameters and replicate results.",
        },
        {
            "question": "How should the innovation claims be interpreted conservatively?",
            "response": "The innovation metrics are presented as quantitative evidence for knowledge-structure and citation-dynamics analysis, not as a substitute for expert interpretation. Final scientific conclusions remain the responsibility of the researcher.",
        },
        {
            "question": "What extra value does structural-hole brokerage add beyond standard centrality metrics?",
            "response": "Brokerage analysis does not replace cluster or disruption metrics. Instead, it highlights bridge-like papers that connect relatively separated knowledge communities, offering a complementary perspective on how ideas may travel across subfields.",
        },
    ]

    reproducibility_responses = [
        "The dataset-level record count, year range, and field coverage are preserved in exported snapshots.",
        "All citation/network thresholds can be documented through the reproducibility checklist and result packages.",
        "Ranked tables are exported as CSV files so reviewers can inspect the exact evidence behind each claim.",
    ]
    if submission_methods_paragraph:
        reproducibility_responses.append(
            "Submission-oriented methods note is preserved in the structured snapshot for manuscript drafting."
        )
    if submission_execution_policy:
        reproducibility_responses.append(
            "If large-sample export acceleration was used, see the recorded execution_policy for the exact robustness setting."
        )

    evidence_mapping = []
    for item in figure_package_snapshot.get("figure_items", []):
        evidence_mapping.append(
            {
                "item_id": item["id"],
                "item_type": "figure",
                "name": item["name"],
                "supports_claim": item["result_mapping"],
                "reviewer_note": item["reviewer_note"],
            }
        )
    for item in figure_package_snapshot.get("table_items", []):
        evidence_mapping.append(
            {
                "item_id": item["id"],
                "item_type": "table",
                "name": item["name"],
                "supports_claim": item["result_mapping"],
                "reviewer_note": item["reviewer_note"],
            }
        )

    limitations = [
        "The disruption index is computed on the internal citation network induced by the uploaded dataset rather than a global citation graph.",
        "Bibliographic coupling strength depends on the completeness and formatting quality of cited-reference metadata.",
        "Automatically generated textual materials are structured drafting aids and should be checked by the researcher before submission.",
    ]
    limitations.extend(
        item for item in submission_snapshot.get("stated_limitations", [])
        if item not in limitations
    )

    return {
        "innovation_claims": innovation_claims,
        "anticipated_questions": anticipated_questions,
        "reproducibility_responses": reproducibility_responses,
        "evidence_mapping": evidence_mapping,
        "limitations": limitations,
    }


def build_reviewer_response_report(submission_snapshot, figure_package_snapshot, reviewer_snapshot):
    """
    Build a reviewer-response drafting report from the available analysis outputs.
    """
    lines = [
        "# Reviewer Response Material Package",
        "",
        "## 1. Suggested Innovation Claims",
    ]
    lines.extend(
        [
            f"### {index + 1}. {item['theme']}",
            f"- Claim: {item['claim']}",
            f"- Evidence: {item['evidence']}",
            "",
        ]
        for index, item in enumerate(reviewer_snapshot["innovation_claims"])
    )

    flattened_lines = []
    for block in lines:
        if isinstance(block, list):
            flattened_lines.extend(block)
        else:
            flattened_lines.append(block)
    lines = flattened_lines

    lines.extend([
        "## 2. Anticipated Reviewer Questions",
    ])
    for index, item in enumerate(reviewer_snapshot["anticipated_questions"], start=1):
        lines.extend(
            [
                f"### Q{index}. {item['question']}",
                f"- Draft response: {item['response']}",
                "",
            ]
        )

    lines.extend([
        "## 3. Reproducibility Response Notes",
    ])
    lines.extend(f"- {item}" for item in reviewer_snapshot["reproducibility_responses"])

    lines.extend([
        "",
        "## 4. Evidence Mapping",
    ])
    for item in reviewer_snapshot["evidence_mapping"]:
        lines.extend(
            [
                f"### {item['item_id']} ({item['item_type']}) {item['name']}",
                f"- Supports claim: {item['supports_claim']}",
                f"- Reviewer note: {item['reviewer_note']}",
                "",
            ]
        )

    lines.extend([
        "## 5. Stated Limitations",
    ])
    lines.extend(f"- {item}" for item in reviewer_snapshot["limitations"])

    lines.extend([
        "",
        "## 6. Author Reminder",
        "- Adapt each draft response to the exact wording of the reviewer comment.",
        "- Keep the final response evidence-based and cite the exported tables/figures explicitly.",
        "- Do not overstate innovation claims beyond what the exported metrics directly support.",
        "",
    ])
    return "\n".join(lines)


def build_research_report_snapshot(
    manuscript_snapshot,
    reproducibility_snapshot,
    innovation_snapshot,
    submission_snapshot,
    figure_package_snapshot,
    reviewer_snapshot,
    journal_submission_snapshot=None,
):
    """
    Aggregate the major exported materials into a single research-report snapshot.
    """
    year_range = manuscript_snapshot.get("year_range", [])
    year_range_label = (
        f"{year_range[0]}-{year_range[1]}"
        if len(year_range) == 2
        else "N/A"
    )
    changed_parameters = [
        item for item in reproducibility_snapshot.get("analysis_parameters", [])
        if item.get("changed")
    ]
    scale_profile = _classify_dataset_scale(manuscript_snapshot.get("records", 0))
    top_keywords = manuscript_snapshot.get("top_keywords", [])[:5]
    top_keyword_labels = ", ".join(item.get("label", "") for item in top_keywords if item.get("label")) or "N/A"
    top_journals = manuscript_snapshot.get("top_journals", [])[:3]
    top_journal_labels = ", ".join(item.get("label", "") for item in top_journals if item.get("label")) or "N/A"
    mean_di = (
        submission_snapshot.get("innovation_metrics", {})
        .get("disruption_index", {})
        .get("summary", {})
        .get("mean_di", 0.0)
    )
    bc_nodes = (
        submission_snapshot.get("innovation_metrics", {})
        .get("bibliographic_coupling", {})
        .get("network_metrics", {})
        .get("nodes", 0)
    )
    submission_methods_support = submission_snapshot.get("methods_support", {})
    execution_policy = submission_methods_support.get("execution_policy", {})
    journal_submission_snapshot = journal_submission_snapshot or {}
    journal_preferences = journal_submission_snapshot.get(
        "journal_preferences",
        reproducibility_snapshot.get("submission_preferences", {}).get("journal_preferences", {}),
    )
    recommended_template = journal_submission_snapshot.get(
        "recommended_template",
        reproducibility_snapshot.get("submission_preferences", {}).get("recommended_template", {}),
    )
    headline_findings = [
        f"Dataset spans {year_range_label} with {manuscript_snapshot.get('records', 0)} records.",
        (
            f"Keyword space includes {manuscript_snapshot.get('unique_keywords', 0)} unique terms "
            f"and {reproducibility_snapshot.get('keyword_statistics', {}).get('cooccurrence_pairs', 0)} co-occurrence pairs."
        ),
        (
            f"Bibliographic coupling network contains "
            f"{submission_snapshot.get('innovation_metrics', {}).get('bibliographic_coupling', {}).get('network_metrics', {}).get('nodes', 0)} nodes."
        ),
        (
            f"Disruption analysis reports mean DI="
            f"{submission_snapshot.get('innovation_metrics', {}).get('disruption_index', {}).get('summary', {}).get('mean_di', 0.0):.4f}."
        ),
        (
            f"Structural-hole brokerage identifies "
            f"{submission_snapshot.get('innovation_metrics', {}).get('structural_hole', {}).get('summary', {}).get('core_brokers', 0)} core brokers."
        ),
    ]
    if journal_preferences:
        headline_findings.append(
            "Journal submission preferences are recorded for "
            f"{journal_preferences.get('main_text_policy', 'balanced')} main text, "
            f"{journal_preferences.get('supplement_policy', 'standard')} supplement, "
            f"{journal_preferences.get('review_intensity', 'standard')} review intensity, and "
            f"{journal_preferences.get('article_format', 'full_article')} article format."
        )
    narrative_templates = [
        (
            "The dataset covers "
            f"{year_range_label} and comprises {manuscript_snapshot.get('records', 0)} records, "
            f"suggesting a {scale_profile['tier']}-scale evidence base for the current bibliometric investigation."
        ),
        (
            f"Core publication outlets include {top_journal_labels}, while the dominant keyword profile is characterized by {top_keyword_labels}."
        ),
        (
            f"Structural analysis shows a bibliographic coupling network with {bc_nodes} nodes, supporting interpretation of shared knowledge bases across focal studies."
        ),
        (
            f"Innovation-oriented citation dynamics are summarized by a mean disruption index of {mean_di:.4f}, which can be used to contrast more disruptive versus consolidating publications."
        ),
        (
            f"Bridge evidence is captured through structural-hole brokerage, which highlights papers that connect otherwise separated knowledge communities."
        ),
    ]
    manuscript_blueprint = [
        {
            "section": "Introduction",
            "purpose": "Define the research scope and motivate why a bibliometric investigation is needed for the uploaded corpus.",
            "suggested_evidence": "Use dataset size, year range, and journal concentration as contextual anchors.",
        },
        {
            "section": "Methods",
            "purpose": "Report data source provenance, preprocessing rules, and non-default analysis parameters.",
            "suggested_evidence": "Reuse the reproducibility checklist, changed-parameter summary, and target-journal preference record.",
        },
        {
            "section": "Results",
            "purpose": "Present descriptive patterns, knowledge structure, and innovation indicators in a logical order.",
            "suggested_evidence": "Use publication trends, top keywords, bibliographic coupling, and disruption index outputs.",
        },
        {
            "section": "Discussion",
            "purpose": "Interpret the relationship between structural patterns and innovation dynamics without overclaiming causality.",
            "suggested_evidence": "Link innovation claims, stated limitations, and reviewer-facing conservative interpretations.",
        },
    ]
    methods_support = {
        "methods_evidence_map": reproducibility_snapshot.get("methods_evidence_map", []),
        "methods_writing_pack": reproducibility_snapshot.get("methods_writing_pack", {}),
        "parameter_change_summary": reproducibility_snapshot.get("parameter_change_summary", []),
        "submission_methods_note": submission_methods_support.get("methods_paragraph", ""),
        "execution_policy": execution_policy,
    }
    output_assembly = []
    preferred_outputs = submission_snapshot.get("recommended_output_sequence", [])
    if preferred_outputs:
        for idx, item in enumerate(preferred_outputs, start=1):
            output_kind = item.get("output_kind", "figure")
            name = item.get("name", f"{output_kind.title()} {idx}")
            placement = _classify_output_placement(name, output_kind)
            output_assembly.append(
                {
                    "type": output_kind,
                    "order": idx,
                    "priority_rank": item.get("priority_rank", idx),
                    "priority_reason": item.get("priority_reason", ""),
                    "name": name,
                    "caption": item.get("caption", ""),
                    "target_section": placement["section"],
                    "placement_reason": placement["reason"],
                }
            )
    else:
        for idx, item in enumerate(submission_snapshot.get("recommended_figures", []), start=1):
            name = item.get("name", f"Figure {idx}")
            placement = _classify_output_placement(name, "figure")
            output_assembly.append(
                {
                    "type": "figure",
                    "order": idx,
                    "name": name,
                    "caption": item.get("caption", ""),
                    "target_section": placement["section"],
                    "placement_reason": placement["reason"],
                }
            )
        for idx, item in enumerate(submission_snapshot.get("recommended_tables", []), start=1):
            name = item.get("name", f"Table {idx}")
            placement = _classify_output_placement(name, "table")
            output_assembly.append(
                {
                    "type": "table",
                    "order": idx,
                    "name": name,
                    "caption": item.get("caption", ""),
                    "target_section": placement["section"],
                    "placement_reason": placement["reason"],
                }
            )
    type_specific_templates = _build_type_specific_narrative_templates(
        manuscript_snapshot,
        reproducibility_snapshot,
        submission_snapshot,
    )

    return {
        "dataset_overview": {
            "records": manuscript_snapshot.get("records", 0),
            "year_range": year_range,
            "year_range_label": year_range_label,
            "scale_profile": scale_profile,
            "unique_journals": manuscript_snapshot.get("unique_journals", 0),
            "unique_keywords": manuscript_snapshot.get("unique_keywords", 0),
            "doi_coverage": manuscript_snapshot.get("doi_coverage", 0.0),
            "abstract_coverage": manuscript_snapshot.get("abstract_coverage", 0.0),
            "author_coverage": manuscript_snapshot.get("author_coverage", 0.0),
        },
        "component_counts": {
            "changed_parameters": len(changed_parameters),
            "recommended_figures": len(submission_snapshot.get("recommended_figures", [])),
            "recommended_tables": len(submission_snapshot.get("recommended_tables", [])),
            "figure_guides": len(figure_package_snapshot.get("figure_items", [])),
            "table_guides": len(figure_package_snapshot.get("table_items", [])),
            "reviewer_questions": len(reviewer_snapshot.get("anticipated_questions", [])),
            "journal_template_variants": len(journal_submission_snapshot.get("template_variants", [])),
            "chapter_target_sections": len(submission_snapshot.get("chapter_target_output_plan", [])),
            "methods_mapping_rows": len(reproducibility_snapshot.get("methods_evidence_map", [])),
        },
        "headline_findings": headline_findings,
        "result_narrative_templates": narrative_templates,
        "figure_type_narrative_templates": type_specific_templates,
        "manuscript_blueprint": manuscript_blueprint,
        "output_assembly_plan": output_assembly,
        "chapter_target_output_plan": submission_snapshot.get("chapter_target_output_plan", []),
        "methods_support": methods_support,
        "non_default_parameters": changed_parameters,
        "submission_preferences": {
            "journal_preferences": journal_preferences,
            "recommended_template": recommended_template,
        },
        "innovation_claims": reviewer_snapshot.get("innovation_claims", []),
        "recommended_outputs": {
            "figures": submission_snapshot.get("recommended_figures", []),
            "tables": submission_snapshot.get("recommended_tables", []),
        },
        "limitations": reviewer_snapshot.get("limitations", []),
        "component_snapshots": {
            "manuscript_case": manuscript_snapshot,
            "reproducibility": reproducibility_snapshot,
            "innovation_metrics": innovation_snapshot,
            "submission_result": submission_snapshot,
            "figure_explanation": figure_package_snapshot,
            "reviewer_response": reviewer_snapshot,
            "journal_submission": journal_submission_snapshot,
        },
    }


def build_research_report(
    research_snapshot,
    manuscript_report,
    reproducibility_report,
    innovation_report,
    submission_report,
    figure_package_report,
    reviewer_report,
):
    """
    Build a one-click master research report by organizing previously exported reports.
    """
    overview = research_snapshot["dataset_overview"]
    counts = research_snapshot["component_counts"]
    scale_profile = overview.get("scale_profile", {"label": "Unknown", "guidance": "No guidance available."})
    lines = [
        "# One-Click Bibliometrics Research Report",
        "",
        "## 1. Executive Overview",
        f"- Records analyzed: {overview['records']}",
        f"- Year range: {overview['year_range_label']}",
        f"- Dataset scale: {scale_profile['label']}",
        f"- Unique journals: {overview['unique_journals']}",
        f"- Unique keywords: {overview['unique_keywords']}",
        f"- DOI coverage: {overview['doi_coverage']:.1%}",
        f"- Abstract coverage: {overview['abstract_coverage']:.1%}",
        f"- Author coverage: {overview['author_coverage']:.1%}",
        "",
        "## 2. Reporting Strategy",
        f"- Scale-specific guidance: {scale_profile['guidance']}",
        "- Recommended writing mode: treat this report as a structured drafting baseline and refine claims manually before submission.",
        "- Suggested author workflow: select core figures first, adapt result narrative templates second, and finalize parameter reporting third.",
        "",
        "## 3. Research Package Summary",
        f"- Changed parameters recorded: {counts['changed_parameters']}",
        f"- Recommended figures: {counts['recommended_figures']}",
        f"- Recommended tables: {counts['recommended_tables']}",
        f"- Figure guidance notes: {counts['figure_guides']}",
        f"- Table guidance notes: {counts['table_guides']}",
        f"- Reviewer questions prepared: {counts['reviewer_questions']}",
        f"- Journal template variants available: {counts.get('journal_template_variants', 0)}",
        f"- Chapter-target sections prepared: {counts.get('chapter_target_sections', 0)}",
        f"- Methods mapping rows: {counts.get('methods_mapping_rows', 0)}",
        "",
        "## 4. Target Journal Submission Strategy",
    ]
    submission_preferences = research_snapshot.get("submission_preferences", {})
    journal_preferences = submission_preferences.get("journal_preferences", {})
    recommended_template = submission_preferences.get("recommended_template", {})
    if journal_preferences:
        lines.extend(
            [
                f"- Main text policy: {journal_preferences.get('main_text_policy', 'balanced')}",
                f"- Supplement policy: {journal_preferences.get('supplement_policy', 'standard')}",
                f"- Review intensity: {journal_preferences.get('review_intensity', 'standard')}",
                f"- Article format: {journal_preferences.get('article_format', 'full_article')}",
            ]
        )
    else:
        lines.append("- No target-journal submission preferences were recorded.")
    if recommended_template:
        lines.extend(
            [
                f"- Recommended template: {recommended_template.get('template_id', '')} / {recommended_template.get('template_name', '')}",
                f"- Template note: {recommended_template.get('editor_note', '')}",
            ]
        )

    lines.extend([
        "",
        "## 5. Headline Findings",
    ])
    lines.extend(f"- {item}" for item in research_snapshot["headline_findings"])
    lines.extend([
        "",
        "## 6. Result Narrative Templates",
    ])
    lines.extend(f"- {item}" for item in research_snapshot["result_narrative_templates"])
    lines.extend([
        "",
        "## 7. Figure-Type Narrative Templates",
    ])
    if research_snapshot.get("figure_type_narrative_templates"):
        for item in research_snapshot["figure_type_narrative_templates"]:
            lines.extend(
                [
                    f"### {item['output_kind'].title()}: {item['name']}",
                    f"- Narrative type: {item['narrative_type']}",
                    f"- Template: {item['template']}",
                    f"- Interpretation focus: {item['focus']}",
                    "",
                ]
            )
    else:
        lines.append("- No figure-type narrative templates are available.")

    lines.extend([
        "## 8. Manuscript Blueprint",
    ])
    for item in research_snapshot["manuscript_blueprint"]:
        lines.extend(
            [
                f"### {item['section']}",
                f"- Purpose: {item['purpose']}",
                f"- Suggested evidence: {item['suggested_evidence']}",
                "",
            ]
        )
    lines.extend([
        "## 9. Methods Evidence Pack",
    ])
    methods_support = research_snapshot.get("methods_support", {})
    methods_map = methods_support.get("methods_evidence_map", [])
    methods_writing_pack = methods_support.get("methods_writing_pack", {})
    parameter_change_summary = methods_support.get("parameter_change_summary", [])
    if methods_map:
        lines.extend([
            "| Step | Algorithm or Rule | Key Parameters | Evidence Output | Manuscript Use |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for item in methods_map:
            lines.append(
                f"| {item['step']} | {item['algorithm_or_rule']} | {item['key_parameters']} | {item['evidence_output']} | {item['manuscript_use']} |"
            )
    else:
        lines.append("- No methods evidence map is available.")
    if parameter_change_summary:
        lines.extend([
            "",
            "| Module | Changed Parameters | Current Settings | Default Settings | Methods Note |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ])
        for item in parameter_change_summary:
            lines.append(
                f"| {item['module']} | {item['changed_parameters']} | {item['current_settings']} | {item['default_settings']} | {item['methods_note']} |"
            )
    if methods_writing_pack:
        lines.extend([
            "",
            f"- Data source paragraph: {methods_writing_pack.get('data_source_paragraph', '')}",
            f"- Pipeline paragraph: {methods_writing_pack.get('pipeline_paragraph', '')}",
            f"- Parameter paragraph: {methods_writing_pack.get('parameter_paragraph', '')}",
            f"- Reproducibility paragraph: {methods_writing_pack.get('reproducibility_paragraph', '')}",
            f"- Submission alignment paragraph: {methods_writing_pack.get('submission_alignment_paragraph', '')}",
        ])
    if methods_support.get("execution_policy"):
        lines.append("- Submission execution settings are retained in the structured snapshot; use the journal submission guide for the concise large-sample export notice.")

    lines.extend([
        "",
        "## 10. Innovation Claims Snapshot",
    ])
    for item in research_snapshot["innovation_claims"]:
        lines.extend(
            [
                f"### {item['theme']}",
                f"- Claim: {item['claim']}",
                f"- Evidence: {item['evidence']}",
                "",
            ]
        )

    lines.extend([
        "## 11. Non-Default Parameters to Report",
    ])
    if research_snapshot["non_default_parameters"]:
        lines.extend(
            f"- {item['group']} / {item['label']}: {item['value']} (default={item['default']})"
            for item in research_snapshot["non_default_parameters"]
        )
    else:
        lines.append("- No non-default parameters were recorded in the current session.")

    lines.extend([
        "",
        "## 12. Chapter-Target Output Plan",
        "- This view reorganizes the same evidence package for different manuscript chapters so you can draft section by section.",
    ])
    for chapter in research_snapshot.get("chapter_target_output_plan", []):
        lines.extend([
            f"### Chapter: {chapter['chapter']}",
            f"- Chapter note: {chapter['chapter_note']}",
        ])
        for item in chapter.get("recommended_items", []):
            lines.append(
                f"- C{item.get('chapter_rank', 0)} / P{item.get('priority_rank', 0)} {item['output_kind'].title()} {item['name']}: {item.get('chapter_reason', '')}"
            )
        lines.append("")

    lines.extend([
        "## 13. Recommended Output Assembly Plan",
        "- This plan organizes your visual evidence by suggested manuscript section based on common bibliometric reporting standards.",
    ])
    
                        
    assembly_plan = research_snapshot.get("output_assembly_plan", [])
    grouped_plan = {}
    for item in assembly_plan:
        section = item.get("target_section", "Other")
        if section not in grouped_plan:
            grouped_plan[section] = []
        grouped_plan[section].append(item)
    
                                      
    section_order = [
        "Introduction or Results",
        "Methods or Supplementary",
        "Results (Structural)",
        "Results (Relational/Innovation)",
        "Results",
        "Discussion",
        "Results or Supplementary",
        "Other"
    ]
    
    for section in section_order:
        if section in grouped_plan:
            lines.extend([
                f"### Section: {section}",
            ])
            for item in grouped_plan[section]:
                label = f"P{item.get('priority_rank', item.get('order', 0))}"
                lines.extend([
                    f"- **{item['type'].title()} {label}**: {item['name']}",
                    f"  - Caption: {item['caption']}",
                    f"  - Reasoning: {item.get('placement_reason', 'Standard placement.')}",
                ])
                if item.get("priority_reason"):
                    lines.append(f"  - Priority rationale: {item['priority_reason']}")
            lines.append("")

    lines.extend([
        "## 14. Stated Limitations",
    ])
    lines.extend(f"- {item}" for item in research_snapshot["limitations"])

    appendix_sections = [
        ("Appendix A. Manuscript Case Report", manuscript_report),
        ("Appendix B. Reproducibility Checklist", reproducibility_report),
        ("Appendix C. Innovation Metrics Report", innovation_report),
        ("Appendix D. Submission Result Package Report", submission_report),
        ("Appendix E. Figure Explanation Package", figure_package_report),
        ("Appendix F. Reviewer Response Material Package", reviewer_report),
    ]
    for title, body in appendix_sections:
        lines.extend([
            "",
            f"## {title}",
            "",
            body,
        ])

    lines.append("")
    return "\n".join(lines)
