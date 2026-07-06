from collections import Counter
import re

import pandas as pd

from modules.data_pipeline import _extract_country_from_affiliation
from modules.keyword_pipeline import split_semicolon_terms


INSTITUTION_HINTS = (
    "univ",
    "univers",
    "institute",
    "institut",
    "college",
    "academy",
    "acad",
    "hospital",
    "centre",
    "center",
    "lab",
    "laboratory",
    "school",
)

SUBUNIT_HINTS = (
    "dept",
    "department",
    "faculty",
    "school",
    "college",
    "lab",
    "laboratory",
    "center",
    "centre",
    "program",
    "programme",
    "division",
    "unit",
)


def _select_representative_institution(addr_text):
    segments = [segment.strip().rstrip(".") for segment in addr_text.split(",") if segment.strip()]
    if not segments:
        return ""

    for segment in segments:
        lowered = segment.lower()
        if any(hint in lowered for hint in INSTITUTION_HINTS):
            return segment

    for idx, segment in enumerate(segments[:-1]):
        lowered = segment.lower()
        if any(hint in lowered for hint in SUBUNIT_HINTS):
            next_segment = segments[idx + 1]
            if next_segment:
                return next_segment

    return segments[0]


def extract_institutions_from_affiliation(aff_str):
    if pd.isna(aff_str) or str(aff_str).strip() in ("", "nan"):
        return []

    institutions = set()
    text = str(aff_str).strip()
    addr_blocks = re.split(r";\s*(?=\[)", text)
    if len(addr_blocks) <= 1:
        addr_blocks = [text]

    for block in addr_blocks:
        block = block.strip()
        if not block:
            continue
        if "]" in block:
            institution = block.split("]", 1)[1].strip()
        else:
            institution = block
        institution = _select_representative_institution(institution)
        if institution and len(institution) > 1:
            institutions.add(institution)

    return sorted(institutions)


def build_top_entity_table(counter, entity_label, top_n=10):
    total = sum(counter.values())
    rows = []
    for rank, (entity, count) in enumerate(counter.most_common(top_n), start=1):
        rows.append(
            {
                "Rank": rank,
                entity_label: entity,
                "Papers": count,
                "Share (%)": round(count * 100 / total, 2) if total > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_top_journal_table(df, top_n=10):
    if "Journal" not in df.columns:
        return pd.DataFrame()
    journal_series = df["Journal"].fillna("").astype(str).str.strip()
    journal_series = journal_series[(journal_series != "") & (journal_series != "nan")]
    return build_top_entity_table(Counter(journal_series.tolist()), "Journal", top_n=top_n)


def build_top_country_table(df, top_n=10):
    if "Affiliations" not in df.columns:
        return pd.DataFrame()

    country_freq = Counter()
    for aff_value in df["Affiliations"].tolist():
        countries_str = _extract_country_from_affiliation(aff_value)
        if not countries_str:
            continue
        for country in [item.strip() for item in countries_str.split(";") if item.strip()]:
            country_freq[country] += 1
    return build_top_entity_table(country_freq, "Country", top_n=top_n)


def build_top_institution_table(df, top_n=10):
    if "Affiliations" not in df.columns:
        return pd.DataFrame()

    institution_freq = Counter()
    for aff_value in df["Affiliations"].tolist():
        for institution in extract_institutions_from_affiliation(aff_value):
            institution_freq[institution] += 1
    return build_top_entity_table(institution_freq, "Institution", top_n=top_n)


def calculate_frequency(df, column):
    """Calculate frequency of items in a column where items are semicolon-separated."""
    if column not in df.columns:
        return Counter()

    counter = Counter()
    for value in df[column].dropna().tolist():
        terms = split_semicolon_terms(value)
        for t in terms:
            counter[t] += 1
    return counter


def calculate_simple_frequency(df, column):
    """Calculate frequency of items in a column where each row has one item."""
    if column not in df.columns:
        return Counter()

    series = df[column].fillna("").astype(str).str.strip()
    series = series[(series != "") & (series != "nan")]
    return Counter(series.tolist())


def calculate_category_frequency(df):
    """Special logic for category frequency: WoS_Categories or Research_Areas."""
    has_wos_cat = "WoS_Categories" in df.columns
    has_research_areas = "Research_Areas" in df.columns

    cat_freq = Counter()
    if has_wos_cat:
        cat_freq = calculate_frequency(df, "WoS_Categories")

    if has_research_areas and not cat_freq:
        cat_freq = calculate_frequency(df, "Research_Areas")

    return cat_freq
