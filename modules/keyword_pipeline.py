import re
from collections import Counter

import pandas as pd

OPTIONAL_DOMAIN_PATTERNS = {}

LOWERCASE_CONNECTORS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on",
    "or", "the", "to", "with", "via", "vs",
}

TEXT_STOPWORDS = LOWERCASE_CONNECTORS | {
    "about", "after", "against", "among", "approach", "associated", "based",
    "between", "case", "cases", "clinical", "comparison", "control", "controls",
    "data", "dataset", "datasets", "effect", "effects", "evaluation", "evidence",
    "findings", "focus", "group", "groups", "high", "human", "humans", "impact",
    "improved", "improving", "including", "increase", "increased", "method",
    "methods", "model", "models", "paper", "patient", "patients", "result",
    "results", "review", "reviews", "sample", "samples", "significant", "study",
    "studies", "system", "systems", "using",
}

GENERIC_NOISE_TERMS = {
    "Analysis", "Data", "Method", "Methods", "Model", "Models", "Paper",
    "Research", "Result", "Results", "Review", "Reviews", "Study", "Studies",
    "System", "Systems",
}

CANONICAL_TERM_PATTERNS = [
    (re.compile(r"^covid[\s-]?19$", re.IGNORECASE), "COVID-19"),
    (re.compile(r"^sars[\s-]?cov[\s-]?2$", re.IGNORECASE), "SARS-CoV-2"),
    (re.compile(r"^h([1-9][0-9]?)n([1-9][0-9]?)$", re.IGNORECASE), lambda m: f"H{m.group(1)}N{m.group(2)}"),
    (re.compile(r"^rt[\s-]?pcr$", re.IGNORECASE), "RT-PCR"),
    (re.compile(r"^mrna$", re.IGNORECASE), "mRNA"),
    (re.compile(r"^rna$", re.IGNORECASE), "RNA"),
    (re.compile(r"^dna$", re.IGNORECASE), "DNA"),
    (re.compile(r"^bibliometric(s)?$", re.IGNORECASE), "Bibliometrics"),
    (re.compile(r"^co[\s-]?citation$", re.IGNORECASE), "Co-citation"),
    (re.compile(r"^co[\s-]?occurrence$", re.IGNORECASE), "Co-occurrence"),
]

UPPERCASE_TOKENS = {
    "dna", "hiv", "mers", "ngs", "np",
    "pcr", "pb1", "pb2", "pa", "r0", "rna", "rsv", "sars", "tc",
}

COMPILED_OPTIONAL_DOMAIN_PATTERNS = {
    keyword: re.compile(pattern, re.IGNORECASE)
    for keyword, pattern in OPTIONAL_DOMAIN_PATTERNS.items()
}


def _split_governance_entries(raw_text):
    if raw_text is None:
        return []
    return [
        entry.strip()
        for entry in re.split(r"[;\n\r]+", str(raw_text))
        if entry and entry.strip()
    ]


def split_semicolon_terms(value):
    if pd.isna(value) or str(value).strip() in ("", "nan"):
        return []
    return [term.strip() for term in str(value).split(";") if term.strip()]


def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def _strip_inline_markup(text):
    # Some WoS-derived keyword fields contain inline markup such as
    # <italic>...</italic>. Remove the tags before canonicalization so tagged
    # and plain-text variants collapse into the same keyword.
    cleaned = re.sub(r"<[^>]+>", " ", str(text))
    return _normalize_whitespace(cleaned)


def normalize_keyword(term):
    if term is None:
        return None
    cleaned = _strip_inline_markup(term)
    if not cleaned or cleaned.lower() == "nan":
        return None
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s*/\s*", " ", cleaned)
    cleaned = re.sub(r"\s*-\s*", "-", cleaned)
    cleaned = cleaned.strip(" ;,:.[](){}\"'")
    if not cleaned:
        return None

    for pattern, replacement in CANONICAL_TERM_PATTERNS:
        match = pattern.fullmatch(cleaned)
        if match:
            return replacement(match) if callable(replacement) else replacement

    parts = cleaned.split(" ")
    normalized_parts = []
    for index, part in enumerate(parts):
        token = part.strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in LOWERCASE_CONNECTORS and index != 0:
            normalized_parts.append(lowered)
            continue
        if lowered in UPPERCASE_TOKENS or re.fullmatch(r"[a-z]*\d+[a-z0-9-]*", lowered):
            normalized_parts.append(token.upper())
            continue
        if "-" in token:
            sub_parts = [sub for sub in token.split("-") if sub]
            if sub_parts:
                formatted_sub_parts = [_format_subtoken(sub) for sub in sub_parts]
                if all(re.fullmatch(r"[A-Za-z]+", sub) for sub in sub_parts):
                    normalized_parts.append(" ".join(formatted_sub_parts))
                else:
                    normalized_parts.append("-".join(formatted_sub_parts))
                continue
        normalized_parts.append(_format_subtoken(token))

    normalized = " ".join(normalized_parts).strip()
    if not normalized or normalized in GENERIC_NOISE_TERMS:
        return None
    return normalized


def _format_subtoken(token):
    lowered = token.lower()
    if lowered in UPPERCASE_TOKENS or re.fullmatch(r"[a-z]*\d+[a-z0-9-]*", lowered):
        return token.upper()
    if len(token) <= 2 and token.isalpha():
        return token.upper()
    return lowered.capitalize()


def _deduplicate_terms(terms):
    results = []
    seen = set()
    for term in terms:
        canonical = normalize_keyword(term)
        if canonical and canonical not in seen:
            results.append(canonical)
            seen.add(canonical)
    return results


def _prepare_blocked_terms(blocked_terms_text=None):
    blocked_terms = set()
    for entry in _split_governance_entries(blocked_terms_text):
        canonical = normalize_keyword(entry)
        blocked_terms.add(canonical if canonical else _normalize_whitespace(entry))
    return blocked_terms


def _prepare_replacement_map(replacement_map_text=None):
    replacement_map = {}
    for entry in _split_governance_entries(replacement_map_text):
        if "=>" not in entry:
            continue
        alias, canonical = entry.split("=>", 1)
        alias = alias.strip()
        canonical = canonical.strip()
        if not alias or not canonical:
            continue
        alias_key = normalize_keyword(alias)
        canonical_value = normalize_keyword(canonical)
        if alias_key and canonical_value:
            replacement_map[alias_key] = canonical_value
    return replacement_map


def apply_keyword_governance(records, blocked_terms_text=None, replacement_map_text=None):
    blocked_terms = _prepare_blocked_terms(blocked_terms_text)
    replacement_map = _prepare_replacement_map(replacement_map_text)
    governed_records = []

    for terms in records:
        governed_terms = []
        seen = set()
        for term in terms:
            canonical = normalize_keyword(term)
            if not canonical:
                continue
            canonical = replacement_map.get(canonical, canonical)
            if canonical in blocked_terms or canonical in seen:
                continue
            governed_terms.append(canonical)
            seen.add(canonical)
        governed_records.append(governed_terms)
    return governed_records


KEYWORD_SOURCE_FIELDS = {
    "DE": ("Author_Keywords",),
    "ID": ("Keywords_Plus",),
    "DE+ID": ("Author_Keywords", "Keywords_Plus"),
}


def _normalize_keyword_source(keyword_source):
    normalized = str(keyword_source or "DE+ID").strip().upper()
    return normalized if normalized in KEYWORD_SOURCE_FIELDS else "DE+ID"


def _should_fallback_to_automatic_terms(keyword_source, metadata_terms):
    normalized_source = _normalize_keyword_source(keyword_source)
    if metadata_terms:
        return False
    # When users explicitly choose DE or ID for cross-tool alignment, keep the
    # ranking strictly on the selected metadata field instead of mixing in
    # title/abstract-derived automatic terms.
    return normalized_source == "DE+ID"


def _extract_metadata_terms(row, keyword_source="DE+ID"):
    terms = []
    for field in KEYWORD_SOURCE_FIELDS[_normalize_keyword_source(keyword_source)]:
        for term in split_semicolon_terms(row.get(field, "")):
            terms.append(term)
    return _deduplicate_terms(terms)


def _tokenize(text):
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", str(text).lower())


def _extract_acronyms(text):
    acronyms = []
    for match in re.findall(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)*\b|\b[A-Z]\d+[A-Z0-9-]*\b", str(text)):
        canonical = normalize_keyword(match)
        if canonical:
            acronyms.append(canonical)
    return acronyms


def _extract_ngram_candidates(text, max_ngram=3, title_boost=1):
    scores = Counter()
    segments = re.split(r"[\.;:,\(\)\[\]\{\}\n\t]+", str(text))
    for segment in segments:
        tokens = [token for token in _tokenize(segment) if token not in TEXT_STOPWORDS]
        if not tokens:
            continue
        for ngram_size in range(1, max_ngram + 1):
            if len(tokens) < ngram_size:
                continue
            for start in range(len(tokens) - ngram_size + 1):
                phrase_tokens = tokens[start:start + ngram_size]
                if any(len(token) < 3 and not re.search(r"\d", token) for token in phrase_tokens):
                    continue
                candidate = normalize_keyword(" ".join(phrase_tokens))
                if not candidate or candidate in GENERIC_NOISE_TERMS:
                    continue
                increment = title_boost if ngram_size > 1 else max(1, title_boost - 1)
                scores[candidate] += increment
    return scores


def _extract_automatic_terms(title, abstract, max_terms=8):
    scores = Counter()
    title_text = "" if pd.isna(title) else str(title)
    abstract_text = "" if pd.isna(abstract) else str(abstract)

    scores.update(_extract_ngram_candidates(title_text, max_ngram=3, title_boost=4))
    scores.update(_extract_ngram_candidates(abstract_text, max_ngram=2, title_boost=1))

    for acronym in _extract_acronyms(f"{title_text} {abstract_text}"):
        scores[acronym] += 3

    ordered_terms = [
        term for term, score in scores.most_common()
        if score > 0 and term not in GENERIC_NOISE_TERMS
    ]
    return ordered_terms[:max_terms]


def _extract_plugin_terms(title, abstract, compiled_patterns=None):
    text = f"{'' if pd.isna(title) else title} {'' if pd.isna(abstract) else abstract}"
    lowered_text = text.lower()
    compiled_patterns = compiled_patterns or {}
    return [
        canonical
        for canonical, pattern in compiled_patterns.items()
        if pattern.search(lowered_text)
    ]


def extract_keywords_from_dataframe(
    df,
    min_metadata_terms=3,
    max_auto_terms=8,
    blocked_terms_text=None,
    replacement_map_text=None,
    keyword_source="DE+ID",
    enable_optional_domain_plugin=False,
):
    if df is None or len(df) == 0:
        return []

    records = []
    titles = df["Title"].fillna("").astype(str).tolist() if "Title" in df.columns else [""] * len(df)
    abstracts = df["Abstract"].fillna("").astype(str).tolist() if "Abstract" in df.columns else [""] * len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        metadata_terms = _extract_metadata_terms(row, keyword_source=keyword_source)
        automatic_terms = []
        plugin_terms = []
        if _should_fallback_to_automatic_terms(keyword_source, metadata_terms):
            automatic_terms = _extract_automatic_terms(
                titles[idx] if idx < len(titles) else "",
                abstracts[idx] if idx < len(abstracts) else "",
                max_terms=max_auto_terms,
            )
            if enable_optional_domain_plugin:
                plugin_terms = _extract_plugin_terms(
                    titles[idx] if idx < len(titles) else "",
                    abstracts[idx] if idx < len(abstracts) else "",
                    compiled_patterns=COMPILED_OPTIONAL_DOMAIN_PATTERNS,
                )
        records.append(_deduplicate_terms(metadata_terms + automatic_terms + plugin_terms))

    return apply_keyword_governance(
        records,
        blocked_terms_text=blocked_terms_text,
        replacement_map_text=replacement_map_text,
    )
