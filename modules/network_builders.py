from collections import Counter

import pandas as pd
import streamlit as st

from modules.data_pipeline import safe_year


@st.cache_data
def build_cooccurrence_network(keywords_list, min_cooccurrence=2):
    cooccurrence = Counter()
    keyword_freq = Counter()
    for kws in keywords_list:
        unique_kws = sorted(set(kws))
        for kw in unique_kws:
            keyword_freq[kw] += 1
        for i in range(len(unique_kws)):
            for j in range(i + 1, len(unique_kws)):
                cooccurrence[(unique_kws[i], unique_kws[j])] += 1
    if min_cooccurrence > 1:
        cooccurrence = Counter(
            {
                pair: weight
                for pair, weight in cooccurrence.items()
                if weight >= min_cooccurrence
            }
        )
    return keyword_freq, cooccurrence


@st.cache_data
def build_journal_network(df):
    journal_year = {}
    journals = df["Journal"].tolist() if "Journal" in df.columns else []
    years = df["Year"].tolist() if "Year" in df.columns else []
    for journal_value, year in zip(journals, years):
        journal = str(journal_value).strip()
        normalized_year = safe_year(year) if pd.notna(year) else None
        if journal and journal != "nan" and normalized_year is not None:
            journal_year.setdefault(journal, {})
            journal_year[journal][normalized_year] = journal_year[journal].get(normalized_year, 0) + 1
    return journal_year


@st.cache_data
def build_keyword_journal_cooccurrence(df, keywords_list, top_n_keywords=20, top_n_journals=15):
    kw_journal_cooccur = Counter()
    journal_freq = Counter()
    keyword_freq_local = Counter()
    journals = df["Journal"].tolist() if "Journal" in df.columns else []
    for idx, journal_value in enumerate(journals):
        journal = str(journal_value).strip()
        if not journal or journal == "nan":
            continue
        journal_freq[journal] += 1
        kws = keywords_list[idx] if idx < len(keywords_list) else []
        for kw in set(kws):
            keyword_freq_local[kw] += 1
            kw_journal_cooccur[(kw, journal)] += 1

    top_keywords = [kw for kw, _ in keyword_freq_local.most_common(top_n_keywords)]
    top_journals = [journal for journal, _ in journal_freq.most_common(top_n_journals)]
    top_keyword_set = set(top_keywords)
    top_journal_set = set(top_journals)
    filtered_cooccur = {
        (kw, journal): weight
        for (kw, journal), weight in kw_journal_cooccur.items()
        if kw in top_keyword_set and journal in top_journal_set
    }
    return top_keywords, top_journals, filtered_cooccur, keyword_freq_local, journal_freq
