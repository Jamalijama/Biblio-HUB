import re

import pandas as pd

import modules.data_pipeline as data_pipeline
import modules.keyword_pipeline as keyword_pipeline
from modules.data_pipeline import (
    _extract_country_from_affiliation,
    clean_year_column,
    deduplicate_dataframe,
    process_global_data,
    safe_year,
)
from modules.keyword_pipeline import (
    apply_keyword_governance,
    extract_keywords_from_dataframe,
    normalize_keyword,
    split_semicolon_terms,
)


def test_split_semicolon_terms_handles_empty_values():
    assert split_semicolon_terms("") == []
    assert split_semicolon_terms("Network; Citation Analysis ; Topic Mapping") == ["Network", "Citation Analysis", "Topic Mapping"]


def test_normalize_keyword_consolidates_common_variants():
    assert normalize_keyword("covid 19") == "COVID-19"
    assert normalize_keyword("co-citation") == "Co-citation"
    assert normalize_keyword("aedes-aegypti") == "Aedes Aegypti"
    assert normalize_keyword("<italic>aedes Aegypti< Italic>") == "Aedes Aegypti"
    assert normalize_keyword("rt pcr") == "RT-PCR"
    assert normalize_keyword("study") is None


def test_extract_keywords_prefers_metadata_and_deduplicates():
    df = pd.DataFrame(
        [
            {
                "Title": "Citation analysis for bibliometric mapping",
                "Abstract": "A bibliometric study of citation analysis.",
                "Author_Keywords": "citation-analysis; Bibliometrics",
                "Keywords_Plus": "BIBLIOMETRICS; Citation Analysis",
            }
        ]
    )

    keywords = extract_keywords_from_dataframe(df)

    assert keywords == [["Citation Analysis", "Bibliometrics"]]


def test_extract_keywords_respects_keyword_source_selection():
    df = pd.DataFrame(
        [
            {
                "Title": "Citation analysis for bibliometric mapping",
                "Abstract": "A bibliometric study of citation analysis.",
                "Author_Keywords": "Citation Analysis; Explainability",
                "Keywords_Plus": "Bibliometrics; Knowledge Mapping",
            }
        ]
    )

    de_keywords = extract_keywords_from_dataframe(df, keyword_source="DE")
    id_keywords = extract_keywords_from_dataframe(df, keyword_source="ID")
    combined_keywords = extract_keywords_from_dataframe(df, keyword_source="DE+ID")

    assert de_keywords == [["Citation Analysis", "Explainability"]]
    assert id_keywords == [["Bibliometrics", "Knowledge Mapping"]]
    assert combined_keywords == [["Citation Analysis", "Explainability", "Bibliometrics", "Knowledge Mapping"]]


def test_extract_keywords_uses_automatic_terms_when_metadata_is_missing():
    df = pd.DataFrame(
        [
            {
                "Title": "Network analysis for cancer diagnosis and survival prediction",
                "Abstract": "Network analysis improves cancer diagnosis and survival prediction in clinical workflows.",
                "Author_Keywords": "",
                "Keywords_Plus": "",
            }
        ]
    )

    keywords = extract_keywords_from_dataframe(df, min_metadata_terms=2, max_auto_terms=6)

    assert "Network Analysis" in keywords[0]
    assert "Cancer Diagnosis" in keywords[0] or "Survival Prediction" in keywords[0]


def test_extract_keywords_retains_optional_plugin_matches_when_enabled(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "Title": "Knowledge graph mapping for citation networks",
                "Abstract": "Citation network mapping highlights collaboration structure and graph-based discovery.",
                "Author_Keywords": "",
                "Keywords_Plus": "",
            }
        ]
    )

    monkeypatch.setattr(
        keyword_pipeline,
        "COMPILED_OPTIONAL_DOMAIN_PATTERNS",
        {
            "Knowledge Graph": re.compile(r"knowledge graph", re.IGNORECASE),
            "Citation Network": re.compile(r"citation network", re.IGNORECASE),
            "Collaboration Structure": re.compile(r"collaboration structure", re.IGNORECASE),
        },
    )

    keywords = extract_keywords_from_dataframe(
        df,
        min_metadata_terms=3,
        max_auto_terms=4,
        enable_optional_domain_plugin=True,
    )

    assert "Knowledge Graph" in keywords[0]
    assert "Citation Network" in keywords[0]
    assert "Collaboration Structure" in keywords[0]


def test_apply_keyword_governance_supports_blocklist_and_alias_mapping():
    records = [["COVID-19", "Citation Analysis", "Bibliometrics"]]

    governed = apply_keyword_governance(
        records,
        blocked_terms_text="Bibliometrics",
        replacement_map_text="citation analysis => Knowledge Mapping",
    )

    assert governed == [["COVID-19", "Knowledge Mapping"]]


def test_extract_keywords_from_dataframe_applies_vocabulary_governance():
    df = pd.DataFrame(
        [
            {
                "Title": "Citation analysis for bibliometric mapping",
                "Abstract": "A bibliometric study of citation analysis.",
                "Author_Keywords": "citation-analysis; Bibliometrics",
                "Keywords_Plus": "BIBLIOMETRICS; Citation Analysis",
            }
        ]
    )

    keywords = extract_keywords_from_dataframe(
        df,
        blocked_terms_text="Bibliometrics",
        replacement_map_text="citation analysis => Knowledge Mapping",
    )

    assert keywords == [["Knowledge Mapping"]]


def test_safe_year_and_clean_year_column_filter_invalid_values():
    assert safe_year("2024") == 2024
    assert safe_year("2024.0") == 2024
    assert safe_year("85") is None
    assert safe_year("2200") is None
    assert safe_year("bad") is None

    df = pd.DataFrame([{"Year": "2020"}, {"Year": "85"}, {"Year": "nan"}, {"Year": "2021.0"}])
    cleaned = clean_year_column(df)

    assert cleaned["Year"].tolist() == [2020, 2021]


def test_deduplicate_dataframe_merges_duplicate_doi_rows():
    df = pd.DataFrame(
        [
            {"Title": "Paper A", "Authors": "Smith", "Year": 2020, "DOI": "10.1/a", "Abstract": "short"},
            {"Title": "Paper A expanded", "Authors": "Smith", "Year": 2020, "DOI": "10.1/a", "Abstract": "a much longer abstract"},
        ]
    )

    deduped, report = deduplicate_dataframe(df)

    assert len(deduped) == 1
    assert report == {"original": 2, "removed": 1, "final": 1}
    assert deduped.iloc[0]["Abstract"] == "a much longer abstract"


def test_extract_country_from_affiliation_normalizes_common_aliases():
    value = "[Smith J] Univ X, Dept Y, USA; [Lee K] Univ Z, Hong Kong"

    countries = _extract_country_from_affiliation(value)

    assert countries == "China; United States"


def test_extract_country_from_affiliation_handles_wos_single_block_format():
                                                                                      
    value = "[Seymour, Robert L.; Adams, A. Paige; Leal, Grace; Alcorn, Maria D. H.; Weaver, Scott C.] Univ Texas Med Branch, Dept Pathol, Sealy Ctr Vaccine Dev, Inst Human Infect & Immun, Galveston, TX 77555 USA"
    countries = _extract_country_from_affiliation(value)
    assert countries == "United States"

    value2 = "[Ferreira, Qesya Rodrigues; Lemos, Fabian Fellipe Bueno; Moura, Matheus Nascimento; Nascimento, Jessica Oliveira de Souza; Novaes, Ana Flavia; Barcelos, Isadora Souza; Fernandes, Larissa Alves; Amaral, Liliany Souza de Brito; Barreto, Fernanda Khouri; Melo, Fabricio Freire de] Univ Fed Bahia, Inst Multidisciplinar Saude, BR-45029094 Vitoria Da Conquista, Brazil"
    countries2 = _extract_country_from_affiliation(value2)
    assert countries2 == "Brazil"


def test_extract_country_from_affiliation_uses_controlled_country_names():
    assert _extract_country_from_affiliation("Viruses-Basel") == ""

    value = "INST PASTEUR,BANGUI,CENT AFR REPUBL; INST NATL SANTE PUBL,ABIDJAN,COTE IVOIRE"
    countries = _extract_country_from_affiliation(value)
    assert countries == "Central African Republic; Cote d'Ivoire"

    value2 = "Minist Hlth & Human Serv, Majuro, Marshall Island; Univ X, Santo Domingo, Dominican Rep"
    countries2 = _extract_country_from_affiliation(value2)
    assert countries2 == "Dominican Republic; Marshall Islands"


def test_extract_country_from_affiliation_merges_additional_common_country_aliases():
    value = "Dept A, Univ A, Boston, U.S.; Dept B, Inst C, Beijing, Peoples R of China; Dept D, Lab E, London, U.K."

    countries = _extract_country_from_affiliation(value)

    assert countries == "China; United Kingdom; United States"


def test_process_global_data_builds_keywords_and_networks(monkeypatch):
    df = pd.DataFrame(
        [
            {"Title": "Network mapping in medicine", "Abstract": "Citation clustering for diagnosis pathways", "Year": 2020, "Journal": "Journal A"},
            {"Title": "Graph analytics for imaging", "Abstract": "Visualization methods for scan interpretation", "Year": 2021, "Journal": "Journal B"},
        ]
    )
    monkeypatch.setattr(data_pipeline, "load_data", lambda uploaded_files=None, _file_names=None, _dir_fingerprint="": (df, {"removed": 0}))

    result = (
        process_global_data.__wrapped__(enable_optional_domain_plugin=False)
        if hasattr(process_global_data, "__wrapped__")
        else process_global_data(enable_optional_domain_plugin=False)
    )

    assert result[0] is not None
    assert result[1]["removed"] == 0
    assert len(result[2]) == 2
    assert result[3]
    assert isinstance(result[4], dict)
    assert "Journal A" in result[5]


def test_process_global_data_respects_vocabulary_governance(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "Title": "Network mapping in medicine",
                "Abstract": "Citation analysis for diagnosis",
                "Author_Keywords": "Citation Analysis; Bibliometrics",
                "Keywords_Plus": "",
                "Year": 2020,
                "Journal": "Journal A",
            },
        ]
    )
    monkeypatch.setattr(
        data_pipeline,
        "load_data",
        lambda uploaded_files=None, _file_names=None, _dir_fingerprint="": (df, {"removed": 0}),
    )

    result = (
        process_global_data.__wrapped__(
            blocked_terms_text="Bibliometrics",
            replacement_map_text="Citation Analysis => Knowledge Mapping",
        )
        if hasattr(process_global_data, "__wrapped__")
        else process_global_data(
            blocked_terms_text="Bibliometrics",
            replacement_map_text="Citation Analysis => Knowledge Mapping",
        )
    )

    assert result[2] == [["Knowledge Mapping"]]


def test_process_global_data_respects_keyword_source_selection(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "Title": "Network mapping in medicine",
                "Abstract": "Citation analysis for diagnosis",
                "Author_Keywords": "Citation Analysis; Explainability",
                "Keywords_Plus": "Bibliometrics; Knowledge Mapping",
                "Year": 2020,
                "Journal": "Journal A",
            },
        ]
    )
    monkeypatch.setattr(
        data_pipeline,
        "load_data",
        lambda uploaded_files=None, _file_names=None, _dir_fingerprint="", local_file_paths=None, _local_path_fingerprint="": (df, {"removed": 0}),
    )

    result = (
        process_global_data.__wrapped__(keyword_source="ID")
        if hasattr(process_global_data, "__wrapped__")
        else process_global_data(keyword_source="ID")
    )

    assert result[2] == [["Bibliometrics", "Knowledge Mapping"]]


def test_load_data_reads_wos_tab_delimited_txt_via_dataframe_path(tmp_path):
    sample_path = tmp_path / "sample_wos.txt"
    sample_path.write_text("PT\tAU\tTI\tSO\tPY\tCR\nJ\tSmith, J\tPaper A\tJournal A\t2020\tLEE K, 2019, NATURE\n", encoding="utf-8")

    original_data_dir = data_pipeline.DATA_DIR
    data_pipeline.DATA_DIR = str(tmp_path)
    try:
        loaded_df, report = data_pipeline.load_data.__wrapped__()
    finally:
        data_pipeline.DATA_DIR = original_data_dir

    assert loaded_df is not None
    assert loaded_df.iloc[0]["Title"] == "Paper A"
    assert loaded_df.iloc[0]["Journal"] == "Journal A"
    assert report["final"] == 1
