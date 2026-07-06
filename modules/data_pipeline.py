import difflib
import os
import re
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from modules.keyword_pipeline import split_semicolon_terms

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
)

WOS_TO_INTERNAL = {
    "PT": "PT", "AU": "Authors", "AF": "Authors_Full", "BA": "BA", "BE": "BE",
    "GP": "GP", "BF": "BF", "CA": "CA", "TI": "Title", "SO": "Journal",
    "SE": "SE", "BS": "BS", "LA": "Language", "DT": "DocType", "CT": "CT",
    "CY": "CY", "CL": "CL", "SP": "SP", "HO": "HO",
    "DE": "Author_Keywords", "ID": "Keywords_Plus", "AB": "Abstract",
    "C1": "Affiliations", "C3": "C3", "RP": "Reprint_Address", "EM": "EM",
    "RI": "RI", "OI": "OI", "FU": "Funding", "FP": "FP", "FX": "FX",
    "CR": "Cited_References", "NR": "Num_References", "TC": "Times_Cited",
    "Z9": "Z9", "U1": "U1", "U2": "U2", "PU": "Publisher", "PI": "PI",
    "PA": "PA", "SN": "ISSN", "EI": "EI", "BN": "BN", "J9": "J9", "JI": "JI",
    "PD": "PD", "PY": "Year", "VL": "Volume", "IS": "Issue", "PN": "PN",
    "SU": "SU", "SI": "SI", "MA": "MA", "BP": "Start_Page", "EP": "End_Page",
    "AR": "AR", "DI": "DOI", "DL": "DL", "D2": "D2", "EA": "EA", "PG": "PG",
    "WC": "WoS_Categories", "WE": "WE", "SC": "Research_Areas", "GA": "GA",
    "PM": "PMID", "OA": "OA", "HC": "HC", "HP": "HP", "DA": "DA", "UT": "UT",
}

CSV_TO_INTERNAL = {
    "Title": "Title", "Abstract": "Abstract", "Year": "Year",
    "Journal": "Journal", "DOI": "DOI", "PMID": "PMID", "PMCID": "PMCID",
    "Authors": "Authors", "Author_Keywords": "Author_Keywords",
    "Keywords_Plus": "Keywords_Plus", "Times_Cited": "Times_Cited",
    "Affiliations": "Affiliations", "Cited_References": "Cited_References",
    "WoS_Categories": "WoS_Categories", "Research_Areas": "Research_Areas",
    "DocType": "DocType", "Funding": "Funding", "Language": "Language",
}

COUNTRY_ALIASES = {
    "USA": "United States", "U S A": "United States", "U.S.A.": "United States",
    "US": "United States", "U S": "United States", "U.S.": "United States", "U.S": "United States",
    "UNITED STATES": "United States", "UNITED STATES AMERICA": "United States",
    "United States of America": "United States",
    "PEOPLES R CHINA": "China", "PR CHINA": "China", "P R CHINA": "China",
    "PEOPLES REP CHINA": "China", "PEOPLES REPUBLIC CHINA": "China",
    "PEOPLES R OF CHINA": "China", "P R OF CHINA": "China",
    "P R C": "China", "PRC": "China", "MAINLAND CHINA": "China",
    "HONG KONG": "China", "MACAU": "China",
    "U K": "United Kingdom", "UK": "United Kingdom", "U.K.": "United Kingdom", "U.K": "United Kingdom", "GB": "United Kingdom",
    "UNITED KINGDOM": "United Kingdom",
    "ENGLAND": "United Kingdom", "SCOTLAND": "United Kingdom",
    "WALES": "United Kingdom", "N IRELAND": "United Kingdom",
    "NORTH IRELAND": "United Kingdom", "GREAT BRITAIN": "United Kingdom",
    "FED REP GER": "Germany", "FEDERAL REP GER": "Germany",
    "W GERMANY": "Germany", "GERMANY": "Germany", "GER DEM REP": "Germany",
    "RUSSIA": "Russia", "RUSSIAN FEDERATION": "Russia", "USSR": "Russia",
    "CZECH REPUBLIC": "Czech Republic", "CZECHOSLOVAKIA": "Czech Republic",
    "SOUTH KOREA": "South Korea", "REPUBLIC OF KOREA": "South Korea",
    "KOREA": "South Korea",
    "IRAN": "Iran", "ISLAMIC REPUBLIC IRAN": "Iran",
    "TAIWAN": "China", "REPUBLIC OF CHINA": "China",
    "BRAZIL": "Brazil", "BRASIL": "Brazil",
    "AUSTRALIA": "Australia", "CANADA": "Canada", "FRANCE": "France",
    "JAPAN": "Japan", "INDIA": "India", "ITALY": "Italy", "SPAIN": "Spain",
    "NETHERLANDS": "Netherlands", "THE NETHERLANDS": "Netherlands",
    "SWEDEN": "Sweden", "SWITZERLAND": "Switzerland", "NORWAY": "Norway",
    "DENMARK": "Denmark", "FINLAND": "Finland", "BELGIUM": "Belgium",
    "AUSTRIA": "Austria", "POLAND": "Poland", "PORTUGAL": "Portugal",
    "GREECE": "Greece", "TURKEY": "Turkey", "TÜRKIYE": "Turkey", "TURKIYE": "Turkey",
    "MEXICO": "Mexico", "MÉXICO": "Mexico", "ARGENTINA": "Argentina",
    "CHILE": "Chile", "COLOMBIA": "Colombia", "SOUTH AFRICA": "South Africa",
    "EGYPT": "Egypt", "SAUDI ARABIA": "Saudi Arabia", "THAILAND": "Thailand",
    "VIETNAM": "Vietnam", "VIET NAM": "Vietnam", "SINGAPORE": "Singapore",
    "MALAYSIA": "Malaysia", "INDONESIA": "Indonesia", "PHILIPPINES": "Philippines",
    "PAKISTAN": "Pakistan", "BANGLADESH": "Bangladesh", "NEW ZEALAND": "New Zealand",
    "IRELAND": "Ireland", "ISRAEL": "Israel", "CROATIA": "Croatia",
    "SERBIA": "Serbia", "ROMANIA": "Romania", "HUNGARY": "Hungary",
    "CUBA": "Cuba", "NEPAL": "Nepal", "ETHIOPIA": "Ethiopia", "KENYA": "Kenya",
    "NIGERIA": "Nigeria", "TANZANIA": "Tanzania", "CAMBODIA": "Cambodia",
    "MYANMAR": "Myanmar", "PERU": "Peru", "ECUADOR": "Ecuador",
    "VENEZUELA": "Venezuela", "URUGUAY": "Uruguay", "BELARUS": "Belarus",
    "UKRAINE": "Ukraine", "LAOS": "Laos", "ANGOLA": "Angola", "ALGERIA": "Algeria",
    "BOTSWANA": "Botswana", "BURKINA FASO": "Burkina Faso", "CAMEROON": "Cameroon",
    "GABON": "Gabon", "GAMBIA": "Gambia", "GUINEA": "Guinea",
    "MADAGASCAR": "Madagascar", "MALI": "Mali", "MOROCCO": "Morocco",
    "NAMIBIA": "Namibia", "NIGER": "Niger", "REP CONGO": "Republic of the Congo",
    "DEM REP CONGO": "Democratic Republic of the Congo", "SENEGAL": "Senegal",
    "TOGO": "Togo", "UGANDA": "Uganda", "ZAMBIA": "Zambia", "ZIMBABWE": "Zimbabwe",
    "BENIN": "Benin", "GHANA": "Ghana", "MOZAMBIQUE": "Mozambique",
    "GEORGIA": "Georgia", "ESTONIA": "Estonia",
    "CENT AFR REPUBL": "Central African Republic",
    "CENT AFR REP": "Central African Republic",
    "CENTRAL AFR REPUBLIC": "Central African Republic",
    "COTE IVOIRE": "Cote d'Ivoire",
    "COTE D IVOIRE": "Cote d'Ivoire",
    "COTE D'IVOIRE": "Cote d'Ivoire",
    "DOMINICAN REP": "Dominican Republic",
    "MARSHALL ISLAND": "Marshall Islands",
    "SRI LANKA": "Sri Lanka",
    "SOMALIA": "Somalia",
    "SUDAN": "Sudan",
    "U ARAB EMIRATES": "United Arab Emirates",
    "TRINIDAD TOBAGO": "Trinidad and Tobago",
    "SAO TOME & PRIN": "Sao Tome and Principe",
    "ST KITTS & NEVI": "Saint Kitts and Nevis",
    "ST LUCIA": "Saint Lucia",
    "NETH ANTILLES": "Netherlands Antilles",
    "BOSNIA & HERCEG": "Bosnia and Herzegovina",
}

VALID_COUNTRY_NAMES = {
    "Afghanistan", "Albania", "Algeria", "Angola", "Anguilla", "Argentina", "Armenia",
    "Aruba", "Australia", "Austria", "Azerbaijan", "Bahrain", "Bangladesh", "Barbados",
    "Belgium", "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina",
    "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Cambodia", "Cameroon",
    "Canada", "Cape Verde", "Cayman Islands", "Central African Republic", "Chad", "Chile",
    "China", "Colombia", "Comoros", "Cook Islands", "Costa Rica", "Cote d'Ivoire",
    "Croatia", "Cuba", "Curacao", "Cyprus", "Czech Republic",
    "Democratic Republic of the Congo", "Denmark", "Djibouti", "Dominica",
    "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Estonia", "Ethiopia",
    "Fiji", "Finland", "France", "French Guiana", "Gabon", "Gambia", "Georgia",
    "Germany", "Ghana", "Gibraltar", "Greece", "Grenada", "Guatemala", "Guinea",
    "Haiti", "Honduras", "Hungary", "India", "Indonesia", "Iran", "Iraq", "Ireland",
    "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati",
    "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Liberia", "Libya", "Lithuania",
    "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta",
    "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova",
    "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nepal",
    "Netherlands", "Netherlands Antilles", "New Caledonia", "New Zealand", "Nicaragua",
    "Niger", "Nigeria", "North Macedonia", "Norway", "Oman", "Pakistan", "Palestine",
    "Panama", "Paraguay", "Peru", "Philippines", "Poland", "Portugal",
    "Republic of the Congo", "Qatar", "Romania", "Russia", "Rwanda", "Samoa",
    "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles",
    "Sierra Leone", "Singapore", "Sint Maarten", "Slovenia", "Solomon Islands",
    "Somalia", "South Africa", "South Korea", "Spain", "Sri Lanka",
    "Saint Kitts and Nevis", "Saint Lucia", "Sudan", "Suriname", "Sweden",
    "Switzerland", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga",
    "Trinidad and Tobago", "Tunisia", "Turkey", "Uganda", "Ukraine",
    "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Vanuatu",
    "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe",
}

VALID_COUNTRY_LOOKUP = {name.upper(): name for name in VALID_COUNTRY_NAMES}
COUNTRY_ALIAS_LOOKUP = {alias.upper(): standard for alias, standard in COUNTRY_ALIASES.items()}

US_STATE_ABBREVS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR",
}


def safe_year(value):
    try:
        year = int(float(str(value)))
    except (ValueError, TypeError):
        return None
    current_year = date.today().year
    if 1900 <= year <= current_year + 1:
        return year
    return None


def clean_year_column(df):
    df = df.copy()
    df["_safe_year"] = df["Year"].apply(safe_year)
    df = df[df["_safe_year"].notna()].copy()
    df["Year"] = df["_safe_year"].astype(int)
    return df.drop(columns=["_safe_year"])


def _find_default_data():
    if not os.path.isdir(DATA_DIR):
        return []
    default_files = []
    for ext in [".txt", ".csv"]:
        for fname in sorted(os.listdir(DATA_DIR)):
            if fname.lower().endswith(ext):
                default_files.append((os.path.join(DATA_DIR, fname), ext))
    return default_files


def _detect_wos_format(content_sample):
    first_line = content_sample.split("\n")[0].strip()
    wos_fields = {"PT", "AU", "AF", "TI", "SO", "AB", "DE", "ID", "CR", "TC", "C1", "PY", "DI"}
    header_fields = set(first_line.split("\t"))
    return len(header_fields & wos_fields) >= 5


def _parse_wos_authors(author_str):
    if pd.isna(author_str) or str(author_str).strip() in ("", "nan"):
        return []
    return [author.strip() for author in str(author_str).split(";") if author.strip()]


def _parse_wos_keywords(kw_str):
    return split_semicolon_terms(kw_str)


def _count_semicolon_terms(series):
    counter = Counter()
    for value in series.tolist():
        counter.update(_parse_wos_keywords(value))
    return counter


def _normalize_country_name(raw_name):
    name = raw_name.strip().rstrip(".").rstrip(",")
    if not name:
        return None
    name_upper = name.upper()
    if name_upper in COUNTRY_ALIAS_LOOKUP:
        return COUNTRY_ALIAS_LOOKUP[name_upper]
    if re.match(r"^([A-Z]{2})\s+\d{5}(-\d{4})?\s+USA$", name, re.IGNORECASE):
        return "United States"
    if re.match(r"^[A-Z]{2}\s+USA$", name, re.IGNORECASE):
        return "United States"
    if name_upper.endswith(" USA") or name_upper == "USA":
        return "United States"
    for alias, standard in COUNTRY_ALIASES.items():
        if name_upper.endswith(alias):
            return standard
    if re.match(r"^[A-Z]{2}\s+\d{5}(-\d{4})?$", name_upper):
        return None
    if re.match(r"^[A-Z]{2}$", name_upper) and name_upper in US_STATE_ABBREVS:
        return None
    if re.match(r"^[A-Z]{1,2}\s+\d", name_upper):
        return None
    if re.match(r"^[A-Z]\d{2}\s+\d", name_upper):
        return None
    if re.match(r"^\d", name):
        return None
    if len(name) <= 2 and name_upper not in COUNTRY_ALIASES:
        return None
    if re.match(r"^[A-Z][a-z]+\s+[A-Z]\.?$", name):
        return None
    if re.match(r"^[A-Z][a-z]+-[A-Z][a-z]+,\s*[A-Z][a-z]+", name):
        return None
    return VALID_COUNTRY_LOOKUP.get(name_upper)


def _is_valid_country(candidate):
    if not candidate:
        return False
    candidate = candidate.strip()
    candidate_upper = candidate.upper()
    if _normalize_country_name(candidate) is not None:
        return True
    if len(candidate) < 3:
        return False

    if re.match(r"^[A-Z][a-z]+\s+[A-Z]\.?$", candidate):
        return False
    if re.match(r"^[A-Z][a-z]+-[A-Z][a-z]+,\s*[A-Z][a-z]+", candidate):
        return False
    if re.match(r"^[A-Z]{1,2}\s+\d", candidate_upper):
        return False
    if re.match(r"^\d", candidate):
        return False
    if re.match(r"^[A-Z]{2}$", candidate_upper) and candidate_upper in US_STATE_ABBREVS:
        return False
    return False


def _looks_like_author_name(block_text):
    stripped = block_text.strip()
    if stripped.startswith("["):
        return False
    if re.match(r"^[A-Z][a-zA-Z\s\-]+,\s*[A-Z][a-zA-Z\-\.]+(\s+[A-Z]\.)*$", stripped):
        return True
    if re.match(r"^[A-Z][a-z]+-[A-Z][a-z]+,\s*[A-Z][a-z]+", stripped):
        return True
    return False


def _extract_country_from_affiliation(aff_str):
    if pd.isna(aff_str) or str(aff_str).strip() in ("", "nan"):
        return ""
    text = re.sub(r"\[[^\]]*\]", "", str(aff_str)).strip()
    countries = set()
    addr_blocks = [s.strip() for s in text.split(";") if s.strip()]
    for addr in addr_blocks:
        if not addr or _looks_like_author_name(addr):
            continue
        addr_parts = [x.strip() for x in addr.split(",")]
        if not addr_parts:
            continue
        last_part = addr_parts[-1].strip().rstrip(".")
        if not last_part:
            continue
        if _is_valid_country(last_part):
            normalized = _normalize_country_name(last_part)
            if normalized:
                countries.add(normalized)
    return "; ".join(sorted(countries))


def _standardize_columns(df, source="csv"):
    rename_source = WOS_TO_INTERNAL if source == "wos" else CSV_TO_INTERNAL
    rename_map = {col: rename_source[col] for col in df.columns if col in rename_source}
    df = df.rename(columns=rename_map)

    for col in ["Title", "Abstract", "Year", "Journal", "DOI", "Authors"]:
        if col not in df.columns:
            df[col] = None
    return df


def _merge_rows(row1, row2):
    merged = {}
    for col in row1.index:
        val1 = row1[col]
        val2 = row2[col]
        if pd.isna(val1) and pd.isna(val2):
            merged[col] = np.nan
        elif pd.isna(val1):
            merged[col] = val2
        elif pd.isna(val2):
            merged[col] = val1
        else:
            merged[col] = val1 if len(str(val1)) >= len(str(val2)) else val2
    return pd.Series(merged)


def deduplicate_dataframe(df):
    if df is None or len(df) == 0:
        return df, {"original": 0, "removed": 0, "final": 0}

    original_count = len(df)

    if "DOI" in df.columns:
        df["_valid_doi"] = df["DOI"].apply(
            lambda x: str(x).strip().lower()
            if pd.notna(x) and str(x).strip() not in ("", "nan")
            else None
        )
        doi_groups = df.groupby("_valid_doi")
        to_drop = []
        for _, group in doi_groups:
            if len(group) > 1:
                merged_row = group.iloc[0]
                for i in range(1, len(group)):
                    merged_row = _merge_rows(merged_row, group.iloc[i])
                df.loc[group.index[0]] = merged_row
                to_drop.extend(group.index[1:])
        df = df.drop(index=to_drop).drop(columns=["_valid_doi"])

    if "Title" in df.columns and "Year" in df.columns:
        if "Authors" in df.columns:
            df["_title_author_clean"] = df.apply(
                lambda row: re.sub(r"[^a-z0-9]", "", (str(row["Title"]) + str(row["Authors"])).lower())
                if pd.notna(row["Title"])
                else "",
                axis=1,
            )
        else:
            df["_title_author_clean"] = df["Title"].apply(
                lambda x: re.sub(r"[^a-z0-9]", "", str(x).lower()) if pd.notna(x) else ""
            )

        df["_year_str"] = df["Year"].apply(lambda x: str(x).strip()[:4] if pd.notna(x) else "")
        df["_title_author_prefix"] = df["_title_author_clean"].apply(lambda value: value[:20] if value else "")
        to_drop = []
        for year in df["_year_str"].unique():
            year_df = df[df["_year_str"] == year]
            if len(year_df) < 2:
                continue
            for _, prefix_df in year_df.groupby("_title_author_prefix", sort=False):
                if len(prefix_df) < 2:
                    continue
                indices = prefix_df.index.tolist()
                merged_indices = set()
                for i in range(len(indices)):
                    if indices[i] in merged_indices:
                        continue
                    t1 = prefix_df.loc[indices[i], "_title_author_clean"]
                    if not t1 or len(t1) < 10:
                        continue
                    for j in range(i + 1, len(indices)):
                        if indices[j] in merged_indices:
                            continue
                        t2 = prefix_df.loc[indices[j], "_title_author_clean"]
                        if not t2 or len(t2) < 10:
                            continue
                        if abs(len(t1) - len(t2)) > 30:
                            continue
                        if difflib.SequenceMatcher(None, t1, t2).ratio() > 0.85:
                            df.loc[indices[i]] = _merge_rows(df.loc[indices[i]], df.loc[indices[j]])
                            to_drop.append(indices[j])
                            merged_indices.add(indices[j])
        df = df.drop(index=to_drop).drop(columns=["_title_author_clean", "_title_author_prefix", "_year_str"])

    df = df.reset_index(drop=True)
    final_count = len(df)
    return df, {"original": original_count, "removed": original_count - final_count, "final": final_count}


def _parse_wos_text_dataframe(content):
    lines = content.split("\n")
    header = lines[0].strip().split("\t")
    rows = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        fields = line.split("\t")
        row_dict = {}
        for i, header_name in enumerate(header):
            row_dict[header_name] = fields[i] if i < len(fields) else ""
        rows.append(row_dict)
    return pd.DataFrame(rows, columns=header)


def _get_data_dir_fingerprint():
    """Generate a fingerprint based on file names and modification times in DATA_DIR."""
    if not os.path.isdir(DATA_DIR):
        return ""
    files = []
    for f in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, f)
        if os.path.isfile(path):
            files.append(f"{f}:{os.path.getmtime(path)}")
    return "|".join(files)


def _get_local_paths_fingerprint(local_file_paths):
    if not local_file_paths:
        return ""
    records = []
    for raw_path in local_file_paths:
        path = str(raw_path or "").strip().strip('"')
        if not path:
            continue
        if os.path.exists(path):
            stat = os.stat(path)
            records.append(f"{os.path.abspath(path)}:{stat.st_mtime_ns}:{stat.st_size}")
        else:
            records.append(f"{os.path.abspath(path)}:missing")
    return "|".join(sorted(records))


@st.cache_data
def load_data(uploaded_files=None, _file_names=None, _dir_fingerprint="", local_file_paths=None, _local_path_fingerprint=""):
    encodings = ["utf-8-sig", "utf-8", "latin-1", "gbk", "cp1252"]
    all_dfs = []

    if local_file_paths and len(local_file_paths) > 0:
        for raw_path in local_file_paths:
            default_path = str(raw_path or "").strip().strip('"')
            if not default_path:
                continue
            if not os.path.exists(default_path):
                raise FileNotFoundError(f"Local file not found: {default_path}")
            default_ext = os.path.splitext(default_path)[1].lower()
            df = None
            if default_ext == ".txt":
                for enc in encodings:
                    try:
                        df = _standardize_columns(
                            pd.read_csv(default_path, encoding=enc, sep="\t", low_memory=False),
                            source="wos",
                        )
                        break
                    except (UnicodeDecodeError, Exception):
                        continue
                if df is None:
                    for enc in encodings:
                        try:
                            with open(default_path, encoding=enc) as file_obj:
                                df = _standardize_columns(
                                    _parse_wos_text_dataframe(file_obj.read()),
                                    source="wos",
                                )
                                break
                        except (UnicodeDecodeError, Exception):
                            continue
            else:
                for enc in encodings:
                    try:
                        sample = pd.read_csv(default_path, encoding=enc, nrows=0, low_memory=False)
                        keep_cols = [c for c in sample.columns if not c.startswith("Unnamed")]
                        df = _standardize_columns(
                            pd.read_csv(default_path, encoding=enc, usecols=keep_cols, low_memory=False),
                            source="csv",
                        )
                        break
                    except (UnicodeDecodeError, Exception):
                        continue
            if df is None:
                raise ValueError(f"Failed to parse local file: {default_path}")
            all_dfs.append(df)
    elif uploaded_files and len(uploaded_files) > 0:
        for uploaded_file, name in zip(uploaded_files, _file_names):
            df = None
            is_wos = name.lower().endswith(".txt")
            if not is_wos:
                try:
                    uploaded_file.seek(0)
                    sample_bytes = uploaded_file.read(4096)
                    uploaded_file.seek(0)
                    sample_text = sample_bytes.decode("utf-8-sig", errors="replace")
                    if _detect_wos_format(sample_text):
                        is_wos = True
                except Exception:
                    pass
            if is_wos:
                for enc in encodings:
                    try:
                        uploaded_file.seek(0)
                        df = _standardize_columns(
                            pd.read_csv(uploaded_file, encoding=enc, sep="\t", low_memory=False),
                            source="wos",
                        )
                        break
                    except (UnicodeDecodeError, Exception):
                        continue
                if df is None:
                    for enc in encodings:
                        try:
                            uploaded_file.seek(0)
                            df = _standardize_columns(
                            _parse_wos_text_dataframe(uploaded_file.read().decode(enc)),
                            source="wos",
                        )
                            break
                        except (UnicodeDecodeError, Exception):
                            continue
            if df is None:
                for enc in encodings:
                    try:
                        uploaded_file.seek(0)
                        df = _standardize_columns(
                            pd.read_csv(uploaded_file, encoding=enc, low_memory=False),
                            source="csv",
                        )
                        break
                    except (UnicodeDecodeError, Exception):
                        continue
            if df is not None:
                all_dfs.append(df)
    else:
        for default_path, default_ext in _find_default_data():
            df = None
            if default_ext == ".txt":
                for enc in encodings:
                    try:
                        df = _standardize_columns(
                            pd.read_csv(default_path, encoding=enc, sep="\t", low_memory=False),
                            source="wos",
                        )
                        break
                    except (UnicodeDecodeError, Exception):
                        continue
                if df is None:
                    for enc in encodings:
                        try:
                            with open(default_path, encoding=enc) as file_obj:
                                df = _standardize_columns(
                                    _parse_wos_text_dataframe(file_obj.read()),
                                    source="wos",
                                )
                                break
                        except (UnicodeDecodeError, Exception):
                            continue
            else:
                for enc in encodings:
                    try:
                        sample = pd.read_csv(default_path, encoding=enc, nrows=0, low_memory=False)
                        keep_cols = [c for c in sample.columns if not c.startswith("Unnamed")]
                        df = _standardize_columns(
                            pd.read_csv(default_path, encoding=enc, usecols=keep_cols, low_memory=False),
                            source="csv",
                        )
                        break
                    except (UnicodeDecodeError, Exception):
                        continue
            if df is not None:
                all_dfs.append(df)

    if not all_dfs:
        return None, None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df[[c for c in combined_df.columns if not c.startswith("Unnamed")]]
    return deduplicate_dataframe(combined_df)


def process_global_data(
    uploaded_files=None,
    _file_names=None,
    blocked_terms_text=None,
    replacement_map_text=None,
    keyword_source="DE+ID",
    enable_optional_domain_plugin=False,
    _dir_fingerprint="",
    local_file_paths=None,
    _local_path_fingerprint="",
    _progress_callback=None,
    defer_heavy_analysis=False,
):
    from modules.keyword_pipeline import extract_keywords_from_dataframe
    from modules.network_builders import build_cooccurrence_network, build_journal_network

    if _progress_callback:
        _progress_callback(0.10, "Reading files and deduplicating records...")

    load_kwargs = {
        "_file_names": _file_names,
        "_dir_fingerprint": _dir_fingerprint,
    }
    if local_file_paths:
        load_kwargs["local_file_paths"] = local_file_paths
        load_kwargs["_local_path_fingerprint"] = _local_path_fingerprint
    df, dedup_report = load_data(uploaded_files, **load_kwargs)
    if df is None:
        return None, None, None, None, None, None

    if defer_heavy_analysis:
        if _progress_callback:
            _progress_callback(
                1.0,
                "Data loading completed. Keyword and network analytics are deferred until a heavy analysis module is opened.",
            )
        return df, dedup_report, [], Counter(), Counter(), {}

    if _progress_callback:
        _progress_callback(0.40, "Extracting keywords from the loaded dataset...")

    keywords_list = (
        extract_keywords_from_dataframe.__wrapped__(
            df,
            blocked_terms_text=blocked_terms_text,
            replacement_map_text=replacement_map_text,
            keyword_source=keyword_source,
            enable_optional_domain_plugin=enable_optional_domain_plugin,
        )
        if hasattr(extract_keywords_from_dataframe, "__wrapped__")
        else extract_keywords_from_dataframe(
            df,
            blocked_terms_text=blocked_terms_text,
            replacement_map_text=replacement_map_text,
            keyword_source=keyword_source,
            enable_optional_domain_plugin=enable_optional_domain_plugin,
        )
    )
    if _progress_callback:
        _progress_callback(0.68, "Building keyword co-occurrence structures...")
    keyword_freq, cooccurrence = (
        build_cooccurrence_network.__wrapped__(keywords_list)
        if hasattr(build_cooccurrence_network, "__wrapped__")
        else build_cooccurrence_network(keywords_list)
    )
    if _progress_callback:
        _progress_callback(0.86, "Preparing journal-year support tables...")
    journal_year = (
        build_journal_network.__wrapped__(df)
        if hasattr(build_journal_network, "__wrapped__")
        else build_journal_network(df)
    )
    if _progress_callback:
        _progress_callback(1.0, "Data loading completed.")
    return df, dedup_report, keywords_list, keyword_freq, cooccurrence, journal_year
