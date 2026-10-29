# -*- coding: utf-8 -*-
"""
Global (worldwide) country matching for the four WOS files.
Unlike the EU/EEA-only reanalysis, this extracts ALL countries from C1 addresses.
Method: country is the terminal segment of each address block (WoS convention),
with a dedicated rule for USA (state + ZIP + USA). Whole-counting of papers
(a paper with authors from N countries counts once for each of the N).
Outputs per-virus global country publication/citation tables + match audit.
"""
import csv, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import os
DATA_ROOT = Path(os.environ.get("BIBLIO_DATA_ROOT", Path.cwd())).resolve()


BASE = DATA_ROOT
OUT = BASE / "06_全球分析"
OUT.mkdir(parents=True, exist_ok=True)
END_YEAR = 2024
FILES = {
    "Chikungunya": (BASE / "02_原始数据_WoS" / "Merged_Chikungunya_WOS_Full_桌面6月7日原始版_77MB.txt", 2008),
    "Zika": (BASE / "02_原始数据_WoS" / "四病毒补充WOS" / "Merged_ZIKA_WOS_Full.txt", 2015),
    "Dengue": (BASE / "02_原始数据_WoS" / "四病毒补充WOS" / "Merged_dengue_WOS_Full.txt", 2008),
    "West Nile": (BASE / "02_原始数据_WoS" / "四病毒补充WOS" / "Merged_west_nile_WOS_Full.txt", 2008),
}

# ISO3-ish code -> (canonical name, aliases as they appear at END of WOS address blocks)
C = {
 # ---- Americas (high priority for arboviruses) ----
 "USA": ("United States", ("USA", "United States", "United States of America", "U S A")),
 "CAN": ("Canada", ("Canada",)),
 "BRA": ("Brazil", ("Brazil",)),
 "MEX": ("Mexico", ("Mexico",)),
 "ARG": ("Argentina", ("Argentina",)),
 "COL": ("Colombia", ("Colombia",)),
 "PER": ("Peru", ("Peru",)),
 "VEN": ("Venezuela", ("Venezuela", "Bolivarian Republic of Venezuela")),
 "CHL": ("Chile", ("Chile",)),
 "CUB": ("Cuba", ("Cuba",)),
 "PRI": ("Puerto Rico", ("Puerto Rico",)),
 "DOM": ("Dominican Republic", ("Dominican Republic", "Dominican Rep")),
 "HTI": ("Haiti", ("Haiti",)),
 "JAM": ("Jamaica", ("Jamaica",)),
 "TTO": ("Trinidad and Tobago", ("Trinidad and Tobago", "Trinidad & Tobago", "Trinidad Tobago")),
 "BRB": ("Barbados", ("Barbados",)),
 "BHS": ("Bahamas", ("Bahamas", "The Bahamas")),
 "GUY": ("Guyana", ("Guyana",)),
 "SUR": ("Suriname", ("Suriname",)),
 "GUF": ("French Guiana", ("French Guiana", "Guyane Francaise")),
 "GTM": ("Guatemala", ("Guatemala",)),
 "HND": ("Honduras", ("Honduras",)),
 "SLV": ("El Salvador", ("El Salvador",)),
 "NIC": ("Nicaragua", ("Nicaragua",)),
 "CRI": ("Costa Rica", ("Costa Rica",)),
 "PAN": ("Panama", ("Panama",)),
 "BOL": ("Bolivia", ("Bolivia", "Plurinational State of Bolivia")),
 "ECU": ("Ecuador", ("Ecuador",)),
 "PRY": ("Paraguay", ("Paraguay",)),
 "URY": ("Uruguay", ("Uruguay",)),
 "BLZ": ("Belize", ("Belize",)),
 "MTQ": ("Martinique", ("Martinique",)),
 "GLP": ("Guadeloupe", ("Guadeloupe",)),
 "ANT": ("Netherlands Antilles", ("Netherlands Antilles", "Curacao", "Aruba", "Sint Maarten", "Bonaire")),
 "LCA": ("Saint Lucia", ("Saint Lucia", "St Lucia")),
 "VCT": ("Saint Vincent and the Grenadines", ("Saint Vincent and the Grenadines", "St Vincent")),
 "GRD": ("Grenada", ("Grenada",)),
 "ATG": ("Antigua and Barbuda", ("Antigua and Barbuda", "Antigua", "Antigua & Barbu")),
 "KNA": ("Saint Kitts and Nevis", ("Saint Kitts and Nevis", "St Kitts Nevis", "St Kitts", "St Kitts & Nevi")),
 "DMA": ("Dominica", ("Dominica",)),
 "CYM": ("Cayman Islands", ("Cayman Islands",)),
 "VIR": ("Virgin Islands", ("Virgin Islands", "US Virgin Islands", "British Virgin Islands")),
 "TCA": ("Turks and Caicos Islands", ("Turks and Caicos Islands", "Turks & Caicos")),
 "BMU": ("Bermuda", ("Bermuda",)),
 # ---- Europe (EU30 + UK + others) ----
 "AT": ("Austria", ("Austria",)), "BE": ("Belgium", ("Belgium",)),
 "BG": ("Bulgaria", ("Bulgaria",)), "CY": ("Cyprus", ("Cyprus",)),
 "CZ": ("Czechia", ("Czech Republic", "Czechia",)), "DE": ("Germany", ("Germany",)),
 "DK": ("Denmark", ("Denmark",)), "EE": ("Estonia", ("Estonia",)),
 "EL": ("Greece", ("Greece",)), "ES": ("Spain", ("Spain",)),
 "FI": ("Finland", ("Finland",)), "FR": ("France", ("France",)),
 "HR": ("Croatia", ("Croatia",)), "HU": ("Hungary", ("Hungary",)),
 "IE": ("Ireland", ("Ireland",)), "IS": ("Iceland", ("Iceland",)),
 "IT": ("Italy", ("Italy",)), "LI": ("Liechtenstein", ("Liechtenstein",)),
 "LT": ("Lithuania", ("Lithuania",)), "LU": ("Luxembourg", ("Luxembourg",)),
 "LV": ("Latvia", ("Latvia",)), "MT": ("Malta", ("Malta",)),
 "NL": ("Netherlands", ("Netherlands", "The Netherlands")),
 "NO": ("Norway", ("Norway",)), "PL": ("Poland", ("Poland",)),
 "PT": ("Portugal", ("Portugal",)), "RO": ("Romania", ("Romania",)),
 "SE": ("Sweden", ("Sweden",)), "SI": ("Slovenia", ("Slovenia",)),
 "SK": ("Slovakia", ("Slovakia", "Slovak Republic",)),
 "GBR": ("United Kingdom", ("United Kingdom", "England", "Scotland", "Wales",
                            "Northern Ireland", "N Ireland", "North Ireland", "Great Britain", "Britain")),
 "CHE": ("Switzerland", ("Switzerland",)), "SRB": ("Serbia", ("Serbia", "Republic of Serbia")),
 "RUS": ("Russia", ("Russia", "Russian Federation")), "UKR": ("Ukraine", ("Ukraine",)),
 "BLR": ("Belarus", ("Belarus",)), "MDA": ("Moldova", ("Moldova", "Republic of Moldova")),
 "MKD": ("North Macedonia", ("North Macedonia", "Macedonia", "FYR Macedonia")),
 "ALB": ("Albania", ("Albania",)), "BIH": ("Bosnia and Herzegovina", ("Bosnia and Herzegovina", "Bosnia Hercegovina", "Bosnia & Herceg", "Bosnia & Hercegovina")),
 "MNE": ("Montenegro", ("Montenegro",)), "XKX": ("Kosovo", ("Kosovo",)),
 "GEO": ("Georgia", ("Georgia",)), "ARM": ("Armenia", ("Armenia",)), "AZE": ("Azerbaijan", ("Azerbaijan",)),
 "FIN": ("Finland", ("Finland",)),
 # ---- Asia ----
 "CHN": ("China", ("Peoples R China", "China", "People's Republic of China")),
 "TWN": ("Taiwan", ("Taiwan",)),
 "KOR": ("South Korea", ("South Korea", "Republic of Korea", "Korea")),
 "JPN": ("Japan", ("Japan",)),
 "IND": ("India", ("India",)),
 "THA": ("Thailand", ("Thailand",)),
 "VNM": ("Vietnam", ("Viet Nam", "Vietnam")),
 "IDN": ("Indonesia", ("Indonesia",)),
 "MYS": ("Malaysia", ("Malaysia",)),
 "SGP": ("Singapore", ("Singapore",)),
 "PHL": ("Philippines", ("Philippines",)),
 "LKA": ("Sri Lanka", ("Sri Lanka",)),
 "BGD": ("Bangladesh", ("Bangladesh",)),
 "PAK": ("Pakistan", ("Pakistan",)),
 "IRN": ("Iran", ("Iran", "Iran, Islamic Republic of")),
 "SAU": ("Saudi Arabia", ("Saudi Arabia",)),
 "ISR": ("Israel", ("Israel",)),
 "TUR": ("Turkey", ("Turkey", "Turkiye")),
 "IRQ": ("Iraq", ("Iraq",)), "LBN": ("Lebanon", ("Lebanon",)),
 "JOR": ("Jordan", ("Jordan",)), "KWT": ("Kuwait", ("Kuwait",)),
 "ARE": ("United Arab Emirates", ("United Arab Emirates", "UAE", "U Arab Emirates")),
 "QAT": ("Qatar", ("Qatar",)), "OMN": ("Oman", ("Oman",)),
 "YEM": ("Yemen", ("Yemen",)), "NPL": ("Nepal", ("Nepal",)),
 "KHM": ("Cambodia", ("Cambodia", "Khmer Republic")),
 "LAO": ("Laos", ("Laos", "Lao People's Democratic Republic")),
 "MMR": ("Myanmar", ("Myanmar", "Burma")),
 "MNG": ("Mongolia", ("Mongolia",)),
 "KAZ": ("Kazakhstan", ("Kazakhstan",)), "UZB": ("Uzbekistan", ("Uzbekistan",)),
 "KGZ": ("Kyrgyzstan", ("Kyrgyzstan",)), "TJK": ("Tajikistan", ("Tajikistan",)),
 "TKM": ("Turkmenistan", ("Turkmenistan",)),
 "AFG": ("Afghanistan", ("Afghanistan",)), "SYR": ("Syria", ("Syria", "Syrian Arab Republic")),
 "PSE": ("Palestine", ("Palestine", "State of Palestine")),
 "BHR": ("Bahrain", ("Bahrain",)),
 # ---- Africa ----
 "ZAF": ("South Africa", ("South Africa",)),
 "EGY": ("Egypt", ("Egypt",)),
 "KEN": ("Kenya", ("Kenya",)),
 "NGA": ("Nigeria", ("Nigeria",)),
 "ETH": ("Ethiopia", ("Ethiopia",)),
 "TZA": ("Tanzania", ("Tanzania", "United Republic of Tanzania")),
 "SEN": ("Senegal", ("Senegal",)),
 "UGA": ("Uganda", ("Uganda",)),
 "GHA": ("Ghana", ("Ghana",)),
 "CMR": ("Cameroon", ("Cameroon",)),
 "MAR": ("Morocco", ("Morocco",)),
 "DZA": ("Algeria", ("Algeria",)),
 "TUN": ("Tunisia", ("Tunisia",)),
 "SDN": ("Sudan", ("Sudan",)),
 "MLI": ("Mali", ("Mali",)),
 "BFA": ("Burkina Faso", ("Burkina Faso",)),
 "CIV": ("Cote d'Ivoire", ("Cote d'Ivoire", "Ivory Coast", "Cote DIvoire", "Cote Ivoire")),
 "AGO": ("Angola", ("Angola",)),
 "MOZ": ("Mozambique", ("Mozambique",)),
 "ZWE": ("Zimbabwe", ("Zimbabwe",)),
 "ZMB": ("Zambia", ("Zambia",)),
 "MWI": ("Malawi", ("Malawi",)),
 "BWA": ("Botswana", ("Botswana",)),
 "NAM": ("Namibia", ("Namibia",)),
 "RWA": ("Rwanda", ("Rwanda",)),
 "BDI": ("Burundi", ("Burundi",)),
 "MDG": ("Madagascar", ("Madagascar",)),
 "MUS": ("Mauritius", ("Mauritius",)),
 "REU": ("Reunion", ("Reunion", "La Reunion")),
 "COD": ("DR Congo", ("Democratic Republic of the Congo", "Democratic Republic Congo",
                        "DR Congo", "Congo Democratic Republic", "Zaire", "DEM REP CONGO")),
 "COG": ("Congo", ("Republic of the Congo", "Congo", "Congo Republic", "Rep Congo")),
 "GAB": ("Gabon", ("Gabon",)),
 "BEN": ("Benin", ("Benin", "Dahomey")),
 "TGO": ("Togo", ("Togo",)),
 "NER": ("Niger", ("Niger",)),
 "TCD": ("Chad", ("Chad",)),
 "CAF": ("Central African Republic", ("Central African Republic", "Cent Afr Republ")),
 "SOM": ("Somalia", ("Somalia",)),
 "DJI": ("Djibouti", ("Djibouti",)),
 "ERI": ("Eritrea", ("Eritrea",)),
 "GMB": ("Gambia", ("Gambia", "The Gambia")),
 "SLE": ("Sierra Leone", ("Sierra Leone",)),
 "LBR": ("Liberia", ("Liberia",)),
 "GIN": ("Guinea", ("Guinea", "Republique de Guinee")),
 "GNB": ("Guinea-Bissau", ("Guinea-Bissau", "Guinea Bissau")),
 "GNQ": ("Equatorial Guinea", ("Equatorial Guinea",)),
 "MRT": ("Mauritania", ("Mauritania",)),
 "LBY": ("Libya", ("Libya",)),
 "LSO": ("Lesotho", ("Lesotho",)),
 "SWZ": ("Eswatini", ("Eswatini", "Swaziland",)),
 "PNG": ("Papua New Guinea", ("Papua New Guinea", "Papua N Guinea")),
 "NCL": ("New Caledonia", ("New Caledonia",)),
 "COM": ("Comoros", ("Comoros", "Comoros Islands")),
 "BRN": ("Brunei", ("Brunei", "Brunei Darussalam")),
 "CPV": ("Cape Verde", ("Cape Verde", "Cabo Verde")),
 "SYC": ("Seychelles", ("Seychelles",)),
 "MDV": ("Maldives", ("Maldives",)),
 "BTN": ("Bhutan", ("Bhutan",)),
 "FSM": ("Micronesia", ("Micronesia", "Federated States of Micronesia")),
 "PRK": ("North Korea", ("North Korea", "Democratic People's Republic of Korea")),
 "TLS": ("Timor-Leste", ("Timor-Leste", "East Timor")),
 "MHL": ("Marshall Islands", ("Marshall Islands", "Marshall Island")),
 "FJI": ("Fiji", ("Fiji",)),
 "NZL": ("New Zealand", ("New Zealand",)),
 "AUS": ("Australia", ("Australia",)),
 "WSM": ("Samoa", ("Samoa",)),
 "TON": ("Tonga", ("Tonga",)),
 "SLB": ("Solomon Islands", ("Solomon Islands",)),
 "VUT": ("Vanuatu", ("Vanuatu",)),
}

# compile: terminal-of-block match; longer aliases first
ALIAS = []
for code, (_, aliases) in C.items():
    for a in aliases:
        ALIAS.append((code, a))
ALIAS.sort(key=lambda x: -len(x[1]))
US_RX = re.compile(r"(?:^|[,;])\s*(?:[A-Z]{2}\s+)?(?:\d{4,5}(?:-\d{4})?\s+)?(?:USA|United States|U\.S\.A\.?)\s*\.?$", re.I)
def block_country(block: str):
    b = block.strip().rstrip(";").strip()
    if US_RX.search(b):
        return "USA"
    for code, a in ALIAS:
        # alias must be the final segment after a comma/semicolon (or whole block)
        rx = re.compile(r"(?:^|[,;])\s*" + re.escape(a) + r"\s*\.?$", re.I)
        if rx.search(b):
            return code
    return None

def split_blocks(c1: str):
    out = []
    for chunk in c1.split("["):
        if "]" in chunk:
            out.append(chunk.split("]", 1)[1])
    return out

def run(virus):
    path, start = FILES[virus]
    pubs = Counter(); cites = Counter(); seen = set()
    eligible = matched = 0
    unmatched_tails = Counter(); unmatched_n = 0
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for rn, row in enumerate(reader, 2):
            py = (row.get("PY") or "").strip()
            if not py.isdigit() or not (start <= int(py) <= END_YEAR):
                continue
            uid = (row.get("UT") or "").strip() or f"ROW:{rn}"
            if uid in seen:
                continue
            seen.add(uid); eligible += 1
            try:
                tc = int(float((row.get("TC") or "0").strip() or 0))
            except ValueError:
                tc = 0
            codes = set()
            for block in split_blocks(row.get("C1") or ""):
                code = block_country(block)
                if code:
                    codes.add(code)
                else:
                    unmatched_n += 1
                    tail = re.split(r"[,;]", block.strip())[-1].strip().rstrip(".")
                    tail = re.sub(r"\s+", " ", tail)
                    tail = " ".join(tail.split()[-3:])
                    if tail:
                        unmatched_tails[tail] += 1
            if codes:
                matched += 1
            for code in codes:
                pubs[code] += 1; cites[code] += tc
    rows = [{"Virus": virus, "Code": code, "Country": C[code][0],
             "Publications": pubs[code], "Total_citations": cites[code]} for code in pubs]
    df = pd_sorted(rows)
    df.to_csv(OUT / f"global_country_pubs_{virus.replace(' ','')}.csv", index=False, encoding="utf-8-sig")
    audit = {"Virus": virus, "eligible": eligible, "matched": matched,
             "match_rate": round(matched / max(eligible, 1), 4), "unmatched_blocks": unmatched_n}
    print(f"\n=== {virus} ===")
    print(f"eligible={eligible} matched={matched} rate={matched/max(eligible,1):.3%} "
          f"unmatched_blocks={unmatched_n} countries={len(pubs)}")
    print("Top 15:", [(C[c][0], pubs[c]) for c, _ in pubs.most_common(15)])
    print("Top unmatched tails:", unmatched_tails.most_common(20))
    return audit, unmatched_tails

def pd_sorted(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    return df.sort_values("Publications", ascending=False)

if __name__ == "__main__":
    import pandas as pd  # noqa
    targets = [sys.argv[1]] if len(sys.argv) > 1 else list(FILES)
    audits = []
    for v in targets:
        a, _ = run(v)
        audits.append(a)
    import pandas as pd
    pd.DataFrame(audits).to_csv(OUT / "global_match_audit.csv", index=False, encoding="utf-8-sig")
