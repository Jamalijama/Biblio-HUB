import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest
import pandas as pd
from collections import Counter


@pytest.fixture
def minimal_df():
    return pd.DataFrame(
        [
            {
                "Title": "Paper A",
                "Year": 2020,
                "Journal": "Journal of Network Science",
                "Authors": "Smith, J; Lee, K",
                "DOI": "10.1234/a",
                "Abstract": "This is about network mapping and citation analysis.",
                "Author_Keywords": "Network; Citation; Data",
                "Keywords_Plus": "Network Analysis; Citation Analysis",
                "Cited_References": "SMITH J, 2018, SCIENCE; LEE K, 2019, NATURE",
            },
            {
                "Title": "Paper B",
                "Year": 2021,
                "Journal": "Journal of Citation Studies",
                "Authors": "Chen, Q",
                "DOI": "",
                "Abstract": "This is about citation analysis and data processing.",
                "Author_Keywords": "Citation; Data",
                "Keywords_Plus": "Citation Analysis; Data Science",
                "Cited_References": "SMITH J, 2018, SCIENCE; CHEN Q, 2017, CELL",
            },
            {
                "Title": "Paper C",
                "Year": 2022,
                "Journal": "Journal of Data",
                "Authors": "Wang, L; Zhang, Z",
                "DOI": "10.1234/c",
                "Abstract": "This is about Data.",
                "Author_Keywords": "Data",
                "Keywords_Plus": "Big Data",
                "Cited_References": "WANG L, 2016, PNAS",
            },
        ]
    )


@pytest.fixture
def keyword_freq():
    return {"Network": 4, "Citation": 3, "Data": 2}


@pytest.fixture
def cooccurrence():
    return Counter({("Network", "Citation"): 2, ("Citation", "Data"): 1})


@pytest.fixture
def dedup_report():
    return {"original": 4, "removed": 1, "final": 3}
