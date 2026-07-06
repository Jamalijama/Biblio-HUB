import pandas as pd

from modules.entity_analysis import (
    build_top_country_table,
    build_top_institution_table,
    build_top_journal_table,
    extract_institutions_from_affiliation,
)


def test_extract_institutions_from_affiliation_prefers_organization_name():
    aff_str = (
        "[Alice] Department of Data Science, University of Test, New York, USA; "
        "[Bob] Institute of Metrics, Beijing, PR CHINA"
    )

    institutions = extract_institutions_from_affiliation(aff_str)

    assert "University of Test" in institutions
    assert "Institute of Metrics" in institutions
    assert "Department of Data Science" not in institutions


def test_build_top_entity_tables_return_ranked_frames():
    df = pd.DataFrame(
        [
            {
                "Journal": "Journal A",
                "Affiliations": "[Alice] University of Test, New York, USA; [Bob] Institute of Metrics, Beijing, PR CHINA",
            },
            {
                "Journal": "Journal A",
                "Affiliations": "[Carol] Department of Data Science, University of Test, New York, USA",
            },
            {
                "Journal": "Journal B",
                "Affiliations": "[Dan] Institute of Metrics, Beijing, PR CHINA",
            },
        ]
    )

    country_df = build_top_country_table(df, top_n=10)
    institution_df = build_top_institution_table(df, top_n=10)
    journal_df = build_top_journal_table(df, top_n=10)

    assert list(country_df.columns) == ["Rank", "Country", "Papers", "Share (%)"]
    assert country_df.iloc[0]["Country"] == "China"
    assert country_df.iloc[0]["Papers"] == 2
    assert institution_df.iloc[0]["Institution"] in {"Institute of Metrics", "University of Test"}
    assert institution_df.iloc[0]["Papers"] == 2
    assert journal_df.iloc[0]["Journal"] == "Journal A"
    assert journal_df.iloc[0]["Papers"] == 2
