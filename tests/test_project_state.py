
import pandas as pd

from modules.project_state import (
    apply_loaded_project_state,
    clear_loaded_project_state,
    collect_project_session_state,
    load_project_state,
    save_project_state,
    stash_loaded_project_state,
)


def test_save_and_load_project_state():
                      
    df = pd.DataFrame({
        "Title": ["Paper 1", "Paper 2", "Paper 3"],
        "Year": [2020, 2021, 2022],
        "Journal": ["Journal A", "Journal B", "Journal A"],
        "Authors": ["Smith J", "Lee K", "Wang H"]
    })
    keywords_list = ["keyword1", "keyword2", "keyword3"]
    keyword_freq = {"keyword1": 10, "keyword2": 5, "keyword3": 3}
    cooccurrence = {("keyword1", "keyword2"): 4, ("keyword2", "keyword3"): 2}
    journal_year = None
    dedup_report = {"original": 3, "removed": 0, "final": 3}
    session_state = {"selected_page": "Dataset Overview", "vos_topn": 20}

                
    project_bytes = save_project_state(
        df,
        keywords_list,
        keyword_freq,
        cooccurrence,
        journal_year,
        dedup_report,
        session_state
    )

                
    (
        loaded_df,
        loaded_keywords_list,
        loaded_keyword_freq,
        loaded_cooccurrence,
        loaded_journal_year,
        loaded_dedup_report,
        loaded_session_state
    ) = load_project_state(project_bytes)

            
    pd.testing.assert_frame_equal(df, loaded_df)
    assert loaded_keywords_list == keywords_list
    assert loaded_keyword_freq == keyword_freq
    assert loaded_cooccurrence == cooccurrence
    assert loaded_journal_year == journal_year
    assert loaded_dedup_report == dedup_report
    assert loaded_session_state == session_state


def test_collect_project_session_state_keeps_known_prefixes_and_page():
    session_state = {
        "vos_topn": 30,
        "auth_min_papers": 2,
        "unrelated_key": "skip",
    }

    collected = collect_project_session_state(session_state, "Relational Network Analysis")

    assert collected["vos_topn"] == 30
    assert collected["auth_min_papers"] == 2
    assert "unrelated_key" not in collected
    assert collected["selected_page"] == "Relational Network Analysis"


def test_stash_apply_and_clear_loaded_project_state_updates_session_state():
    session_state = {}
    df = pd.DataFrame({"Title": ["Paper 1"]})
    loaded_inner_state = {"vos_topn": 40, "selected_page": "Dataset Overview"}

    stash_loaded_project_state(
        session_state,
        df,
        ["kw"],
        {"kw": 1},
        {("kw", "kw2"): 2},
        None,
        {"original": 1, "removed": 0, "final": 1},
        loaded_inner_state,
    )

    assert "loaded_df" in session_state
    assert session_state["loaded_keywords_list"] == ["kw"]

    apply_loaded_project_state(session_state)
    assert session_state["project_loaded_active"] is True
    assert session_state["vos_topn"] == 40
    assert session_state["selected_page"] == "Dataset Overview"

    clear_loaded_project_state(session_state)
    assert "project_loaded_active" not in session_state
    assert "loaded_df" not in session_state
