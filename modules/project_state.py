
import io
import json
from typing import Any, Dict, Optional, Tuple

import pandas as pd

PROJECT_SESSION_PREFIXES = [
    "vos_",
    "kj_",
    "auth_",
    "inst_",
    "bc_",
    "cocite_",
    "bt_",
    "flow_",
    "cs_",
    "burst_",
    "entity_",
    "map_",
    "matrix_",
    "thematic_",
    "journal_",
    "language_",
    "structure_",
    "vocab_",
    "publication_bundle_",
    "lightweight_mode_",
]

LOADED_PROJECT_KEYS = [
    "project_loaded_active",
    "loaded_df",
    "loaded_keywords_list",
    "loaded_keyword_freq",
    "loaded_cooccurrence",
    "loaded_journal_year",
    "loaded_dedup_report",
    "loaded_session_state",
]


def _encode_mapping(mapping: Any) -> Any:
    if not isinstance(mapping, dict):
        return mapping
    return [
        {"key": list(key) if isinstance(key, tuple) else key, "value": value}
        for key, value in mapping.items()
    ]


def _decode_mapping(payload: Any) -> Any:
    if not isinstance(payload, list):
        return payload
    restored = {}
    for item in payload:
        key = item["key"]
        if isinstance(key, list):
            key = tuple(key)
        restored[key] = item["value"]
    return restored


def save_project_state(
    df: pd.DataFrame,
    keywords_list: list,
    keyword_freq: Dict,
    cooccurrence: Dict,
    journal_year: Any,
    dedup_report: Optional[Dict] = None,
    st_session_state: Optional[Dict] = None
) -> bytes:
    state = {
        "df": df.to_json(orient="split", date_format="iso"),
        "keywords_list": keywords_list,
        "keyword_freq": _encode_mapping(dict(keyword_freq)),
        "cooccurrence": _encode_mapping(cooccurrence),
        "journal_year": journal_year,
        "dedup_report": dedup_report,
        "session_state": st_session_state if st_session_state else {}
    }
    return json.dumps(state, ensure_ascii=False, indent=2).encode("utf-8")


def load_project_state(json_bytes: bytes) -> Tuple[pd.DataFrame, list, Dict, Dict, Any, Optional[Dict], Dict]:
    state = json.loads(json_bytes.decode("utf-8"))
    df = pd.read_json(io.StringIO(state["df"]), orient="split")
    keywords_list = state["keywords_list"]
    keyword_freq = _decode_mapping(state["keyword_freq"])
    cooccurrence = _decode_mapping(state["cooccurrence"])
    journal_year = state.get("journal_year")
    dedup_report = state.get("dedup_report")
    session_state = state.get("session_state", {})
    return df, keywords_list, keyword_freq, cooccurrence, journal_year, dedup_report, session_state


def collect_project_session_state(session_state: Dict[str, Any], page: str) -> Dict[str, Any]:
    saved_session_state = {}
    for key in session_state:
        if any(key.startswith(prefix) for prefix in PROJECT_SESSION_PREFIXES):
            saved_session_state[key] = session_state[key]
    saved_session_state["selected_page"] = page
    return saved_session_state


def stash_loaded_project_state(
    session_state: Dict[str, Any],
    loaded_df: pd.DataFrame,
    loaded_keywords_list: list,
    loaded_keyword_freq: Dict,
    loaded_cooccurrence: Dict,
    loaded_journal_year: Any,
    loaded_dedup_report: Optional[Dict],
    loaded_session_state: Dict[str, Any],
) -> None:
    session_state["loaded_df"] = loaded_df
    session_state["loaded_keywords_list"] = loaded_keywords_list
    session_state["loaded_keyword_freq"] = loaded_keyword_freq
    session_state["loaded_cooccurrence"] = loaded_cooccurrence
    session_state["loaded_journal_year"] = loaded_journal_year
    session_state["loaded_dedup_report"] = loaded_dedup_report
    session_state["loaded_session_state"] = loaded_session_state


def apply_loaded_project_state(session_state: Dict[str, Any]) -> None:
    for key, value in session_state.get("loaded_session_state", {}).items():
        session_state[key] = value
    session_state["project_loaded_active"] = True


def clear_loaded_project_state(session_state: Dict[str, Any]) -> None:
    for key in LOADED_PROJECT_KEYS:
        session_state.pop(key, None)


def render_project_management_sidebar(
    st: Any,
    page: str,
    df: pd.DataFrame,
    keywords_list: list,
    keyword_freq: Dict,
    cooccurrence: Dict,
    journal_year: Any,
    dedup_report: Optional[Dict],
    log_exception_callback: Optional[Any] = None,
) -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💾 Project Management")

    if st.session_state.get("project_loaded_active"):
        st.sidebar.info("Using a loaded project snapshot.")
        if st.sidebar.button("Clear Loaded Project", key="btn_clear_loaded_project"):
            clear_loaded_project_state(st.session_state)
            st.rerun()

    if st.sidebar.button("Save Current Project", key="btn_save_project"):
        saved_session_state = collect_project_session_state(st.session_state, page)
        project_bytes = save_project_state(
            df,
            keywords_list,
            keyword_freq,
            cooccurrence,
            journal_year,
            dedup_report,
            saved_session_state,
        )
        st.sidebar.download_button(
            label="Download Project JSON",
            data=project_bytes,
            file_name="biblio_hub_project.json",
            mime="application/json",
            key="btn_download_project",
        )
        st.sidebar.success("Project prepared for download!")

    st.sidebar.markdown("---")
    uploaded_project = st.sidebar.file_uploader(
        "Load Saved Project",
        type=["json"],
        key="project_uploader",
    )
    if uploaded_project is None:
        st.sidebar.markdown("---")
        return

    try:
        (
            loaded_df,
            loaded_keywords_list,
            loaded_keyword_freq,
            loaded_cooccurrence,
            loaded_journal_year,
            loaded_dedup_report,
            loaded_session_state,
        ) = load_project_state(uploaded_project.read())
        stash_loaded_project_state(
            st.session_state,
            loaded_df,
            loaded_keywords_list,
            loaded_keyword_freq,
            loaded_cooccurrence,
            loaded_journal_year,
            loaded_dedup_report,
            loaded_session_state,
        )
        st.sidebar.success("Project loaded successfully! Click 'Apply Loaded Project' to use it.")
        if st.sidebar.button("Apply Loaded Project", key="btn_apply_project"):
            apply_loaded_project_state(st.session_state)
            st.rerun()
    except Exception as exc:
        if log_exception_callback is not None:
            log_exception_callback("load_project_state", exc)
        st.sidebar.error(f"Failed to load project: {exc}")

    st.sidebar.markdown("---")
