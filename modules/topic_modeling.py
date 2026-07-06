import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from modules.export_bundle import SCIENTIFIC_COLORWAY


_BERTOPIC_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bertopic-worker")
_BERTOPIC_JOBS: dict[str, dict] = {}
_BERTOPIC_JOBS_LOCK = Lock()


def _most_common_items(values, limit):
    if hasattr(values, "most_common"):
        return values.most_common(limit)
    return sorted(values.items(), key=lambda item: item[1], reverse=True)[:limit]


def get_bertopic_profile_settings(record_count, profile_name="quick_preview"):
    record_count = max(int(record_count or 0), 0)
    normalized_profile = profile_name if profile_name in {"quick_preview", "full_analysis"} else "quick_preview"
    if normalized_profile == "full_analysis":
        if record_count > 6000:
            doc_cap = 600
        elif record_count > 3000:
            doc_cap = 900
        else:
            doc_cap = min(record_count, 1500)
        include_topics_over_time = doc_cap <= 600
        summary = (
            f"Full analysis: {doc_cap}/{record_count} records"
            + (", with topic evolution." if include_topics_over_time else ", topic evolution skipped for speed.")
        )
    else:
        if record_count > 6000:
            doc_cap = 250
        elif record_count > 3000:
            doc_cap = 400
        else:
            doc_cap = min(record_count, 600)
        include_topics_over_time = False
        summary = f"Quick preview: {doc_cap}/{record_count} records, topic evolution skipped."
    return {
        "profile_name": normalized_profile,
        "doc_cap": min(doc_cap, record_count),
        "include_topics_over_time": include_topics_over_time,
        "summary": summary,
        "downsampled": min(doc_cap, record_count) < record_count,
    }


def recommend_bertopic_doc_cap(record_count, lightweight_mode=True):
    profile_name = "quick_preview" if lightweight_mode else "full_analysis"
    return get_bertopic_profile_settings(record_count, profile_name)["doc_cap"]


def should_compute_bertopic_evolution(analyzed_record_count, lightweight_mode=True):
    analyzed_record_count = max(int(analyzed_record_count or 0), 0)
    if analyzed_record_count < 10:
        return False
    if lightweight_mode:
        return False
    return analyzed_record_count <= 600


@st.cache_resource
def get_sentence_transformer_model(model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _execute_bertopic_job(df, min_docs=3, include_topics_over_time=True):
    runner = run_bertopic_analysis.__wrapped__ if hasattr(run_bertopic_analysis, "__wrapped__") else run_bertopic_analysis
    return runner(df, min_docs=min_docs, include_topics_over_time=include_topics_over_time)


def submit_bertopic_background_job(
    df,
    min_docs=3,
    include_topics_over_time=True,
    profile_name="quick_preview",
):
    job_id = str(uuid.uuid4())
    future = _BERTOPIC_EXECUTOR.submit(
        _execute_bertopic_job,
        df.copy(),
        min_docs,
        include_topics_over_time,
    )
    with _BERTOPIC_JOBS_LOCK:
        _BERTOPIC_JOBS[job_id] = {
            "job_id": job_id,
            "future": future,
            "profile_name": profile_name,
            "min_docs": int(min_docs),
            "include_topics_over_time": bool(include_topics_over_time),
            "analyzed_records": int(len(df)),
            "submitted_at": time.time(),
            "resolved_at": None,
            "result": None,
            "error": None,
        }
    return job_id


def get_bertopic_background_job(job_id):
    with _BERTOPIC_JOBS_LOCK:
        job = _BERTOPIC_JOBS.get(job_id)
    if not job:
        return {"job_id": job_id, "status": "missing"}

    future = job["future"]
    status = "running"
    if future.cancelled():
        status = "cancelled"
    elif future.done():
        try:
            result = future.result()
            with _BERTOPIC_JOBS_LOCK:
                job["result"] = result
                job["resolved_at"] = job["resolved_at"] or time.time()
                job["error"] = None
            status = "done"
        except Exception as exc:
            with _BERTOPIC_JOBS_LOCK:
                job["error"] = str(exc)
                job["resolved_at"] = job["resolved_at"] or time.time()
            status = "error"

    with _BERTOPIC_JOBS_LOCK:
        current_job = dict(_BERTOPIC_JOBS.get(job_id, {}))
    current_job.pop("future", None)
    current_job["status"] = status
    return current_job


def discard_bertopic_background_job(job_id):
    with _BERTOPIC_JOBS_LOCK:
        job = _BERTOPIC_JOBS.pop(job_id, None)
    if not job:
        return False
    future = job.get("future")
    if future and not future.done():
        future.cancel()
    return True


@st.cache_data
def run_bertopic_analysis(df, min_docs=3, include_topics_over_time=True):
    try:
        import bertopic
        import hdbscan
        import sentence_transformers
        import umap
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from umap import UMAP
    except ImportError as exc:
        print(f"BERTopic dependencies missing: {exc}")
        return "MISSING_LIB", str(exc), None

    docs = []
    years = []
    for _, row in df.iterrows():
        title = str(row.get("Title", "")).strip()
        abstract = str(row.get("Abstract", "")).strip()
        text = f"{title}. {abstract}" if abstract and abstract != "nan" else title
        if text and text != "nan" and len(text) > 20:
            docs.append(text)
            year = row.get("Year", None)
            try:
                years.append(int(float(str(year))))
            except (ValueError, TypeError):
                years.append(None)

    if len(docs) < 10:
        return None, None, None

    sentence_model = get_sentence_transformer_model("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=False)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(
        min_cluster_size=max(min_docs, 2),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=False,
        language="english",
    )

    valid_mask = [year is not None for year in years]
    valid_docs = [doc for doc, keep in zip(docs, valid_mask) if keep]
    valid_years = [year for year, keep in zip(years, valid_mask) if keep]
    valid_embeddings = embeddings[valid_mask]
    if len(valid_docs) < 10:
        return None, None, None

    topic_model.fit_transform(valid_docs, embeddings=valid_embeddings)
    topics_over_time = None
    if include_topics_over_time and valid_years and len(set(valid_years)) >= 3:
        try:
            topics_over_time = topic_model.topics_over_time(valid_docs, valid_years, nr_bins=10)
        except Exception:
            topics_over_time = None
    topic_info = topic_model.get_topic_info()
    return topic_model, topic_info, topics_over_time


def render_bertopic_overview(topic_info, top_n=15):
    if topic_info is None or len(topic_info) <= 1:
        return None

    display_info = topic_info[topic_info["Topic"] != -1].head(top_n)
    if len(display_info) == 0:
        return None

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for idx, (_, row) in enumerate(display_info.iterrows()):
        label = row.get("Name", f"Topic {row['Topic']}")
        count = row["Count"]
        fig.add_trace(
            go.Bar(
                name=label,
                x=[count],
                y=[label],
                orientation="h",
                marker_color=colors[idx % len(colors)],
                text=f"{count} docs",
                textposition="inside",
                hovertemplate=f"<b>{label}</b><br>Documents: {count}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=dict(text="Semantic Topic Distribution (BERTopic)", font=dict(size=16)),
        xaxis_title="Number of Documents",
        height=max(400, len(display_info) * 35 + 100),
        margin=dict(l=250, r=30, t=60, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="lightgray")
    fig.update_yaxes(gridcolor="lightgray")
    return fig


def render_bertopic_evolution(topics_over_time, top_n=10):
    if topics_over_time is None or len(topics_over_time) == 0:
        return None

    valid = topics_over_time[topics_over_time["Topic"] != -1]
    if len(valid) == 0:
        return None

    top_topics = valid.groupby("Topic")["Frequency"].sum().nlargest(top_n).index.tolist()
    filtered = valid[valid["Topic"].isin(top_topics)]
    if len(filtered) == 0:
        return None

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for idx, topic_id in enumerate(top_topics):
        topic_data = filtered[filtered["Topic"] == topic_id].sort_values("Timestamp")
        if len(topic_data) == 0:
            continue
        name = topic_data["Name"].iloc[0] if "Name" in topic_data.columns else f"Topic {topic_id}"
        timestamps = topic_data["Timestamp"]
        if hasattr(timestamps.dtype, "kind") and timestamps.dtype.kind == "M":
            x_vals = timestamps.dt.year
        else:
            x_vals = timestamps
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=topic_data["Frequency"],
                name=name,
                mode="lines+markers",
                line=dict(width=2.5, color=colors[idx % len(colors)]),
                marker=dict(size=7),
                hovertemplate=f"<b>{name}</b><br>Year: %{{x}}<br>Documents: %{{y}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text="Semantic Topic Evolution Over Time (BERTopic)", font=dict(size=16)),
        xaxis_title="Year",
        yaxis_title="Number of Documents",
        hovermode="x unified",
        height=550,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=10)),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="lightgray")
    fig.update_yaxes(gridcolor="lightgray")
    return fig


def render_bertopic_comparison(keyword_freq, topic_info, top_n=15):
    if topic_info is None or len(topic_info) <= 1:
        return None

    keyword_topics = set()
    for _, row in topic_info[topic_info["Topic"] != -1].head(top_n).iterrows():
        name = str(row.get("Name", ""))
        parts = name.split("_")
        for part in parts[1:]:
            token = part.lower().strip()
            if len(token) > 2:
                keyword_topics.add(token)

    keyword_cooccurrence = set()
    for keyword, _ in _most_common_items(keyword_freq, top_n * 3):
        keyword_cooccurrence.add(keyword.lower().strip())

    overlap = keyword_topics & keyword_cooccurrence
    only_cooccurrence = keyword_cooccurrence - keyword_topics
    only_bertopic = keyword_topics - keyword_cooccurrence

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Shared Topics", "Only in Co-occurrence", "Only in BERTopic"),
        horizontal_spacing=0.08,
    )
    shared_list = sorted(overlap)[:10]
    cooccurrence_only_list = sorted(only_cooccurrence)[:10]
    bertopic_only_list = sorted(only_bertopic)[:10]

    fig.add_trace(
        go.Bar(
            x=[keyword_freq.get(keyword.title(), 0) for keyword in shared_list],
            y=[keyword.title() for keyword in shared_list],
            orientation="h",
            marker_color=SCIENTIFIC_COLORWAY[2],
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=[keyword_freq.get(keyword.title(), 0) for keyword in cooccurrence_only_list],
            y=[keyword.title() for keyword in cooccurrence_only_list],
            orientation="h",
            marker_color=SCIENTIFIC_COLORWAY[0],
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=[1] * len(bertopic_only_list),
            y=[keyword.title() for keyword in bertopic_only_list],
            orientation="h",
            marker_color=SCIENTIFIC_COLORWAY[3],
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.update_layout(
        title=dict(text="Co-occurrence vs. BERTopic: Method Comparison", font=dict(size=18, color="black")),
        height=max(400, max(len(shared_list), len(cooccurrence_only_list), len(bertopic_only_list)) * 30 + 150),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray", title_text="Count", title_font=dict(size=14, color="black"), tickfont=dict(color="black"))
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", tickfont=dict(color="black"))
    return fig
