from collections import Counter
import re

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from modules.data_pipeline import _extract_country_from_affiliation, clean_year_column
from modules.entity_analysis import extract_institutions_from_affiliation
from modules.keyword_pipeline import normalize_keyword

BURST_GENERIC_TERMS = {
    "Arbovirus",
    "Arboviruses",
    "Dengue Virus",
    "Disease",
    "Epidemiology",
    "Fever",
    "Infection",
    "Transmission",
    "Virus",
}


def _most_common_items(values, limit):
    if hasattr(values, "most_common"):
        return values.most_common(limit)
    return sorted(values.items(), key=lambda item: item[1], reverse=True)[:limit]


def _fallback_burst_info(label, freqs, year_range, label_column="Keyword"):
    peak_freq = max(freqs)
    if peak_freq <= 0:
        return None

    peak_index = freqs.index(peak_freq)
    return {
        label_column: label,
        "Burst Strength": 1,
        "Burst Weight": round(peak_freq / max(1, sum(freqs) / len(freqs)), 2),
        "Start": year_range[peak_index],
        "End": year_range[peak_index],
        "Max Freq": peak_freq,
    }


def _compute_burst_llr(freqs, expected, start_idx, end_idx):
    if expected <= 0:
        return 0.0
    score = 0.0
    for idx in range(start_idx, end_idx + 1):
        observed = float(freqs[idx])
        if observed <= 0:
            continue
        score += observed * np.log(observed / expected) - (observed - expected)
    return max(0.0, float(score))


def _compute_adjusted_burst_score(raw_score, duration, alpha=0.75):
    return max(0.0, float(raw_score)) / (max(1, int(duration)) ** float(alpha))


def _build_year_slices(year_values, slice_count):
    unique_years = sorted({int(year) for year in year_values})
    if len(unique_years) < 2:
        return []

    slice_count = max(2, min(slice_count, len(unique_years)))
    slices = []
    for chunk in np.array_split(unique_years, slice_count):
        if len(chunk) == 0:
            continue
        chunk_years = [int(year) for year in chunk.tolist()]
        start_year = chunk_years[0]
        end_year = chunk_years[-1]
        label = str(start_year) if start_year == end_year else f"{start_year}-{end_year}"
        slices.append({"label": label, "years": set(chunk_years)})
    return slices


def _extract_period_topics(period_keywords, top_keywords, max_topics_per_slice, keywords_per_topic):
    keyword_counter = Counter()
    pair_counter = Counter()

    for keywords in period_keywords:
        filtered = [kw for kw in dict.fromkeys(keywords) if kw in top_keywords]
        if not filtered:
            continue
        keyword_counter.update(filtered)
        for idx, left_kw in enumerate(filtered):
            for right_kw in filtered[idx + 1:]:
                pair_counter[tuple(sorted((left_kw, right_kw)))] += 1

    if not keyword_counter:
        return []

    graph = nx.Graph()
    for keyword, weight in keyword_counter.items():
        graph.add_node(keyword, weight=weight)
    for (left_kw, right_kw), weight in pair_counter.items():
        graph.add_edge(left_kw, right_kw, weight=weight)

    if graph.number_of_edges() > 0:
        communities = list(nx.algorithms.community.greedy_modularity_communities(graph, weight="weight"))
    else:
        communities = [{keyword} for keyword, _ in keyword_counter.most_common(max_topics_per_slice)]

    topics = []
    for community in communities:
        ranked_keywords = sorted(community, key=lambda kw: (-keyword_counter[kw], kw))
        if not ranked_keywords:
            continue
        keyword_weights = {kw: keyword_counter[kw] for kw in ranked_keywords}
        topics.append(
            {
                "keywords": set(ranked_keywords),
                "keyword_weights": keyword_weights,
                "label_keywords": ranked_keywords[:keywords_per_topic],
                "weight": sum(keyword_weights.values()),
            }
        )

    topics.sort(key=lambda item: (-item["weight"], sorted(item["label_keywords"])))
    return topics[:max_topics_per_slice]


def _build_period_topic_slices(
    df,
    keywords_list,
    keyword_freq,
    *,
    slice_count=4,
    top_n_keywords=30,
    max_topics_per_slice=5,
    keywords_per_topic=3,
):
    df_valid = clean_year_column(df)
    if len(df_valid) < 4:
        return []

    year_slices = _build_year_slices(df_valid["Year"].tolist(), slice_count)
    if len(year_slices) < 2:
        return []

    top_keywords = {kw for kw, _ in _most_common_items(keyword_freq, top_n_keywords)}
    period_topics = []
    for year_slice in year_slices:
        slice_keywords = []
        for row_idx in df_valid.index:
            if row_idx >= len(keywords_list):
                continue
            year = int(df_valid.at[row_idx, "Year"])
            if year not in year_slice["years"]:
                continue
            slice_keywords.append(keywords_list[row_idx] or [])

        topics = _extract_period_topics(
            slice_keywords,
            top_keywords=top_keywords,
            max_topics_per_slice=max_topics_per_slice,
            keywords_per_topic=keywords_per_topic,
        )
        if topics:
            period_topics.append({"period": year_slice["label"], "topics": topics})
    return period_topics


def render_alluvial_topic_flow(
    df,
    keywords_list,
    keyword_freq,
    slice_count=4,
    top_n_keywords=30,
    max_topics_per_slice=5,
    keywords_per_topic=3,
):
    period_topics = _build_period_topic_slices(
        df,
        keywords_list,
        keyword_freq,
        slice_count=slice_count,
        top_n_keywords=top_n_keywords,
        max_topics_per_slice=max_topics_per_slice,
        keywords_per_topic=keywords_per_topic,
    )
    if len(period_topics) < 2:
        return None

    labels = []
    colors = []
    node_index = {}
    node_x = []
    node_y = []
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel + px.colors.qualitative.Safe

    for period_idx, period_data in enumerate(period_topics):
        period_color = palette[period_idx % len(palette)]
        period_x = 0.02 + 0.96 * (period_idx / max(1, len(period_topics) - 1))
        for topic_idx, topic in enumerate(period_data["topics"]):
            label = " / ".join(topic["label_keywords"])
            node_key = (period_idx, topic_idx)
            node_index[node_key] = len(labels)
            labels.append(label)
            colors.append(period_color)
            node_x.append(period_x)
            node_y.append(0.08 + 0.84 * (topic_idx / max(1, len(period_data["topics"]) - 1)))

    source = []
    target = []
    value = []
    link_labels = []

    for period_idx in range(len(period_topics) - 1):
        current_topics = period_topics[period_idx]["topics"]
        next_topics = period_topics[period_idx + 1]["topics"]
        for current_idx, current_topic in enumerate(current_topics):
            for next_idx, next_topic in enumerate(next_topics):
                shared_keywords = sorted(current_topic["keywords"] & next_topic["keywords"])
                if not shared_keywords:
                    continue
                overlap_weight = sum(
                    min(
                        current_topic["keyword_weights"].get(keyword, 0),
                        next_topic["keyword_weights"].get(keyword, 0),
                    )
                    for keyword in shared_keywords
                )
                if overlap_weight <= 0:
                    continue
                source.append(node_index[(period_idx, current_idx)])
                target.append(node_index[(period_idx + 1, next_idx)])
                value.append(overlap_weight)
                link_labels.append(", ".join(shared_keywords[:4]))

    if not value:
        return None

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=18,
                    thickness=18,
                    line=dict(color="rgba(80,80,80,0.35)", width=0.5),
                    label=labels,
                    color=colors,
                    x=node_x,
                    y=node_y,
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color="rgba(120, 120, 120, 0.25)",
                    customdata=link_labels,
                    hovertemplate=(
                        "<b>Shared Keywords</b>: %{customdata}<br>"
                        "<b>Flow Weight</b>: %{value}<extra></extra>"
                    ),
                ),
            )
        ]
    )
    fig.update_layout(
        title="Alluvial Topic Flow Across Time Slices",
        font=dict(size=13),                      
        height=max(650, 160 * max(len(period_data["topics"]) for period_data in period_topics)),                   
        margin=dict(l=40, r=40, t=80, b=40),                          
    )
    for period_idx, period_data in enumerate(period_topics):
        period_x = 0.02 + 0.96 * (period_idx / max(1, len(period_topics) - 1))
        fig.add_annotation(
            x=period_x,
            y=1.06,                  
            xref="paper",
            yref="paper",
            text=f"<b>{period_data['period']}</b>",         
            showarrow=False,
            font=dict(size=14, color="#44546A"),                  
            xanchor="center",
        )
    return fig


def _extract_entity_values(row, entity_type):
    if entity_type == "journal":
        value = str(row.get("Journal", "")).strip()
        return [value] if value and value != "nan" else []
    if entity_type == "country":
        countries_str = _extract_country_from_affiliation(row.get("Affiliations", ""))
        return [item.strip() for item in str(countries_str).split(";") if item and item.strip()]
    if entity_type == "institution":
        return extract_institutions_from_affiliation(row.get("Affiliations", ""))
    return []


def build_entity_forecast_tables(
    df,
    *,
    entity_type="journal",
    top_n_entities=15,
    forecast_horizon=3,
    lookback_years=6,
    min_total_occurrences=3,
):
    label_map = {
        "journal": "Journal",
        "country": "Country",
        "institution": "Institution",
    }
    entity_label = label_map.get(entity_type, "Entity")
    df_valid = clean_year_column(df)
    year_range = sorted(df_valid["Year"].unique())
    if len(year_range) < 4:
        return pd.DataFrame(), pd.DataFrame(), entity_label

    entity_totals = Counter()
    entity_year_counts = {}
    for _, row in df_valid.iterrows():
        entities = set(_extract_entity_values(row, entity_type))
        if not entities:
            continue
        year = int(row["Year"])
        for entity in entities:
            entity_totals[entity] += 1
            entity_year_counts.setdefault(entity, Counter())[year] += 1

    candidates = [
        entity for entity, total in entity_totals.most_common(top_n_entities * 3)
        if total >= min_total_occurrences
    ][: max(top_n_entities, 1) * 3]
    if not candidates:
        return pd.DataFrame(), pd.DataFrame(), entity_label

    summary_rows = []
    series_rows = []
    future_years = np.arange(year_range[-1] + 1, year_range[-1] + forecast_horizon + 1)
    for entity in candidates:
        counter = entity_year_counts.get(entity, Counter())
        actual_counts = np.array([float(counter.get(year, 0)) for year in year_range], dtype=float)
        if actual_counts.sum() < min_total_occurrences:
            continue

        fit_years = np.array(year_range[-lookback_years:], dtype=float) if len(year_range) > lookback_years else np.array(year_range, dtype=float)
        fit_counts = actual_counts[-lookback_years:] if len(year_range) > lookback_years else actual_counts
        slope, intercept = np.polyfit(fit_years, fit_counts, 1)
        forecast_counts = np.maximum(0.0, slope * future_years + intercept)
        latest_count = float(actual_counts[-1])
        next_count = float(forecast_counts[0]) if len(forecast_counts) > 0 else latest_count
        baseline = max(latest_count, 1.0)
        growth_rate = ((next_count - latest_count) / baseline) * 100.0

        summary_rows.append(
            {
                entity_label: entity,
                "Total Papers": int(actual_counts.sum()),
                "Latest Year": int(year_range[-1]),
                "Latest Count": round(latest_count, 2),
                "Projected Next Count": round(next_count, 2),
                "Projected Growth (%)": round(growth_rate, 2),
                "Trend Slope": round(float(slope), 4),
                "Active Years": int(np.count_nonzero(actual_counts)),
            }
        )

        for year, count in zip(year_range, actual_counts):
            series_rows.append(
                {
                    "Year": int(year),
                    entity_label: entity,
                    "Count": round(float(count), 2),
                    "Series": "Actual",
                }
            )
        for year, count in zip(future_years, forecast_counts):
            series_rows.append(
                {
                    "Year": int(year),
                    entity_label: entity,
                    "Count": round(float(count), 2),
                    "Series": "Forecast",
                }
            )

    if not summary_rows:
        return pd.DataFrame(), pd.DataFrame(), entity_label

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["Projected Growth (%)", "Projected Next Count", "Total Papers", entity_label],
        ascending=[False, False, False, True],
    ).head(top_n_entities).reset_index(drop=True)
    keep_entities = summary_df[entity_label].tolist()
    series_df = pd.DataFrame(series_rows)
    if not series_df.empty:
        series_df = series_df[series_df[entity_label].isin(keep_entities)].reset_index(drop=True)
    return summary_df, series_df, entity_label


def render_entity_forecast_rank_figure(summary_df, entity_label, top_n=10):
    if summary_df is None or summary_df.empty:
        return None

    plural_label = {"Country": "Countries", "Institution": "Institutions", "Journal": "Journals"}.get(
        entity_label,
        f"{entity_label}s",
    )
    plot_df = summary_df.head(top_n).copy().sort_values("Projected Growth (%)", ascending=True)
    fig = go.Figure()
    for _, row in plot_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[0, row["Projected Growth (%)"]],
                y=[row[entity_label], row[entity_label]],
                mode="lines",
                line=dict(color="rgba(94,55,157,0.35)", width=3),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=plot_df["Projected Growth (%)"],
            y=plot_df[entity_label],
            mode="markers",
            marker=dict(color="#5E379D", size=12, line=dict(color="white", width=1.1)),
            customdata=plot_df[["Projected Next Count", "Latest Count"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                + "Projected Growth: %{x:.2f}%<br>"
                + "Projected Next Count: %{customdata[0]}<br>"
                + "Latest Count: %{customdata[1]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(
        title=f"Exploratory Growth Projection for Top {plural_label}",
        xaxis_title="Exploratory Next-Year Growth (%)",
        yaxis_title=entity_label,
        height=max(420, 36 * len(plot_df) + 140),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def render_entity_forecast_trajectory_figure(series_df, entity_label, top_n=5):
    if series_df is None or series_df.empty:
        return None

    totals = (
        series_df[series_df["Series"] == "Actual"]
        .groupby(entity_label, as_index=False)["Count"]
        .sum()
        .sort_values(["Count", entity_label], ascending=[False, True])
        .head(top_n)
    )
    selected_entities = totals[entity_label].tolist()
    plot_df = series_df[series_df[entity_label].isin(selected_entities)].copy()
    if plot_df.empty:
        return None

    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Safe
    color_lookup = {entity: palette[idx % len(palette)] for idx, entity in enumerate(selected_entities)}
    fig = go.Figure()
    for entity in selected_entities:
        entity_df = plot_df[plot_df[entity_label] == entity].sort_values(["Series", "Year"])
        actual_df = entity_df[entity_df["Series"] == "Actual"].sort_values("Year")
        forecast_df = entity_df[entity_df["Series"] == "Forecast"].sort_values("Year")
        color = color_lookup[entity]
        if not actual_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=actual_df["Year"],
                    y=actual_df["Count"],
                    mode="lines+markers",
                    name=entity,
                    line=dict(color=color, width=1.8),
                    marker=dict(size=5, color=color),
                    hovertemplate=f"<b>{entity}</b><br>Year: %{{x}}<br>Count: %{{y}}<br>Series: Actual<extra></extra>",
                )
            )
        if not forecast_df.empty:
            forecast_x = [actual_df["Year"].max()] + forecast_df["Year"].tolist() if not actual_df.empty else forecast_df["Year"].tolist()
            forecast_y = [actual_df["Count"].iloc[-1]] + forecast_df["Count"].tolist() if not actual_df.empty else forecast_df["Count"].tolist()
            fig.add_trace(
                go.Scatter(
                    x=forecast_x,
                    y=forecast_y,
                    mode="lines+markers",
                    name=f"{entity} Forecast",
                    line=dict(color=color, width=1.5, dash="dash"),
                    marker=dict(size=4.5, color=color),
                    showlegend=False,
                    hovertemplate=f"<b>{entity}</b><br>Year: %{{x}}<br>Count: %{{y}}<br>Series: Forecast<extra></extra>",
                )
            )

    fig.update_layout(
        title=f"{entity_label} Exploratory Forecast Trajectories",
        xaxis_title="Year",
        yaxis_title="Papers",
        height=580,
        hovermode="x unified",
        legend_title=entity_label,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", rangemode="tozero")
    return fig


def summarize_entity_forecast_signals(summary_df, entity_label, top_n=5):
    if summary_df is None or summary_df.empty:
        return f"Current data do not provide enough stable year-by-{entity_label.lower()} continuity to support an exploratory forward projection."

    top_df = summary_df.head(top_n).copy()
    leaders = top_df[entity_label].tolist()
    strongest = leaders[:2]
    rising_fast = top_df[top_df["Projected Growth (%)"] > 0][entity_label].tolist()[:2]
    cooling = top_df[top_df["Projected Growth (%)"] < 0][entity_label].tolist()[:1]

    message_parts = []
    if strongest:
        message_parts.append(f"Exploratory forward candidates among {entity_label.lower()}s: " + "; ".join(strongest))
    if rising_fast:
        message_parts.append("Most likely to keep gaining momentum under the current projection: " + "; ".join(rising_fast))
    if cooling:
        message_parts.append("Potential near-term cooling under the current projection: " + "; ".join(cooling))
    return " | ".join(message_parts)


def build_entity_leadership_shift_tables(
    df,
    *,
    entity_type="country",
    top_n_entities=12,
    recent_year_window=4,
    min_total_occurrences=3,
):
    label_map = {
        "country": "Country",
        "institution": "Institution",
    }
    entity_label = label_map.get(entity_type, "Entity")
    df_valid = clean_year_column(df)
    year_range = sorted(df_valid["Year"].unique())
    if len(year_range) < 3:
        return pd.DataFrame(), pd.DataFrame(), entity_label

    recent_years = year_range[-min(recent_year_window, len(year_range)) :]
    focus_years = recent_years[-min(2, len(recent_years)) :]
    baseline_years = recent_years[:-len(focus_years)]
    if not baseline_years:
        baseline_years = recent_years[:1]

    yearly_publications = {
        int(year): int(count)
        for year, count in df_valid["Year"].value_counts().sort_index().items()
    }
    entity_totals = Counter()
    entity_year_counts = {}
    for _, row in df_valid.iterrows():
        entities = set(_extract_entity_values(row, entity_type))
        if not entities:
            continue
        year = int(row["Year"])
        for entity in entities:
            entity_totals[entity] += 1
            entity_year_counts.setdefault(entity, Counter())[year] += 1

    candidates = [
        entity for entity, total in entity_totals.most_common(top_n_entities * 3)
        if total >= min_total_occurrences
    ][: max(1, top_n_entities) * 3]
    if not candidates:
        return pd.DataFrame(), pd.DataFrame(), entity_label

    summary_rows = []
    series_rows = []
    for entity in candidates:
        counter = entity_year_counts.get(entity, Counter())
        total_count = int(sum(counter.values()))
        if total_count < min_total_occurrences:
            continue

        yearly_shares = {}
        for year in year_range:
            publications = max(yearly_publications.get(int(year), 0), 1)
            count = int(counter.get(int(year), 0))
            share = (count / publications) * 100.0
            yearly_shares[int(year)] = share
            series_rows.append(
                {
                    "Year": int(year),
                    entity_label: entity,
                    "Count": count,
                    "Leadership Share (%)": round(share, 3),
                }
            )

        recent_share = float(np.mean([yearly_shares[int(year)] for year in focus_years]))
        baseline_share = float(np.mean([yearly_shares[int(year)] for year in baseline_years]))
        share_shift = recent_share - baseline_share
        baseline_denominator = max(abs(baseline_share), 0.1)
        momentum = (share_shift / baseline_denominator) * 100.0

        summary_rows.append(
            {
                entity_label: entity,
                "Total Papers": total_count,
                "Recent Share (%)": round(recent_share, 3),
                "Baseline Share (%)": round(baseline_share, 3),
                "Share Shift (pp)": round(share_shift, 3),
                "Leadership Momentum (%)": round(momentum, 2),
                "Latest Count": int(counter.get(int(year_range[-1]), 0)),
                "Active Years": int(sum(1 for year in year_range if counter.get(int(year), 0) > 0)),
            }
        )

    if not summary_rows:
        return pd.DataFrame(), pd.DataFrame(), entity_label

    summary_df = pd.DataFrame(summary_rows)
    share_threshold = float(summary_df["Recent Share (%)"].median())
    signal_types = []
    for _, row in summary_df.iterrows():
        if row["Recent Share (%)"] >= share_threshold and row["Share Shift (pp)"] >= 0:
            signal_types.append("Rising Leaders")
        elif row["Recent Share (%)"] < share_threshold and row["Share Shift (pp)"] >= 0:
            signal_types.append("Emerging Challengers")
        elif row["Recent Share (%)"] >= share_threshold and row["Share Shift (pp)"] < 0:
            signal_types.append("Established but Slipping")
        else:
            signal_types.append("Peripheral / Stable")
    summary_df["Signal Type"] = signal_types
    summary_df = summary_df.sort_values(
        ["Share Shift (pp)", "Recent Share (%)", "Total Papers", entity_label],
        ascending=[False, False, False, True],
    ).head(top_n_entities).reset_index(drop=True)

    keep_entities = summary_df[entity_label].tolist()
    series_df = pd.DataFrame(series_rows)
    if not series_df.empty:
        series_df = series_df[series_df[entity_label].isin(keep_entities)].reset_index(drop=True)
    return summary_df, series_df, entity_label


def render_entity_leadership_shift_figure(summary_df, entity_label, top_n=10):
    if summary_df is None or summary_df.empty:
        return None

    plot_df = summary_df.head(top_n).copy().sort_values("Share Shift (pp)", ascending=True)
    color_map = {
        "Rising Leaders": "#D45959",
        "Emerging Challengers": "#2F74B8",
        "Established but Slipping": "#F2B382",
        "Peripheral / Stable": "#7A7A7A",
    }
    fig = px.scatter(
        plot_df,
        x="Share Shift (pp)",
        y=entity_label,
        size="Recent Share (%)",
        color="Signal Type",
        hover_name=entity_label,
        color_discrete_map=color_map,
        title=f"{entity_label} Leadership Shift",
    )
    fig.update_traces(
        marker=dict(line=dict(color="white", width=1)),
        customdata=plot_df[["Recent Share (%)", "Baseline Share (%)", "Leadership Momentum (%)"]],
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            + "Share Shift: %{x:.3f} pp<br>"
            + "Recent Share: %{customdata[0]:.3f}%<br>"
            + "Baseline Share: %{customdata[1]:.3f}%<br>"
            + "Leadership Momentum: %{customdata[2]:.2f}%<extra></extra>"
        ),
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#666666", opacity=0.7)
    fig.update_layout(
        height=max(460, 36 * len(plot_df) + 160),
        xaxis_title="Leadership Share Shift (percentage points)",
        yaxis_title=entity_label,
        legend_title="Signal Type",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=False)
    return fig


def render_entity_leadership_trajectory_figure(series_df, summary_df, entity_label, top_n=5):
    if series_df is None or series_df.empty or summary_df is None or summary_df.empty:
        return None

    selected_entities = summary_df.head(top_n)[entity_label].tolist()
    plot_df = series_df[series_df[entity_label].isin(selected_entities)].copy()
    if plot_df.empty:
        return None

    fig = px.line(
        plot_df,
        x="Year",
        y="Leadership Share (%)",
        color=entity_label,
        markers=True,
        title=f"{entity_label} Leadership Share Over Time",
        color_discrete_sequence=px.colors.qualitative.Set2 + px.colors.qualitative.Safe,
    )
    fig.update_layout(
        height=580,
        hovermode="x unified",
        xaxis_title="Year",
        yaxis_title="Leadership Share of Papers (%)",
        legend_title=entity_label,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", rangemode="tozero")
    return fig


def summarize_entity_leadership_shift(summary_df, entity_label, top_n=5):
    if summary_df is None or summary_df.empty:
        return f"Current data do not provide enough year-by-{entity_label.lower()} continuity to interpret leadership shifts."

    top_df = summary_df.head(top_n).copy()
    rising = top_df[top_df["Signal Type"].isin(["Rising Leaders", "Emerging Challengers"])][entity_label].tolist()[:2]
    slipping = top_df[top_df["Signal Type"] == "Established but Slipping"][entity_label].tolist()[:2]

    message_parts = []
    if rising:
        message_parts.append("Leadership is shifting toward: " + "; ".join(rising))
    if slipping:
        message_parts.append("Previously dominant but now slipping: " + "; ".join(slipping))
    if not message_parts:
        strongest = top_df.iloc[0][entity_label]
        message_parts.append(f"Most stable leadership signal currently comes from {strongest}.")
    return " | ".join(message_parts)


def build_theme_migration_forecast_tables(
    df,
    keywords_list,
    keyword_freq,
    *,
    slice_count=4,
    top_n_keywords=30,
    max_topics_per_slice=5,
    keywords_per_topic=3,
):
    period_topics = _build_period_topic_slices(
        df,
        keywords_list,
        keyword_freq,
        slice_count=slice_count,
        top_n_keywords=top_n_keywords,
        max_topics_per_slice=max_topics_per_slice,
        keywords_per_topic=keywords_per_topic,
    )
    if len(period_topics) < 2:
        return pd.DataFrame(), pd.DataFrame()

    nodes = {}
    best_parent = {}
    overlap_lookup = {}
    for period_idx, period_data in enumerate(period_topics):
        for topic_idx, topic in enumerate(period_data["topics"]):
            node_id = (period_idx, topic_idx)
            nodes[node_id] = {
                "period_idx": period_idx,
                "period": period_data["period"],
                "weight": int(topic["weight"]),
                "keywords": set(topic["keywords"]),
                "label_keywords": list(topic["label_keywords"]),
            }

    for period_idx in range(len(period_topics) - 1):
        current_topics = period_topics[period_idx]["topics"]
        next_topics = period_topics[period_idx + 1]["topics"]
        for current_idx, current_topic in enumerate(current_topics):
            for next_idx, next_topic in enumerate(next_topics):
                shared_keywords = current_topic["keywords"] & next_topic["keywords"]
                if not shared_keywords:
                    continue
                overlap_weight = sum(
                    min(
                        current_topic["keyword_weights"].get(keyword, 0),
                        next_topic["keyword_weights"].get(keyword, 0),
                    )
                    for keyword in shared_keywords
                )
                if overlap_weight <= 0:
                    continue
                prev_id = (period_idx, current_idx)
                next_id = (period_idx + 1, next_idx)
                overlap_lookup[(prev_id, next_id)] = overlap_weight
                existing = best_parent.get(next_id)
                if existing is None or overlap_weight > existing[1]:
                    best_parent[next_id] = (prev_id, overlap_weight)

    chain_map = {}
    next_chain_id = 1
    for period_idx, period_data in enumerate(period_topics):
        for topic_idx, _ in enumerate(period_data["topics"]):
            node_id = (period_idx, topic_idx)
            parent_info = best_parent.get(node_id)
            if parent_info and parent_info[0] in chain_map:
                chain_map[node_id] = chain_map[parent_info[0]]
            else:
                chain_map[node_id] = next_chain_id
                next_chain_id += 1

    chain_period_rows = []
    for node_id, chain_id in chain_map.items():
        node = nodes[node_id]
        chain_period_rows.append(
            {
                "Chain ID": chain_id,
                "Theme Cluster": f"Theme {chain_id}: " + " / ".join(node["label_keywords"]),
                "Period Index": node["period_idx"],
                "Period": node["period"],
                "Theme Weight": node["weight"],
                "Representative Keywords": ", ".join(node["label_keywords"]),
            }
        )
    chain_period_df = pd.DataFrame(chain_period_rows)
    if chain_period_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary_rows = []
    for chain_id, group in chain_period_df.groupby("Chain ID", sort=True):
        group = group.sort_values("Period Index")
        weights = group["Theme Weight"].astype(float).to_numpy()
        period_indices = group["Period Index"].astype(float).to_numpy()
        latest_weight = float(weights[-1])
        previous_weight = float(weights[-2]) if len(weights) >= 2 else 0.0
        baseline = max(previous_weight, 1.0)
        recent_growth = ((latest_weight - previous_weight) / baseline) * 100.0 if len(weights) >= 2 else 0.0
        slope = float(np.polyfit(period_indices, weights, 1)[0]) if len(weights) >= 2 else 0.0
        projected_next_weight = max(0.0, latest_weight + slope)
        age = int(len(group))
        if age <= 2 and recent_growth > 0:
            signal_type = "Emerging Theme"
        elif recent_growth >= 15:
            signal_type = "Rising Hotspot"
        elif recent_growth < -10:
            signal_type = "Cooling Theme"
        else:
            signal_type = "Stable Core"

        summary_rows.append(
            {
                "Chain ID": int(chain_id),
                "Theme Cluster": group["Theme Cluster"].iloc[-1],
                "Latest Slice": group["Period"].iloc[-1],
                "Latest Weight": round(latest_weight, 2),
                "Projected Next Weight": round(projected_next_weight, 2),
                "Recent Growth (%)": round(recent_growth, 2),
                "Trend Slope": round(slope, 3),
                "Covered Slices": age,
                "Representative Keywords": " | ".join(group["Representative Keywords"].tail(3).tolist()),
                "Signal Type": signal_type,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["Projected Next Weight", "Recent Growth (%)", "Covered Slices", "Chain ID"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return summary_df, chain_period_df


def render_theme_migration_trajectory_figure(chain_period_df, summary_df, top_n=6):
    if chain_period_df is None or chain_period_df.empty or summary_df is None or summary_df.empty:
        return None

    top_chain_ids = summary_df.head(top_n)["Chain ID"].tolist()
    plot_df = chain_period_df[chain_period_df["Chain ID"].isin(top_chain_ids)].copy()
    if plot_df.empty:
        return None

    fig = px.line(
        plot_df,
        x="Period",
        y="Theme Weight",
        color="Theme Cluster",
        markers=True,
        line_group="Chain ID",
        title="Theme Cluster Migration Trajectories",
        color_discrete_sequence=px.colors.qualitative.Set2 + px.colors.qualitative.Safe,
    )
    fig.update_traces(
        customdata=plot_df[["Representative Keywords"]],
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            + "Period: %{x}<br>"
            + "Theme Weight: %{y}<br>"
            + "Keywords: %{customdata[0]}<extra></extra>"
        ),
    )
    fig.update_layout(
        height=600,
        xaxis_title="Time Slice",
        yaxis_title="Theme Weight",
        hovermode="x unified",
        legend_title="Theme Chain",
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray", type="category")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", rangemode="tozero")
    return fig


def render_theme_migration_opportunity_map(summary_df, top_n=12):
    if summary_df is None or summary_df.empty:
        return None

    plot_df = summary_df.head(top_n).copy()
    color_map = {
        "Rising Hotspot": "#D45959",
        "Emerging Theme": "#2F74B8",
        "Stable Core": "#5E379D",
        "Cooling Theme": "#7A7A7A",
    }
    fig = px.scatter(
        plot_df,
        x="Recent Growth (%)",
        y="Projected Next Weight",
        size="Latest Weight",
        color="Signal Type",
        hover_name="Theme Cluster",
        text="Theme Cluster",
        color_discrete_map=color_map,
        title="Theme Cluster Future Hotspot Map",
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(line=dict(color="white", width=1)),
        customdata=plot_df[["Representative Keywords", "Latest Slice"]],
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            + "Recent Growth: %{x:.2f}%<br>"
            + "Projected Next Weight: %{y:.2f}<br>"
            + "Latest Slice: %{customdata[1]}<br>"
            + "Keywords: %{customdata[0]}<extra></extra>"
        ),
    )
    fig.update_layout(
        height=620,
        xaxis_title="Recent Growth (%)",
        yaxis_title="Projected Next Theme Weight",
        legend_title="Signal Type",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", rangemode="tozero")
    return fig


def summarize_theme_migration_signals(summary_df, top_n=5):
    if summary_df is None or summary_df.empty:
        return "Current data do not provide enough stable theme-chain continuity to interpret future hotspot migration."

    top_df = summary_df.head(top_n).copy()
    rising = top_df[top_df["Signal Type"].isin(["Rising Hotspot", "Emerging Theme"])]["Theme Cluster"].tolist()
    stable = top_df[top_df["Signal Type"] == "Stable Core"]["Theme Cluster"].tolist()
    cooling = top_df[top_df["Signal Type"] == "Cooling Theme"]["Theme Cluster"].tolist()

    message_parts = []
    if rising:
        message_parts.append("Likely shifting toward core hotspots: " + "; ".join(rising[:2]))
    if stable:
        message_parts.append("Stable core themes: " + "; ".join(stable[:2]))
    if cooling:
        message_parts.append("Potentially cooling themes: " + "; ".join(cooling[:2]))
    if not message_parts:
        strongest = top_df.iloc[0]["Theme Cluster"]
        message_parts.append(f"Most stable forward signal currently comes from {strongest}.")
    return " | ".join(message_parts)


def render_citespace_timeline(df, keywords_list, keyword_freq, top_n=20):
    df_valid = clean_year_column(df)
    top_keywords = [kw for kw, _ in _most_common_items(keyword_freq, top_n)]
    timeline_data = {kw: {} for kw in top_keywords}
    for row_idx, row in df_valid.iterrows():
        year = int(row["Year"])
        kws = keywords_list[row_idx] if row_idx < len(keywords_list) else []
        for kw in set(kws):
            if kw in timeline_data:
                timeline_data[kw][year] = timeline_data[kw].get(year, 0) + 1

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for idx, (kw, year_data) in enumerate(timeline_data.items()):
        if not year_data:
            continue
        years = sorted(year_data.keys())
        counts = [year_data[year] for year in years]
        fig.add_trace(
            go.Scatter(
                x=years,
                y=counts,
                name=kw,
                mode="lines+markers",
                line=dict(width=1.25, color=colors[idx % len(colors)]),
                marker=dict(size=4.5),
                opacity=0.9,
            )
        )
    fig.update_layout(
        title=dict(text="Temporal Keyword Evolution", font=dict(size=16)),
        xaxis_title="Year",
        yaxis_title="Frequency",
        hovermode="x unified",
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02, font=dict(size=11)),
        margin=dict(l=40, r=180, t=80, b=80),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="lightgray", showgrid=True)
    fig.update_yaxes(gridcolor="lightgray", showgrid=True)
    return fig


def parse_selected_keywords(raw_text):
    if raw_text is None:
        return []

    selected_keywords = []
    seen = set()
    for entry in re.split(r"[,;\n\r]+", str(raw_text)):
        canonical = normalize_keyword(entry)
        if canonical and canonical not in seen:
            selected_keywords.append(canonical)
            seen.add(canonical)
    return selected_keywords


def _build_keyword_year_count_lookup(df, keywords_list, selected_keywords):
    df_valid = clean_year_column(df)
    year_range = sorted(df_valid["Year"].unique())
    if not year_range or not selected_keywords:
        return df_valid, year_range, {}, {}

    keyword_set = set(selected_keywords)
    year_keyword_counts = {year: Counter() for year in year_range}
    year_publication_totals = df_valid["Year"].value_counts().sort_index().to_dict()

    for row_idx, row in df_valid.iterrows():
        year = int(row["Year"])
        keywords = keywords_list[row_idx] if row_idx < len(keywords_list) else []
        matched_keywords = set(keywords) & keyword_set
        if not matched_keywords:
            continue
        for keyword in matched_keywords:
            year_keyword_counts[year][keyword] += 1

    return df_valid, year_range, year_keyword_counts, year_publication_totals


def build_selected_keyword_share_table(df, keywords_list, selected_keywords):
    _, year_range, year_keyword_counts, _ = _build_keyword_year_count_lookup(df, keywords_list, selected_keywords)
    rows = []
    for year in year_range:
        year_counter = year_keyword_counts.get(year, Counter())
        selected_total = sum(year_counter.values())
        if selected_total <= 0:
            continue
        for keyword in selected_keywords:
            count = int(year_counter.get(keyword, 0))
            rows.append(
                {
                    "Year": int(year),
                    "Keyword": keyword,
                    "Count": count,
                    "Selected Total": int(selected_total),
                    "Share (%)": round(count / selected_total * 100.0, 2),
                }
            )
    return pd.DataFrame(rows)


def render_selected_keyword_share_figure(share_df, selected_keywords):
    if share_df is None or share_df.empty:
        return None

    plot_df = share_df.copy()
    plot_df["Keyword"] = pd.Categorical(plot_df["Keyword"], categories=selected_keywords, ordered=True)
    fig = px.bar(
        plot_df,
        x="Year",
        y="Share (%)",
        color="Keyword",
        category_orders={"Keyword": selected_keywords},
        title="Selected Keyword Share by Year",
        text="Share (%)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(
        customdata=plot_df[["Count", "Selected Total"]],
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            + "Year: %{x}<br>"
            + "Share: %{y:.2f}%<br>"
            + "Count: %{customdata[0]} / %{customdata[1]} selected-keyword occurrences<extra></extra>"
        ),
        texttemplate="%{y:.1f}%",
        textposition="inside",
    )
    fig.update_layout(
        barmode="stack",
        height=560,
        yaxis_title="Share Within Selected Keywords (%)",
        xaxis_title="Year",
        legend_title="Keyword",
        hovermode="x unified",
    )
    fig.update_xaxes(type="category", showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(range=[0, 100], showgrid=True, gridcolor="lightgray")
    return fig


def build_keyword_growth_table(
    df,
    keywords_list,
    keyword_freq,
    *,
    candidate_top_n=60,
    min_total_occurrences=3,
    min_latest_count=2,
):
    candidate_keywords = [
        keyword
        for keyword, total_freq in _most_common_items(keyword_freq, candidate_top_n)
        if total_freq >= min_total_occurrences
    ]
    if not candidate_keywords:
        return pd.DataFrame(), pd.DataFrame()

    _, year_range, year_keyword_counts, year_publication_totals = _build_keyword_year_count_lookup(
        df,
        keywords_list,
        candidate_keywords,
    )
    if len(year_range) < 2:
        return pd.DataFrame(), pd.DataFrame()

    yearly_rows = []
    for year in year_range:
        publications = int(year_publication_totals.get(year, 0))
        for keyword in candidate_keywords:
            count = int(year_keyword_counts.get(year, Counter()).get(keyword, 0))
            share_pct = (count / publications * 100.0) if publications > 0 else 0.0
            yearly_rows.append(
                {
                    "Year": int(year),
                    "Keyword": keyword,
                    "Count": count,
                    "Publications": publications,
                    "Paper Share (%)": round(share_pct, 4),
                }
            )
    yearly_df = pd.DataFrame(yearly_rows)
    if yearly_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    latest_year = int(max(year_range))
    previous_year = int(sorted(year_range)[-2])
    summary_rows = []
    for keyword, group in yearly_df.groupby("Keyword", sort=False):
        group = group.sort_values("Year")
        latest_row = group[group["Year"] == latest_year]
        previous_row = group[group["Year"] == previous_year]
        if latest_row.empty or previous_row.empty:
            continue

        latest_count = int(latest_row["Count"].iloc[0])
        previous_count = int(previous_row["Count"].iloc[0])
        if latest_count < min_latest_count:
            continue

        latest_share = float(latest_row["Paper Share (%)"].iloc[0])
        previous_share = float(previous_row["Paper Share (%)"].iloc[0])
        previous_publications = max(int(previous_row["Publications"].iloc[0]), 1)
        smoothing_share = 100.0 / previous_publications
        baseline_share = max(previous_share, smoothing_share)
        growth_rate = ((latest_share - previous_share) / baseline_share) * 100.0
        trend_slope = 0.0
        if group["Paper Share (%)"].nunique() > 1:
            trend_slope = float(np.polyfit(group["Year"].astype(float), group["Paper Share (%)"].astype(float), 1)[0])

        summary_rows.append(
            {
                "Keyword": keyword,
                "Latest Year": latest_year,
                "Previous Year": previous_year,
                "Latest Count": latest_count,
                "Previous Count": previous_count,
                "Latest Share (%)": round(latest_share, 3),
                "Previous Share (%)": round(previous_share, 3),
                "Growth Rate (%)": round(growth_rate, 2),
                "Net Change (pp)": round(latest_share - previous_share, 3),
                "Trend Slope (pp/year)": round(trend_slope, 4),
                "Total Frequency": int(group["Count"].sum()),
                "Active Years": int((group["Count"] > 0).sum()),
            }
        )

    if not summary_rows:
        return pd.DataFrame(), yearly_df

    growth_df = pd.DataFrame(summary_rows).sort_values(
        ["Growth Rate (%)", "Net Change (pp)", "Latest Count", "Total Frequency", "Keyword"],
        ascending=[False, False, False, False, True],
    )
    return growth_df.reset_index(drop=True), yearly_df


def render_keyword_growth_leader_figure(growth_df, top_n=12):
    if growth_df is None or growth_df.empty:
        return None

    plot_df = growth_df.head(top_n).copy()
    plot_df = plot_df.sort_values("Growth Rate (%)", ascending=True)
    fig = go.Figure()
    for _, row in plot_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[0, row["Growth Rate (%)"]],
                y=[row["Keyword"], row["Keyword"]],
                mode="lines",
                line=dict(color="rgba(47,116,184,0.35)", width=3),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=plot_df["Growth Rate (%)"],
            y=plot_df["Keyword"],
            mode="markers",
            marker=dict(color="#2F74B8", size=12, line=dict(color="white", width=1.1)),
            customdata=plot_df[["Net Change (pp)", "Latest Count", "Latest Year"]],
            hovertemplate=(
                "<b>%{y}</b><br>"
                + "Growth Rate: %{x:.2f}%<br>"
                + "Net Change: %{customdata[0]:.3f} pp<br>"
                + "Latest Count (%{customdata[2]}): %{customdata[1]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Fastest-Growing Keywords",
        xaxis_title="Latest-Year Growth Rate (%)",
        yaxis_title="Keyword",
        height=max(420, 36 * len(plot_df) + 140),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def render_keyword_growth_trend_figure(yearly_df, growth_df, top_n=6):
    if yearly_df is None or yearly_df.empty or growth_df is None or growth_df.empty:
        return None

    top_keywords = growth_df.head(top_n)["Keyword"].tolist()
    plot_df = yearly_df[yearly_df["Keyword"].isin(top_keywords)].copy()
    if plot_df.empty:
        return None

    fig = px.line(
        plot_df,
        x="Year",
        y="Paper Share (%)",
        color="Keyword",
        markers=True,
        category_orders={"Keyword": top_keywords},
        title="Growth-Leader Keyword Trajectories",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(
        customdata=plot_df[["Count", "Publications"]],
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            + "Year: %{x}<br>"
            + "Paper Share: %{y:.3f}%<br>"
            + "Count: %{customdata[0]} / %{customdata[1]} papers<extra></extra>"
        ),
    )
    fig.update_layout(
        height=560,
        xaxis_title="Year",
        yaxis_title="Share of Papers (%)",
        hovermode="x unified",
        legend_title="Keyword",
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", rangemode="tozero")
    return fig


def build_publication_forecast_frame(df, *, forecast_horizon=4, lookback_years=8):
    df_valid = clean_year_column(df)
    year_counts = df_valid["Year"].value_counts().sort_index()
    if len(year_counts) < 4:
        return pd.DataFrame(), {}

    actual_years = year_counts.index.astype(int).to_numpy()
    actual_counts = year_counts.astype(float).to_numpy()
    if len(actual_years) > lookback_years:
        fit_years = actual_years[-lookback_years:]
        fit_counts = actual_counts[-lookback_years:]
    else:
        fit_years = actual_years
        fit_counts = actual_counts

    slope, intercept = np.polyfit(fit_years.astype(float), fit_counts, 1)
    future_years = np.arange(actual_years[-1] + 1, actual_years[-1] + forecast_horizon + 1)
    future_counts = np.maximum(0.0, slope * future_years + intercept)

    actual_df = pd.DataFrame(
        {"Year": actual_years, "Publications": actual_counts.round(2), "Series": "Actual"}
    )
    forecast_df = pd.DataFrame(
        {"Year": future_years, "Publications": future_counts.round(2), "Series": "Forecast"}
    )
    combined_df = pd.concat([actual_df, forecast_df], ignore_index=True)
    metadata = {
        "latest_year": int(actual_years[-1]),
        "latest_publications": int(actual_counts[-1]),
        "forecast_horizon": int(forecast_horizon),
        "slope": float(slope),
        "fit_start_year": int(fit_years[0]),
        "fit_end_year": int(fit_years[-1]),
    }
    return combined_df, metadata


def render_publication_forecast_figure(forecast_df):
    if forecast_df is None or forecast_df.empty:
        return None

    plot_df = forecast_df.copy()
    fig = go.Figure()
    for series_name, color, dash in [
        ("Actual", "#5E379D", "solid"),
        ("Forecast", "#D45959", "dash"),
    ]:
        part = plot_df[plot_df["Series"] == series_name]
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["Year"],
                y=part["Publications"],
                mode="lines+markers",
                name=series_name,
                line=dict(color=color, width=1.8, dash=dash),
                marker=dict(size=5, color=color),
            )
        )
    fig.update_layout(
        title="Publication Forecast",
        xaxis_title="Year",
        yaxis_title="Publications",
        height=560,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title="Series",
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", rangemode="tozero")
    return fig


def summarize_publication_forecast(forecast_df, metadata):
    if forecast_df is None or forecast_df.empty or not metadata:
        return "Current data do not provide enough year continuity to interpret future publication direction."

    forecast_only = forecast_df[forecast_df["Series"] == "Forecast"].copy()
    if forecast_only.empty:
        return "Current data do not provide enough forecast points to interpret publication direction."

    latest_publications = float(metadata.get("latest_publications", 0))
    next_publications = float(forecast_only.iloc[0]["Publications"])
    horizon_publications = float(forecast_only.iloc[-1]["Publications"])
    baseline = max(latest_publications, 1.0)
    next_growth = ((next_publications - latest_publications) / baseline) * 100.0
    direction = "continued expansion" if next_growth >= 0 else "near-term slowdown"
    return (
        f"Publication trend suggests {direction}: latest output is {latest_publications:.1f} papers in {metadata.get('latest_year')}, "
        f"the next-year projection is {next_publications:.1f}, and the {int(metadata.get('forecast_horizon', len(forecast_only)))}-year horizon reaches about {horizon_publications:.1f} papers."
    )


def build_keyword_opportunity_map_frame(
    df,
    keywords_list,
    keyword_freq,
    *,
    top_n_keywords=40,
    recent_year_window=4,
    min_total_occurrences=3,
):
    growth_df, yearly_df = build_keyword_growth_table(
        df,
        keywords_list,
        keyword_freq,
        candidate_top_n=top_n_keywords,
        min_total_occurrences=min_total_occurrences,
        min_latest_count=1,
    )
    if yearly_df.empty:
        return pd.DataFrame()

    year_range = sorted(yearly_df["Year"].unique())
    if len(year_range) < 3:
        return pd.DataFrame()

    recent_years = year_range[-min(recent_year_window, len(year_range)) :]
    recent_focus_years = recent_years[-min(2, len(recent_years)) :]
    baseline_years = recent_years[:-len(recent_focus_years)]
    if not baseline_years:
        baseline_years = recent_years[:1]

    rows = []
    for keyword, group in yearly_df.groupby("Keyword", sort=False):
        recent_part = group[group["Year"].isin(recent_focus_years)]
        baseline_part = group[group["Year"].isin(baseline_years)]
        if recent_part.empty or baseline_part.empty:
            continue

        recent_share = float(recent_part["Paper Share (%)"].mean())
        baseline_share = float(baseline_part["Paper Share (%)"].mean())
        baseline_publications = max(int(baseline_part["Publications"].max()), 1)
        smoothing_share = 100.0 / baseline_publications
        growth_rate = ((recent_share - baseline_share) / max(baseline_share, smoothing_share)) * 100.0
        recent_count = int(recent_part["Count"].sum())
        total_frequency = int(group["Count"].sum())
        if recent_count <= 0 or total_frequency < min_total_occurrences:
            continue
        rows.append(
            {
                "Keyword": keyword,
                "Recent Share (%)": round(recent_share, 3),
                "Baseline Share (%)": round(baseline_share, 3),
                "Growth Rate (%)": round(growth_rate, 2),
                "Recent Count": recent_count,
                "Total Frequency": total_frequency,
            }
        )

    if not rows:
        return pd.DataFrame()

    opportunity_df = pd.DataFrame(rows)
    share_threshold = float(opportunity_df["Recent Share (%)"].median())
    categories = []
    for _, row in opportunity_df.iterrows():
        if row["Recent Share (%)"] >= share_threshold and row["Growth Rate (%)"] >= 0:
            categories.append("Core Hotspots")
        elif row["Recent Share (%)"] < share_threshold and row["Growth Rate (%)"] >= 0:
            categories.append("Emerging Signals")
        elif row["Recent Share (%)"] >= share_threshold and row["Growth Rate (%)"] < 0:
            categories.append("Mature but Cooling")
        else:
            categories.append("Niche / Weakening")
    opportunity_df["Signal Type"] = categories
    return opportunity_df.sort_values(
        ["Growth Rate (%)", "Recent Share (%)", "Recent Count", "Keyword"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def render_keyword_opportunity_map(opportunity_df, top_n=20):
    if opportunity_df is None or opportunity_df.empty:
        return None

    plot_df = opportunity_df.head(top_n).copy()
    color_map = {
        "Core Hotspots": "#D45959",
        "Emerging Signals": "#2F74B8",
        "Mature but Cooling": "#F2B382",
        "Niche / Weakening": "#7A7A7A",
    }
    top_labels = set(plot_df.sort_values(["Recent Count", "Growth Rate (%)"], ascending=[False, False]).head(10)["Keyword"])
    fig = px.scatter(
        plot_df,
        x="Recent Share (%)",
        y="Growth Rate (%)",
        size="Recent Count",
        color="Signal Type",
        text=plot_df["Keyword"].where(plot_df["Keyword"].isin(top_labels), ""),
        hover_name="Keyword",
        color_discrete_map=color_map,
        title="Keyword Opportunity Map",
    )
    fig.update_traces(
        marker=dict(line=dict(color="white", width=1)),
        textposition="top center",
        customdata=plot_df[["Recent Count", "Total Frequency"]],
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            + "Recent Share: %{x:.3f}%<br>"
            + "Growth Rate: %{y:.2f}%<br>"
            + "Recent Count: %{customdata[0]}<br>"
            + "Total Frequency: %{customdata[1]}<extra></extra>"
        ),
    )
    fig.update_layout(
        height=620,
        xaxis_title="Recent Share of Papers (%)",
        yaxis_title="Recent Growth Rate (%)",
        legend_title="Signal Type",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="lightgray", rangemode="tozero")
    fig.update_yaxes(showgrid=True, gridcolor="lightgray")
    return fig


def summarize_keyword_opportunity_map(opportunity_df, top_n=10):
    if opportunity_df is None or opportunity_df.empty:
        return "Current data do not provide enough temporal keyword continuity to interpret forward opportunity signals."

    top_df = opportunity_df.head(top_n).copy()
    core_hotspots = top_df[top_df["Signal Type"] == "Core Hotspots"]["Keyword"].tolist()[:2]
    emerging = top_df[top_df["Signal Type"] == "Emerging Signals"]["Keyword"].tolist()[:2]
    cooling = top_df[top_df["Signal Type"] == "Mature but Cooling"]["Keyword"].tolist()[:2]

    message_parts = []
    if core_hotspots:
        message_parts.append("Likely future core hotspots: " + "; ".join(core_hotspots))
    if emerging:
        message_parts.append("Most notable emerging signals: " + "; ".join(emerging))
    if cooling:
        message_parts.append("Potentially cooling mature topics: " + "; ".join(cooling))
    if not message_parts:
        strongest = top_df.iloc[0]["Keyword"]
        message_parts.append(f"Strongest forward keyword signal currently comes from {strongest}.")
    return " | ".join(message_parts)


def summarize_forward_signals_overview(
    publication_summary="",
    keyword_summary="",
    entity_summary="",
    leadership_summary="",
    theme_summary="",
):
    summary_items = []
    if publication_summary:
        summary_items.append(f"Publication Trend: {publication_summary}")
    if keyword_summary:
        summary_items.append(f"Keyword Hotspots: {keyword_summary}")
    if entity_summary:
        summary_items.append(f"Entity Growth: {entity_summary}")
    if leadership_summary:
        summary_items.append(f"Leadership Shift: {leadership_summary}")
    if theme_summary:
        summary_items.append(f"Theme Migration: {theme_summary}")

    if not summary_items:
        return "Current data do not yet provide enough stable forward-looking signals for an integrated assessment."

    return "\n".join(f"- {item}" for item in summary_items)


def _kleinberg_burst(sequence, s=2, gamma=1.0):
    if not sequence or max(sequence) == 0:
        return []
    n = len(sequence)
    total = sum(sequence)
    if total == 0:
        return []
    expected = total / n
    if expected == 0:
        return []

    max_state = 1
    for idx in range(2, 100):
        if (expected * (s ** idx)) > max(sequence) * 1.5:
            break
        max_state = idx
    num_states = max_state + 1
    tau = [expected * (s ** idx) for idx in range(num_states)]
    alpha = gamma * np.log(num_states) if num_states > 1 else 0

    cost = np.full((num_states, n), np.inf)
    parent = np.full((num_states, n), -1, dtype=int)
    for idx in range(num_states):
        transition = alpha * idx
        if tau[idx] > 0:
            cost[idx, 0] = (
                sequence[0] * np.log(sequence[0] / tau[idx]) + tau[idx] - sequence[0]
                if sequence[0] > 0
                else tau[idx]
            )
            cost[idx, 0] += transition

    for t in range(1, n):
        for state_idx in range(num_states):
            best_cost = np.inf
            best_parent = 0
            for prev_idx in range(num_states):
                transition = alpha * abs(state_idx - prev_idx)
                candidate_cost = cost[prev_idx, t - 1] + transition
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_parent = prev_idx
            emit = (
                sequence[t] * np.log(sequence[t] / tau[state_idx]) + tau[state_idx] - sequence[t]
                if tau[state_idx] > 0 and sequence[t] > 0
                else (tau[state_idx] if tau[state_idx] > 0 else np.inf)
            )
            cost[state_idx, t] = best_cost + emit
            parent[state_idx, t] = best_parent

    final_state = np.argmin(cost[:, -1])
    path = [0] * n
    state = final_state
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = parent[state, t]

    bursts = []
    in_burst = False
    burst_start = 0
    burst_max_state = 0
    for t, state in enumerate(path):
        if state > 0:
            if not in_burst:
                in_burst = True
                burst_start = t
                burst_max_state = state
            else:
                burst_max_state = max(burst_max_state, state)
        elif in_burst:
            burst_end = t - 1
            duration = max(1, burst_end - burst_start + 1)
            bursts.append(
                {
                    "start": burst_start,
                    "end": burst_end,
                    "strength": burst_max_state,
                    "weight": sum(sequence[burst_start:t]) / duration / expected,
                    "duration": duration,
                    "score": _compute_burst_llr(sequence, expected, burst_start, burst_end),
                }
            )
            in_burst = False
            burst_max_state = 0
    if in_burst:
        burst_end = n - 1
        duration = max(1, burst_end - burst_start + 1)
        bursts.append(
            {
                "start": burst_start,
                "end": burst_end,
                "strength": burst_max_state,
                "weight": sum(sequence[burst_start:n]) / duration / expected,
                "duration": duration,
                "score": _compute_burst_llr(sequence, expected, burst_start, burst_end),
            }
        )
    return bursts


def build_burst_table(label_sequences, year_range, label_column="Keyword", top_n=20, include_fallback=False):
    columns = [
        label_column,
        "Burst Strength",
        "Adjusted Burst Score",
        "Burst Weight",
        "Start",
        "End",
        "Duration",
        "Max Freq",
    ]
    if len(year_range) < 3:
        return pd.DataFrame(columns=columns)

    burst_info = []
    for label, year_counts in label_sequences.items():
        freqs = [year_counts.get(year, 0) for year in year_range]
        if max(freqs) == 0:
            continue
        bursts = _kleinberg_burst(freqs, s=2, gamma=1.0)
        for burst in bursts:
            raw_score = round(burst["score"], 4)
            duration = int(year_range[burst["end"]] - year_range[burst["start"]] + 1)
            burst_info.append(
                {
                    label_column: label,
                    "Burst Strength": raw_score,
                    "Adjusted Burst Score": round(_compute_adjusted_burst_score(raw_score, duration), 4),
                    "Burst Weight": round(burst["weight"], 2),
                    "Start": year_range[burst["start"]],
                    "End": year_range[burst["end"]],
                    "Duration": duration,
                    "Max Freq": max(freqs),
                }
            )
        if include_fallback and not bursts:
            fallback_info = _fallback_burst_info(label, freqs, year_range, label_column=label_column)
            if fallback_info:
                fallback_duration = int(fallback_info["End"] - fallback_info["Start"] + 1)
                fallback_info["Adjusted Burst Score"] = round(
                    _compute_adjusted_burst_score(fallback_info["Burst Strength"], fallback_duration),
                    4,
                )
                fallback_info["Duration"] = fallback_duration
                burst_info.append(fallback_info)

    burst_info.sort(
        key=lambda item: (
            item["Adjusted Burst Score"],
            item["Burst Strength"],
            item["Burst Weight"],
            -item["Start"],
            item["Max Freq"],
        ),
        reverse=True,
    )
    if not burst_info:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(burst_info[:top_n], columns=columns)


def _filter_year_window(df_valid, start_year=None, end_year=None):
    if df_valid.empty:
        return df_valid

    year_min = int(df_valid["Year"].min())
    year_max = int(df_valid["Year"].max())
    if start_year is None:
        start_year = year_min
    if end_year is None:
        end_year = year_max

    start_year = max(int(start_year), year_min)
    end_year = min(int(end_year), year_max)
    if start_year > end_year:
        return df_valid.iloc[0:0].copy()

    return df_valid[(df_valid["Year"] >= start_year) & (df_valid["Year"] <= end_year)].copy()


def _canonicalize_keyword_record(keywords):
    canonical_keywords = []
    seen = set()
    for keyword in keywords or []:
        canonical = normalize_keyword(keyword)
        if canonical and canonical not in seen:
            canonical_keywords.append(canonical)
            seen.add(canonical)
    return canonical_keywords


def _is_corpus_anchor_keyword(keyword, total_occurrences, active_years, frequency_rank, total_years):
    normalized = normalize_keyword(keyword)
    if not normalized:
        return False
    if len(normalized.split()) != 1:
        return False
    if int(frequency_rank or 0) > 3:
        return False
    if int(total_occurrences or 0) < 100:
        return False
    return (int(active_years or 0) / max(1, int(total_years or 1))) >= 0.75


def _should_suppress_burst_keyword(keyword, total_occurrences, active_years, frequency_rank, total_years):
    normalized = normalize_keyword(keyword)
    if not normalized:
        return True
    if normalized in BURST_GENERIC_TERMS:
        return True
    return _is_corpus_anchor_keyword(
        normalized,
        total_occurrences=total_occurrences,
        active_years=active_years,
        frequency_rank=frequency_rank,
        total_years=total_years,
    )


def render_burst_figure(burst_df, label_column="Keyword", title="Burst Detection (Kleinberg Algorithm)", year_tick_step=2):
    if burst_df is None or burst_df.empty:
        return None

    max_strength = max(burst_df["Burst Strength"])
    red_gradient = [
        "#FE5757",
        "#EA1E20",
        "#BE0208",
        "#9C000D",
        "#790105",
    ]
    y_labels = list(burst_df[label_column])
    fig = go.Figure()
    for y_pos, (_, info) in enumerate(burst_df.iterrows()):
        bar_width = info["End"] - info["Start"] + 0.8
        red_intensity = min(1.0, info["Burst Strength"] / max(max_strength, 1))
        color_idx = min(len(red_gradient) - 1, int(round(red_intensity * (len(red_gradient) - 1))))
        color = red_gradient[color_idx]
        fig.add_trace(
            go.Bar(
                name=f"{info[label_column]} (raw={info['Burst Strength']}, adj={info['Adjusted Burst Score']})",
                x=[bar_width],
                y=[y_pos],
                base=[info["Start"] - 0.4],
                orientation="h",
                marker=dict(color=color, line=dict(color="#C23D3D", width=0.8)),
                text=(
                    f"{info['Start']}-{info['End']} | Raw={info['Burst Strength']:.2f}"
                    f" | Adj={info['Adjusted Burst Score']:.2f}"
                ),
                textposition="inside",
                textfont=dict(size=10, color="white"),
                hovertemplate=(
                    f"<b>{info[label_column]}</b><br>Period: {info['Start']}-{info['End']}"
                    f"<br>Raw Burst Strength: {info['Burst Strength']}"
                    f"<br>Adjusted Burst Score: {info['Adjusted Burst Score']}"
                    f"<br>Burst Weight: {info['Burst Weight']}"
                    f"<br>Duration: {info['Duration']}"
                    f"<br>Max Freq: {info['Max Freq']}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    min_year = int(burst_df["Start"].min())
    max_year = int(burst_df["End"].max())
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        barmode="overlay",
        xaxis_title="Year",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(y_labels))),
            ticktext=y_labels,
            dtick=1,
            fixedrange=True,
            autorange="reversed",
        ),
        xaxis=dict(
            gridcolor="lightgray",
            range=[min_year - 1, max_year + 1],
            fixedrange=True,
            tickmode="linear",
            tick0=min_year,
            dtick=max(1, int(year_tick_step or 1)),
        ),
        height=max(400, len(y_labels) * 30 + 150),
        margin=dict(l=180, r=30, t=60, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        dragmode=False,
        hovermode="closest",
    )
    fig.add_annotation(
        text="Bar order uses Adjusted Burst Score (duration-penalized) | burst ranking suppresses generic umbrella terms and corpus-anchor single-token terms | color intensity uses raw Burst Strength | Kleinberg's two-state automaton model (s=2, γ=1.0)",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.08,
        showarrow=False,
        font=dict(size=10, color="gray"),
    )
    return fig


def build_keyword_burst_table(
    df,
    keywords_list,
    keyword_freq,
    top_n=20,
    start_year=None,
    end_year=None,
    min_total_occurrences=3,
    min_active_years=2,
):
    df_valid = _filter_year_window(clean_year_column(df), start_year=start_year, end_year=end_year)
    if df_valid.empty:
        return pd.DataFrame(columns=["Keyword", "Burst Strength", "Burst Weight", "Start", "End", "Max Freq"])

    year_range = sorted(df_valid["Year"].unique())
    if len(year_range) < 3:
        return pd.DataFrame(columns=["Keyword", "Burst Strength", "Burst Weight", "Start", "End", "Max Freq"])

    filtered_keyword_freq = Counter()
    filtered_keyword_years = {}
    for row_idx in df_valid.index:
        kws = _canonicalize_keyword_record(keywords_list[row_idx] if row_idx < len(keywords_list) else [])
        year = int(df_valid.at[row_idx, "Year"])
        for kw in kws:
            filtered_keyword_freq[kw] += 1
            filtered_keyword_years.setdefault(kw, Counter())[year] += 1

    candidate_keywords = [
        kw
        for kw, total in filtered_keyword_freq.items()
        if total >= max(1, int(min_total_occurrences))
        and len(filtered_keyword_years.get(kw, {})) >= max(1, int(min_active_years))
    ]
    if not candidate_keywords:
        candidate_keywords = [kw for kw, _ in _most_common_items(filtered_keyword_freq or keyword_freq, max(top_n * 3, top_n))]

    frequency_rank_lookup = {
        kw: rank
        for rank, (kw, _) in enumerate(filtered_keyword_freq.most_common(), start=1)
    }
    total_years = len(year_range)
    filtered_candidates = [
        kw
        for kw in candidate_keywords
        if not _should_suppress_burst_keyword(
            kw,
            total_occurrences=filtered_keyword_freq.get(kw, 0),
            active_years=len(filtered_keyword_years.get(kw, {})),
            frequency_rank=frequency_rank_lookup.get(kw, len(filtered_keyword_freq) + 1),
            total_years=total_years,
        )
    ]
    if filtered_candidates:
        candidate_keywords = filtered_candidates

    keyword_year_freq = {kw: {year: 0 for year in year_range} for kw in candidate_keywords}
    for row_idx, row in df_valid.iterrows():
        year = int(row["Year"])
        kws = _canonicalize_keyword_record(keywords_list[row_idx] if row_idx < len(keywords_list) else [])
        for kw in kws:
            if kw in keyword_year_freq:
                keyword_year_freq[kw][year] += 1
    return build_burst_table(
        keyword_year_freq,
        year_range,
        label_column="Keyword",
        top_n=top_n,
        include_fallback=False,
    )


def render_burst_detection(df, keywords_list, keyword_freq, top_n=15, start_year=None, end_year=None):
    burst_df = build_keyword_burst_table(
        df,
        keywords_list,
        keyword_freq,
        top_n=max(20, top_n),
        start_year=start_year,
        end_year=end_year,
    )
    if burst_df.empty:
        return None
    return render_burst_figure(
        burst_df.head(top_n),
        label_column="Keyword",
        title="Burst Detection (Kleinberg Algorithm)",
        year_tick_step=2,
    )
