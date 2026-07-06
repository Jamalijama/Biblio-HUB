import base64
import io
import os
import re
from typing import Any

import streamlit as st

from modules.export_bundle import (
    A4_LANDSCAPE_HEIGHT_PX,
    A4_LANDSCAPE_WIDTH_PX,
    PUBLICATION_EXPORT_FORMATS,
    plotly_figure_to_bytes,
    sanitize_filename,
    style_publication_figure,
)
from modules.network_visualization import render_network_publication_figure


def render_html_iframe(html_content, height=820):
    if not html_content:
        return
    html_b64 = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
    st.iframe(
        f"data:text/html;base64,{html_b64}",
        height=height,
    )


def download_html_button(html_content, filename="network.html", label="Download HTML"):
    b64 = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")
    href = (
        f'<a href="data:text/html;base64,{b64}" download="{filename}" '
        'style="display:inline-block;padding:6px 16px;background-color:#8796A5;color:white;'
        f'border-radius:4px;text-decoration:none;font-size:14px;margin:4px 0;">{label}</a>'
    )
    st.markdown(href, unsafe_allow_html=True)


def download_plotly_button(
    fig,
    filename="chart.png",
    label="Download PNG",
    width=A4_LANDSCAPE_WIDTH_PX,
    height=A4_LANDSCAPE_HEIGHT_PX,
):
    if fig is None:
        return
    export_height = int(height or A4_LANDSCAPE_HEIGHT_PX)
    filename_stem, _ = os.path.splitext(filename)
    readable_title = filename_stem.replace("_", " ").strip().title()
    st.markdown("---")
    st.markdown(f"#### Static Exports for {readable_title}")
    prepare_requested = st.button(
        "Prepare Static Exports",
        key=f"plotly_exports_prepare_{sanitize_filename(filename_stem)}",
        use_container_width=True,
    )
    if not prepare_requested:
        return
    export_fig = style_publication_figure(fig, height=export_height)
    cols = st.columns(len(PUBLICATION_EXPORT_FORMATS))
    exported = False
    for idx, export_format in enumerate(PUBLICATION_EXPORT_FORMATS):
        try:
            file_name = f"{filename_stem}.{export_format}"
            mime = (
                "image/png"
                if export_format == "png"
                else "image/svg+xml" if export_format == "svg" else "application/pdf"
            )
            data = plotly_figure_to_bytes(export_fig, export_format=export_format, width=width, height=export_height)
            cols[idx].download_button(
                label=f"Download {export_format.upper()}",
                data=data,
                file_name=file_name,
                mime=mime,
                key=f"plotly_export_{sanitize_filename(filename_stem)}_{export_format}",
            )
            exported = True
        except Exception:
            continue
    if not exported:
        st.caption("Install kaleido for static image export: pip install kaleido")


def render_plotly_chart(fig, **kwargs):
    if fig is None:
        return
    kwargs.setdefault("theme", None)
    st.plotly_chart(fig, **kwargs)


def download_matplotlib_button(fig, filename="chart.png", label="Download PNG"):
    if fig is None:
        return
    filename_stem, _ = os.path.splitext(filename)
    readable_title = filename_stem.replace("_", " ").strip().title()
    st.markdown("---")
    st.markdown(f"#### Static Exports for {readable_title}")
    prepare_requested = st.button(
        "Prepare Static Exports",
        key=f"matplotlib_exports_prepare_{sanitize_filename(filename_stem)}",
        use_container_width=True,
    )
    if not prepare_requested:
        return
    cols = st.columns(len(PUBLICATION_EXPORT_FORMATS))
    for idx, export_format in enumerate(PUBLICATION_EXPORT_FORMATS):
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format=export_format,
            dpi=300 if export_format == "png" else None,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="white",
        )
        buf.seek(0)
        mime = (
            "image/png"
            if export_format == "png"
            else "image/svg+xml" if export_format == "svg" else "application/pdf"
        )
        cols[idx].download_button(
            label=f"Download {export_format.upper()}",
            data=buf.getvalue(),
            file_name=f"{filename_stem}.{export_format}",
            mime=mime,
            key=f"matplotlib_export_{sanitize_filename(filename_stem)}_{export_format}",
        )


def render_and_download_network_figure(
    graph,
    filename_stem,
    title,
    node_groups=None,
    legend_label="Frequency",
    layout_mode="auto",
    size_range=(24, 84),
    label_max_len=40,
    max_visible_labels=24,
    label_font_size=17,
    width=A4_LANDSCAPE_WIDTH_PX,
    height=A4_LANDSCAPE_HEIGHT_PX,
    mode="publication",
    edge_color_override=None,
    edge_weight_attr="weight",
    edge_width_scale=1.0,
    edge_alpha_scale=1.0,
    precomputed_fig=None,
):
    if graph is None or graph.number_of_nodes() == 0:
        return None

    fig = precomputed_fig
    if fig is None:
        fig = render_network_publication_figure(
            graph,
            node_groups=node_groups,
            title=title,
            size_range=size_range,
            label_max_len=label_max_len,
            legend_label=legend_label,
            layout_mode=layout_mode,
            max_visible_labels=max_visible_labels,
            label_font_size=label_font_size,
            mode=mode,
            edge_color_override=edge_color_override,
            edge_weight_attr=edge_weight_attr,
            edge_width_scale=edge_width_scale,
            edge_alpha_scale=edge_alpha_scale,
        )
    if fig is None:
        return None

    prepare_requested = st.button(
        "Prepare Static Exports",
        key=f"network_exports_prepare_{sanitize_filename(filename_stem)}",
        use_container_width=True,
    )
    if not prepare_requested:
        return fig

    cols = st.columns(len(PUBLICATION_EXPORT_FORMATS))
    exported = False
    for idx, export_format in enumerate(PUBLICATION_EXPORT_FORMATS):
        try:
            file_name = f"{filename_stem}.{export_format}"
            mime = (
                "image/png"
                if export_format == "png"
                else "image/svg+xml" if export_format == "svg" else "application/pdf"
            )
            data = plotly_figure_to_bytes(fig, export_format=export_format, width=width, height=height)
            cols[idx].download_button(
                label=f"Download {export_format.upper()}",
                data=data,
                file_name=file_name,
                mime=mime,
                key=f"network_export_{sanitize_filename(filename_stem)}_{export_format}",
            )
            exported = True
        except Exception:
            continue
    if not exported:
        st.caption("Install kaleido for static network export: pip install kaleido")
    return fig


def show_cluster_report(stats, node_type_label="nodes"):
    if "cluster_report" not in stats:
        return
    report = stats["cluster_report"]
    if not report:
        return

    st.markdown("### Cluster Report")
    for group_key in sorted(report.keys()):
        group_info = report[group_key]
        color = group_info["color"]
        members = group_info["members"]
        label = group_info.get("label", "")
        if label and label.endswith(": "):
            cluster_name = f"Cluster {group_key + 1}"
        else:
            cluster_name = label if label else f"Cluster {group_key + 1}"
        preview = ", ".join(members[:10])
        if len(members) > 10:
            preview += f" ... (+{len(members) - 10} more)"
        with st.expander(f"{cluster_name} ({len(members)} {node_type_label})", expanded=len(members) <= 6):
            st.markdown(
                f'<span style="display:inline-block;width:14px;height:14px;background-color:{color};'
                f'border-radius:3px;margin-right:6px;vertical-align:middle;"></span>'
                f"**Preview:** {preview}",
                unsafe_allow_html=True,
            )
            st.caption("Hover labels in the network for full names; this panel is condensed to keep the layout readable.")


def _sync_integer_control_from_slider(base_key: str) -> None:
    slider_key = f"{base_key}__slider"
    input_key = f"{base_key}__input"
    if slider_key not in st.session_state:
        fallback_value = int(st.session_state.get(input_key, st.session_state.get(base_key, 0)))
        st.session_state[base_key] = fallback_value
        st.session_state[input_key] = fallback_value
        return
    value = int(st.session_state[slider_key])
    st.session_state[base_key] = value
    st.session_state[input_key] = value


def _sync_integer_control_from_input(base_key: str, min_value: int, input_max: int) -> None:
    input_key = f"{base_key}__input"
    slider_key = f"{base_key}__slider"
    slider_max_key = f"{base_key}__slider_max"
    value = int(st.session_state.get(input_key, st.session_state.get(base_key, min_value)))
    value = max(min_value, min(input_max, value))
    st.session_state[base_key] = value
    slider_max = int(st.session_state.get(slider_max_key, input_max))
    st.session_state[slider_key] = min(value, slider_max)
    st.session_state[input_key] = value


def integer_control(
    st_container: Any,
    label: str,
    min_value: int,
    default_value: int,
    key: str,
    *,
    input_max: int | None = None,
    slider_max: int | None = None,
    slider_soft_cap: int | None = None,
) -> int:
    effective_input_max = max(min_value, int(input_max if input_max is not None else default_value))
    if slider_max is None:
        if slider_soft_cap is None:
            effective_slider_max = effective_input_max
        else:
            effective_slider_max = min(effective_input_max, slider_soft_cap)
    else:
        effective_slider_max = max(min_value, min(effective_input_max, int(slider_max)))

    default_clamped = max(min_value, min(effective_input_max, int(default_value)))
    if key not in st.session_state:
        st.session_state[key] = default_clamped

    slider_key = f"{key}__slider"
    input_key = f"{key}__input"
    slider_max_key = f"{key}__slider_max"
    st.session_state[slider_max_key] = effective_slider_max

    current_value = max(min_value, min(effective_input_max, int(st.session_state[key])))
    st.session_state[key] = current_value
    if slider_key not in st.session_state:
        st.session_state[slider_key] = min(current_value, effective_slider_max)
    else:
        st.session_state[slider_key] = min(max(min_value, int(st.session_state[slider_key])), effective_slider_max)
    if input_key not in st.session_state:
        st.session_state[input_key] = current_value
    else:
        st.session_state[input_key] = max(min_value, min(effective_input_max, int(st.session_state[input_key])))

    control_col1, control_col2 = st_container.columns([4, 1.35])
    with control_col1:
        st_container.slider(
            label,
            min_value=min_value,
            max_value=effective_slider_max,
            key=slider_key,
            on_change=_sync_integer_control_from_slider,
            args=(key,),
        )
    with control_col2:
        st_container.number_input(
            f"{label} value",
            min_value=min_value,
            max_value=effective_input_max,
            step=1,
            key=input_key,
            label_visibility="collapsed",
            on_change=_sync_integer_control_from_input,
            args=(key, min_value, effective_input_max),
        )
    return int(st.session_state[key])


def _normalize_hex_color(value: str, fallback: str) -> str:
    text = str(value or "").strip()
    return text.upper() if re.fullmatch(r"#[0-9A-Fa-f]{6}", text) else fallback


def _render_color_control(key_prefix: str, label: str, default_color: str) -> str:
    preset_options = {
        "Default": default_color,
        "Blue": "#4C78A8",
        "Red": "#C95C5C",
        "Gray": "#8796A5",
        "Custom": None,
    }
    preset = st.selectbox(
        f"{label} Preset",
        list(preset_options.keys()),
        key=f"{key_prefix}_preset",
    )
    if preset == "Custom":
        custom_value = st.text_input(
            f"{label} HEX",
            value=st.session_state.get(f"{key_prefix}_hex", default_color),
            key=f"{key_prefix}_hex",
            help="Use a 6-digit HEX color such as #4C78A8.",
        )
        color = _normalize_hex_color(custom_value, default_color)
    else:
        color = preset_options[preset]
    st.caption(f"Current color: `{color}`")
    return color


def render_plot_style_controls(
    key_prefix: str,
    *,
    default_primary: str,
    default_secondary: str | None = None,
    default_height: int = 500,
    show_legend_default: bool = False,
    allow_color_controls: bool = True,
    preserve_original_colors: bool = False,
) -> dict[str, int | str | bool | None]:
    with st.expander("Figure Style", expanded=False):
        st.caption("Keep analytical thresholds in the main panel; open this section only when you need visual fine-tuning.")
        st.caption("Changes in this panel are applied only after you click Apply Figure Style.")
        with st.form(key=f"{key_prefix}_style_form", clear_on_submit=False):
            font_size = st.slider("Font Size", 10, 28, 14, key=f"{key_prefix}_font_size")
            title_size = st.slider("Title Size", 12, 32, 20, key=f"{key_prefix}_title_size")
            axis_size = st.slider("Axis Size", 10, 26, 13, key=f"{key_prefix}_axis_size")
            figure_height = st.slider("Figure Height", 320, 1000, default_height, step=20, key=f"{key_prefix}_figure_height")
            show_legend = st.checkbox("Show Legend", value=show_legend_default, key=f"{key_prefix}_show_legend")
            primary_color = default_primary
            secondary_color = default_secondary
            if allow_color_controls:
                primary_color = _render_color_control(f"{key_prefix}_primary", "Primary Color", default_primary)
                secondary_color = None
                if default_secondary:
                    secondary_color = _render_color_control(f"{key_prefix}_secondary", "Secondary Color", default_secondary)
            elif preserve_original_colors:
                st.caption("Color adjustment is disabled for this figure to preserve its original multi-element palette.")
            st.form_submit_button("Apply Figure Style", use_container_width=True)
    return {
        "font_size": font_size,
        "title_size": title_size,
        "axis_size": axis_size,
        "figure_height": figure_height,
        "show_legend": show_legend,
        "primary_color": primary_color,
        "secondary_color": secondary_color,
        "preserve_original_colors": preserve_original_colors,
    }


def apply_publication_style_with_overrides(fig, style: dict[str, int | str | bool | None]):
    if fig is None:
        return fig

    styled_fig = style_publication_figure(fig, height=int(style["figure_height"]))
    styled_fig.update_layout(
        showlegend=bool(style["show_legend"]),
        font=dict(size=int(style["font_size"]), color="#222222"),
        title=dict(font=dict(size=int(style["title_size"]), color="#222222")),
        legend=dict(
            bgcolor="white",
            bordercolor="#D9D9D9",
            borderwidth=1,
            font=dict(color="#222222"),
            title=dict(font=dict(color="#222222")),
        ),
    )
    styled_fig.update_xaxes(
        title_font=dict(size=int(style["axis_size"]), color="#222222"),
        tickfont=dict(size=int(style["axis_size"]), color="#222222"),
        showgrid=True,
    )
    styled_fig.update_yaxes(
        title_font=dict(size=int(style["axis_size"]), color="#222222"),
        tickfont=dict(size=int(style["axis_size"]), color="#222222"),
        showgrid=True,
    )
    if hasattr(styled_fig.layout, "coloraxis") and styled_fig.layout.coloraxis:
        styled_fig.update_layout(
            coloraxis_colorbar=dict(
                tickfont=dict(color="#222222"),
                title=dict(font=dict(color="#222222")),
            )
        )

    if not style.get("preserve_original_colors", False):
        secondary_color = style.get("secondary_color")
        for index, trace in enumerate(styled_fig.data):
            trace_color = style["primary_color"] if index % 2 == 0 or not secondary_color else secondary_color
            trace_type = getattr(trace, "type", "")
            if trace_type in {"bar", "histogram", "box", "violin"} and hasattr(trace, "marker"):
                trace.marker.color = trace_color
            elif trace_type in {"scatter", "scattergl"}:
                mode = getattr(trace, "mode", "") or ""
                if "lines" in mode and hasattr(trace, "line"):
                    trace.line.color = trace_color
                if hasattr(trace, "marker"):
                    trace.marker.color = trace_color
    return styled_fig
