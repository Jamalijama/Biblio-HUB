import pytest

import modules.export_bundle as export_bundle
import modules.figure_export_bundle as figure_export_bundle


@pytest.fixture(autouse=True)
def stub_plotly_static_export(monkeypatch):
    def fake_plotly_figure_to_bytes(*args, **kwargs):
        return b"fake-plotly-bytes"

    monkeypatch.setattr(export_bundle, "plotly_figure_to_bytes", fake_plotly_figure_to_bytes)
    monkeypatch.setattr(
        figure_export_bundle,
        "plotly_figure_to_bytes",
        fake_plotly_figure_to_bytes,
        raising=False,
    )
