import datetime
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd

def _text_bytes(content):
    return str(content).encode("utf-8")

def _json_bytes(payload):
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_copyright_package(df, root_dir=None):
    root_path = Path(root_dir).resolve() if root_dir else PROJECT_ROOT
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        files_to_include = [
            "CODE_SUMMARY.md",
            "MODULE_DESIGN.md",
            "VERSION_HISTORY.md",
            "COPYRIGHT_AND_IP.md",
            "THIRD_PARTY_NOTICES.md",
            "README.md"
        ]
        for filename in files_to_include:
            file_path = root_path / filename
            if file_path.exists():
                with file_path.open("r", encoding="utf-8") as f:
                    zf.writestr(f"01_technical_docs/{filename}", _text_bytes(f.read()))
        core_modules = [
            "app.py",
            "modules/data_pipeline.py",
            "modules/experiment_framework.py",
            "modules/topic_modeling.py",
            "modules/export_orchestrator.py"
        ]
        for mod in core_modules:
            mod_path = root_path / mod
            if mod_path.exists():
                with mod_path.open("r", encoding="utf-8") as f:
                    content = f.read()
                    zf.writestr(f"02_source_code_samples/{mod.replace('/', '_')}", _text_bytes(content))
        app_summary = {
            "application_name": "Biblio-HUB",
            "export_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_stats": {
                "total_records": len(df),
                "columns": list(df.columns)
            },
            "environment": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}+",
                "core_dependencies": ["streamlit", "pandas", "plotly", "networkx", "bertopic"]
            }
        }
        zf.writestr("00_application_summary.json", _json_bytes(app_summary))
        manual_content = f"""# User Manual Draft

## 1. Software Overview
This software is an integrated bibliometric analysis platform designed to support the full workflow from data cleaning and network construction to semantic topic modeling and innovation-oriented metrics.

## 2. Runtime Environment
- Operating system: Windows/Linux/macOS
- Runtime language: Python 3.10+
- Core framework: Streamlit

## 3. Core Functional Modules
- **Data management**: supports WoS/Scopus imports with deduplication and normalization workflows.
- **Visual analytics**: supports publication trends, core-journal analysis, keyword clouds, and related descriptive outputs.
- **Network analysis**: supports keyword co-occurrence, bibliographic coupling, co-authorship, and other network construction workflows.
- **Semantic modeling**: integrates BERTopic for large-scale topic extraction and topic-evolution analysis.
- **Innovation assessment**: provides forward-looking indicators such as the Disruption Index (DI) and Structural Holes.
- **Export center**: generates figure bundles and methodological evidence packages in one click.

## 4. Current Data Snapshot
- Processed record count: {len(df)}
- Snapshot generated at: {app_summary['export_date']}
"""
        zf.writestr("03_manual_draft/user_manual_draft.md", _text_bytes(manual_content))
    return bundle.getvalue()
