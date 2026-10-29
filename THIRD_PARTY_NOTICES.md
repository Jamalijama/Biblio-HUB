# Third-Party Notices

This document records visible third-party dependencies and bundled frontend resources for release review and compliance follow-up.

## Python Dependencies

| Dependency | Purpose | License | Version Floor |
|---|---|---|---|
| Streamlit | web UI | Apache-2.0 | >=1.20.0 |
| pandas | data processing | BSD-3-Clause | >=1.5.0 |
| numpy | numerical computing | BSD-3-Clause | >=1.23.0 |
| networkx | network analysis | BSD-3-Clause | >=2.8 |
| matplotlib | static plotting | PSF-2.0 | >=3.5.0 |
| plotly | interactive and static chart export | MIT | >=5.0.0 |
| kaleido | Plotly static export | MIT | >=0.2.1 |
| pyvis | interactive network rendering | BSD-3-Clause | >=0.3.2 |
| python-louvain | community detection | BSD-3-Clause | >=0.16 |
| wordcloud | word-cloud rendering | MIT | >=1.8.0 |
| scikit-learn | statistical helpers | BSD-3-Clause | >=1.0.0 |
| sentence-transformers | topic-modeling extras | Apache-2.0 | >=2.0.0 |
| umap-learn | topic-modeling dimensionality reduction | BSD-3-Clause | >=0.5.0 |
| hdbscan | topic-modeling clustering | BSD-3-Clause | >=0.8.0 |
| BERTopic | topic modeling | MIT | >=0.16.0 |
| pytest | testing | MIT | >=8.0.0 |

## Bundled Frontend Assets

| Path | Asset | Purpose | Review Status |
|---|---|---|---|
| `lib/vis-9.1.2/vis-network.min.js` | vis-network | interactive network visualization | verify upstream license text before public redistribution |
| `lib/vis-9.1.2/vis-network.css` | vis-network | interactive network styling | verify upstream license text before public redistribution |
| `lib/tom-select/tom-select.complete.min.js` | Tom Select | enhanced selectors | verify upstream license text before public redistribution |
| `lib/tom-select/tom-select.css` | Tom Select | selector styling | verify upstream license text before public redistribution |
| `lib/bindings/utils.js` | helper binding script | frontend integration | confirm provenance before broad release |

## Release Checklist

- confirm exact upstream license texts for bundled frontend files
- keep project-level `LICENSE` in the repository root
- record any future external model weights or API terms separately if such integrations are added
