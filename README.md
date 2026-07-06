# Biblio-HUB
Bibliometric analysis is widely used to map research landscapes, identify hotspots, and trace thematic change, yet the practical workflow behind many studies remains fragmented across multiple software environments.


`Biblio-HUB` 是一个基于 `Python + Streamlit` 的一体化文献计量分析平台，覆盖数据导入、字段清洗、关键词治理、合作网络、主题演化、结构指标、创新信号和导出归档等完整流程。

当前仓库已经同步到最新可迁移版本，适合作为公开代码仓库维护，也支持生成一个可直接上传 GitHub 的精简发布目录。

## 功能概览

- 数据导入：支持 `Web of Science` 纯文本导出和标准化 `CSV`
- 数据治理：字段标准化、年份清洗、重复记录处理、关键词屏蔽与同义归并
- 关系网络：关键词共现、关键词-期刊关联、作者合作、机构合作、国家合作、文献共被引、作者共被引、期刊共被引、文献耦合
- 结构分析：Descriptive、Journal Analysis、Keyword Analysis、Thematic Map、Hierarchical Cluster Heatmap、Top Entities、Authors Over Time、Three-Field Plot、Lotka's Law、Language Distribution
- 时间与前沿：关键词时间线、Burst Detection、Alluvial Topic Flow、Forward Signals、Disruption、Brokerage
- 导出中心：图件、表格、引文、网络文件和多类一键打包导出

## 快速开始

### 1. 安装运行依赖

```bash
pip install -r requirements-runtime.txt
```

如需完整功能：

```bash
pip install -r requirements.txt
```

### 2. 启动应用

```bash
streamlit run app.py --server.port 8501 --server.headless true
```

Windows 用户也可以直接双击 `start.bat`。

默认访问地址：

```text
http://localhost:8501
```

## 数据格式

### Web of Science 文本

- 支持 `.txt`
- 自动解析常见 WoS 字段，如 `AU`、`TI`、`SO`、`AB`、`DE`、`ID`、`CR`、`TC`、`C1`、`PY`、`DI`

### CSV 文件

- 支持常见列名映射
- 可读取标题、作者、期刊、摘要、关键词、被引次数、机构、国家等字段

### 编码识别

- 自动尝试 `UTF-8`、`Latin-1`、`GBK`、`CP1252`

## 主要模块

### Dataset Overview

- 核心指标总览
- 年度发文趋势
- Top journals
- 关键词词云与高频词表

### Relational Network Analysis

- Keyword Co-occurrence
- Keyword-Journal Bipartite
- Author Collaboration
- Institution Collaboration
- Country Collaboration
- Country Publication and Citation Impact Quadrant
- Co-citation / Bibliographic Coupling

其中 `Country Publications and Citation Impact Quadrant Analysis` 位于 `Relational Network Analysis` 板块中，显示名称为 `Country Publication and Citation Impact Quadrant`。

### Temporal Trend & Topic Evolution

- Keyword Timeline
- Burst Detection
- Alluvial Topic Flow
- BERTopic Workspace
- Forward Signals

### Bibliometric Structure & Performance

- Descriptive
- Journal Analysis
- Keyword Analysis
- Thematic Map
- Hierarchical Cluster Heatmap
- Top Entities
- Authors Over Time
- Three-Field Plot
- Lotka's Law
- Language Distribution

### Citation, Category & Source Analysis

- Citation Analysis
- Subject Categories
- Document Types
- Funding Analysis
- Reference Analysis
- RPYS
- Reference Burst Detection
- Publisher Analysis
- Disruption Index

### Export Center

- 批量导出图件
- 统一选择导出格式
- 一键研究包 / 投稿包 /结果包
- 导出状态日志与缺失提示

## 可迁移性说明

- 默认关键词流程优先使用数据本身的 `DE / ID` 元数据
- 自动标题摘要补词仅在元数据不足时启用
- 可选领域词典插件现已默认关闭，避免对其他研究领域产生隐性偏置
- 如需启用额外领域词典，可在侧栏 `Vocabulary Governance` 中手动开启实验选项

## 依赖分层

- `requirements-runtime.txt`：Web 端日常运行所需
- `requirements-topic-modeling.txt`：BERTopic 与语义主题建模增强依赖
- `requirements-dev.txt`：测试与开发依赖
- `requirements.txt`：完整安装集合

## 项目结构

```text
bibliometrics-5.26/
├── app.py
├── modules/
├── assets/
├── lib/
├── tests/
├── data/
├── .github/workflows/
├── .streamlit/
├── start.bat
├── run_tests.py
├── requirements-runtime.txt
├── requirements-topic-modeling.txt
├── requirements-dev.txt
├── requirements.txt
├── README.md
└── README_EN.md
```

## GitHub 发布

仓库根目录可生成一个精简发布目录：

```text
github_release/Biblio-HUB
```

该目录仅保留运行所需代码、资源、测试、许可证和文档，不包含原始数据、投稿中间产物、调试缓存和归档性大文件。

## 测试

```bash
python run_tests.py
```

最近一次核心回归已通过。

## 技术栈

| 类别 | 技术 |
|------|------|
| Web | Streamlit |
| 数据处理 | Pandas, NumPy |
| 网络分析 | NetworkX, python-louvain |
| 交互网络 | PyVis |
| 图表渲染 | Plotly, Matplotlib |
| 词云 | WordCloud |
| 主题建模 | BERTopic, sentence-transformers, UMAP, HDBSCAN |

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | >= 3.10 |
| 内存 | 建议 4 GB 以上 |
| 浏览器 | Chrome / Edge / Firefox |

## 说明

- `examples/legacy_export_demo/` 为归档示例，不属于主运行链路
- `raw data/` 为研究归档目录，不建议随公开代码仓库一并发布
- 版权与第三方依赖说明见 `COPYRIGHT_AND_IP.md` 和 `THIRD_PARTY_NOTICES.md`

## License

本项目使用 `MIT License`。


`Biblio-HUB` is an integrated `Python + Streamlit` bibliometrics platform for data ingestion, cleaning, network analysis, temporal exploration, structural interpretation, innovation signals, and publication-ready export workflows.

The repository is now aligned with the latest portable version and can also generate a stripped `github_release/Biblio-HUB` directory for direct public upload.

## Quick Start

### Runtime install

```bash
pip install -r requirements-runtime.txt
streamlit run app.py --server.port 8501 --server.headless true
```

For the full stack, including topic modeling and developer tooling:

```bash
pip install -r requirements.txt
```

Windows users can also double-click `start.bat`, then open [http://localhost:8501](http://localhost:8501).

## Supported Input

- `Web of Science` plain-text export (`.txt`)
- standardized `CSV`
- automatic encoding detection for `UTF-8`, `Latin-1`, `GBK`, and `CP1252`

## Main Modules

### Dataset Overview

- key metrics
- annual publications
- top journals
- keyword word cloud
- top keyword table

### Relational Network Analysis

- keyword co-occurrence
- keyword-journal bipartite network
- author collaboration
- institution collaboration
- country collaboration
- country publication and citation impact quadrant
- co-citation
- bibliographic coupling

`Country Publications and Citation Impact Quadrant Analysis` is located in the `Relational Network Analysis` module and appears in the UI as `Country Publication and Citation Impact Quadrant`.

### Temporal Trend & Topic Evolution

- keyword timeline
- burst detection
- alluvial topic flow
- BERTopic workspace
- forward signals

### Bibliometric Structure & Performance

- Descriptive
- Journal Analysis
- Keyword Analysis
- Thematic Map
- Hierarchical Cluster Heatmap
- Top Entities
- Authors Over Time
- Three-Field Plot
- Lotka's Law
- Language Distribution

### Citation, Category & Source Analysis

- citation analysis
- subject categories
- document types
- funding analysis
- reference analysis
- RPYS
- reference burst detection
- publisher analysis
- disruption index

### Export Center

- figure export
- table export
- citation export
- selected-figure bundle
- one-click research bundle
- result and submission packaging

## Portability Notes

- keyword workflows prioritize dataset-native metadata fields (`DE / ID`)
- automatic title/abstract term extraction is used only when metadata is sparse
- the optional domain keyword plugin is now disabled by default to avoid hidden discipline-specific bias
- if you intentionally need extra domain dictionaries, enable the experimental plugin in `Vocabulary Governance`

## Dependency Layers

- `requirements-runtime.txt`: daily web-app runtime
- `requirements-topic-modeling.txt`: BERTopic and semantic topic modeling extras
- `requirements-dev.txt`: test and development tools
- `requirements.txt`: full installation

## Project Structure

```text
bibliometrics-5.26/
├── app.py
├── modules/
├── assets/
├── lib/
├── tests/
├── data/
├── .streamlit/
├── .github/workflows/
├── start.bat
├── run_tests.py
├── requirements-runtime.txt
├── requirements-topic-modeling.txt
├── requirements-dev.txt
├── requirements.txt
├── README.md
└── README_EN.md
```

## GitHub-Ready Release

The repository can generate a stripped release directory:

```text
github_release/Biblio-HUB
```

This directory keeps only runtime code, assets, tests, licenses, and documentation. It excludes raw data, manuscript artifacts, temporary scripts, caches, and other nonessential materials.

## Tests

```bash
python run_tests.py
```

## Tech Stack

| Category | Technology |
|----------|------------|
| Web Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Network Analysis | NetworkX, python-louvain |
| Interactive Visualization | PyVis |
| Static Visualization | Plotly, Matplotlib |
| Word Cloud | WordCloud |
| Topic Modeling | BERTopic, sentence-transformers, UMAP, HDBSCAN |

## System Requirements

| Item | Minimum |
|------|---------|
| Python | >= 3.10 |
| Memory | 4 GB RAM recommended |
| Browser | Chrome / Firefox / Edge |

## Notes

- `examples/legacy_export_demo/` is archived and not part of the main runtime path
- `raw data/` is a research archive directory and is not recommended for public GitHub release
- copyright and third-party notices are documented in `COPYRIGHT_AND_IP.md` and `THIRD_PARTY_NOTICES.md`
