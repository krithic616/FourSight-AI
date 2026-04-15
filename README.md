# FourSight AI

**Analytics intelligence and business insight copilot for structured CSV datasets.**

FourSight AI is a Streamlit-based analytics application that moves beyond a static dashboard workflow. It ingests a CSV file, profiles the dataset, evaluates business readiness, generates deterministic analytics across multiple layers, and adds a grounded local AI analyst experience for concise insight generation and report support.

The project is designed for business analysis scenarios where users need more than charts. Instead of only visualizing data, FourSight AI helps interpret data quality, structure, readiness, diagnostic patterns, directional forecasts, recommended actions, and compact AI-assisted summaries.

## Why It Is Different From Normal Dashboards

- It is workflow-oriented rather than chart-oriented.
- It combines deterministic analytics with a grounded local AI layer.
- It evaluates whether deeper analytics are justified before showing them.
- It emphasizes evidence-backed business interpretation, not just visualization.
- It can still produce useful deterministic output when local AI is unavailable or constrained.

## Core Features

- CSV upload and safe tabular ingestion
- Data quality checks for duplicates and missing values
- Column type profiling and preprocessing summaries
- Dataset-type detection and analytics readiness assessment
- Deterministic descriptive, diagnostic, predictive, and prescriptive layers
- Local AI Analyst powered through Ollama
- Compact fallback summaries when local generation fails
- Downloadable TXT and HTML business insight reports

## Analytics Layers

- Data Quality
- Column Profiling
- Dataset Intelligence
- Descriptive Analytics
- Diagnostic Analytics
- Predictive Analytics
- Prescriptive Analytics
- AI Analyst
- Download Report

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Ollama for local LLM serving

## Project Workflow

1. Upload a CSV dataset.
2. Profile the dataset and compute core KPIs.
3. Evaluate data quality and structural readiness.
4. Detect dataset type and determine which analytics layers should be enabled.
5. Generate deterministic business analysis across descriptive, diagnostic, predictive, and prescriptive stages.
6. Optionally request a grounded local AI summary from the AI Analyst tab.
7. Export the resulting report as TXT or HTML.

## Key Design Principles

- **Grounded outputs:** the AI layer is constrained to computed app context rather than raw free-form dataset interpretation.
- **Progressive analytics:** advanced layers are enabled only when the data supports them.
- **Deterministic first:** business value remains available even if local AI is slow or unavailable.
- **Low-friction local use:** designed to run as a local analytics copilot with Ollama rather than depend on hosted AI APIs.
- **Modular architecture:** analytics, AI, reporting, core processing, and UI logic are kept separated for maintainability.

## Local AI / Ollama Note

FourSight AI supports local AI generation through Ollama. The app is designed to keep AI responses grounded in computed analytics context and to fall back gracefully when local model generation is unavailable, times out, or exceeds available system memory.

For low-resource systems, smaller models such as `tinyllama` are generally more practical than heavier local models. Ollama must be installed and running separately on your machine.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/krithic616/FourSight-AI.git
cd FourSight-AI