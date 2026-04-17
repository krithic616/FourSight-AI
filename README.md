FourSight AI

Transform raw CSV files into structured business intelligence with analytics layers, AI-assisted interpretation, and export-ready reporting.

FourSight AI is a practical analytics workspace that converts raw datasets into actionable insights through multiple layers including diagnostics, prescriptive analytics, AI-driven interpretation, and report generation.

Application Preview

<p align="center">
  <img src="./dashboard main.jpeg" width="950"/>
</p>

This is the main workspace where users upload data and trigger the full analytics pipeline.

What This Project Does

- Upload CSV datasets into an analytics workflow  
- Perform data quality checks and profiling  
- Generate descriptive, diagnostic, predictive, and prescriptive insights  
- Provide AI-assisted interpretation using a local model  
- Visualize KPIs and recommendation drivers  
- Export structured reports in TXT, HTML, and PDF  

Core Workflow

The system is structured into layered analytics modules:

- Dataset Intelligence  
- Descriptive Analytics  
- Diagnostic Analytics  
- Predictive Analytics  
- Prescriptive Analytics  
- AI Analyst  
- Report Export  

Prescriptive Analytics Layer

![Prescriptive Analytics](./dashboard2.jpeg)

This layer evaluates readiness and converts analytical signals into business recommendations.

Includes:

- prescriptive readiness scoring  
- evidence-based recommendation logic  
- business decision framing  

KPI and Visualization Layer

![KPI Visualization](./dashboard3.jpeg)

This section presents key business metrics and recommendation drivers.

Includes:

- growth opportunities  
- risk controls  
- efficiency improvements  
- monitoring priorities  
- recommendation basis chart (diagnostic vs predictive vs combined)  

This is where raw analysis becomes visually interpretable for decision-making.

AI Analyst Layer

![AI Analyst](./dashboard5.jpeg)

This layer integrates a local AI model to generate narrative insights.

Includes:

- system status (local model availability)  
- model selection  
- controlled AI interaction  

AI Prompting and Insight Controller

![AI Prompting](./dashboard3.jpeg)

This section enables structured interaction with the AI Analyst.

Includes:

- response modes (Executive, Analyst, Action-focused)  
- custom prompt input  
- quick insight buttons:
  - Key Business Insights  
  - Top Risks  
  - Recommended Actions  
- controlled generation of business narratives  

This transforms analytics into explainable insights.

Report Preview and Export

![Report Preview](./dashboard6.jpeg)

Final output layer that converts analytics into structured reports.

Includes:

- full report preview  
- structured business summary  
- export options:
  - TXT  
  - HTML  
  - PDF  

Designed for stakeholder communication and decision reporting.

Key Features

- End-to-end analytics pipeline from upload to report  
- Multi-layer business insight generation  
- KPI visualization and recommendation breakdown  
- Prescriptive analytics engine  
- AI-assisted narrative interpretation  
- Prompt-controlled insight generation  
- Multi-format report export system  
- Local AI integration with fallback support  

Tech Stack

- Python  
- Streamlit  
- Pandas  
- Plotly / visualization layer  
- Ollama (local LLM)  
- HTML / PDF report generation  

Run Locally

```bash
git clone https://github.com/krithic616/FourSight-AI.git
cd FourSight-AI
pip install -r requirements.txt
streamlit run app/main.p
