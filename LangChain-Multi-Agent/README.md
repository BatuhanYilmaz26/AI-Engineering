# LangChain Multi-Agent Research Assistant

A research application that turns a question into a source-based report and an editorial review. Built with LangChain and Streamlit, the project separates searching, reading, writing, and reviewing into a sequential workflow.

## Workflow

1. **Search:** Find relevant web sources with Tavily and collect their URLs and research notes.
2. **Read:** Select up to three complementary sources and extract supporting content.
3. **Write:** Produce a structured report with an introduction, key findings, conclusion, and sources.
4. **Review:** Assess the report's clarity, reasoning, organization, and source attribution.

The search and reader stages use tool-enabled agents. The writer and critic use prompt-based chains. The critic provides an editorial assessment rather than independent factual verification.

## Interface

- Focused research form with stage-by-stage progress.
- Separate tabs for the report, editorial review, and research evidence.
- Markdown report downloads.
- Session-based results that survive ordinary interface reruns.
- Preserved completed outputs when a later stage fails.
- Clear messages for missing credentials, provider quotas, and interrupted research.

## Technology

| Component | Technology |
| --- | --- |
| Interface | Streamlit |
| Agents and chains | LangChain |
| Language model | GPT-4o mini through ChatOpenAI |
| Web search | Tavily |
| Content extraction | Trafilatura, readability-lxml, Beautiful Soup |
| Testing | unittest and Streamlit AppTest |

## Project structure

```text
LangChain-Multi-Agent/
|-- app.py                   # Streamlit research workspace
|-- requirements.txt         # Project dependencies
|-- .env.example             # Environment variable template
|-- .streamlit/
|   `-- config.toml          # Interface theme
|-- src/
|   |-- agents/
|   |   `-- agents.py        # Search/reader agents and writer/critic chains
|   |-- pipelines/
|   |   `-- pipeline.py      # Workflow orchestration and progress reporting
|   `-- tools/
|       `-- tools.py         # Web search and page extraction tools
|-- tests/
|   |-- test_app.py          # Interface behavior and failure recovery
|   `-- test_pipeline.py     # Stage wiring and output handling
`-- README.md
```

The offline tests use simulated provider responses to exercise interface behavior and pipeline integration without external API calls.
