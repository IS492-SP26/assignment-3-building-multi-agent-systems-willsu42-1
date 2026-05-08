[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/SEjAoIAq)
# Multi-Agent Research System - Assignment 3

Starter scaffold for a multi-agent deep-research assistant on HCI topics. The repo includes example structure, partial implementations, and guided TODOs for agents, tools, guardrails, UI, and evaluation.

## Project Structure

```text
.
├── src/
│   ├── agents/
│   │   └── autogen_agents.py          # AutoGen agent creation + tool wiring
│   ├── autogen_orchestrator.py        # Multi-agent orchestration scaffold
│   ├── guardrails/
│   │   ├── safety_manager.py          # Safety coordination scaffold
│   │   ├── input_guardrail.py         # Input validation scaffold
│   │   └── output_guardrail.py        # Output validation scaffold
│   ├── tools/
│   │   ├── web_search.py              # Tavily / Brave search
│   │   ├── paper_search.py            # Semantic Scholar search
│   │   └── citation_tool.py           # Citation formatting utilities
│   ├── evaluation/
│   │   ├── judge.py                   # LLM-as-a-Judge scaffold
│   │   └── evaluator.py               # Batch evaluation scaffold
│   └── ui/
│       ├── cli.py                     # Interactive CLI
│       └── streamlit_app.py           # Streamlit web UI
├── data/
│   ├── example_queries.json           # Primary evaluation dataset
│   └── test_queries_sample.json       # Alternate/fallback dataset
├── docs/
│   └── TODO_AUDIT_AND_SOLUTIONS.md    # TODO inventory + guidance notes
├── config.yaml
├── requirements.txt
├── .env.example
├── example_autogen.py
└── main.py
```

## Setup

### 1) Prerequisites

- Python 3.9+
- `uv` (recommended) or `pip`

### 2) Install dependencies

Using `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Using `pip`:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3) Configure environment variables

```bash
cp .env.example .env
```

Minimum required keys:

- One model API path:
  - `OPENAI_API_KEY` (+ `OPENAI_BASE_URL` for vLLM/OpenAI-compatible endpoints), or
  - `GROQ_API_KEY`
- One search API:
  - `TAVILY_API_KEY` or `BRAVE_API_KEY`

Optional:

- `SEMANTIC_SCHOLAR_API_KEY` (recommended for higher paper-search rate limits)

## Running

### AutoGen example mode (default)

```bash
python main.py
# or
python main.py --mode autogen
```

### CLI

```bash
python main.py --mode cli
```

### Streamlit web UI

```bash
python main.py --mode web
# or
streamlit run src/ui/streamlit_app.py
```

### Batch evaluation scaffold

```bash
python main.py --mode evaluate
```

By default, this path only runs a simple test query until students complete the evaluation TODOs in `src/evaluation/` and wire them through `main.py`.

## Assignment Checklist (What Students Still Need To Complete)

- [ ] Finalize agent prompts/roles and end-to-end orchestration behavior.
- [ ] Finish tool integration and evidence formatting.
- [ ] Complete safety/guardrail logic and connect it to runtime flow.
- [ ] Surface safety outcomes clearly in the UI.
- [ ] Finish LLM-as-a-Judge scoring and batch evaluation reporting.
- [ ] Ensure CLI/web interfaces show traces and citations clearly.
- [ ] Document reproducible demo steps and representative outputs.

## Notes

- Some modules are intentionally partial and include TODO markers for students to complete.
- Use `ASSIGNMENT_INSTRUCTIONS.md` as the primary guide for where each requirement should be implemented.

## References

- [AutoGen documentation](https://microsoft.github.io/autogen/)
- [Tavily API](https://docs.tavily.com/)
- [Semantic Scholar API](https://api.semanticscholar.org/)
- [Guardrails AI](https://docs.guardrailsai.com/)
- [NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/)

## Demo  
- UI
<img width="1506" height="798" alt="image" src="https://github.com/user-attachments/assets/8fcd287b-1ea4-4c54-b295-53da4d1e5403" />  
- Using one of the example queries to run  
<img width="1493" height="767" alt="image" src="https://github.com/user-attachments/assets/63a1f89b-891c-4ea5-9c8e-7e5704ffd51d" />  
- Get response from AI model(using Groq here)
<img width="1497" height="831" alt="image" src="https://github.com/user-attachments/assets/b43c0476-c389-45c9-bc31-95075b344adc" />  
- Response the Citations  
<img width="1498" height="838" alt="image" src="https://github.com/user-attachments/assets/e5c0fc58-1ba0-467e-aef3-d35fc328074c" />  
- Show how many sources used and quality score  
<img width="1444" height="837" alt="image" src="https://github.com/user-attachments/assets/bfe30b8c-0418-4e44-aeac-4f3b60c0b371" />  
- Agent traces  
<img width="1114" height="594" alt="image" src="https://github.com/user-attachments/assets/f83f80f2-ad61-453a-85b5-b327155a513f" />
- Detail of agent traces - User & Planner  
<img width="632" height="785" alt="image" src="https://github.com/user-attachments/assets/912f8f01-4a32-4240-840f-76ad5eaed199" />  
- Detail of agent traces - Researcher  
<img width="609" height="764" alt="image" src="https://github.com/user-attachments/assets/b88acdc4-8799-4b59-96ed-dc786e8db12a" />  
- Detail of agent traces - Writer & Critic  
<img width="609" height="678" alt="image" src="https://github.com/user-attachments/assets/cdb1f7ee-2dc2-4274-b86a-69d8b47e21f9" />  
- Query history & safety log  
<img width="1046" height="540" alt="image" src="https://github.com/user-attachments/assets/0d36b9fe-84b0-42f4-b7a0-3e2416a19071" />






