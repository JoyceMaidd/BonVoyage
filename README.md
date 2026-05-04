# BonVoyage ✈️

An AI travel planning agent for solo travelers. Plan attractions, find hostels, lookup discounts, and export to Google My Maps.

## Setup

1. Clone the repo
2. Copy `.env.example` to `.env` and add your API keys:

   ```bash
   cp .env.example .env
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Web UI (Streamlit):**
```bash
streamlit run app.py
```

**CLI (ReAct loop):**
```bash
python -m bonvoyage.agent.react_loop
```

**Run evaluations:**
```bash
python -m bonvoyage.eval.run_evals
```

## Project Structure

- `agent/` — ReAct loop, intent extraction, controller
- `tools/` — Search, geocoding, exporting
- `models/` — Data models (TripState, Attraction, etc.)
- `prompts/` — System prompts (YAML)
- `eval/` — Test harness

## Environment Variables

See [.env.example](.env.example) for required API keys (Gemini, Tavily).
