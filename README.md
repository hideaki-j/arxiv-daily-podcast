# ArXiv Dialy Newsletter & Podcast

Daily arXiv newsletter and podcast for automatic evaluation research. Fetches recent cs.IR and cs.CL papers plus keyword-matched papers, stores them in a persistent pool, scores unsent papers with LLM, generates podcast-style audio summaries, and sends email digests.

## Vibe Code Alert

This project was 99% vibe coded as a fun Saturday hack because I wanted to explore the usefulness of receiving an LLM-based, daily podcast-style summary of recent arXiv papers in my field, especially given the tsunami of academic papers being published every day. (Yes, I follow Andrej's [practice](https://github.com/karpathy/llm-council).)

## Quickstart

```bash
# 1. Create environment and install dependencies
uv venv && uv sync

# 2. Configure environment variables
# Create a .env file with OPENAI_API_KEY (plus Gmail creds if email is enabled)

# 3. Customize settings (optional)
# Edit my_config/config.yaml, my_config/keywords.yaml

# 4. Run
uv run -m ir_arxiv_ranker --config my_config/config.yaml

# 5. (Optional) Setup GitHub Actions for automatic daily runs
# See "GitHub Actions schedule" section below
```
## Setup Daily Newsletter & Podcast with GitHub Actions

1. Push this repo to GitHub (or fork it).
2. Go to Settings -> Secrets and variables -> Actions, then add `OPENAI_API_KEY`, `GMAIL_ADDRESS`, and `GMAIL_APP_PASSWORD`.
3. Edit the `cron` in `.github/workflows/arxiv-newsletter.yml` for your desired time (GitHub uses UTC).
4. Open Actions -> ArXiv Newsletter -> Run workflow, or wait for the schedule; check logs if it fails.

## Configuration essentials

Edit `my_config/config.yaml` to control the run:

- `influence_filter_model`, `influence_score_threshold`: model and minimum score (0–5) for author influence reporting; keep this at `3` to include scores `3`, `4`, and `5`.
- `ranking_aspects_path`, `ranking_max_workers`: separate YAML file for ranking aspect weights and the parallel worker count for per-paper ranking scores.
- `ranking_model`, `podcast_model`: main LLMs for ranking and transcripts.
- `podcast_provider`: set to `openrouter` to use OpenRouter for podcast generation (requires `OPENROUTER_API_KEY` in .env), or leave unset for default (`openai`).
- `top_n`, `top_n_tts`: how many papers to rank vs. generate audio for.
- `generate_transcript`, `use_tts`: enable/disable transcripts and mp3 audio.
- `include_keyword_papers`: set to `false` for IR/CL only mode (excludes keyword-matched papers).
- `email_enabled`: enable/disable Gmail delivery.
- `keywords_path`: keyword list for discovery.

Required `.env` entries when email is enabled:

```bash
OPENAI_API_KEY=...
GMAIL_ADDRESS=...
GMAIL_APP_PASSWORD=...
```

💡 Get a Gmail App Password by enabling 2-Step Verification and generating one in Google Account settings: [Google Account help page](https://support.google.com/accounts/answer/185833)

More details (full config list, pricing, outputs, structure) are in
[`docs/README.md`](docs/README.md).

## Dataflow

```mermaid
flowchart TD
  A["config.yaml<br>+ keywords.yaml<br>+ .env<br/>(__main__.py)"] --> B["Fetch arXiv papers<br/>cs.IR + cs.CL + keywords<br/>(arxiv_client.py)"]
  B --> C["LLM author influence scoring<br/>(influence_filter.py)"]
  C --> D["Persistent paper pool<br/>with sent flag<br/>(paper_state.py)"]
  D --> E["LLM per-paper aspect scoring<br/>+ aggregate ranking<br/>(ranking.py)"]
  E --> F["Write rankings.csv + results.json<br/>(output.py)"]
  E --> G["Download top PDF<br/>(output.py)"]
  G --> H["LLM transcripts<br/>(podcast.py)"]
  H --> I["TTS mp3s<br/>(tts.py)"]
  E --> J["HTML newsletter<br/>(output.py)"]
  J --> K["Email send (optional)<br/>(emailer.py)"]
  I --> K
```

Optional steps: transcripts, TTS, and email are controlled by config flags.

## FAQ

**Q. Can I use other LLMs?**
- A. Yes. Swap `ranking_model`, `podcast_model`, and `tts_model` in `my_config/config.yaml` to any supported OpenAI models, and keep `my_config/pricing.json` in sync. You can also set `podcast_provider: openrouter` to use OpenRouter specifically for the podcast generation.


**Q. How much does it cost to run?**
- A. In general, ~$0.50 per run. Details: It depends on the models and how many papers/audio you generate (author-influence scoring adds an extra LLM pass). Costs are tracked during a run and printed at the end; edit `my_config/pricing.json`, `top_n`, `top_n_tts`, and transcript/TTS flags to control spend.


**Q. Can I customize the keywords/retrieval/domain?**
- A. Keywords are fully configurable in `my_config/keywords.yaml`, and limits are in `my_config/config.yaml`.
  - Changing the base arXiv categories (currently `cs.IR` and `cs.CL`) requires a small code tweak in `src/ir_arxiv_ranker/__main__.py` / `src/ir_arxiv_ranker/arxiv_client.py`.
  - Changing scoring criteria requires edits in `prompt/prompt_scoring.j2` and `my_config/ranking_aspects.yaml` (and, for arXiv query filters, `src/ir_arxiv_ranker/arxiv_client.py`).


In either case, the only thing you need to do is open Claude Code/Cursor/Antigravity/Copilot/Cline/etc. and **just vibe code it**. 😎
