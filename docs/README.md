# Project Details

## What it does

The pipeline has two explicit stages:

1. **Fetch and score** (`--stage fetch-score`)
   - Fetches papers from arXiv (`cs.IR`, `cs.CL`, and optional keyword-matched papers) sorted by last updated date.
   - Merges discovered papers into `state/discovered_papers.json`.
   - Scores new papers for author influence when `PRIORITY_AUTHORS` is configured.
   - Extracts affiliations for new or changed papers.
   - Scores unsent pooled papers on configurable ranking aspects and stores aggregate scores.
2. **Publish** (`--stage publish`)
   - Reads the stored paper pool from `state/discovered_papers.json`.
   - Selects the highest-ranked unsent paper.
   - Downloads the selected PDF.
   - Generates a selected-paper description for the newsletter.
   - Optionally generates a manga-style image.
   - Optionally generates podcast transcripts and TTS audio.
   - Sends email with HTML newsletter and attachments when email is enabled.
   - Marks the selected paper as sent only after email succeeds.

`--stage all` runs both stages in one process and is the default for local manual runs.

Cost note: author-influence scoring adds an extra LLM call; ensure your `pricing.json` includes the chosen `influence_filter.model`.

Each run creates a timestamped directory under `data/YYMMDD-HHMMSS/` with:
- `rankings.csv` and `results.json` - ranking results with TL;DRs and `author_influence_threshold`
- `papers/` - downloaded PDFs
- `transcript/` - generated podcast transcripts
- `podcast/` - synthesized mp3 files
- `manga/` - generated selected-paper image, when enabled
- `newsletter/` - HTML newsletter

Fetch-only runs may create a run directory only when affiliation PDF downloads are needed. Publish runs create the newsletter artifacts.

## Email statistics

The publish email includes five statistics computed from the current unsent
stored pool before the selected paper is marked sent:

| Label | Definition |
|-------|------------|
| `Unsent pool` | Total unsent papers currently in `state/discovered_papers.json` |
| `Fetched 24h` | Unsent papers whose `last_seen_at` is within the last 24 hours |
| `Unique 24h` | Papers from the last-24-hour fetches whose `first_seen_at` is also within the last 24 hours, shown as `unique/fetched` |
| `Added 7d` | Unsent papers whose `first_seen_at` is within the last 7 days |
| `Author pass` | Unsent papers with `influence_score >= influence_score_threshold`, shown as `pass/unsent` |

## Running

```bash
# Full local pipeline
uv run -m ir_arxiv_ranker --config my_config/config.yaml

# Split stages, matching GitHub Actions
uv run -m ir_arxiv_ranker --config my_config/config.yaml --stage fetch-score
uv run -m ir_arxiv_ranker --config my_config/config.yaml --stage publish
```

## GitHub Actions

The automation is intentionally split into two workflow files:

| Workflow | File | Purpose | Schedule |
|----------|------|---------|----------|
| ArXiv Fetch and Score | `.github/workflows/arxiv-fetch-score.yml` | Fetch from arXiv, enrich, score, and commit `state/discovered_papers.json` to the `paper-state` branch | Same weekday hourly retry/check window as before: `0 13-23 * * 1-5`; skips after one scheduled success per UTC cycle and caps scheduled attempts at six |
| ArXiv Publish Newsletter | `.github/workflows/arxiv-publish-newsletter.yml` | Read stored state, generate newsletter/transcript/audio/image, send email, and mark the selected paper sent | 10:00 `America/Toronto` every day, including weekends and holidays; no retry dispatcher |

Both workflows use the same `paper-state` concurrency group so state commits do not overlap. The app code is checked out from `master`; the persistent paper state is checked out from the `paper-state` branch.

## Configuration

Settings in `my_config/config.yaml`:

| Setting | Current Value | Description |
|---------|---------------|-------------|
| `email_enabled` | `true` | Enable/disable email sending |
| `generate_transcript` | `true` | Enable/disable transcript generation |
| `generate_manga_image` | `true` | Enable/disable selected-paper image generation |
| `filter_since_last_schedule` | `true` | Ignored in bucket mode; paper-state deduplication controls repeat notifications |
| `use_tts` | `true` | Enable/disable audio synthesis |
| `influence_filter.provider` | `openai` | Provider for author-influence scoring |
| `influence_filter.model` | `gpt-5-mini-2025-08-07` | Model for author-influence scoring |
| `ranking.provider` | `openai` | Provider for per-paper ranking |
| `ranking.model` | `gpt-5-2025-08-07` | Model for paper ranking and selected-paper summaries |
| `podcast.provider` | `openai` | Provider for transcript generation |
| `podcast.model` | `gpt-5.5` | Model for transcript generation |
| `manga_planner.provider` | `openai` | Provider for selected-paper image planning |
| `manga_planner.model` | `gpt-5.5` | Model for selected-paper image planning |
| `manga_image.provider` | `openai` | Provider for image generation; must be `openai` |
| `manga_image.model` | `gpt-image-2` | Model for image generation |
| `manga_image.size` | `1024x1536` | Generated portrait smartphone-oriented image size |
| `manga_image.quality` | `high` | Generated image quality |
| `manga_image.output_format` | `png` | Generated image format |
| `manga_image.char_cutoff` | `30000` | Max selected-paper text characters for image planning |
| `tts.provider` | `gemini` | Provider for audio synthesis |
| `tts.model` | `gemini-2.5-flash-preview-tts` | Model for audio synthesis |
| `tts.voice` | `Zephyr` | Voice ID for TTS |
| `affiliation.provider` | `gemini` | Provider for affiliation extraction |
| `affiliation.model` | `gemini-3-flash-preview` | Model for affiliation extraction |
| `influence_score_threshold` | `3` | Minimum author-influence score (0–5) to report as influential; includes scores 3, 4, and 5 |
| `ranking_aspects_path` | `my_config/ranking_aspects.yaml` | Separate YAML file with positive/negative ranking aspects and weights |
| `ranking_max_workers` | `150` | Parallel worker count for per-paper ranking aspect scoring |
| `influence_max_workers` | _unset_ | Optional parallel worker count for influence scoring (default 150) |
| `compress_to_64kbps` | `true` | Compress mp3 to 64 kbps (requires `ffmpeg`) |
| `pricing_path` | `my_config/pricing.json` | Path to model pricing JSON |
| `ir_limit` | `50` | cs.IR papers to fetch (max 50) |
| `nlp_limit` | `50` | cs.CL papers to fetch (max 50) |
| `others_limit` | `50` | Keyword papers to fetch (max 50) |
| `include_keyword_papers` | `false` | Include keyword-matched papers (set to `false` for IR/CL only mode) |
| `keywords_path` | `my_config/keywords.yaml` | Path to keywords YAML |
| `top_n` | `3` | Number of pooled papers to include in ranking outputs |
| `top_n_tts` | `3` | Enables transcript/audio generation for the selected paper when greater than 0 (must be <= `top_n`) |
| `abst_word_cutoff` | `200` | Max abstract words in ranking prompt |
| `transcript_word_cutoff` | `1000` | Max words from PDF for transcript prompt |
| `arxiv_timeout` | `30` | arXiv request timeout in seconds |
| `openai_timeout` | `360` | OpenAI/provider request timeout in seconds |

### Keywords

Edit `my_config/keywords.yaml` to customize paper discovery:

```yaml
keywords:
  - automatic evaluation
  - large language models
  - llm-as-a-judge
```

### Pricing

Model costs are configured in `my_config/pricing.json`:

```json
{
  "gpt-5-2025-08-07": {
    "provider": "openai",
    "input_usd_per_1m_tokens": 1.25,
    "cached_input_usd_per_1m_tokens": 0.125,
    "output_usd_per_1m_tokens": 10.0
  },
  "gpt-5-mini-2025-08-07": {
    "provider": "openai",
    "input_usd_per_1m_tokens": 0.25,
    "cached_input_usd_per_1m_tokens": 0.025,
    "output_usd_per_1m_tokens": 2.0
  },
  "gpt-image-2": {
    "provider": "openai",
    "text_input_usd_per_1m_tokens": 5.0,
    "text_cached_input_usd_per_1m_tokens": 1.25,
    "image_input_usd_per_1m_tokens": 8.0,
    "image_cached_input_usd_per_1m_tokens": 2.0,
    "image_output_usd_per_1m_tokens": 30.0
  },
  "gemini-2.5-flash-preview-tts": {
    "provider": "gemini",
    "input_usd_per_1m_text_tokens": 0.5,
    "output_usd_per_1m_audio_tokens": 10.0
  }
}
```

### Environment Variables

Common `.env` entries:
- `OPENAI_API_KEY`

Required when email is enabled:
- `GMAIL_ADDRESS`
- `GMAIL_APP_PASSWORD`

Required only when the matching provider is enabled:
- `OPENROUTER_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`

Optional:
- `PRIORITY_AUTHORS` - semicolon-delimited canonical author names.

In GitHub Actions, the same values should be stored as repository Actions secrets. The fetch-score workflow only writes API/provider secrets needed for scoring/enrichment. The publish workflow also writes Gmail secrets because it can send email.

## Project Structure

```
src/ir_arxiv_ranker/
  __main__.py        # CLI entry point
  arxiv_client.py    # arXiv API queries and paper fetching
  paper_state.py     # Persistent discovered/sent paper state
  ranking.py         # LLM-based paper ranking
  selected_summary.py # Selected-paper newsletter description
  podcast.py         # Transcript generation from PDFs
  tts.py             # Text-to-speech synthesis
  manga_image.py     # Selected-paper image planning and generation
  influence_filter.py # Author influence scoring
  affiliations.py    # Author affiliation extraction
  emailer.py         # Gmail SMTP sending
  output.py          # File I/O (CSV, JSON, downloads)
  models.py          # Data models (Paper, Rankings)
  schedule.py        # Cron parsing helpers retained for compatibility
src/utils/
  call_llm.py        # OpenAI API wrapper
  costs.py           # Cost tracking
  naming.py          # File naming utilities
  timezone.py        # Timezone helpers
prompt/
  prompt_scoring.j2  # Per-paper scoring prompt template
  prompt_selected_summary.j2  # Selected-paper newsletter summary template
  prompt_podcast.j2  # Podcast transcript template
  tts_instructions.txt
  prompt_influence_filter.j2
  prompt_manga.j2
  prompt_manga_planner.j2
template/
  newsletter.j2      # HTML email template
my_config/
  config.yaml
  keywords.yaml
  pricing.json
  ranking_aspects.yaml
.github/workflows/
  arxiv-fetch-score.yml
  arxiv-publish-newsletter.yml
```

## Dependencies

- Python >= 3.10
- `feedparser`, `httpx`, `jinja2`, `openai`, `pypdf`, `pyyaml`, `tqdm`, `python-dotenv`
- Optional: `ffmpeg` (for mp3 compression)
