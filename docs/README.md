# Project Details

## What it does

1. **Fetches papers** from arXiv (cs.IR, cs.CL, and keyword-matched) sorted by last updated date
2. **Filters papers** with an author-influence LLM stage (0–4 Likert gate)
3. **Ranks papers** using LLM based on relevance to automatic evaluation
4. **Downloads PDFs** of top-ranked papers
5. **Generates transcripts** for podcast-style summaries using LLM
6. **Synthesizes audio** (TTS) from transcripts
7. **Sends email** with HTML newsletter and mp3 attachments
8. **Tracks costs** across all LLM/TTS calls

Cost note: the author-influence pre-filter adds an extra LLM call; ensure your `pricing.json` includes the chosen `influence_filter_model`.

Each run creates a timestamped directory under `data/YYMMDD-HHMMSS/` with:
- `rankings.csv` and `results.json` - ranking results with TL;DRs and `author_influence_threshold`
- `papers/` - downloaded PDFs
- `transcript/` - generated podcast transcripts
- `podcast/` - synthesized mp3 files
- `newsletter/` - HTML newsletter

## Configuration

Settings in `my_config/config.yaml`:

| Setting | Current Value | Description |
|---------|---------------|-------------|
| `email_enabled` | `true` | Enable/disable email sending |
| `generate_transcript` | `true` | Enable/disable transcript generation |
| `filter_since_last_schedule` | `true` | Filter papers to those updated after the last scheduled GitHub Action run (cron in `.github/workflows/arxiv-newsletter.yml`) |
| `use_tts` | `true` | Enable/disable audio synthesis |
| `ranking_model` | `gpt-5-2025-08-07` | Model for paper ranking |
| `podcast_model` | `gpt-5-2025-08-07` | Model for transcript generation |
| `tts_model` | `gpt-4o-mini-tts-2025-03-20` | Model for audio synthesis |
| `tts_voice` | `marin` | Voice ID for TTS |
| `tts_instructions_path` | `prompt/tts_instructions.txt` | Path to TTS style instructions |
| `influence_filter_model` | `gpt-5-mini-2025-08-07` | Model for author influence pre-filter |
| `influence_prompt_path` | `prompt/prompt_influence_filter.j2` | Prompt template for author influence scoring |
| `influence_score_threshold` | `3` | Minimum author influence score (0–4) required to keep a paper |
| `influence_batch_size` | _unset_ | Optional batch size for influence scoring (default 150) |
| `compress_to_64kbps` | `true` | Compress mp3 to 64 kbps (requires `ffmpeg`) |
| `pricing_path` | `my_config/pricing.json` | Path to model pricing JSON |
| `ir_limit` | `50` | cs.IR papers to fetch (max 50) |
| `nlp_limit` | `50` | cs.CL papers to fetch (max 50) |
| `others_limit` | `50` | Keyword papers to fetch (max 50) |
| `keywords_path` | `my_config/keywords.yaml` | Path to keywords YAML |
| `top_n` | `5` | Papers to rank and download |
| `top_n_tts` | `5` | Papers to generate audio for (<= `top_n`) |
| `abst_word_cutoff` | `200` | Max abstract words in ranking prompt |
| `transcript_word_cutoff` | `1000` | Max words from PDF for transcript prompt |
| `arxiv_timeout` | `30` | arXiv request timeout in seconds |
| `openai_timeout` | `180` | OpenAI request timeout in seconds |

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
    "input_usd_per_1m_tokens": 1.25,
    "cached_input_usd_per_1m_tokens": 0.125,
    "output_usd_per_1m_tokens": 10.0
  },
  "gpt-5-mini-2025-08-07": {
    "input_usd_per_1m_tokens": 0.25,
    "cached_input_usd_per_1m_tokens": 0.025,
    "output_usd_per_1m_tokens": 2.0
  },
  "gpt-4o-mini-tts-2025-03-20": {
    "input_usd_per_1m_text_tokens": 0.6,
    "output_usd_per_1m_audio_tokens": 12.0,
    "estimated_usd_per_minute": 0.015
  }
}
```

### Environment Variables

Required in `.env` when email is enabled:
- `OPENAI_API_KEY`
- `GMAIL_ADDRESS`
- `GMAIL_APP_PASSWORD`

## Project Structure

```
src/ir_arxiv_ranker/
  __main__.py        # CLI entry point
  arxiv_client.py    # arXiv API queries and paper fetching
  ranking.py         # LLM-based paper ranking
  podcast.py         # Transcript generation from PDFs
  tts.py             # Text-to-speech synthesis
  affiliations.py    # Author affiliation extraction
  emailer.py         # Gmail SMTP sending
  output.py          # File I/O (CSV, JSON, downloads)
  models.py          # Data models (Paper, Rankings)
src/utils/
  call_llm.py        # OpenAI API wrapper
  costs.py           # Cost tracking
  naming.py          # File naming utilities
prompt/
  prompt_ranking.j2  # Ranking prompt template
  prompt_podcast.j2  # Podcast transcript template
  tts_instructions.txt
  prompt_influence_filter.j2
template/
  newsletter.j2      # HTML email template
my_config/
  config.yaml
  keywords.yaml
  pricing.json
```

## Dependencies

- Python >= 3.10
- `feedparser`, `httpx`, `jinja2`, `openai`, `pypdf`, `pyyaml`, `tqdm`, `python-dotenv`
- Optional: `ffmpeg` (for mp3 compression)
