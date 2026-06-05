# Current Pipeline Notes

The project is split into two operational stages that share the persistent paper
state in `state/discovered_papers.json`.

## Stage 1: Fetch And Score

Command:

```bash
uv run -m ir_arxiv_ranker --config my_config/config.yaml --stage fetch-score
```

Responsibilities:

- Fetch recent arXiv papers from `cs.IR`, `cs.CL`, and optional keyword queries.
- Merge discovered papers into the persistent pool.
- Run author-influence scoring for new papers when `PRIORITY_AUTHORS` is set.
- Extract affiliations for new or changed records.
- Run ranking-aspect scoring for pooled unsent records that need scores.
- Save the updated scored pool.

GitHub Actions workflow:

- `.github/workflows/arxiv-fetch-score.yml`
- Uses the lower-traffic weekday hourly check window: `0 6-16 * * 1-5`.
- Skips after one successful scheduled run in the UTC cycle.
- Caps scheduled attempts at six.

## Stage 2: Publish Newsletter

Command:

```bash
uv run -m ir_arxiv_ranker --config my_config/config.yaml --stage publish
```

Responsibilities:

- Read the stored scored paper pool.
- Select the highest-ranked unsent paper.
- Generate the selected-paper summary, transcript/audio, and optional image.
- Write newsletter artifacts.
- Send email when enabled.
- Mark the selected paper sent after email succeeds.

GitHub Actions workflow:

- `.github/workflows/arxiv-publish-newsletter.yml`
- Runs at 10:00 `America/Toronto` every day, including weekends and holidays.
- Has no retry dispatcher.

## Shared State

Both workflows check out app code from `master` and paper state from the
`paper-state` branch. Both use the `paper-state` concurrency group so state
commits do not overlap.
