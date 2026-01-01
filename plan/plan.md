# Plan: Add author influence pre-filter

Context (scope: newly added author influence pre-filter only): Papers are fetched in `__main__.py` via `arxiv_client.fetch_recent_papers` / `fetch_keyword_papers`, optionally date-filtered with `updated_after`, then ranked by `ranking.py` using `prompt/prompt_ranking.j2`. We need a new LLM stage that scores author influence on a 0–4 Likert scale, runs immediately after retrieval (before any date-based filtering), uses `gpt-5-mini`, emits structured JSON, drops papers with missing/low scores, and gates the downstream ranking.

Work plan
1) Structured prompt & schema  
   - Create a dedicated prompt template (e.g., `prompt/prompt_influence_filter.j2`) encoding the Likert rubric for author influence only:  

     | Score | Description | Example |
     | --- | --- | --- |
     | 4 | Majority are highly influential (authors from famous frontier labs or >2,000 in-field citations) | Lead author from DeepMind with >5k in-field citations; coauthors from Meta |
     | 3 | Includes strong influential figure(s) but not a majority | One top PI from Google Research plus mostly mid-career coauthors |
     | 2 | Some active field authors (e.g., consistently publish at the field’s flagship venues) | Multiple authors with recurring ACL/EMNLP papers but not widely cited yet |
     | 1 | Few slightly active names; mostly not active | One workshop/short-paper author plus several newcomers |
     | 0 | None recognized by typical field researchers | Unknown names with no visible flagship presence |

   - Define a JSON schema/response_format (per paper: `id`, `author_influence_threshold`, optional rationale) and enforce it via `call_llm_json`/`response_format` (see `ranking.py` pattern) to guarantee structured JSON. Plan to batch 150 papers safely within token limits and use only arXiv title and author names (no abstract/body) as input to the model.

2) Config plumbing  
   - Extend `my_config/config.yaml` / `config.Settings` to carry `influence_filter_model` (default `gpt-5-mini-2025-08-07`), `influence_prompt_path` (default to the new prompt), `influence_score_threshold` (default 3; drop papers with `author_influence_threshold < 3`), and optional `influence_batch_size`.  
   - Validate ranges, wire pricing lookup (model already present in `my_config/pricing.json`), and surface defaults in docs.

3) Filtering module  
   - Add `src/ir_arxiv_ranker/influence_filter.py` with a dataclass for scores and a function to evaluate a batch of `Paper` objects using `call_llm_json`.  
   - Parse/validate outputs against the schema, drop papers with missing/invalid payloads, and return (scores_by_id, kept_papers). Log failures and costs via `CostTracker`.

4) Reorder pipeline (after retrieval, before date filters)  
   - In `__main__.py`, fetch the capped IR/CL/keyword sets without `updated_after`, dedupe, then run the author influence filter on the full set (all ~150).  
   - After scoring, apply the threshold, then apply the existing date-based filtering (`filter_since_last_schedule` / fallback cascade) to the already-filtered set to honor the ordering constraint. Handle “not enough papers” errors/fallbacks accordingly.

5) Persist scores and surface context  
   - Thread scores into downstream structures (`rank_papers` inputs, `write_results_json`, `write_csv`), storing `author_influence_threshold` for transparency.  
   - Optionally attach a brief note in console/email/newsletter summarizing how many papers passed/failed the influence gate; ensure missing-score papers are simply omitted.

6) Documentation updates  
   - Update `README.md` and `docs/README.md` (dataflow diagram, config table, run description) to mention the new pre-filter stage, scoring rubric, model/default threshold, and failure handling.  
   - Add the new prompt file to the listed artifacts and note cost implications of the extra LLM pass.

Planned file touch map (ballpark deltas; will update after implementation)
| File | Adds (est.) | Reductions (est.) | Function/role | Notes |
| --- | --- | --- | --- | --- |
| src/ir_arxiv_ranker/__main__.py | +60–90 | -10–25 | Pipeline orchestration | Insert influence filter stage post-retrieval, adjust date filtering order, thread scores to downstream steps. |
| src/ir_arxiv_ranker/influence_filter.py (new) | +130–170 | 0 | Influence scoring | Implement batching, 5k-char cap, schema validation, thresholding, logging. |
| src/ir_arxiv_ranker/config.py | +30–50 | -5–10 | Config loading | New settings (model, prompt path, threshold, batch size), validation, defaults. |
| my_config/config.yaml | +5–10 | 0 | Config defaults | Add influence filter config defaults. |
| prompt/prompt_influence_filter.j2 (new) | +50–70 | 0 | Prompt template | Likert rubrics, structured JSON instructions. |
| src/ir_arxiv_ranker/ranking.py | +5–15 | -0–5 | Ranking plumbing | Accept/propagate author influence scores as needed. |
| src/ir_arxiv_ranker/output.py | +15–25 | -5–10 | Output persistence | Persist author influence scores into CSV/JSON outputs. |
| docs/README.md | +20–40 | -5–10 | Docs | Document new stage, config, cost note. |
| README.md | +10–20 | -2–6 | Docs | High-level dataflow and feature note. |
| plan/plan.md | 0 | 0 | Plan | (Current file; no further planned edits.) |
