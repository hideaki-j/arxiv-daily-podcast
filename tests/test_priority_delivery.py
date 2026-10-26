from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest

from ir_arxiv_ranker import __main__ as pipeline
from ir_arxiv_ranker.config import load_config
from ir_arxiv_ranker.paper_state import (
    load_paper_state, pooled_records, refresh_priority_author_gate, save_paper_state,
)
from ir_arxiv_ranker.priority_authors import matched_priority_authors


def record(name, sent=False, score=4, legal_score=2, base_id=None):
    base_id = base_id or name
    return {
        "base_arxiv_id": base_id, "latest_arxiv_id": base_id + "v1", "title": name,
        "authors": [name, "Other Author"], "sent": sent, "in_pool": True,
        "influence_score": score,
        "scoring_scores": {"author_influence_score": score, "legal_domain": legal_score},
        "scoring_total_score": score + legal_score,
    }


def test_matching_requires_full_name_but_ignores_case_accents_and_spacing():
    assert matched_priority_authors(["  AKÁRI   Asai "], ["Akari Asai"]) == ["Akari Asai"]
    for name in ["Asai", "A. Asai", "Akari Asaiki", "Someone Akari Asai"]:
        assert matched_priority_authors([name], ["Akari Asai"]) == []


def test_entire_history_is_rechecked_without_resending_sent_papers():
    state = {"schema_version": 1, "pooled_papers": {
        "new": record("Akari Asai", score=1),
        "famous": record("Famous Unregistered", score=6),
        "sent": {**record("Akari Asai", sent=True), "sent_at": "yesterday"},
    }}
    refresh_priority_author_gate(state, ["Akari Asai"])
    assert len(pooled_records(state)) == 1
    assert state["pooled_papers"]["new"]["scoring_total_score"] == 8
    assert state["pooled_papers"]["famous"]["in_pool"] is False
    assert state["pooled_papers"]["sent"]["sent_at"] == "yesterday"
    refresh_priority_author_gate(state, ["Akari Asai"])
    assert state["pooled_papers"]["new"]["scoring_total_score"] == 8
    refresh_priority_author_gate(state, [])
    assert pooled_records(state) == []
    refresh_priority_author_gate(state, ["Famous Unregistered"])
    assert pooled_records(state)[0]["authors"][0] == "Famous Unregistered"
    refresh_priority_author_gate(state, [], require_match=False)
    assert len(pooled_records(state)) == 2


@pytest.mark.parametrize("targets", ["", "Akari Asai"])
def test_publish_with_no_match_skips_all_generation_and_delivery(tmp_path, monkeypatch, targets):
    state_path = tmp_path / "state.json"
    save_paper_state(state_path, {"schema_version": 1, "pooled_papers": {
        "outsider": record("Famous Unregistered", score=6, legal_score=100),
    }})
    monkeypatch.setenv("PRIORITY_AUTHORS", targets)
    monkeypatch.setenv("PAPER_STATE_PATH", str(state_path))
    monkeypatch.setattr(pipeline, "load_dotenv", lambda: None)
    monkeypatch.setattr(pipeline, "_parse_args", lambda: Namespace(
        stage="publish", config=Path("my_config/config.yaml")))

    def forbidden(*args, **kwargs):
        pytest.fail("No matching author must stop before clients, generation, or email")

    for name in ["OpenAI", "download_papers", "generate_selected_summaries_batch", "send_email"]:
        monkeypatch.setattr(pipeline, name, forbidden)
    pipeline.main()
    assert not load_paper_state(state_path)["pooled_papers"]["outsider"]["in_pool"]


def test_publish_selects_matching_paper_over_higher_scoring_outsider(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    save_paper_state(state_path, {"schema_version": 1, "pooled_papers": {
        "outsider": record("Famous Unregistered", score=6, legal_score=100, base_id="outsider"),
        "target": record("Akari Asai", score=1, base_id="target"),
    }})
    settings = replace(load_config(Path("my_config/config.yaml")),
                       generate_transcript=False, use_tts=False)
    monkeypatch.setenv("PRIORITY_AUTHORS", "Akari Asai")
    monkeypatch.setenv("PAPER_STATE_PATH", str(state_path))
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("GMAIL_ADDRESS", "test@example.com")
    monkeypatch.setenv("GMAIL_APP_PASSWORD", "test")
    monkeypatch.setattr(pipeline, "load_dotenv", lambda: None)
    monkeypatch.setattr(pipeline, "load_config", lambda _: settings)
    monkeypatch.setattr(pipeline, "_parse_args", lambda: Namespace(
        stage="publish", config=Path("my_config/config.yaml")))
    monkeypatch.setattr(pipeline, "OpenAI", lambda: object())
    monkeypatch.setattr(pipeline.genai, "Client", lambda **kwargs: object())
    directories = [tmp_path / name for name in ["run", "pdf", "transcripts", "audio", "newsletter"]]
    for directory in directories:
        directory.mkdir()
    monkeypatch.setattr(pipeline, "create_run_dir", lambda: tuple(directories))
    monkeypatch.setattr(pipeline, "download_papers", lambda *args: [tmp_path / "paper.pdf"])
    monkeypatch.setattr(pipeline, "generate_selected_summaries_batch", lambda **kwargs: {})
    sent = []
    monkeypatch.setattr(pipeline, "send_email", lambda **kwargs: sent.append(kwargs))
    pipeline.main()
    assert len(sent) == 1
    assert "<strong>Akari Asai</strong>, Other Author" in sent[0]["html_body"]
    assert "Famous Unregistered" not in sent[0]["body"]
    assert "**Akari Asai**" in sent[0]["body"]
    assert "Affiliations:" not in sent[0]["html_body"]
    state = load_paper_state(state_path)["pooled_papers"]
    assert state["target"]["sent"] is True
    assert state["outsider"]["sent"] is False
