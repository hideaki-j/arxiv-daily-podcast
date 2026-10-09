from pathlib import Path

from ir_arxiv_ranker import podcast
from ir_arxiv_ranker.models import Paper


def test_podcast_prompts_and_tts_text_include_all_registered_authors(monkeypatch):
    names = ["Akari Asai", "Sarah Wiegreffe", "Sajad Ebrahimi"]
    paper = Paper("P1", "2406.1", "A Paper", ["Other Author", *names], "", "", "", "")
    template = Path("prompt/prompt_podcast.j2").read_text()
    prompt = podcast._render_prompt(template, paper, "Paper text", "gemini", names)
    for name in names:
        assert name in prompt.split("Required names:")[1].splitlines()[0]
    assert "well-known" not in prompt
    assert "Do not mention affiliations" in prompt
    monkeypatch.setattr(podcast, "_extract_pdf_text", lambda _: "Paper text")
    greeting = "Welcome back to the Automatic Evaluation podcast. Thanks for tuning in."
    generated = greeting + " We all know search matters."
    monkeypatch.setattr(podcast, "batch_call_llm_text", lambda **kwargs: [generated])
    monkeypatch.setattr(podcast, "call_llm_text", lambda **kwargs: generated)
    batch = podcast.generate_transcripts_batch(None, "gemini", template, [paper], [Path("fake.pdf")], priority_authors=names)
    single = podcast.generate_transcript(None, "gemini", template, paper, Path("fake.pdf"), priority_authors=names)
    assert batch == [single]
    assert single.startswith(greeting)
    for name in names:
        assert name in single
    assert podcast._ensure_required_authors(single, paper, names) == single
