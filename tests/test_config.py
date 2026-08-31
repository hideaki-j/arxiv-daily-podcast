from pathlib import Path

from ir_arxiv_ranker.config import load_config


def test_project_config_sets_minimum_email_score_to_seven():
    settings = load_config(Path("my_config/config.yaml"))

    assert settings.minimum_email_score == 7.0


def test_missing_minimum_email_score_preserves_previous_behavior(tmp_path):
    config_text = Path("my_config/config.yaml").read_text().replace(
        "minimum_email_score: 7\n", ""
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)

    settings = load_config(config_path)

    assert settings.minimum_email_score is None
