from pathlib import Path

import pytest

from ir_arxiv_ranker.config import load_config


def test_project_config_sets_minimum_email_score_to_seven():
    settings = load_config(Path("my_config/config.yaml"))

    assert settings.minimum_email_score == 7.0
    assert settings.require_priority_author_match is True


def test_missing_minimum_email_score_preserves_previous_behavior(tmp_path):
    config_text = Path("my_config/config.yaml").read_text().replace(
        "minimum_email_score: 7\n", ""
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text)

    settings = load_config(config_path)

    assert settings.minimum_email_score is None


def test_author_gate_defaults_to_enabled_when_omitted(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(Path("my_config/config.yaml").read_text().replace(
        "require_priority_author_match: true\n", ""))
    assert load_config(path).require_priority_author_match is True


def test_author_gate_rejects_non_boolean_values(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(Path("my_config/config.yaml").read_text().replace(
        "require_priority_author_match: true", 'require_priority_author_match: "false"'))
    with pytest.raises(SystemExit, match="must be a boolean"):
        load_config(path)
