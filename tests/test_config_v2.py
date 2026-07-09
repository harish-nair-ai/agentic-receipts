from receipts.config import Config, JudgeProvider, check_independence


def _cfg(provider, model="m", **kw):
    return Config(provider=provider, model=model, api_key="k", **kw)


def test_new_env_vars_parsed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-x")
    monkeypatch.setenv("RECEIPTS_SCORE_PASSES", "5")
    monkeypatch.setenv("RECEIPTS_MIN_CONFIDENCE", "0.75")
    monkeypatch.setenv("RECEIPTS_CHECKER_MODEL", "gemini-custom")
    monkeypatch.setenv("RECEIPTS_AUTOFIX", "1")
    cfg = Config.from_env()
    assert cfg.score_passes == 5
    assert cfg.min_confidence == 0.75
    assert cfg.checker_model == "gemini-custom"
    assert cfg.autofix is True


def test_defaults_when_unset(monkeypatch):
    for v in ("RECEIPTS_SCORE_PASSES", "RECEIPTS_MIN_CONFIDENCE",
              "RECEIPTS_CHECKER_MODEL", "RECEIPTS_AUTOFIX"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    cfg = Config.from_env()
    assert cfg.score_passes == 3
    assert cfg.min_confidence == 0.6
    assert cfg.autofix is False
    assert cfg.checker_model == cfg.model  # falls back to model


def test_score_passes_floor_and_confidence_clamp(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("RECEIPTS_SCORE_PASSES", "0")
    monkeypatch.setenv("RECEIPTS_MIN_CONFIDENCE", "5")
    cfg = Config.from_env()
    assert cfg.score_passes == 1       # floored to 1
    assert cfg.min_confidence == 1.0   # clamped to [0,1]


def test_supports_logprobs_by_provider():
    assert _cfg(JudgeProvider.OPENAI).supports_logprobs is True
    assert _cfg(JudgeProvider.GEMINI).supports_logprobs is True
    assert _cfg(JudgeProvider.ANTHROPIC).supports_logprobs is False


def test_check_independence():
    # Auditing Claude Code (anthropic family) with an Anthropic checker → NOT independent.
    assert check_independence(_cfg(JudgeProvider.ANTHROPIC), "claude-code") is False
    # Gemini checker auditing Claude Code → independent.
    assert check_independence(_cfg(JudgeProvider.GEMINI), "claude-code") is True
    # Unknown agent → assume independent (can't prove overlap).
    assert check_independence(_cfg(JudgeProvider.ANTHROPIC), "some-other-agent") is True
