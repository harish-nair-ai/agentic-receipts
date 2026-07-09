from unittest.mock import patch

from receipts.config import Config, JudgeProvider
from receipts.models import ScoreMethod
from receipts.scorer import ScorePass, generate_critique, score_pass


def _cfg(provider, model="m"):
    return Config(provider=provider, model=model, api_key="k")


def test_score_pass_logprob_openai_parses_top_logprobs():
    fake = ScorePass(score01=0.75, peakiness=0.9, method=ScoreMethod.LOGPROB, letter="T")
    with patch("receipts.scorer._score_logprob_openai", return_value=fake) as m:
        out = score_pass("prompt", _cfg(JudgeProvider.OPENAI))
    assert out is fake
    m.assert_called_once()


def test_score_pass_uses_sampled_for_anthropic():
    fake = ScorePass(score01=0.5, peakiness=1.0, method=ScoreMethod.SAMPLED, letter="K")
    with patch("receipts.scorer._score_sampled", return_value=fake) as m:
        out = score_pass("prompt", _cfg(JudgeProvider.ANTHROPIC))
    assert out.method == ScoreMethod.SAMPLED
    m.assert_called_once()


def test_score_pass_returns_none_on_exception():
    with patch("receipts.scorer._score_logprob_openai", side_effect=RuntimeError("boom")):
        assert score_pass("prompt", _cfg(JudgeProvider.OPENAI)) is None


def test_generate_critique_returns_empty_on_exception():
    with patch("receipts.scorer._critique_call", side_effect=RuntimeError("net")):
        assert generate_critique("c", "e", "ctx", "refuted", _cfg(JudgeProvider.OPENAI)) == ""
