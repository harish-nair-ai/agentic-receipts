"""LLM-as-Judge for fuzzy claim verification."""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from receipts.config import Config, JudgeProvider
from receipts.models import Claim, Evidence, Verdict


@dataclass
class JudgeResult:
    """Result of an LLM judge verification."""
    verdict: Verdict
    reasoning: str
    evidence_text: str


def judge_claim(
    claim: Claim,
    evidence: list[Evidence],
    transcript_context: str,
    config: Config
) -> JudgeResult:
    """Verify a claim using an LLM judge.

    Args:
        claim: The claim to verify.
        evidence: Deterministic evidence already gathered.
        transcript_context: Stringified relevant parts of the transcript.
        config: Judge configuration (provider, model, API key).

    Returns:
        JudgeResult with verdict and reasoning.
    """
    prompt = _build_prompt(claim, evidence, transcript_context)

    try:
        if config.provider == JudgeProvider.GEMINI:
            return _judge_gemini(prompt, config)
        elif config.provider == JudgeProvider.OPENAI:
            return _judge_openai(prompt, config)
        elif config.provider == JudgeProvider.ANTHROPIC:
            return _judge_anthropic(prompt, config)
        else:
            return JudgeResult(
                verdict=Verdict.SKIPPED,
                reasoning=f"Unsupported provider: {config.provider}",
                evidence_text="",
            )
    except Exception as e:
        return JudgeResult(
            verdict=Verdict.SKIPPED,
            reasoning=f"Judge API error: {str(e)}",
            evidence_text="",
        )


def _build_prompt(claim: Claim, evidence: list[Evidence], transcript_context: str) -> str:
    """Construct the prompt for the LLM judge."""
    evidence_text = "\n".join(f"- {e.source.value}: {e.content}" for e in evidence)
    if not evidence_text:
        evidence_text = "No deterministic evidence found."

    return f"""You are an independent auditor verifying claims made by an AI coding agent.
Your job is to check if the agent actually did what it claimed to do, based ONLY on the provided transcript of its actions.

CLAIM TO VERIFY:
"{claim.text}"

DETERMINISTIC EVIDENCE FOUND:
{evidence_text}

TRANSCRIPT CONTEXT (Tool Calls & Results):
{transcript_context}

Evaluate the claim against the transcript.
Did the agent actually execute the necessary tool calls (e.g., bash commands, file edits) to make this claim true?
Or is it hallucinating success?

Respond in JSON format:
{{
  "verdict": "verified" | "unverified" | "refuted",
  "reasoning": "Brief explanation of why",
  "evidence": "Specific tool call or output that proves/disproves the claim"
}}

Respond ONLY with the JSON object.
"""


def _judge_gemini(prompt: str, config: Config) -> JudgeResult:
    """Call Google Gemini API."""
    url = f"{config.api_url}/models/{config.model}:generateContent?key={config.api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json"
        }
    }
    
    with httpx.Client(timeout=config.timeout) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            result = json.loads(text)
            return JudgeResult(
                verdict=Verdict(result.get("verdict", "skipped")),
                reasoning=result.get("reasoning", "No reasoning provided"),
                evidence_text=result.get("evidence", ""),
            )
        except (KeyError, IndexError, json.JSONDecodeError, ValueError) as e:
             return JudgeResult(Verdict.SKIPPED, f"Failed to parse Gemini response: {e}", "")


def _judge_openai(prompt: str, config: Config) -> JudgeResult:
    """Call OpenAI API."""
    url = f"{config.api_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    with httpx.Client(timeout=config.timeout) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        try:
            text = data["choices"][0]["message"]["content"]
            result = json.loads(text)
            return JudgeResult(
                verdict=Verdict(result.get("verdict", "skipped")),
                reasoning=result.get("reasoning", "No reasoning provided"),
                evidence_text=result.get("evidence", ""),
            )
        except (KeyError, IndexError, json.JSONDecodeError, ValueError) as e:
             return JudgeResult(Verdict.SKIPPED, f"Failed to parse OpenAI response: {e}", "")


def _judge_anthropic(prompt: str, config: Config) -> JudgeResult:
    """Call Anthropic API."""
    url = f"{config.api_url}/messages"
    headers = {
        "x-api-key": config.api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    payload = {
        "model": config.model,
        "max_tokens": 1024,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
        "system": "You are a JSON-only API. Output ONLY valid JSON."
    }
    
    with httpx.Client(timeout=config.timeout) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        try:
            text = data["content"][0]["text"]
            # Anthropic doesn't have guaranteed JSON mode yet, so strip backticks if present
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
                
            result = json.loads(text.strip())
            return JudgeResult(
                verdict=Verdict(result.get("verdict", "skipped")),
                reasoning=result.get("reasoning", "No reasoning provided"),
                evidence_text=result.get("evidence", ""),
            )
        except (KeyError, IndexError, json.JSONDecodeError, ValueError) as e:
             return JudgeResult(Verdict.SKIPPED, f"Failed to parse Anthropic response: {e}", "")
