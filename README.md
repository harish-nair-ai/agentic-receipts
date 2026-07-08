<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/harish-nair-ai/receipts/main/assets/logo-dark.svg">
    <img src="https://raw.githubusercontent.com/harish-nair-ai/receipts/main/assets/logo.svg" alt="Receipts" width="400">
  </picture>
</p>

<p align="center">
  <em>The verified-done layer for AI coding agents.</em><br>
  <em>Your agent says "Done!" — Receipts checks if that's actually true.</em>
</p>

<p align="center">
  <a href="https://github.com/harish-nair-ai/receipts/actions"><img src="https://img.shields.io/github/actions/workflow/status/harish-nair-ai/receipts/ci.yml?branch=main&label=CI" alt="CI"></a>
  <a href="https://pypi.org/project/agent-receipts/"><img src="https://img.shields.io/pypi/v/agent-receipts" alt="PyPI"></a>
  <a href="https://pypi.org/project/agent-receipts/"><img src="https://img.shields.io/pypi/pyversions/agent-receipts" alt="Python"></a>
  <a href="https://github.com/harish-nair-ai/receipts/blob/main/LICENSE"><img src="https://img.shields.io/github/license/harish-nair-ai/receipts" alt="License"></a>
</p>

---

> **The problem:** AI coding agents confidently claim "Done! All tests pass!" — but ~47% of the time, they haven't actually verified their own claims. You don't find out until three tasks are stacked on top and everything breaks.

## Quick Start

```bash
pip install agent-receipts
```

Install the Claude Code hook:

```bash
receipts install
```

Now, every time Claude Code finishes a task, you get a receipt:

```bash
# Output at the end of a session:
# ┌─────────────────────────────────────────────────────┐
# │  🧾 RECEIPT                          Score: 3/4     │
# ├─────────────────────────────────────────────────────┤
# │  ✅ Created src/utils.py                            │
# │  ✅ Fixed TypeError in parse_config()               │
# │  ⚠️  "Improved error handling" — unverified         │
# │     └─ No evidence of try/except changes found      │
# │  ❌ "All tests pass" — refuted                      │
# │     └─ pytest exit code 1, 2 failures               │
# ├─────────────────────────────────────────────────────┤
# │  Session: abc-123  │  Agent: claude-code            │
# │  Judge: gemini-3-flash  │  1.2s                     │
# └─────────────────────────────────────────────────────┘
```

## How It Works

```text
Agent Session → Claim Extraction → Evidence Matching → Receipt
     │                │                    │              │
     │          Parse "done"         Cross-check      Terminal card
     │          messages for         against actual    with ✅/⚠️/❌
     │          completion claims    transcript        verdicts
```

1. **Claim Extraction** — Parses the agent's final messages for completion claims ("tests pass", "fixed the bug", "feature works").
2. **Evidence Matching** — Cross-checks claims against the actual transcript: Was `pytest` run? What was the exit code? Does the diff touch what was claimed?
3. **The Receipt** — A clear terminal card showing verified ✅ vs unverified ⚠️ claims.
4. **Independent LLM Judge** — For fuzzy claims ("refactored for clarity"), Receipts uses a fast, independent LLM judge (Gemini, OpenAI, or Anthropic) to verify the transcript evidence. Maker and checker never share a model.

## Why Receipts?

| | Manual Review | Observability Tools | **Receipts** |
|---|:---:|:---:|:---:|
| **Zero config** | ❌ | ❌ | ✅ |
| **Catches false "done"** | Sometimes | ❌ | ✅ |
| **No account/API needed** | ✅ | ❌ | ✅ |
| **Works offline** | ✅ | ❌ | ✅ |
| **Evidence-based verdicts** | ❌ | ⚠️ | ✅ |
| **Shareable stats** | ❌ | ❌ | ✅ |

## Configuration

Set your judge provider using environment variables. Receipts auto-detects the first available key.

```bash
export GEMINI_API_KEY="AIza..."     # Fast, cheap judge (recommended)
# OR
export OPENAI_API_KEY="sk-..."      # Uses gpt-4o-mini by default
# OR
export ANTHROPIC_API_KEY="sk-..."   # Uses claude-3-haiku by default
```

To block Claude Code from exiting if there are unverified claims:
```bash
export RECEIPTS_BLOCK=1
```

## Supported Agents

- ✅ **Claude Code** (via hooks, day one)
- 🔜 **Codex CLI** (coming soon)
- 🔜 **Cursor** (coming soon)

## Weekly Stats

Share your agent's real-world accuracy:

```bash
receipts stats

# 🧾 Your agent (last 7 days):
# ├── 23 sessions audited (15 verified perfectly)
# ├── 47 total completion claims
# ├── 14 unverified/refuted claims caught
# ├── 70.2% verification rate
# └── Top unverified claim types:
#     ├── 'test_pass' (8×)
#     └── 'refactored' (6×)
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT — see [LICENSE](LICENSE) for details.
