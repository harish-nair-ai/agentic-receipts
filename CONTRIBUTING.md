# Contributing to Receipts

Thanks for wanting to contribute! Here's how to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/harish-nair-ai/receipts.git
cd receipts

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
pytest --cov=receipts  # with coverage
```

## Code Quality

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check .        # lint
ruff format .       # format
```

## Adding a New Agent Adapter

Want to add support for a new coding agent (Codex, Cursor, etc.)?

1. Create a new file in `src/receipts/adapters/` (e.g., `codex_cli.py`)
2. Implement the `TranscriptAdapter` base class from `adapters/base.py`
3. Register it in `transcript.py`
4. Add tests in `tests/`
5. Update the README

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR
- Add tests for new functionality
- Update documentation if needed
- Follow the existing code style (Ruff will enforce it)

## Reporting Issues

Found a bug or have a feature request? [Open an issue](https://github.com/harish-nair-ai/receipts/issues/new).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
