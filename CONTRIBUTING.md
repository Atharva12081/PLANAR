# Contributing to PLANAR

## Development Setup

1. Use Python `3.11.9`.
2. Create an environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

## Workflow

1. Create a feature branch.
2. Make focused, testable changes.
3. Run:
   ```bash
   pytest -q
   ```
4. Update docs/configs if behavior changes.
5. Open a pull request with:
   - motivation,
   - implementation summary,
   - validation evidence (metrics, plots, or tests).

## Coding Standards

- Prefer explicit type hints.
- Use Google-style docstrings for public functions.
- Keep scientific assumptions documented in code and README.

## Reproducibility Requirements

- New experiments must specify seed and config file.
- Report key metrics and dataset size used.
- Include artifacts path for reproducibility when possible.
