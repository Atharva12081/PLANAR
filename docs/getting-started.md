# Getting Started

## Python Version

Use `Python 3.11.9`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

## Run

```bash
planar run --config configs/default.yaml
planar reproduce --config configs/reproduce.yaml
```
