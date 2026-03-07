PYTHON ?= python

.PHONY: install test run reproduce report

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e ".[dev]"

test:
	pytest -q

run:
	planar run --config configs/default.yaml

reproduce:
	planar reproduce --config configs/reproduce.yaml

report:
	planar report --config configs/default.yaml
