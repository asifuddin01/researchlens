# Every target either prints a number or changes a file. Nothing here is a
# wrapper around a thing you would rather type yourself.

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help install corpus test lint eval ablation up down clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "};{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

install: ## create the venv and install everything
	python3 -m venv $(VENV)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q -e ".[dev]"
	@echo "installed. next: make corpus"

corpus: ## download the evaluation corpus (~25 open-access papers)
	$(PY) scripts/fetch_corpus.py

# Point at any folder of PDFs. Parses once into data/index; later runs reuse it.
ingest: ## parse a folder of papers into the library cache (DIR=/path/to/papers)
	$(PY) -m researchlens.ingest.library $(or $(DIR),data/pdfs)

test: ## run the unit tests
	$(PY) -m pytest -q

lint:
	$(VENV)/bin/ruff check .

label: ## browse the corpus to write ground truth (ARGS="--papers")
	$(PY) scripts/label.py $(ARGS)

search: ## query the corpus from the command line (Q="your question")
	$(PY) scripts/search.py $(Q)

eval: ## score the baseline configuration
	$(PY) -m eval.run --config "dense only"

# The README's ablation table is written by this target and never by hand.
# A table that can be hand-edited is a table that will drift from the code,
# and at that point it is decoration rather than evidence.
ablation: ## regenerate the ablation table into the README
	$(PY) -m eval.run --ablation | tee /tmp/rl-ablation.md
	$(PY) scripts/write_ablation.py /tmp/rl-ablation.md README.md
	@echo "README.md updated."

up: ## run the full local system (no API key needed)
	docker compose up --build

down:
	docker compose down

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache data/index
	find . -name __pycache__ -type d -exec rm -rf {} +
