.PHONY: help setup data splits train train-fast eval tables test all clean

RUN := uv run
MODELS_FAST := lightgbm_published lightgbm char_cnn bigru
MODELS := $(MODELS_FAST) distilbert

help:
	@echo "make setup       create the uv environment (Python 3.12, pinned dependencies)"
	@echo "make data        download the raw inputs (or COPY_FROM=<dir> to copy a dated capture)"
	@echo "make splits      build the train/val/test and OOD sets with the original's code"
	@echo "make train       train every model and seed (~1.5 h on an M4; resumes if interrupted)"
	@echo "make train-fast  everything except DistilBERT (~15 min)"
	@echo "make eval        metrics, bootstrap CIs, ablations -> reports/results.json + figures/"
	@echo "make tables      regenerate the README tables from reports/results.json"
	@echo "make test        unit tests (seconds; the real-data tests skip if data is absent)"
	@echo "make all         data -> splits -> train -> eval -> tables -> test"

setup:
	uv sync

data:
	$(RUN) python scripts/fetch_data.py $(if $(COPY_FROM),--copy-from $(COPY_FROM),)

splits:
	$(RUN) python -m phishsmall.splits

train:
	@for model in $(MODELS); do $(RUN) python -m phishsmall.train --model $$model || exit 1; done

train-fast:
	@for model in $(MODELS_FAST); do $(RUN) python -m phishsmall.train --model $$model || exit 1; done

eval:
	$(RUN) python -m phishsmall.evaluation

tables:
	$(RUN) python -m phishsmall.tables

test:
	$(RUN) pytest -q

all: data splits train eval tables test

clean:
	rm -rf .pytest_cache artifacts data/processed
