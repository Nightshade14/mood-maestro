This folder contains helper scripts used by the data pipeline.

models.py

Usage
<!-- scripts/README.md -->
# set_db (scripts)

An easy-to-run, one-off pipeline that generates semantic embeddings for tracks and uploads documents to MongoDB.
This folder intentionally contains ops-style scripts (quick jobs). The main app logic should live under `src/`.

## TL;DR
- Run from the repository root:

```bash
python -m scripts.set_db
```

Or, if you prefer to run the file directly (make sure the repo root is on PYTHONPATH):

```bash
PYTHONPATH=. python scripts/set_db.py
```

## What this script does
- Loads a CSV dataset (default: `dataset/final_dataset.csv`).
- Cleans and feature-engineers audio / metadata fields.
- Generates text embeddings with a sentence-transformers model and concatenates normalized numeric features.
- Prepares MongoDB documents (using the Pydantic models in `models.py`) and uploads them to a collection.

## Files in this folder
- `set_db.py` — the standalone pipeline script you run.
- `models.py` — Pydantic models for `TrackAudioFeatures` and `TrackDocument` used for validation and shaping.

## Required environment variables
- `MONGO_URI` (required): MongoDB connection string used to upload documents.
- `LOG_LEVEL` (optional): logging level (DEBUG / INFO / WARNING). Default: `INFO`.

Note: `DATA_FILE_PATH` and `MODEL_NAME` are currently set inside the script; if you want them configurable,
I can make the script read them from env vars or CLI flags.

## Quick dry-run / testing tips
- Run the script on a small subset to validate everything without waiting for the full dataset:

```python
# inside scripts/set_db.py, after df is loaded
df = df.head(1000)  # process first 1k rows only
```

- Or set `LOG_LEVEL=DEBUG` to get more verbose logs while testing:

```bash
LOG_LEVEL=DEBUG python -m scripts.set_db
```

## Troubleshooting
- ModuleNotFoundError when running directly?
	- Make sure you run from the repo root. Use `python -m scripts.set_db` to run in package context.
	- Or run with `PYTHONPATH=. python scripts/set_db.py`.
- IndexError: index X out of bounds for axis 0
	- This happens if rows were dropped during preprocessing and the script attempted to access embeddings by original DataFrame index.
	- The script resets the DataFrame index after preprocessing and iterates by positional index to avoid this — if you see the error, ensure you are running the latest script.

## Style & layout notes
- Keep one-off/ops scripts inside `scripts/` to prevent cluttering `src/` with non-core code.
- When a script matures into application logic, move it into `src/` as importable functions and add a small CLI wrapper or package entry point for repeatable execution.

## Want me to improve this further?
- I can add a `--limit` / `--dry-run` CLI flag to the script, convert `DATA_FILE_PATH` and `MODEL_NAME` into configurable env vars/flags,
	or add a small test that runs the pipeline against a synthetic DataFrame. Tell me which and I'll implement it.

