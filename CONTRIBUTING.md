# Contributing to amyAMP

Thanks for your interest in improving amyAMP.

## Before you start

- Generated peptides are **computational designs only**. Do not present model outputs as experimentally confirmed antimicrobial or amyloid agents without wet-lab validation.
- Please open an [issue](https://github.com/anupkprasad/amyAMP/issues) for bugs, documentation gaps, or feature requests before large pull requests.

## Development setup

```bash
git clone https://github.com/anupkprasad/amyAMP.git
cd amyAMP
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Quick sanity check:

```bash
./generate_peptides.sh smoke_test.fasta 10
```

## Pull request workflow

1. Fork the repository and create a feature branch from `main`.
2. Keep changes focused (code, docs, or analysis scripts — not scratch outputs).
3. Do **not** commit:
   - `results/garbage/`
   - local explorer tools (e.g. third-party predictors under `tools/`)
   - large regenerable caches or personal absolute paths
4. Update `README.md` if you change public CLI / script names.
5. Open a PR against `main` with a short description of *why* the change is needed.

## Code style

- Prefer relative paths from the repo root (see `main.py` / `scripts/util.py`).
- Avoid machine-specific absolute paths.
- Match existing Python style in nearby modules.

## Reporting security / data issues

If you discover a problem involving private data, credentials, or unsafe defaults, email **anupkprasad121@gmail.com** instead of opening a public issue.
