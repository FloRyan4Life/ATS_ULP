# ATS ULP V1

This repository contains a small homing model with a Python package layout and a matching LaTeX documentation scaffold.

## Structure

- `homing_core/`: core Python package with models, geometry, pairing logic, visualization, and the demo entrypoint
- `homing_v4.py`: compatibility wrapper that preserves the previous script entrypoint
- `docs/latex/`: LaTeX documentation project
- `docs/screenshots/`: image assets for the documentation

## Run the demo

```bash
python homing_v4.py
```

You can also run the package directly:

```bash
python -m homing_core
```

## Build the documentation

```bash
cd docs/latex
latexmk main.tex
```

The LaTeX project is configured with `pdflatex -shell-escape` via `.latexmkrc`.