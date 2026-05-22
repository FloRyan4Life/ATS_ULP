# ATS ULP V1

This repository contains a small homing model with a Python package layout and a matching LaTeX documentation scaffold.

## Structure

- `homing_code/`: core Python package with models, geometry, pairing logic, visualization, and the demo entrypoint
- `main.py`: top-level entrypoint for the demo
- `docs/latex/`: LaTeX documentation project
- `docs/screenshots/`: image assets for the documentation

## Run the demo

```bash
python main.py
```

You can also run the package directly:

```bash
python -m homing_code
```

## Build the documentation

```bash
cd docs/latex
latexmk main.tex
```

The LaTeX project is configured with `pdflatex -shell-escape` via `.latexmkrc`.