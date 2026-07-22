# SAGE Bewley

A computational lecture in Julia on wellbeing in heterogeneous-agent macroeconomics.
One page, four models: the neoclassical growth model, a real business cycle model, a
SAGE-RBC, and the SAGE-Bewley model, in which households facing uninsurable income
risk divide their time between paid work and contributing to a shared social fabric.
The lecture ends with a financed activation policy that raises consumption while the
social fabric erodes, and with the same experiment run across seven country
calibrations.

Every figure on the page is produced by the code shown next to it. Every parameter is
either sourced from the literature or calibrated to a stated data target.

Live site: https://conway1521.github.io/sage-bewley/

## What this repository is, and is not

This repository holds the lecture and nothing else: the page source (`index.md`), a
vendored copy of the solver (`sage_engine.jl`) so the build is self-contained, the
pinned Julia and Python environments, and the deploy script.

The research programme behind the lecture lives in a separate repository,
[sage_macro](https://github.com/conway1521/sage_macro): the maintained solver, the
working papers, the calibration pipeline and data files, and the country evidence.
The lecture is the readable front door; the research repo is the workshop.

## Build

The book is the classic Sphinx-based Jupyter Book (0.15) with a Julia kernel.

```bash
# Python build tool (needs Python 3.11; 3.13 removed a module Sphinx 5 requires)
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Julia kernel, registered against this repository's own project (run once)
julia --project=. -e 'using Pkg; Pkg.instantiate();
                      using IJulia; IJulia.installkernel("Julia SAGE", "--project="*pwd())'

# build (executes every Julia cell; delete _build first to force a clean run)
.venv/bin/jupyter-book build .
```

The result is static HTML in `_build/html`.

## Gallery Notebook

`make_gallery.py` rewrites the MyST-specific syntax (citation roles, exercise
directives) as plain Markdown, and jupytext turns the result into
`sage_bewley_wellbeing.ipynb`, the executed, self-contained notebook prepared for
QuantEcon's notebook gallery. The notebook fetches `sage_engine.jl` from this
repository if it is not sitting next to it, so it runs anywhere with the Julia
kernel installed.

## Deploy

`bash deploy_site.sh` publishes the built site to this repository's `gh-pages`
branch, which GitHub Pages serves. Details and the one-time Pages setting are in
`DEPLOY.md`.

## Author

Alessandro Conway. The lecture develops the model of my master's thesis, "Wellbeing
and Macroeconomics: a SAGE approach" (Sciences Po, 2020), which operationalises the
SAGE wellbeing framework of Snower and Lima de Miranda in a Bewley-Aiyagari economy.
