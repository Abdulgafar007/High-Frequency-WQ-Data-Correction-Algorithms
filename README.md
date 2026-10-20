# High-frequency water-quality preprocessing benchmark

This repository contains the code and manuscript-specific reproducibility
materials for the fixed-rule comparison of preprocessing methods applied to
high-frequency storm-event water-quality sensor data.

## What is public

- reusable analysis code;
- the main analysis notebook;
- figure-reproduction code;
- fixed experiment configuration;
- the four storm-event windows used in the manuscript, once exported by the
  authors.

## What is NOT public

The complete long-term monitoring archive is part of ongoing research and is
not included in this repository.

Only the four event windows used in the manuscript is placed in
`data/manuscript_storm_events/`.

## Repository structure

- `analysis/main_analysis.ipynb` — main computational workflow.
- `figures/figure_reproduction.ipynb` — plotting/chart code separated from analysis.
- `src/wq_preprocessing/` — reusable algorithms, degradation scenarios and metrics.
- `scripts/extract_manuscript_storm_events.py` — private extraction step that
  creates the four public event-window CSVs from the full internal archive.
- `data/manuscript_storm_events/` — only the four manuscript event windows.
- `results/` — generated analysis outputs.
- `config/` — event definitions and reproducibility settings.

