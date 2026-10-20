# Manuscript reproducibility data

The complete monitoring archive is not included because it supports ongoing
research.

This directory should contain only the four event-window files used in the
published analysis:

- `storm_01.csv`
- `storm_02.csv`
- `storm_03.csv`
- `storm_04.csv`

Create them locally by running:

`scripts/extract_manuscript_storm_events.py`

Expected public columns:

- `datetime_et`
- `storm_id`
- `site`
- `Turbidity`
- `EC`
- `DO`
- `Temp`
- `Radar`

The extraction script intentionally exports only these columns and event windows.
