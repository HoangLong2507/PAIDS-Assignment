
# CO3135 — Programming for Artificial Intelligence and Data Science

This repository contains a static web application presenting results for the coursework assignments:

- **Assignment 1**: Exploratory Data Analysis (EDA) on Tabular, Text, and Image datasets.
- **Assignment 2**: Machine Learning Pipelines, featuring Classification and Regression models.

The UI is built as a fully **static structure** hosted via GitHub Pages. All data and Plotly charts are dynamically loaded as static JSON files via client-side JavaScript.

## What’s inside

- `docs/` — Contains all static HTML, CSS, JS, and JSON data files.
  - `docs/index.html` — Home landing page.
  - `docs/assignment1/` — **Assignment 1: EDA**
    - `tabular/` — Netflix Titles metadata analysis.
    - `text/` — Email Spam classification EDA.
    - `image/` — CUB Image statistics & quality metrics.
  - `docs/assignment2/` — **Assignment 2: ML Pipeline**
    - `tabular/` — Netflix ML Pipeline (Classification).
    - `text/` — TripAdvisor Hotel Reviews classification.
    - `image/` — Caltech-UCSD Birds-200 classification.
- `export_static.py` — Python script used for generation of static site data.

## Hosting locally

To view the site locally, you can use any static server. For example:

```bash
cd docs
python3 -m http.server 8000
```

Then open your browser to:

- **Home**: http://localhost:8000/
- **Assignment 1**: http://localhost:8000/assignment1/
- **Assignment 2**: http://localhost:8000/assignment2/

## Deployment

The `docs` folder is completely static and is deployed via [GitHub Pages](https://pages.github.com/). 
Go to the repository settings, select **Pages**, and choose the **`docs`** folder on your main branch as the source.

## Notes

- Plotly charts are rendered client-side without a back-end.
- Ensure any large generated static JSON files are tracked or regenerated using the `export_static.py` pipeline if source datasets have changed.

