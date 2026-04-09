
# CO3135 — Programming for Artificial Intelligence and Data Science

This repository contains a static web application that presents **EDA (Exploratory Data Analysis)** results for:

- **Tabular dataset**: `netflix_titles.csv` (Netflix titles metadata)
- **Text dataset**: `email_spam.csv` (Email spam classification)
- **Image dataset**: CUB / Image statistics (bounding boxes, masks, quality metrics)

The UI has been migrated from a FastAPI back-end to a fully **static structure** that can be hosted directly via GitHub Pages or any static HTTP server. All data and Plotly charts are dynamically loaded as static JSON files via client-side JavaScript.

## What’s inside

- `docs/` — Contains all static HTML, CSS, JS, and JSON data files ready for deployment.
  - `docs/index.html` — Home landing page
  - `docs/assignment1/index.html` — Assignment 1 landing page
  - `docs/assignment1/text/index.html` — Text EDA results
  - `docs/assignment1/tabular/index.html` — Tabular EDA results
  - `docs/assignment1/image/index.html` — Image EDA results
  - `docs/static/` — CSS styles, fonts, and global assets
- `export_static.py` — Python script used for generation of static site data (if needed for rebuilding).

## Hosting locally

To view the site locally, you do not need FastAPI. You can use any static server. For example, using Python's built-in `http.server`:

```bash
cd docs
python3 -m http.server 8000
```

Then open your browser to:

- Home: http://localhost:8000/
- Assignment 1 landing: http://localhost:8000/assignment1/
- Text EDA: http://localhost:8000/assignment1/text/
- Tabular EDA: http://localhost:8000/assignment1/tabular/
- Image EDA: http://localhost:8000/assignment1/image/

## Deployment

The `docs` folder is completely static and can be deployed straightforwardly via [GitHub Pages](https://pages.github.com/). 
Go to the repository settings, select **Pages**, and choose the **`docs`** folder on your main branch as the source.

## Notes

- Plotly charts are generated client-side by fetching JSON data and rendering them dynamically without a heavy backend.
- Ensure any large generated static JSON files are tracked or regenerated using the `export_static.py` pipeline if source datasets have changed.
