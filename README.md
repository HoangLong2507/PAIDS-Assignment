
# CO3135 — Programming for Artificial Intelligence and Data Science

This repository contains a small **FastAPI** web app that presents **EDA (Exploratory Data Analysis)** results for:

- **Tabular dataset**: `netflix_titles.csv` (Netflix titles metadata)
- **Text dataset**: `email_spam.csv` (spam vs. ham emails)

The UI is rendered with **Jinja2 templates** and charts are generated with **Plotly**.

## What’s inside

- `app/main.py` — FastAPI entrypoint and routes
- `app/templates/` — HTML templates (landing page + Assignment 1 pages)
- `app/static/` — CSS
- `app/templates/assignment1/EDA/Tabular/eda.py` — tabular EDA logic (Netflix)
- `app/templates/assignment1/EDA/Text/eda.py` — text EDA logic (email spam)

## Requirements

- Python 3.10+ (recommended)

Install dependencies from `requirements.txt`.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the web app

Start the server (from the repository root):

```bash
uvicorn app.main:app --reload
```

Then open:

- Home: http://127.0.0.1:8000/
- Assignment 1 landing: http://127.0.0.1:8000/assignment1
- Text EDA: http://127.0.0.1:8000/assignment1/text
- Tabular EDA: http://127.0.0.1:8000/assignment1/tabular

## Datasets

### Netflix (tabular)

Create or export the tabular dataset `netflix_titles.csv` in this location:  `app/templates/assignment1/EDA/Tabular/netflix_titles.csv`

### Email spam (text)

Create or export the text dataset `email_spam.csv` in this location: `app/templates/assignment1/EDA/Text/email_spam.csv`

## Notes

- Plotly is rendered without bundling Plotly.js in every figure (`include_plotlyjs=False`). The page template is expected to include Plotly.js once globally.
- If you move datasets, update the CSV path or keep a copy in the repository root.

