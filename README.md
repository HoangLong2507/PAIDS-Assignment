
# CO3135 — Programming for Artificial Intelligence and Data Science

This repository contains a small **FastAPI** web app that presents **EDA (Exploratory Data Analysis)** results for:

- **Tabular dataset**: `netflix_titles.csv` (Netflix titles metadata)
 - **Image dataset**: `image_statistic.csv` + related CSVs (image / bbox / mask / quality stats)
- 
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

### Image (image statistics)

Place the image-related CSV files in: `app/templates/assignment1/EDA/Image/csv/`.
Common files used by the Image EDA include:

- `image_statistic.csv` (primary image metadata)
- `bbox_statistic.csv` (bounding box records)
- `mask_statistics.csv`, `quality_metrics.csv`, `pixel_distribution.csv`, `shape_distribution.csv`
- `segmentation_boundary_quality_metric.csv`, `spatial_distribution_analytics.csv`, `tsne_embeddings.csv`, `similarity_matrix.csv`

Optional directories read by the Image EDA template:

- `app/templates/assignment1/EDA/Image/parts/` — files representing part locations
- `app/templates/assignment1/EDA/Image/attributes/` — binary attribute files
 - `app/templates/assignment1/EDA/Image/eda.py` — image EDA logic (CUB / image statistics)
 
## Notes

- Plotly is rendered without bundling Plotly.js in every figure (`include_plotlyjs=False`). The page template is expected to include Plotly.js once globally.
- If you move datasets, update the CSV path or keep a copy in the repository root.

