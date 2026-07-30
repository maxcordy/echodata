Echodata project aims to showcase the capabilities of Tabular Foundation Models.

## Installation

```bash
uv sync
uv pip install "autogluon.tabular[mitra]"
```

## Running the App

```bash
uv run main.py
```

## Using the App

1. Upload training and testing CSV
2. Confirm target columns
3. Modify CSV if needed
  - Add header rows
  - Choose to ignore test target column
4. Select a model/data type
  - Classification
  - Regression
5. Select action
  - Fit & Predict (Generate CSV of predictions)
  - Model Comparison (Leaderboard of metrics)
6. Select model(s) to run
7. Click to run
