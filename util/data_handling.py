# util/data_handling.py

import gradio as gr
import pandas as pd
from gradio.utils import NamedString


def read_csv_to_df(file: NamedString | None, add_header: bool) -> pd.DataFrame | None:
    """Reads a CSV and returns a a df."""
    if file is None:
        return None
    if not add_header:
        return pd.read_csv(file)

    # Add header row to file
    df: pd.DataFrame = pd.read_csv(file, header=None)
    df.columns = [f"col{i}" for i in range(df.shape[1])]
    return df


def preview_data(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Preview rows from df."""
    if df is None:
        return None
    return df.head(5)


def infer_train_columns(df: pd.DataFrame | None) -> gr.Dropdown:
    """Infer column labels from df."""
    if df is None:
        return gr.Dropdown(choices=[], value=None)

    cols = list(map(str, df.columns))
    default_label = cols[-1] if cols else None
    return gr.Dropdown(choices=cols, value=default_label)


def infer_test_columns(df: pd.DataFrame | None) -> gr.Dropdown:
    """Infer column labels from df."""
    if df is None:
        return gr.Dropdown(choices=[], value=None)

    cols = ["None"] + list(map(str, df.columns))
    default_label = cols[-1] if cols else None
    return gr.Dropdown(choices=cols, value=default_label)


def analyze_df(df: pd.DataFrame | None) -> str:
    """Gather and present meta-data about df."""
    if df is None:
        return ""

    shape_info: str = f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"
    column_info: str = "\n".join([f"> {col}: {df[col].dtype}" for col in df.columns])

    info: str = f"{shape_info}\nColumn info:\n{column_info}"
    return info
