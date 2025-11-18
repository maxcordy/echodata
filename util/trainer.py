# util/trainer.py

import tempfile
import pandas as pd
import gradio as gr

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    root_mean_squared_error,
)

from models.model_base import BaseModel
from models.model_registry import MODEL_REGISTRY


def evaluation_scores(
    task: str, y_true: pd.DataFrame, y_pred: pd.DataFrame
) -> dict[str, float]:
    """Generate prediction scores."""
    y_pred = y_pred.astype(y_true.dtype)
    scores: dict[str, float] = {}
    if task == "classification":
        scores["Accuracy"] = float(accuracy_score(y_true, y_pred))
        scores["Precision (Macro)"] = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        scores["Recall (Macro)"] = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        scores["F1 (Macro)"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        scores["Precision (Micro)"] = float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        )
        scores["Recall (Micro)"] = float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        )
        scores["F1 (Micro)"] = float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        )
    else:
        scores["MAE"] = float(mean_absolute_error(y_true, y_pred))
        scores["MSE"] = float(mean_squared_error(y_true, y_pred))
        scores["RMSE"] = float(root_mean_squared_error(y_true, y_pred))
        scores["R2"] = float(r2_score(y_true, y_pred))
        scores["Explained Variance"] = float(explained_variance_score(y_true, y_pred))
    return scores


def load_and_validate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_target_col: str,
    test_target_col: str,
    target_empty: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Load and validate dataframes."""
    # Check targets are valid
    if train_target_col not in train_df.columns:
        raise gr.Error("Please select a valid training target column.")
    if test_target_col != "None" and test_target_col not in test_df.columns:
        raise gr.Error("Please select a valid test target column.")

    # Split train
    y_train: pd.DataFrame = train_df[train_target_col].copy()
    X_train: pd.DataFrame = train_df.drop(columns=[train_target_col]).copy()

    # Split test
    if test_target_col != "None":
        y_test: pd.DataFrame | None = test_df[test_target_col].copy()
        X_test: pd.DataFrame = test_df.drop(columns=[test_target_col]).copy()
    else:
        y_test: pd.Series | None = None
        X_test: pd.DataFrame = test_df.copy()

    # If target empty, discard it
    if target_empty:
        y_test: pd.DataFrame | None = None

    # Validate shape consistency
    if X_train.shape[1] + 1 == X_test.shape[1] and test_target_col == "None":
        raise gr.Error(
            "Test target might not be None: mismatched column numbers by one."
        )
    if X_train.shape[1] == X_test.shape[1] + 1 and test_target_col != "None":
        raise gr.Error("Test target might be None: mismatched column numbers by one.")
    if X_train.shape[1] != X_test.shape[1]:
        raise gr.Error(
            "Please ensure training and test have an equal number of columns."
        )

    return X_train, y_train, X_test, y_test


def train_and_predict(
    model_name: str,
    model_task: str,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[BaseModel, pd.DataFrame]:
    """Train and predict using the model"""
    # Get model
    model_cls: type[BaseModel] = MODEL_REGISTRY[model_name]
    model: BaseModel = model_cls(model_task)

    # Fit + predict
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred


def run_training_and_predict(
    model_name: str,
    model_task: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_target_col: str,
    test_target_col: str,
    target_empty: bool,
) -> tuple[pd.DataFrame, str, str, pd.DataFrame]:
    """Run model predictions and generate output csv."""
    # Load and validate data
    X_train, y_train, X_test, y_test = load_and_validate(
        train_df, test_df, train_target_col, test_target_col, target_empty
    )

    # Get model predictions
    model, y_pred = train_and_predict(
        model_name,
        model_task,
        X_train,
        y_train,
        X_test,
    )

    # Build prediction df
    preds_df = test_df.copy()
    # Place predictions adjacent to test value or overwrite or in rightmost column
    # Name predictions column based on target column
    pred_insert_index: int = test_df.shape[1]
    pred_col_name: str = "prediction"
    if test_target_col != "None":
        pred_col_name: str = f"pred_{test_target_col}"
        if target_empty:
            pred_insert_index = test_df.columns.get_loc(test_target_col)
            preds_df.drop(test_target_col, axis=1, inplace=True)
        else:
            pred_insert_index = test_df.columns.get_loc(test_target_col) + 1
    preds_df.insert(pred_insert_index, pred_col_name, y_pred, allow_duplicates=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    preds_df.to_csv(tmp.name, index=False)

    # Construct model and data information
    info = (
        f"Shapes:\n"
        f"> X_train: {X_train.shape}\n> y_train: {y_train.shape}\n"
        f"> X_test: {X_test.shape}\n> y_test: {y_test.shape if y_test is not None else 'N/A'}\n"
        f"Model: {model.model_name}\n"
        f"> model_task: {model.model_task}\n"
    )

    # Construct score table
    scores: pd.DataFrame = pd.DataFrame()
    if y_test is not None:
        scores = pd.DataFrame([evaluation_scores(model_task, y_test, y_pred)])

    return (preds_df.head(10), tmp.name, info, scores)


def run_leaderboard(
    model_names: list[str],
    model_task: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_target_col: str,
    test_target_col: str,
    target_empty: bool,
) -> pd.DataFrame:
    """Run models and compare results."""
    # Load and validate data
    X_train, y_train, X_test, y_test = load_and_validate(
        train_df, test_df, train_target_col, test_target_col, target_empty
    )

    # Ensure we have y_test
    if y_test is None:
        raise gr.Error("Require testing target for Leaderboard.")

    leaderboard_rows = []
    for model_name in model_names:
        try:
            # Get model predictions
            model, y_pred = train_and_predict(
                model_name,
                model_task,
                X_train,
                y_train,
                X_test,
            )
            # Construct score table
            scores = evaluation_scores(model_task, y_test, y_pred)
            row = {"model_name": model_name, **scores}
            leaderboard_rows.append(row)
            gr.Info(f"{model_name} done.")

        except Exception:
            # Add row with model_name indicating failure
            leaderboard_rows.append({"model_name": model_name})
            gr.Info(f"{model_name} failed.")

    # Add rows of data together and return
    leaderboard_df = pd.DataFrame(leaderboard_rows)
    return leaderboard_df
