import pandas as pd
from autogluon.tabular import TabularPredictor
from typing import Any, override
from models.model_base import BaseModel
from models.model_registry import register_model


@register_model
class MitraImpl(BaseModel):
    """Minimal implementation for Mitra."""

    model_name: str = "Mitra"
    model_possible_tasks: list[str] = [
        "classification",
        "regression",
    ]

    @override
    def __init__(self, model_task: str, **kwargs: Any) -> None:
        super().__init__(model_task, **kwargs)
        # Model detects task type automatically
        self._model = TabularPredictor(label="target")

    @override
    def _fit_impl(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        df = X.copy()
        df["target"] = y

        self._model.fit(df, hyperparameters={"MITRA": {"fine_tune": False}})

    @override
    def _predict_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.predict(X)
