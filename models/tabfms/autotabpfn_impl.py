import pandas as pd
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
    AutoTabPFNClassifier,
    AutoTabPFNRegressor,
)
from typing import Any, override
from models.model_base import BaseModel
from models.model_registry import register_model


@register_model
class AutoTabPFNImpl(BaseModel):
    """Minimal implementation for AutoTabPFN."""

    model_name: str = "AutoTabPFN"
    model_possible_tasks: list[str] = [
        "classification",
        "regression",
    ]

    @override
    def __init__(self, model_task: str, **kwargs: Any) -> None:
        super().__init__(model_task, **kwargs)
        if self.model_task == "classification":
            self._model = AutoTabPFNClassifier()
        elif self.model_task == "regression":
            self._model = AutoTabPFNRegressor()

    @override
    def _fit_impl(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        self._model.fit(X, y)

    @override
    def _predict_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._model.predict(X)
