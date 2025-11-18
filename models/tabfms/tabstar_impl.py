import pandas as pd
from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor
from typing import Any, override
from models.model_base import BaseModel
from models.model_registry import register_model


@register_model
class TabSTARRegImpl(BaseModel):
    """Minimal implementation for TabSTAR."""

    model_name: str = "TabSTAR"
    model_possible_tasks: list[str] = [
        "classification",
        "regression",
    ]

    @override
    def __init__(self, model_task: str, **kwargs: Any) -> None:
        super().__init__(model_task, **kwargs)
        if self.model_task == "classification":
            self._model: Any = TabSTARClassifier()
            self._lbl2idx: dict[str, int]
            self._idx2lbl: dict[int, str]
        elif self.model_task == "regression":
            self._model = TabSTARRegressor()

    @override
    def _fit_impl(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        # Encode labels to integers
        # Model fails when classes don't start from 0
        if self.model_task == "classification":
            classes: list[str] = list(pd.Categorical(y).categories)
            self._lbl2idx = {lbl: i for i, lbl in enumerate(classes)}
            self._idx2lbl = {i: lbl for lbl, i in self._lbl2idx.items()}
            y: pd.DataFrame = y.replace(self._lbl2idx).astype("int64")
        self._model.fit(X, pd.Series(y))

    @override
    def _predict_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        preds: pd.DataFrame = pd.DataFrame(self._model.predict(X))
        if self.model_task == "classification":
            preds = preds.replace(self._idx2lbl)
        return preds
