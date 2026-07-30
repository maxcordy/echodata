# models/model_functional.py

from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, Self


class BaseModel(ABC):
    model_name: str
    model_possible_tasks: list[str]

    def __init__(self, model_task: str, **kwargs: Any) -> None:
        if model_task not in self.model_possible_tasks:
            raise ValueError(
                f"Invalid task '{model_task}' for model '{self.model_name}'. "
                + f"Allowed: {self.model_possible_tasks}"
            )
        self.model_task: str = model_task
        self._model: Any = None
        self._is_fit = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> Self:
        """Public fit method."""
        self._fit_impl(X, y)
        self._is_fit: bool = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Public predict method."""
        if not self._is_fit:
            raise RuntimeError(f"Model {self.model_name} not fitted. Call fit() first.")
        return self._predict_impl(X)

    @abstractmethod
    def _fit_impl(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Subclasses implement the fit logic."""
        ...

    @abstractmethod
    def _predict_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        """Subclasses implement predict logic."""
        ...
