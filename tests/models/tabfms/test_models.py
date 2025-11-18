import pytest
import pandas as pd
import numpy as np

import models.tabfms  # import all models
from models.model_base import BaseModel
from models.model_registry import MODEL_REGISTRY

# Less computationally expensive subset
models_simple: dict[str, type[BaseModel]] = {}
for model_name, model_cls in MODEL_REGISTRY.items():
    if model_name in ("TabPFN", "TabICL"):
        models_simple[model_name] = model_cls

# Create some dummy data for testing
X_dummy = pd.DataFrame({"f1": [0.1, 0.2, 0.3, 0.4], "f2": [1.0, 0.9, 0.8, 0.7]})
y_class = pd.DataFrame([0, 1, 0, 1])  # Binary classification
y_reg = pd.DataFrame([10.0, 20.0, 15.0, 25.0])  # Regression targets


@pytest.mark.parametrize("model_name, model_cls", MODEL_REGISTRY.items())
def test_registered_models(model_name, model_cls) -> None:
    # Check that model_cls can be instantiated for each possible task
    for task in model_cls.model_possible_tasks:
        model = model_cls(task)

        # Fit the model
        model.fit(X_dummy, y_class if task == "classification" else y_reg)
        preds = model.predict(X_dummy)
        assert len(preds) == len(X_dummy)
