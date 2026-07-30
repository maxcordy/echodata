# models/model_registry.py

from models.model_base import BaseModel


# MODEL_REGISTRY: list[type[BaseModel]] = []
MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def register_model(cls: type[BaseModel]) -> type[BaseModel]:
    """Decorator to register a model implementation."""
    # Ensure model extends BaseModel
    if not issubclass(cls, BaseModel):
        raise TypeError(f"Cannot register {cls.__name__}: not a subclass of BaseModel")

    # Use model_name as key if defined, otherwise fallback to class name
    name: str = getattr(cls, "model_name", cls.__name__)

    # Ensure each model has a unique name
    if name in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' already registered with {MODEL_REGISTRY[name]}")
    
    MODEL_REGISTRY[name] = cls
    return cls
    # models: dict[str, type[BaseModel]] = {cls.model_name: cls for cls in MODEL_REGISTRY}
