"""
Model Factory for dynamic instantiation of scikit-learn and XGBoost classifiers.

This module provides a centralized way to create machine learning models
by name, ensuring consistent configuration and easy extension.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Type, List
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Set up logging to track which models are created
logger = logging.getLogger(__name__)


class ModelFactory:
    """
    A factory class that creates machine learning model instances based on a name.
    """

    _models: Dict[str, Type[Any]] = {
        "svc": SVC,
        "linear_svc": LinearSVC,
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
        "mlp": MLPClassifier,
        "knn": KNeighborsClassifier,
    }

    # KNN is NOT iterative (it's a lazy learner)
    _iterative_models: List[str] = ["xgboost", "mlp"]

    @classmethod
    def get_model(cls, model_name: str, **kwargs: Any) -> Any:
        model_name = model_name.lower()
        if model_name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(
                f"Model '{model_name}' not recognized. Available models: {available}"
            )

        for key, value in kwargs.items():
            if value == "null" or value == "None":
                kwargs[key] = None

        model_class = cls._models[model_name]
        logger.info(
            f"Instantiating model: {model_name} with parameters: {kwargs}"
        )
        return model_class(**kwargs)

    @classmethod
    def is_iterative(cls, model_name: str) -> bool:
        model_name = model_name.lower()
        is_iter = model_name in cls._iterative_models
        logger.info(f"Model '{model_name}' is iterative: {is_iter}")
        return is_iter

    @classmethod
    def get_fit_params(
        cls,
        model_name: str,
        X_val: Any,
        y_val: Any,
        early_stopping: bool = False,
        patience: int = 10,
        max_epochs: int = 100,
    ) -> Dict[str, Any]:
        model_name = model_name.lower()
        fit_params: Dict[str, Any] = {}

        if cls.is_iterative(model_name):
            logger.info(f"🔄 Detected iterative model: {model_name}.")
            if model_name == "mlp":
                if early_stopping:
                    fit_params["n_iter_no_change"] = patience
            else:
                fit_params["eval_set"] = [(X_val, y_val)]
                fit_params["verbose"] = False
                if early_stopping:
                    fit_params["early_stopping_rounds"] = patience
        else:
            logger.info(f"⚡ Detected batch/lazy model: {model_name}.")

        return fit_params
