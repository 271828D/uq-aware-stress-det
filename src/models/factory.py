"""
Model Factory for dynamic instantiation of scikit-learn and XGBoost classifiers.

This module provides a centralized way to create machine learning models
by name, ensuring consistent configuration and easy extension.
"""

from __future__ import annotations
import inspect
import logging
from typing import Any, Dict, Type, List
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Set up logging to track which models are created
logger = logging.getLogger(__name__)


class ModelFactory:
    """
    A factory class that creates machine learning model instances based on a name.

    Think of this as a menu: you order by name (e.g., 'svc'), and the kitchen
    (this class) prepares the exact dish (model) with the ingredients
    (parameters) you asked for.
    """

    # A dictionary mapping simple names to the actual model "blueprints" (classes)
    # This is the "menu" of available models.
    _models: Dict[str, Type[Any]] = {
        "svc": SVC,
        "linear_svc": LinearSVC,
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
        "mlp": MLPClassifier,
        "knn": KNeighborsClassifier,
    }

    # A list of models that support iterative training (like epochs/early stopping)
    # This helps the trainer know which models can be trained step-by-step.
    # Note: MLP is iterative but uses internal validation for early stopping.
    # We handle it here to allow max_iter control, but disable internal early_stopping
    # in the config to prevent data leakage.
    _iterative_models: List[str] = ["xgboost", "mlp"]

    # Models that REQUIRE eval_set when early_stopping is enabled
    _requires_val: List[str] = ["xgboost"]

    @classmethod
    def requires_validation(cls, model_name: str) -> bool:
        """
        Check if the model requires a validation set for early stopping.

        Args:
            model_name: Name of the model.

        Returns:
            True if the model requires eval_set for early stopping.
        """
        return model_name.lower() in cls._requires_val

    @classmethod
    def get_model(cls, model_name: str, **kwargs: Any) -> Any:
        """
        Create and return a model instance by name.

        Parameters
        ----------
        model_name : str
            The name of the model to create (e.g., 'svc', 'random_forest').
            Must match one of the keys in the _models dictionary.
        **kwargs : Any
            Optional settings (hyperparameters) to pass to the model
            (e.g., C=1.0, random_state=42).

        Returns
        -------
        Any
            An instance of the requested machine learning model, ready to be trained.

        Raises
        ------
        ValueError
            If the requested model_name is not in our menu (_models).
        """

        model_name = model_name.lower()

        if model_name not in cls._models:
            available = ", ".join(cls._models.keys())
            raise ValueError(
                f"Model '{model_name}' not recognized. Available models: {available}"
            )

        for key, value in kwargs.items():
            if value == "null" or value == "None":
                kwargs[key] = None

        # Convert string representation of lists to actual lists for MLP
        if model_name == "mlp" and "hidden_layer_sizes" in kwargs:
            val = kwargs["hidden_layer_sizes"]
            if isinstance(val, str):
                import ast

                try:
                    kwargs["hidden_layer_sizes"] = ast.literal_eval(val)
                    logger.info(
                        f"Converted hidden_layer_sizes string to list: {kwargs['hidden_layer_sizes']}"  # noqa
                    )
                except (ValueError, SyntaxError):
                    logger.warning(
                        f"Failed to parse hidden_layer_sizes '{val}'. Passing as-is."
                    )

        model_class = cls._models[model_name]

        # Filter kwargs to only include parameters the model accepts
        sig = inspect.signature(model_class.__init__)
        params = sig.parameters

        # Check if model accepts **kwargs (e.g., XGBoost)
        accepts_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

        if not accepts_var_keyword:
            valid_keys = set(params.keys()) - {"self"}
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k in valid_keys
            }
            removed = set(kwargs.keys()) - set(filtered_kwargs.keys())
            if removed:
                logger.warning(
                    f"Model '{model_name}' does not accept: {removed}. "
                    f"These parameters were ignored."
                )
            kwargs = filtered_kwargs

        logger.info(
            f"Instantiating model: {model_name} with parameters: {kwargs}"
        )
        model = model_class(**kwargs)

        return model

    @classmethod
    def is_iterative(cls, model_name: str) -> bool:
        """
        Check if a model supports iterative training (epochs, early stopping).

        Parameters
        ----------
        model_name : str
            The name of the model to check.

        Returns
        -------
        bool
            True if the model supports iterative training, False otherwise.
        """
        # Convert to lowercase for consistent comparison
        model_name = model_name.lower()

        # Check if the model is in our list of iterative models
        is_iter = model_name in cls._iterative_models

        # Log the result for transparency
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
        """
        Get the correct training parameters (fit_params) for a specific model.

        This replaces the manual logic in trainer.py with a centralized decision.

        Parameters
        ----------
        model_name : str
            The name of the model being trained.
        X_val : Any
            Validation features (used for eval_set).
        y_val : Any
            Validation labels (used for eval_set).
        early_stopping : bool
            Whether to enable early stopping.
        patience : int
            Number of rounds to wait for improvement before stopping.
        max_epochs : int
            Maximum number of training rounds.

        Returns
        -------
        Dict[str, Any]
            A dictionary of parameters to pass to model.fit().
        """
        # Convert to lowercase for consistent comparison
        model_name = model_name.lower()

        # Start with empty parameters (default for most models like SVC, RandomForest)
        fit_params: Dict[str, Any] = {}

        # Check if this is an iterative model (like XGBoost or MLP)
        if cls.is_iterative(model_name):
            logger.info(
                f"🔄 Detected iterative model: {model_name}. "
                f"Configuring training parameters."
            )

            # MLPClassifier handles early stopping internally via validation_fraction.
            # To prevent data leakage, we DO NOT pass eval_set to MLP.
            # Instead, we rely on max_iter and alpha (regularization) from config.
            if model_name == "mlp":
                logger.info(
                    "MLP detected: Using internal max_iter. "
                    "External eval_set ignored to prevent leakage."
                )
                # No eval_set for MLP. Early stopping is disabled in config.
            else:
                # XGBoost logic (external eval_set)
                fit_params["eval_set"] = [(X_val, y_val)]
                fit_params["verbose"] = False

                if early_stopping:
                    fit_params["early_stopping_rounds"] = patience
                    logger.info(
                        f"✅ XGBoost Early stopping enabled with patience={patience}"
                    )

        else:
            # For non-iterative models (SVC, RandomForest), no special params needed
            logger.info(
                f"⚡ Detected batch model: {model_name}. "
                f"Using default training parameters."
            )

        return fit_params
