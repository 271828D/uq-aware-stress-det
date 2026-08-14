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
from xgboost import XGBClassifier

# Set up logging to track which models are created
logger = logging.getLogger(__name__)


class ModelFactory:
    """
    A factory class that creates machine learning model instances based on a name.
    
    Think of this as a menu: you order by name (e.g., 'svc'), and the kitchen 
    (this class) prepares the exact dish (model) with the ingredients (parameters) you asked for.
    """

    # A dictionary mapping simple names to the actual model "blueprints" (classes)
    # This is the "menu" of available models.
    _models: Dict[str, Type[Any]] = {
        "svc": SVC,
        "linear_svc": LinearSVC,
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
    }

    # A list of models that support iterative training (like epochs/early stopping)
    # This helps the trainer know which models can be trained step-by-step.
    _iterative_models: List[str] = ["xgboost"]

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
        # Convert the name to lowercase to avoid errors if user types 'SVC' instead of 'svc'
        model_name = model_name.lower()

        # Check if the requested model exists in our menu
        if model_name not in cls._models:
            # If not found, list available options to help the user fix the mistake
            available = ", ".join(cls._models.keys())
            raise ValueError(
                f"Model '{model_name}' is not recognized. Available models: {available}"
            )

        # Clean up any "null" or "None" string values from the config
        for key, value in kwargs.items():
            if value == "null" or value == "None":
                kwargs[key] = None
                
        # Get the "blueprint" (class) for the requested model
        model_class = cls._models[model_name]

        # Log which model is being created for debugging/tracking purposes
        logger.info(f"Instantiating model: {model_name} with parameters: {kwargs}")

        # Build the model! 
        # This is like calling the blueprint's constructor with your custom settings.
        # Example: SVC(C=1.0, kernel='rbf')
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
        max_epochs: int = 100
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
        
        # Check if this is an iterative model (like XGBoost)
        if cls.is_iterative(model_name):
            logger.info(f"🔄 Detected iterative model: {model_name}. Configuring training parameters.")
            
            # Set up the validation dataset for monitoring during training
            fit_params["eval_set"] = [(X_val, y_val)]
            
            # Disable verbose output to keep logs clean
            fit_params["verbose"] = False
            
            # If early stopping is enabled, add the patience parameter
            if early_stopping:
                fit_params["early_stopping_rounds"] = patience
                logger.info(f"✅ Early stopping enabled with patience={patience}")
        
        else:
            # For non-iterative models (SVC, RandomForest), no special params needed
            logger.info(f"⚡ Detected batch model: {model_name}. Using default training parameters.")
        
        return fit_params