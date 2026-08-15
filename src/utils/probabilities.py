"""
Utility functions for extracting prediction probabilities from models.

Some models (like SVC) don't directly give probabilities, so we need
to convert their raw scores into probabilities using mathematical formulas.
"""

import logging
from typing import Any
import numpy as np

# Import the sigmoid function to convert scores to probabilities
# Sigmoid turns any number into a value between 0 and 1 (a probability)
from scipy.special import expit

logger = logging.getLogger(__name__)


def get_prediction_probabilities(model: Any, X: Any) -> np.ndarray:
    """
    Get prediction probabilities from a model, handling
    both probabilistic and non-probabilistic models.

    Parameters
    ----------
    model : Any
        The trained machine learning model.
    X : Any
        The feature data to predict on.

    Returns
    -------
    np.ndarray
        Array of probabilities for the positive class (stress).
    """
    # Check if the model can directly give us probabilities
    if hasattr(model, "predict_proba"):
        # If yes, get the probabilities and return the column
        # for the "stress" class (index 1)
        logger.info(
            "✅ Model supports predict_proba. Using direct probabilities."
        )
        return model.predict_proba(X)[:, 1]

    else:
        # If the model cannot give probabilities, we need to convert its raw scores
        logger.info(
            "⚠️ Model lacks predict_proba."
            "Converting decision_function to probabilities."
        )

        # Get the raw decision scores from the model
        # (distance from the decision boundary)
        decision_scores = model.decision_function(X)

        # Convert the scores to probabilities using the sigmoid function
        # Sigmoid squashes any number into a range between 0 and 1
        probabilities = expit(decision_scores)

        return probabilities
