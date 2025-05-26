# src/models/logistic_regression.py

from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from src.models.base_model import BaseModel

DEFAULT_LOGREG_PARAMS = {
    'solver': 'liblinear',
    'C': 1.0,
    'max_iter': 1000
}


class LogisticRegressionModel(BaseModel):
    """
     Wrapper for scikit-learn's LogisticRegression model.
    Inherits the interface from BaseModel, including fit, predict, predict_proba, etc.
    """

    def _build_model(self) -> BaseEstimator:
        """
        Builds and returns a Logistic Regression model using merged default and user parameters.

        Returns:
            BaseEstimator: An instance of scikit-learn's LogisticRegression.
        """
        params = {**DEFAULT_LOGREG_PARAMS, **self.model_params}
        return LogisticRegression(**params)
