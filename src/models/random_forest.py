# src/models/random_forest.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from src.models.base_model import BaseModel

DEFAULT_RF_PARAMS = {
    'n_estimators': 10,
    'n_jobs': -1  # Use all available CPU cores
}


class RandomForestModel(BaseModel):
    """
    Wrapper for scikit-learn's RandomForestClassifier.

    This class inherits from BaseModel and exposes:
        - fit(X, y)
        - predict(X)
        - predict_proba(X)
        - get_model()
        - is_fitted()
        - clone_model()"""

    def _build_model(self) -> BaseEstimator:
        """
        Constructs the RandomForestClassifier using default and user-specified parameters.

        Returns:
            BaseEstimator: A configured scikit-learn RandomForestClassifier instance.
        """
        params = {**DEFAULT_RF_PARAMS, **self.model_params}
        return RandomForestClassifier(**params)
