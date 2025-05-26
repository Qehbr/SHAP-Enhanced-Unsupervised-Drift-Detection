# src/models/base_model.py
from sklearn.base import BaseEstimator
from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self, model_params=None):
        """
        Abstract Base Class for models used in the simulation pipeline.
        Enforces a standardized interface for training and inference using scikit-learn compatible models

        Args:
            model_params (dict, optional): Parameters to configure the model. If None, an empty dict is used.
        """
        self.model_params = model_params if model_params is not None else {}
        self.model: BaseEstimator = self._build_model()
        self.fitted = False

    @abstractmethod
    def _build_model(self) -> BaseEstimator:
        """
        Instantiates and returns the scikit-learn model.

        Returns:
            BaseEstimator: The scikit-learn model instance.
        """
        pass

    def fit(self, X, y):
        """
        Fits the model to the provided features and labels.

        Args:
            X (array-like): Training features.
            y (array-like): Training labels.
        """
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X):
        """
        Generates predictions using the fitted model.

        Args:
            X (array-like): Input features.

        Returns:
            array-like: Model predictions.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self.fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predicts class probabilities using the fitted model.

        Args:
            X (array-like): Input features.

        Returns:
            array-like: Predicted class probabilities.

        Raises:
            RuntimeError: If the model is not yet fitted.
            AttributeError: If the model does not support probability prediction.
        """
        if not self.fitted:
            raise RuntimeError("Model is not fitted yet.")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError(f"{type(self.model).__name__} does not have predict_proba method.")

    def get_model(self):
        """
        Returns the internal scikit-learn model instance.

        Returns:
            BaseEstimator: The model instance.
        """
        return self.model

    def is_fitted(self):
        """
        Checks whether the model has been fitted.

        Returns:
            bool: True if fitted, False otherwise.
        """
        return self.fitted

    def clone_model(self):
        """
        Creates a new, unfitted instance of the same class with identical parameters.

        Returns:
            BaseModel: A new instance of the current model class.
        """
        return type(self)(model_params=self.model_params)
