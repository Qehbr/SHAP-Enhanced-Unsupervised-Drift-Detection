# src/models/mlp.py
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator
from src.models.base_model import BaseModel

DEFAULT_MLP_PARAMS = {
    'hidden_layer_sizes': (100,),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,
    'batch_size': 'auto',
    'learning_rate': 'constant',
    'learning_rate_init': 0.001,
    'max_iter': 500,
    'shuffle': True,
    'early_stopping': True,
    'n_iter_no_change': 10
}


class MLPModel(BaseModel):
    """
    Wrapper for scikit-learn's MLPClassifier.

    Inherits all model interface methods from BaseModel, including:
        - fit(X, y)
        - predict(X)
        - predict_proba(X)
        - is_fitted()
        - get_model()
        - clone_model()
    """

    def _build_model(self) -> BaseEstimator:
        """
        Constructs the MLPClassifier using a combination of default and user-specified parameters.

        Returns:
            BaseEstimator: A configured scikit-learn MLPClassifier instance.
        """
        params = {**DEFAULT_MLP_PARAMS, **self.model_params}
        return MLPClassifier(**params)
