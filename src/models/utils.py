from src.models.logistic_regression import LogisticRegressionModel
from src.models.mlp import MLPModel
from src.models.random_forest import RandomForestModel


def get_model_instance(model_name, model_params=None, seed=None):
    """
    Factory function to instantiate a model wrapper by name.

    Args:
        model_name (str): Name of the model to instantiate. One of:
                          "LogisticRegression", "RandomForest", "MLP".
        model_params (dict, optional): Dictionary of hyperparameters to pass to the model.

    Returns:
        BaseModel: An instance of the requested model class.

    Raises:
        ValueError: If the provided model name is not recognized.
    """
    if model_params is None:
        model_params = dict()
    if seed is not None and "random_state" not in model_params:
        model_params["random_state"] = seed

    if model_name == "LogisticRegression":
        return LogisticRegressionModel(model_params=model_params)
    elif model_name == "RandomForest":
        return RandomForestModel(model_params=model_params)
    elif model_name == "MLP":
        return MLPModel(model_params=model_params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
