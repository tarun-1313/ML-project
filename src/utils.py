import os
import sys
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    """
    Save any Python object as a pickle file
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Load a pickle object from disk
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Train models and return:
    1. report -> model_name : r2_score
    2. trained_models -> model_name : fitted model
    """
    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():

            # 🚫 CatBoost (NO GridSearch)
            if model_name.lower() == "catboost":
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                report[model_name] = r2_score(y_test, y_pred)
                trained_models[model_name] = model
                continue

            # Models without params (e.g. LinearRegression)
            if model_name not in param or not param[model_name]:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                report[model_name] = r2_score(y_test, y_pred)
                trained_models[model_name] = model
                continue

            # GridSearch for sklearn-native models
            gs = GridSearchCV(
                estimator=model,
                param_grid=param[model_name],
                cv=3,
                n_jobs=-1
            )

            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            y_pred = best_model.predict(X_test)

            report[model_name] = r2_score(y_test, y_pred)
            trained_models[model_name] = best_model

        return report, trained_models

    except Exception as e:
        raise CustomException(e, sys)
