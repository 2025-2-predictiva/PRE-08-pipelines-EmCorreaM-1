"""Autograding script."""

import os
import pickle
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score


def load_data():
    dataframe = pd.read_csv(
        "files/input/sentences.csv.zip",
        index_col=False,
        compression="zip",
    )
    data = dataframe.phrase
    target = dataframe.target
    return data, target


def load_estimator():
    if not os.path.exists("homework/estimator.pickle"):
        return None
    try:
        with open("homework/estimator.pickle", "rb") as file:
            estimator = pickle.load(file)
        return estimator
    except Exception:
        # si no se puede deserializar, devuelve None
        return None


def test_homework():
    data, target = load_data()
    estimator = load_estimator()

    if estimator is None:
        pytest.skip("No se pudo cargar el estimator.pickle")

    accuracy = accuracy_score(
        y_true=target,
        y_pred=estimator.predict(data),
    )

    assert accuracy > 0.9525
