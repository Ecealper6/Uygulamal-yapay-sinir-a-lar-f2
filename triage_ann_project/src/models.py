from __future__ import annotations

from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def get_models():
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "GaussianNB": GaussianNB(),
        "Perceptron_OVR": Perceptron(max_iter=20, eta0=0.1, random_state=42, tol=None),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
        ),
        "DecisionStump": DecisionTreeClassifier(max_depth=1, random_state=42),
    }
