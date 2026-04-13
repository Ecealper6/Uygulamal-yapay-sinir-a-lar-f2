from __future__ import annotations

from typing import Dict


def train_models(models: Dict[str, object], X_train, y_train) -> Dict[str, object]:
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained
