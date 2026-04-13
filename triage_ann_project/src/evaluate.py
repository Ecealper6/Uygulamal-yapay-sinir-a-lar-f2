from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_models(models: Dict[str, object], X_test, y_test, output_dir: str) -> Tuple[pd.DataFrame, dict]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = []
    predictions = {}
    class_labels = sorted(set(y_test.tolist()))

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[model_name] = y_pred
        rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "Recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "F1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            }
        )

        report = classification_report(y_test, y_pred, zero_division=0)
        (output_path / f"classification_report_{model_name}.txt").write_text(report, encoding="utf-8")

        cm = confusion_matrix(y_test, y_pred, labels=class_labels)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.colorbar()
        plt.xticks(range(len(class_labels)), class_labels)
        plt.yticks(range(len(class_labels)), class_labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(output_path / f"confusion_matrix_{model_name}.png", dpi=160)
        plt.close()

    results_df = pd.DataFrame(rows).sort_values(by="F1_macro", ascending=False).reset_index(drop=True)
    results_df.to_csv(output_path / "metrics_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Model"], results_df["F1_macro"])
    plt.title("Model Comparison by Macro F1")
    plt.ylabel("Macro F1")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path / "model_comparison_f1.png", dpi=160)
    plt.close()

    return results_df, predictions
