from __future__ import annotations

from pathlib import Path

from src.demo import DEFAULT_SAMPLE, build_demo_message, transform_sample
from src.evaluate import evaluate_models
from src.models import get_models
from src.preprocessing import PreprocessConfig, load_and_preprocess
from src.train import train_models


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "triage.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main() -> None:
    config = PreprocessConfig(
        test_size=0.2,
        random_state=42,
        encoding="onehot",
        scaling="zscore",
        clip_z_threshold=0.0,
    )

    X_train, X_test, y_train, y_test, metadata = load_and_preprocess(str(DATA_PATH), config)

    print("Dataset loaded successfully")
    print(f"Sample count: {metadata['sample_count']}")
    print(f"Feature count after preprocessing: {len(metadata['feature_names'])}")
    print(f"Class distribution: {metadata['class_distribution']}")

    models = get_models()
    trained_models = train_models(models, X_train, y_train)
    results_df, _ = evaluate_models(trained_models, X_test, y_test, str(OUTPUT_DIR))

    print("\nModel comparison:\n")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]
    sample_x = transform_sample(DEFAULT_SAMPLE, str(DATA_PATH), metadata)
    predicted_class = int(best_model.predict(sample_x)[0])

    demo_text = build_demo_message(best_model_name, predicted_class, DEFAULT_SAMPLE)
    print("\n" + demo_text)
    (OUTPUT_DIR / "demo_output.txt").write_text(demo_text, encoding="utf-8")

    summary_lines = [
        "Project run completed successfully.",
        f"Best model by Macro F1: {best_model_name}",
        "",
        "Metrics summary:",
        results_df.to_string(index=False),
        "",
        demo_text,
    ]
    (OUTPUT_DIR / "run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
