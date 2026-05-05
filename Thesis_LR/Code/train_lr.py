"""Train Logistic Regression on the same dataset Kent used for RF comparison.

This is the BEFORE-fix baseline (wind doesn't affect simulation yet).

Usage:
    python train_lr.py
"""

from pathlib import Path
from modules.model_trainer import ModelTrainer


def main():
    code_dir = Path(__file__).resolve().parent
    csv_path = code_dir / "dataFiles" / "synthetic_fire_dataset.csv"
    output_dir = code_dir / "models"

    print("=" * 60)
    print("  LOGISTIC REGRESSION TRAINING (before wind fix)")
    print("  Dataset: Kent's synthetic_fire_dataset.csv")
    print("=" * 60)

    trainer = ModelTrainer(csv_path=str(csv_path), output_dir=str(output_dir), seed=42)

    print(f"\nFeatures: {trainer.feature_names}")
    print(f"Train size: {len(trainer.X_train)} | Test size: {len(trainer.X_test)}")
    print(f"Train positives: {int(trainer.y_train.sum())} | Test positives: {int(trainer.y_test.sum())}")

    trainer.train_logistic_regression()
    metrics = trainer.evaluate()
    saved_path = trainer.save_model(filename="fire_lr_model.joblib")

    print(f"\nSaved model: {saved_path}")
    print(f"\nMetrics summary: {metrics}")

    trainer.feature_importance_report()


if __name__ == "__main__":
    main()
