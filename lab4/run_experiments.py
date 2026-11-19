"""Воспроизводимый запуск итогового эксперимента лабораторной №4."""
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from ensemble_models import run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput

ROOT = Path(__file__).resolve().parent
frame = pd.read_csv(ROOT / "german.csv", sep=";")
X = frame.drop(columns=["Creditability"])
y = frame["Creditability"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
result = run_ensemble_experiment(
    EnsembleInput(X_train, y_train, X_test, y_test, random_seed=42)
)
payload = {
    "model_name": result.model_name,
    "split": {"test_size": 0.2, "random_state": 42, "stratify": True},
    "test_samples": int(len(y_test)),
    "positive_class": 1,
    "threshold": 0.5,
    "metrics": {
        "roc_auc": result.roc_auc,
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
    },
}
(ROOT / "experiment_results.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, ensure_ascii=False, indent=2))
