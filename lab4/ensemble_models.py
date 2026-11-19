"""Воспроизводимый эксперимент с ансамблем моделей для German Credit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


_NUMERIC_COLUMNS = [
    "Duration_of_Credit_monthly", "Credit_Amount", "Age_years",
]


def _as_frame(features):
    """Сохраняет имена столбцов, если они переданы, иначе создаёт DataFrame."""
    if isinstance(features, pd.DataFrame):
        return features.copy()
    array = np.asarray(features)
    if array.ndim != 2:
        raise ValueError("Признаки должны быть двумерной таблицей")
    return pd.DataFrame(array, columns=[f"feature_{i}" for i in range(array.shape[1])])


def _make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    # В German Credit непрерывные по смыслу поля — срок, сумма и возраст.
    # Остальные поля являются кодами категорий и обрабатываются one-hot.
    if all(name in frame.columns for name in _NUMERIC_COLUMNS):
        numeric = _NUMERIC_COLUMNS
    elif frame.shape[1] == 20:
        numeric = [frame.columns[i] for i in (1, 4, 12)]
    else:
        numeric = list(frame.select_dtypes(include=[np.number]).columns)
    categorical = [c for c in frame.columns if c not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("categorical", categorical_pipe, categorical),
        ("numeric", numeric_pipe, numeric),
    ])


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучает soft-voting ансамбль только на train и оценивает test."""
    x_train = _as_frame(data.train_features)
    x_test = _as_frame(data.test_features)
    # При ndarray train/test имена признаков должны быть согласованы.
    x_test.columns = x_train.columns
    y_train = np.asarray(data.train_target).ravel().astype(int)
    y_test = np.asarray(data.test_target).ravel().astype(int)
    seed = int(data.random_seed)

    rf = Pipeline([
        ("preprocess", _make_preprocessor(x_train)),
        ("model", RandomForestClassifier(
            n_estimators=900, min_samples_leaf=3, max_features=0.7,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        )),
    ])
    gb = Pipeline([
        ("preprocess", _make_preprocessor(x_train)),
        ("model", GradientBoostingClassifier(
            n_estimators=180, learning_rate=0.025, max_depth=2,
            min_samples_leaf=8, subsample=0.85, random_state=seed,
        )),
    ])
    lr = Pipeline([
        ("preprocess", _make_preprocessor(x_train)),
        ("model", LogisticRegression(
            C=0.3, max_iter=3000, class_weight="balanced",
            random_state=seed,
        )),
    ])
    model = VotingClassifier(
        estimators=[("random_forest", rf), ("gradient_boosting", gb), ("logistic_regression", lr)],
        voting="soft", weights=[3, 1, 1], n_jobs=-1,
    )
    model.fit(x_train, y_train)
    classes = list(model.classes_)
    positive_index = classes.index(1) if 1 in classes else len(classes) - 1
    probabilities = model.predict_proba(x_test)[:, positive_index]
    predictions = (probabilities >= 0.5).astype(int)

    return EnsembleResult(
        model_name="Soft Voting: Random Forest (weight=3) + Gradient Boosting + Logistic Regression",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        f1=float(f1_score(y_test, predictions, zero_division=0)),
    )
