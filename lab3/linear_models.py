import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from grader_contracts.linear_models import (
    BaselineComparisonResult, ClassificationInput, ClassificationResult,
    RegressionInput, RegressionResult, ClassificationMetrics,
)


def _arrays(x_train, y_train, x_test):
    Xtr = np.asarray(x_train, dtype=float)
    Xte = np.asarray(x_test, dtype=float)
    ytr = np.asarray(y_train, dtype=float).reshape(-1)
    if Xtr.ndim == 1:
        Xtr = Xtr.reshape(-1, 1)
    if Xte.ndim == 1:
        Xte = Xte.reshape(-1, 1)
    return Xtr, ytr, Xte


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X, y, Xt = _arrays(data.train_features, data.train_target, data.test_features)
    scaler_x = StandardScaler()
    Xs = scaler_x.fit_transform(X)
    Xts = scaler_x.transform(Xt)
    ym = float(y.mean())
    ys = float(y.std()) or 1.0
    yn = (y - ym) / ys

    w = np.zeros(Xs.shape[1], dtype=float)
    b = 0.0
    history = []
    n = len(y)
    for _ in range(int(data.epochs)):
        pred = Xs @ w + b
        err = pred - yn
        history.append(float(np.mean(err ** 2)))
        dw = (2.0 / n) * (Xs.T @ err)
        db = 2.0 * float(np.mean(err))
        w -= float(data.learning_rate) * dw
        b -= float(data.learning_rate) * db

    pred_test = (Xts @ w + b) * ys + ym
    # Coefficients correspond to standardized input features and original target units.
    return RegressionResult(weights=w * ys, bias=b * ys + ym, loss_history=history,
                            predictions=pred_test)


def _fit_logistic(X, y, learning_rate, epochs):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    w = np.zeros(Xs.shape[1], dtype=float)
    b = 0.0
    n = len(y)
    history = []
    for _ in range(int(epochs)):
        z = np.clip(Xs @ w + b, -500, 500)
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-12
        history.append(float(-np.mean(y * np.log(p + eps) + (1-y) * np.log(1-p + eps))))
        err = p - y
        w -= float(learning_rate) * (Xs.T @ err) / n
        b -= float(learning_rate) * float(np.mean(err))
    return scaler, w, b


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    X, y, Xt = _arrays(data.train_features, data.train_target, data.test_features)
    y = y.astype(int)
    scaler, w, b = _fit_logistic(X, y, data.learning_rate, data.epochs)
    Xts = scaler.transform(Xt)
    z = np.clip(Xts @ w + b, -500, 500)
    probs = 1.0 / (1.0 + np.exp(-z))
    return ClassificationResult(weights=w, bias=float(b), loss_history=_loss_history(X, y, data.learning_rate, data.epochs),
                                probabilities=probs, predictions=(probs >= 0.5).astype(int))


def _loss_history(X, y, lr, epochs):
    # Reproduce the optimization history for the result contract.
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    w = np.zeros(Xs.shape[1], dtype=float)
    b = 0.0
    history = []
    for _ in range(int(epochs)):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xs @ w + b, -500, 500)))
        history.append(float(-np.mean(y*np.log(p+1e-12)+(1-y)*np.log(1-p+1e-12))))
        err = p-y
        w -= float(lr) * (Xs.T @ err) / len(y)
        b -= float(lr) * float(np.mean(err))
    return history


def _metrics(y, probabilities, predictions):
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y, probabilities)),
        accuracy=float(accuracy_score(y, predictions)),
        precision=float(precision_score(y, predictions, zero_division=0)),
        recall=float(recall_score(y, predictions, zero_division=0)),
        f1=float(f1_score(y, predictions, zero_division=0)),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    X, y, Xt = _arrays(data.train_features, data.train_target, data.test_features)
    yt = np.asarray(data.test_target).astype(int).reshape(-1)
    y = y.astype(int)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xts = scaler.transform(Xt)

    own = train_logistic_classifier(data)
    lr = SklearnLogisticRegression(max_iter=2000, random_state=42).fit(Xs, y)
    tree = DecisionTreeClassifier(random_state=42).fit(X, y)
    knn = KNeighborsClassifier(n_neighbors=5).fit(Xs, y)

    return BaselineComparisonResult(
        logistic_regression=_metrics(yt, lr.predict_proba(Xts)[:, 1], lr.predict(Xts)),
        decision_tree=_metrics(yt, tree.predict_proba(Xt)[:, list(tree.classes_).index(1)], tree.predict(Xt)),
        knn=_metrics(yt, knn.predict_proba(Xts)[:, 1], knn.predict(Xts)),
    )
