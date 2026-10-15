"""Train and evaluate the diabetes classifiers and save the best one.

    python train.py

Leakage-free by design: the data is split FIRST; imputation and scaling live inside each model's Pipeline, so they are
learned from the training rows only; the model is chosen by cross-validated recall on the training set, and the test set
is used once, for reporting.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "diabetes.csv"
MODEL_PATH = ROOT / "models" / "diabetes_model.joblib"
METRICS_PATH = ROOT / "metrics.json"

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]  # a value of 0 is not physiologically possible
SEED = 2


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df[ZERO_MEANS_MISSING] = df[ZERO_MEANS_MISSING].replace(0, np.nan)
    return df


def candidates():
    """name -> (pipeline, hyper-parameter grid). Imputer and scaler are inside the pipeline."""
    def pipe(model):
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", model)])
    return {
        "SVM (Linear)": (pipe(SVC(kernel="linear")), {"model__C": [0.1, 1, 10]}),
        "Logistic Regression": (pipe(LogisticRegression(max_iter=1000)), {"model__C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (pipe(RandomForestClassifier(random_state=SEED)),
                          {"model__n_estimators": [50, 100, 200], "model__max_depth": [None, 5, 10]}),
    }


def split(df):
    X, y = df[FEATURES], df["Outcome"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def train(save=True, verbose=True):
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)
    models, results = {}, {}
    for name, (pipeline, grid) in candidates().items():
        search = GridSearchCV(pipeline, grid, cv=5, scoring="recall", n_jobs=-1).fit(X_train, y_train)
        pred = search.predict(X_test)
        models[name] = search.best_estimator_
        results[name] = {
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "cv_recall": round(float(search.best_score_), 4),
            "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "test_precision": round(float(precision_score(y_test, pred)), 4),
            "test_recall": round(float(recall_score(y_test, pred)), 4),
            "test_f1": round(float(f1_score(y_test, pred)), 4),
        }
        if verbose:
            print(f"{name:<20} cv_recall={results[name]['cv_recall']:.3f}  test: acc={results[name]['test_accuracy']:.3f} "
                  f"prec={results[name]['test_precision']:.3f} rec={results[name]['test_recall']:.3f} f1={results[name]['test_f1']:.3f}")
    best = max(results, key=lambda n: results[n]["cv_recall"])          # chosen on TRAINING cross-validation only
    summary = {"selected_model": best, "selection_rule": "highest 5-fold CV recall on the training set", "models": results,
               "train_accuracy_selected": round(float(accuracy_score(y_train, models[best].predict(X_train))), 4),
               "n_train": len(X_train), "n_test": len(X_test)}
    if verbose:
        print(f"\nSelected: {best}  (train accuracy {summary['train_accuracy_selected']:.3f})")
    if save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump({"models": models, "best": best, "features": FEATURES}, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(summary, indent=2))
    return models, summary


if __name__ == "__main__":
    train()
