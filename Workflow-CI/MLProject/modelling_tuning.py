import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ======================================================
# SET MLFLOW TRACKING (REMOTE - DAGSHUB)
# ======================================================
mlflow.set_tracking_uri(
    "https://dagshub.com/Rodexxx24/LogisticRegression_Tuning.mlflow"
)
mlflow.set_experiment("LogisticRegression_Tuning")

# ======================================================
# LOAD DATASET
# ======================================================
train_df = pd.read_csv("dataset_preprocessing/credit_train_preprocessed.csv")
test_df  = pd.read_csv("dataset_preprocessing/credit_test_preprocessed.csv")

TARGET_COLUMN = "default"

X_train = train_df.drop(columns=[TARGET_COLUMN])
y_train = train_df[TARGET_COLUMN]

X_test  = test_df.drop(columns=[TARGET_COLUMN])
y_test  = test_df[TARGET_COLUMN]

# ======================================================
# HYPERPARAMETER TUNING
# ======================================================
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy"
)

# ======================================================
# TRAINING & MANUAL LOGGING
# ======================================================
with mlflow.start_run(run_name="LogisticRegression_Tuning"):

    print("[INFO] Training model with GridSearchCV...")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # ========================
    # METRICS
    # ========================
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    # ========================
    # LOG PARAMETERS & METRICS
    # ========================
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Metadata dataset (best practice)
    mlflow.log_param("n_train_samples", X_train.shape[0])
    mlflow.log_param("n_test_samples", X_test.shape[0])
    mlflow.log_param("n_features", X_train.shape[1])

    # ========================
    # LOG MODEL (MLFLOW OFFICIAL)
    # ========================
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        registered_model_name="LogisticRegression_Tuning"
    )

    # ==================================================
    # ADDITIONAL ARTEFACTS (ADVANCED REQUIREMENT)
    # ==================================================

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    # 2. Prediction CSV
    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred
    })

    pred_path = "predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    mlflow.log_artifact(pred_path)

    # 3. Metric summary file (non-visual artefact)
    metric_info = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "best_params": grid.best_params_
    }

    metric_info_path = "metric_info.json"
    with open(metric_info_path, "w") as f:
        yaml.dump(metric_info, f)

    mlflow.log_artifact(metric_info_path)

    print("[INFO] Training selesai & artefak berhasil disimpan ke MLflow (DagsHub)")
