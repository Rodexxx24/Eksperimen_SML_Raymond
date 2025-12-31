import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import os
import yaml

# ========================
# SET TRACKING REMOTE (DAGSHUB)
# ========================
mlflow.set_tracking_uri("https://dagshub.com/Rodexxx24/LogisticRegression_Tuning.mlflow")
mlflow.set_experiment("LogisticRegression_Tuning")

# ========================
# LOAD DATA
# ========================
train_df = pd.read_csv("dataset_preprocessing/credit_train_preprocessed.csv")
test_df  = pd.read_csv("dataset_preprocessing/credit_test_preprocessed.csv")

target_column = "default"
X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]
X_test  = test_df.drop(columns=[target_column])
y_test  = test_df[target_column]

# ========================
# HYPERPARAMETER GRID
# ========================
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear", "lbfgs"]
}
grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=3, scoring="accuracy")

# ========================
# TRAINING + MANUAL LOGGING
# ========================
with mlflow.start_run(run_name="LogisticRegression_Tuning"):

    print("[INFO] MLFLOW: Training model...")
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

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # ========================
    # LOG MODEL (struktur folder lengkap)
    # ========================
    model_folder = "model"
    os.makedirs(model_folder, exist_ok=True)
    mlflow.sklearn.save_model(best_model, path=model_folder)

    # Buat conda.yaml / python_env.yml / requirements.txt jika ingin lengkap
    # Contoh minimal conda.yaml
    conda_dict = {
        "name": "logreg_env",
        "channels": ["defaults"],
        "dependencies": [
            "python=3.11",
            "scikit-learn",
            "pandas",
            "matplotlib",
            "pip",
            {"pip": ["mlflow"]}
        ]
    }
    with open(os.path.join(model_folder, "conda.yaml"), "w") as f:
        yaml.dump(conda_dict, f)

    # ========================
    # LOG ARTEFAK TAMBAHAN
    # ========================
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)

    # 2. File prediksi CSV
    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    csv_path = "predictions.csv"
    pred_df.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

    # 3. Folder model lengkap
    mlflow.log_artifacts(model_folder)

    print(f"[INFO] MLFLOW: Training selesai & artefak tersimpan di run")
