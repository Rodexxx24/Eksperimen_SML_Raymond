import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    # ========================
    # AKTIFKAN AUTOLOG
    # ========================
    mlflow.sklearn.autolog()

    # ========================
    # SET TRACKING URI (Lokal / CI / Remote)
    # ========================
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("LogisticRegression_Basic")

    print(f"[INFO] MLflow Tracking URI: {MLFLOW_TRACKING_URI}")

    # ========================
    # LOAD DATA
    # ========================
    train_df = pd.read_csv("dataset_preprocessing/credit_train_preprocessed.csv")
    test_df  = pd.read_csv("dataset_preprocessing/credit_test_preprocessed.csv")

    target_column = "default"

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    # ========================
    # TRAINING
    # ========================
    with mlflow.start_run(run_name="LogisticRegression_Basic"):
        print("[INFO] Training Logistic Regression...")

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # ========================
        # MANUAL METRICS (pelengkap autolog)
        # ========================
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy_manual", acc)
        mlflow.log_metric("precision_manual", prec)
        mlflow.log_metric("recall_manual", rec)
        mlflow.log_metric("f1_score_manual", f1)

        print("[INFO] Training selesai & model tersimpan di MLflow")

if __name__ == "__main__":
    print("[INFO] MLflow CI Training started")
    main()
