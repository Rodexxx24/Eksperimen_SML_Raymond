import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ========================
# AKTIFKAN AUTOLOG
# ========================
mlflow.sklearn.autolog()

# ========================
# SET TRACKING LOKAL
# ========================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("LogisticRegression_Basic")

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
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ========================
    # OPTIONAL: LOG MANUAL METRICS TAMBAHAN
    # ========================
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy_manual", acc)
    mlflow.log_metric("precision_manual", prec)
    mlflow.log_metric("recall_manual", rec)
    mlflow.log_metric("f1_score_manual", f1)

    print("Training selesai & model tersimpan ke MLflow (lokal)")