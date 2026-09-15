import os
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import mlflow

# MLflow runs locally in CI. The workflow starts the MLflow server on port 5000.
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("mlops-training-experiment")

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
ytest = pd.read_csv("ytest.csv").squeeze("columns")

numeric_features = [
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "NumberOfTrips", "NumberOfChildrenVisiting", "MonthlyIncome",
    "PitchSatisfactionScore", "NumberOfFollowups", "DurationOfPitch"
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender", "MaritalStatus",
    "Designation", "ProductPitched", "Passport", "OwnCar"
]

class_counts = ytrain.value_counts()
if 0 not in class_counts.index or 1 not in class_counts.index:
    raise ValueError("Training data must contain both ProdTaken classes 0 and 1.")
scale_pos_weight = class_counts[0] / class_counts[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
)

param_grid = {
    "xgbclassifier__n_estimators": [100],
    "xgbclassifier__max_depth": [3],
    "xgbclassifier__colsample_bytree": [0.5],
    "xgbclassifier__colsample_bylevel": [0.5],
    "xgbclassifier__learning_rate": [0.05],
    "xgbclassifier__reg_lambda": [0.5],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring="f1",
    )
    grid_search.fit(Xtrain, ytrain)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(
        ytrain, y_pred_train, output_dict=True, zero_division=0
    )
    test_report = classification_report(
        ytest, y_pred_test, output_dict=True, zero_division=0
    )

    mlflow.log_param("classification_threshold", classification_threshold)
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1_score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1_score": test_report["1"]["f1-score"],
    })

    model_path = "tourism_project/deployment/best_Tourism-Project_model_v1.joblib"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"Model saved to {model_path}")
    print("Test classification report:")
    print(classification_report(ytest, y_pred_test, zero_division=0))
