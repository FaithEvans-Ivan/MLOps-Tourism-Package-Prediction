# for data manipulation
import pandas as pd
# for building the preprocessing and modeling pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# for model serialization and experiment tracking
import joblib
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")   
mlflow.set_experiment("mlops-training-experiment")     

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric_features = [
    'Age',     # Customer's age
    'CityTier', # The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3)
    'NumberOfPersonVisiting', # Total number of people accompanying the customer on the trip
    'PreferredPropertyStar',  # Preferred hotel rating by the customer
    'NumberOfTrips',     # Average number of trips the customer takes annually
    'NumberOfChildrenVisiting', # Number of children below age 5 accompanying the customer
    'MonthlyIncome', # Gross monthly income of the customer
    'PitchSatisfactionScore', # Score indicating the customer's satisfaction with the sales pitch
    'NumberOfFollowups', # Total number of follow-ups by the salesperson after the sales pitch
    'DurationOfPitch' # Duration of the sales pitch delivered to the customer
]

categorical_features = [
    'TypeofContact', # The method by which the customer was contacted (Company Invited or Self Inquiry)
    'Occupation', # Customer's occupation (e.g., Salaried, Freelancer)
    'Gender', # Gender of the customer (Male, Female)
    'MaritalStatus', # Marital status of the customer (Single, Married, Divorced)
    'Designation', # Customer's designation in their current organization
    'ProductPitched', # The type of product pitched to the customer
    'Passport', # Whether the customer holds a valid passport (0: No, 1: Yes)
    'OwnCar' # Whether the customer owns a car (0: No, 1: Yes)
]

# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)
# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [100],        # Number of boosting trees. More trees can improve performance but increase training time.
    'xgbclassifier__max_depth': [3],           # Maximum depth of each tree. Higher values increase model complexity and risk of overfitting.
    'xgbclassifier__colsample_bytree': [0.5],    # Fraction of features sampled when building each tree.
    'xgbclassifier__colsample_bylevel': [0.5],   # Fraction of features sampled at each tree level.
    'xgbclassifier__learning_rate': [0.05],       # Step size used during boosting. Smaller values may improve generalization but require more trees.
    'xgbclassifier__reg_lambda': [0.5],          # L2 regularization strength. Higher values help reduce overfitting.
}
# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)   

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination tried during the search as a nested run,
    # so all experiments can be compared side by side in the MLflow UI
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    # Log the best hyperparameters in the main run
    mlflow.log_params(grid_search.best_params_)

    # Store the best model
    best_model = grid_search.best_estimator_

    # Set classification threshold
    classification_threshold = 0.45    

    # Make predictions on the training and test data
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Evaluation
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model next to app.py so the Streamlit app can load it directly,
    # and log it as an MLflow artifact for traceability
    model_path = "tourism_project/deployment/best_Tourism_Project_model_v1.joblib" 
    joblib.dump(best_model, model_path) 
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
