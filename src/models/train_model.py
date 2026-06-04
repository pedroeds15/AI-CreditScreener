from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TARGET_COLUMN = "loan_status"

#load the cleaned dataset
def load_data(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


#Split model into 80/20 training/test data
def split_data(X: pd.DataFrame, y: pd.Series):
    return train_test_split( X, y, test_size=0.2, random_state=42, stratify=y )


# Train models and tune Random Forest using GridSearchCV
def train_models(X_train: pd.DataFrame, y_train: pd.Series):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    logistic_model.fit(X_train_scaled, y_train)

    decision_tree_model = DecisionTreeClassifier(random_state=42)
    decision_tree_model.fit(X_train, y_train)

    rf_param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
    rf_grid.fit(X_train, y_train)

    random_forest_model = rf_grid.best_estimator_

    print("Best Random Forest parameters:", rf_grid.best_params_)

    models = {
        "Logistic Regression": logistic_model,
        "Decision Tree": decision_tree_model,
        "Random Forest": random_forest_model
    }

    return models, scaler

#evaluate a single model and return metrics
def evaluate_model(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0)
    }


#evaluate all the trained models on the test set
def evaluate_all_models(models: dict, scaler, X_test: pd.DataFrame, y_test: pd.Series):
    results = {}

    X_test_scaled = scaler.transform(X_test)

    for model_name, model in models.items():
        if model_name == "Logistic Regression":
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
        results[model_name] = evaluate_model(y_test, y_pred)

    return results

#prints metrics
def print_results(results: dict):
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1 Score : {metrics['f1_score']:.4f}")


#finds the best model based on the f1_score metric (tuned random forest off of f1_score)
def find_best_model(results: dict):
    best_model_name = max(results, key=lambda name: results[name]["f1_score"])
    return best_model_name
        
