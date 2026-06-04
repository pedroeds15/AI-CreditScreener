from pathlib import Path

from models.train_model import (
    load_data,
    split_data,
    train_models,
    evaluate_all_models,
    print_results,
    find_best_model,
)
from models.predict_model import make_predictions
from preprocessing_data.pre_process import separate_features_target
from visualization.visualize import (plot_confusion_matrix, results_to_dataframe, plot_metric_comparison)




def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "processed" / "cleaned_loan_data.csv"

    # Load cleaned data
    df = load_data(data_path)
    print("Loaded cleaned dataset successfully.")
    print("Dataset shape:", df.shape)

    # Split into features and target
    X, y = separate_features_target(df)

    # Train/test split
    X_train, X_test, y_train, y_test = split_data(X, y)
    print("\nTrain/Test Split Complete")
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)

    # Train models
    models, scaler = train_models(X_train, y_train)
    print("\nModels trained successfully.")

    # Evaluate models
    results = evaluate_all_models(models, scaler, X_test, y_test)
    print_results(results)

    # Sample predictions for each model
    all_pred = {}
    predictions = {
        "Logistic Regression": scaler.transform(X_test), # We need to scale the features for logistic regression
        "Decision Tree": X_test,
        "Random Forest": X_test
    }

    # Goes through every model and makes predictions on the test set, then plots a confusion matrix for each model and saves the graphs into the reports/figures directory
    for model in predictions:
        print(f"\nSample predictions for {model}:")
        sample_preds = make_predictions(models[model], predictions[model])
        print(sample_preds[:10])
        all_pred[model] = sample_preds

        plot_confusion_matrix(y_test, sample_preds, model)


    # Displays metric results in a table 
    metric_results = results_to_dataframe(results)
    print(f"\nMetric results DataFrame:\n{metric_results}")

    # Plots 4 model performance graphs based on metrics (accuracy, precision, recall, and f1_score) and shows location of saved graphs
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        plot_metric_comparison(metric_results, metric, output_dir="reports/figures")

    # Find best model
    best_model_name = find_best_model(results)
    best_model = models[best_model_name]

    print(f"\nBest Model: {best_model}")


if __name__ == "__main__":
    main()


