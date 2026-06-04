from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numbers
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Converts the results from the model evaluation into a df for easier visulization
def results_to_dataframe(results):
    rows = []

    for model_name, metrics in results.items():
        row = {"Model": model_name}
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, numbers.Number):
                row[metric_name] = metric_value
        rows.append(row)

    df_results = pd.DataFrame(rows)

    if "accuracy" in df_results.columns:
        df_results = df_results.sort_values(by="accuracy", ascending=True)
    
    return df_results

# Plots 4 model performance graphs based on metrics (accuracy, precision, recall, and f1_score)
def plot_metric_comparison(df_results, metric, output_dir):
    if metric not in df_results.columns:
        print(f"Skipping {metric} - not found in DataFrame columns")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.bar(df_results["Model"], df_results[metric])
    plt.title(f"{metric.capitalize()} Comparison Across Models")
    plt.xlabel("Model")
    plt.ylabel(metric.capitalize())
    plt.xticks(rotation=20)
    plt.tight_layout()

    # Saves graphs as a PNG file in the reports/figures directory
    file_path = output_dir / f"{metric}_comparison.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved {metric} plot to: {file_path}")

# Creates confusion matrix for each model 
def plot_confusion_matrix(y_true, y_pred, model_name, output_dir="reports/figures"):
    output_dir = Path(output_dir)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix")

    plt.tight_layout()

    # Save figure into reports/figures directory
    file_path = output_dir / f"{model_name}_confusion_matrix.png"
    plt.savefig(file_path)
    plt.close()

    print(f"Saved confusion matrix: {file_path}")
