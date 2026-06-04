from pathlib import Path
import pandas as pd

# Loads raw CSV data.
def load_data(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)

# Applies the necessary cleaning steps identified in the clean_data.ipynb notebook
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Cap unrealistic ages to 90
    df["person_age"] = df["person_age"].clip(upper=90)

    # Drop gender for fairness reasons
    if "person_gender" in df.columns:
        df = df.drop(columns=["person_gender"])

    return df


# Encode education so it is no longer categorical
def encode_education(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    education_map = {
        "High School": 1,
        "Associate": 2,
        "Bachelor": 3,
        "Master": 4,
        "Doctorate": 5,
    }

    if "person_education" in df.columns:
        df["person_education"] = df["person_education"].map(education_map)

    return df


# Encode binary categorical variables
def encode_binary_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    binary_mappings = {
        "previous_loan_defaults_on_file": {"No": 0, "Yes": 1}
    }

    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    return df


# One-hot encode nominal categorical variables 
def encode_nominal_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    nominal_cols = ["person_home_ownership", "loan_intent"]
    existing_nominal_cols = [col for col in nominal_cols if col in df.columns]

    if existing_nominal_cols:
        df = pd.get_dummies(df, columns=existing_nominal_cols, drop_first=True, dtype=int)

    return df


# Separate predictors and target
def separate_features_target(df: pd.DataFrame):
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]
    return X, y


# Save cleaned dataframe to CSV
def save_processed_data(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    base_dir = Path(__file__).resolve().parents[1]
    raw_path = base_dir / "data" / "raw" / "loan_data.csv"
    processed_path = base_dir / "data" / "processed" / "cleaned_loan_data.csv"

    orig_df = load_data(raw_path)

    df = orig_df.copy()
    df = clean_data(df)
    df = encode_education(df)
    df = encode_binary_variables(df)
    df = encode_nominal_variables(df)

    X, y = separate_features_target(df)

    save_processed_data(df, processed_path)

    print("Processed data saved successfully.")
    print("\nOriginal shape:", orig_df.shape)
    print("Cleaned shape:", df.shape)

    print("\nOriginal preview:")
    print(orig_df.head())

    print("\nCleaned preview:")
    print(df.head())

    print("\nRemaining object columns:")
    print(df.select_dtypes(include=["object"]).columns.tolist())

    print("Feature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)


if __name__ == "__main__":
    main()