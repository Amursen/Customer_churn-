import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input DataFrame by dropping unnecessary columns and encoding categorical variables.

    """
    df.columns = df.columns.str.strip()

    for col in ["customerID", "CustomerID", "customer_id", "customer_id"]:
        if col in df.columns:
            df.drop(columns=[col])


    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df["target_col"].str.strip().map({"yes" : 1, "no" : 0})

    if "totalcharges" in df.columns:
        df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")


    if "seniorcitizen" in df.columns:
        df["seniorcitizen"] = df["seniorcitizen"].fillna(0).astype(int)


    num_cols =df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)


    return df
