import pandas as pd
import os


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    return( pd.read_csv(file_path)
