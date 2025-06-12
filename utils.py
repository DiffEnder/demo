import pandas as pd
import numpy as np


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a summary of a pandas DataFrame.

    This function provides a quick overview of the DataFrame, including:
    - Basic statistics for numeric columns
    - Unique value counts for categorical columns
    - Missing value counts for all columns
    - Sample values for each column

    Parameters:
    df (pd.DataFrame): The input DataFrame to summarize

    Returns:
    pd.DataFrame: A summary DataFrame with statistics for each column
    """
    # Initialize lists to store summary information
    columns = []
    dtypes = []
    non_null_counts = []
    null_counts = []
    unique_counts = []
    numeric_stats = []
    sample_values = []

    for col in df.columns:
        columns.append(col)
        dtypes.append(str(df[col].dtype))
        non_null_counts.append(df[col].count())
        null_counts.append(df[col].isnull().sum())
        unique_counts.append(df[col].nunique())

        if np.issubdtype(df[col].dtype, np.number):
            numeric_stats.append(
                f"min: {df[col].min():.2f}, max: {df[col].max():.2f}, mean: {df[col].mean():.2f}, median: {df[col].median():.2f}"
            )
        else:
            numeric_stats.append("N/A")

        # Add sample values
        sample = df[col].dropna().sample(n=min(3, df[col].count())).tolist()
        sample_values.append(str(sample))

    # Create summary DataFrame
    summary_df = pd.DataFrame(
        {
            "Column": columns,
            "Data Type": dtypes,
            "Non-Null Count": non_null_counts,
            "Null Count": null_counts,
            "Unique Values": unique_counts,
            "Numeric Stats": numeric_stats,
            "Sample Values": sample_values,
        }
    )

    return summary_df
