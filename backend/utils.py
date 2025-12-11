# backend/utils.py
import pandas as pd

def analyze_dataframe_for_api(df: pd.DataFrame):
    """Return JSON-friendly analysis summary of df for frontend to render."""
    summary = {}
    summary['rows'] = int(df.shape[0])
    summary['columns'] = int(df.shape[1])
    summary['dtypes'] = df.dtypes.astype(str).to_dict()
    summary['missing_values'] = df.isnull().sum().to_dict()
    # Basic numeric description
    numeric = df.select_dtypes(include='number')
    summary['describe'] = numeric.describe().to_dict()
    # Correlation matrix (to JSON-friendly dict)
    if numeric.shape[1] > 0:
        corr = numeric.corr()
        # convert to nested dict
        summary['correlation'] = corr.round(4).to_dict()
    else:
        summary['correlation'] = {}
    # simple class distribution if a target column exists
    # frontend can ask for target column name explicitly if needed
    return summary
