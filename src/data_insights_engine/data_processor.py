"""
Data processing functionality for the Data Insights Engine.
Handles data cleaning, transformation, and preprocessing.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def clean_data(df):
    """
    Clean a dataset by handling missing values, duplicates, and outliers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The raw dataset to clean
        
    Returns:
    --------
    pandas.DataFrame
        The cleaned dataset
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Handle missing values
    for col in df_clean.columns:
        # For numeric columns, impute with median
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        # For categorical/object columns, impute with mode
        elif pd.api.types.is_object_dtype(df_clean[col]):
            most_common = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
            df_clean[col] = df_clean[col].fillna(most_common)
    
    # Handle outliers for numeric columns
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        # Skip ID columns
        if 'id' in col.lower():
            continue
            
        # Calculate IQR
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers at the bounds
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Convert date columns to datetime
    for col in df_clean.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df_clean[col] = pd.to_datetime(df_clean[col])
            except:
                pass
    
    return df_clean

def preprocess_data(df):
    """
    Preprocess a dataset for analysis by creating derived features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to preprocess
        
    Returns:
    --------
    pandas.DataFrame
        The preprocessed dataset
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Extract features from date columns
    date_cols = df_processed.select_dtypes(include=['datetime64']).columns
    for col in date_cols:
        df_processed[f'{col}_year'] = df_processed[col].dt.year
        df_processed[f'{col}_month'] = df_processed[col].dt.month
        df_processed[f'{col}_day'] = df_processed[col].dt.day
        df_processed[f'{col}_dayofweek'] = df_processed[col].dt.dayofweek
        df_processed[f'{col}_quarter'] = df_processed[col].dt.quarter
    
    # Create derived features
    if 'sales' in df_processed.columns and 'profit' in df_processed.columns:
        df_processed['profit_margin'] = (df_processed['profit'] / df_processed['sales']).replace([np.inf, -np.inf], 0)
    
    if 'quantity' in df_processed.columns and 'sales' in df_processed.columns:
        df_processed['average_price'] = (df_processed['sales'] / df_processed['quantity']).replace([np.inf, -np.inf], 0)
    
    return df_processed