"""
Statistical analysis functions for the Data Insights Engine.
"""

import pandas as pd
import numpy as np
from scipy import stats

def generate_summary_stats(df):
    """
    Generate summary statistics for a dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing summary statistics
    """
    # Get numerical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Create a summary DataFrame
    summary = pd.DataFrame({
        'Mean': df[numeric_cols].mean(),
        'Median': df[numeric_cols].median(),
        'Std Dev': df[numeric_cols].std(),
        'Min': df[numeric_cols].min(),
        'Max': df[numeric_cols].max(),
        'Range': df[numeric_cols].max() - df[numeric_cols].min(),
        'Missing (%)': df[numeric_cols].isnull().mean() * 100
    })
    
    # Add quartiles
    summary['25th Percentile'] = df[numeric_cols].quantile(0.25)
    summary['75th Percentile'] = df[numeric_cols].quantile(0.75)
    
    # Add skewness and kurtosis
    summary['Skewness'] = df[numeric_cols].skew()
    summary['Kurtosis'] = df[numeric_cols].kurtosis()
    
    return summary.round(2)

def analyze_correlations(df, method='pearson', threshold=0.1):
    """
    Analyze correlations between variables in a dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    method : str, default='pearson'
        The correlation method to use: 'pearson', 'spearman', or 'kendall'
    threshold : float, default=0.1
        The minimum absolute correlation to consider significant
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing significant correlations
    """
    # Get numerical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr(method=method)
    
    # Create a DataFrame for significant correlations
    significant_corrs = []
    
    # Get upper triangle of correlation matrix to avoid duplicates
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            var1 = corr_matrix.columns[i]
            var2 = corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]
            
            # Only include correlations above threshold
            if abs(corr) >= threshold:
                significant_corrs.append({
                    'Variable 1': var1,
                    'Variable 2': var2,
                    'Correlation': corr,
                    'Strength': categorize_correlation_strength(corr)
                })
    
    # Create DataFrame from significant correlations
    if significant_corrs:
        return pd.DataFrame(significant_corrs)
    else:
        return pd.DataFrame(columns=['Variable 1', 'Variable 2', 'Correlation', 'Strength'])

def categorize_correlation_strength(corr_value):
    """Categorize correlation strength based on absolute value."""
    abs_corr = abs(corr_value)
    
    if abs_corr < 0.3:
        return 'Weak'
    elif abs_corr < 0.7:
        return 'Moderate'
    else:
        return 'Strong'

def detect_outliers(df, method='iqr'):
    """
    Detect outliers in a dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to analyze
    method : str, default='iqr'
        The method to use for outlier detection: 'iqr' or 'zscore'
        
    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the detected outliers
    dict
        A dictionary with additional outlier statistics
    """
    # Get numerical columns (excluding ID columns)
    numeric_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns 
                    if not 'id' in col.lower()]
    
    # Dictionary to store results
    outlier_results = {}
    outlier_indices = set()
    
    if method == 'iqr':
        # IQR-based outlier detection
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers for this column
            col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            
            if len(col_outliers) > 0:
                outlier_indices.update(col_outliers)
    
    elif method == 'zscore':
        # Z-score based outlier detection
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df[col].fillna(df[col].median())))
            
            # Consider values with z-score > 3 as outliers
            col_outliers = df.index[z_scores > 3]
            
            if len(col_outliers) > 0:
                outlier_indices.update(col_outliers)
    
    # Create DataFrame with all outliers
    outliers = df.loc[list(outlier_indices)] if outlier_indices else pd.DataFrame()
    
    # Store results
    outlier_results['outlier_count'] = len(outlier_indices)
    outlier_results['outlier_percentage'] = len(outlier_indices) / len(df) * 100 if len(df) > 0 else 0
    
    return outliers, outlier_results