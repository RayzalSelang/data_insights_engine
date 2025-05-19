"""
Data loading functionality for the Data Insights Engine.
Handles importing data from various file formats and providing sample datasets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def load_data(file_path):
    """
    Load data from a file in various formats.
    
    Parameters:
    -----------
    file_path : str
        The path to the file to be loaded
        
    Returns:
    --------
    pandas.DataFrame
        The loaded data as a pandas DataFrame
    """
    # Get the file extension to determine how to load it
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        # Try different encodings and delimiters for CSV files
        try:
            return pd.read_csv(file_path)
        except UnicodeDecodeError:
            # If UTF-8 fails, try other encodings
            return pd.read_csv(file_path, encoding='latin1')
        except pd.errors.ParserError:
            # If comma delimiter fails, try other delimiters
            return pd.read_csv(file_path, sep=';')
    elif file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif file_extension == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def get_sample_data(dataset_name="sales"):
    """
    Generate sample datasets for demonstration purposes.
    
    Parameters:
    -----------
    dataset_name : str, default="sales"
        Name of the sample dataset to generate: "sales", "customer", or "product"
        
    Returns:
    --------
    pandas.DataFrame
        The generated sample dataset
    """
    if dataset_name == "sales":
        return _generate_sales_data()
    elif dataset_name == "customer":
        return _generate_customer_data()
    elif dataset_name == "product":
        return _generate_product_data()
    else:
        raise ValueError(f"Unknown sample dataset: {dataset_name}")

def _generate_sales_data(num_records=500):
    """Generate a sample sales dataset."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate dates for the past year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days)]
    
    # Define regions, categories, and customer segments
    regions = ['North', 'South', 'East', 'West']
    categories = ['Electronics', 'Furniture', 'Office Supplies', 'Clothing']
    customer_segments = ['Consumer', 'Corporate', 'Home Office']
    
    # Generate random data
    df = pd.DataFrame({
        'order_id': [f'ORD-{i:06d}' for i in range(1, num_records + 1)],
        'order_date': np.random.choice(dates, num_records),
        'region': np.random.choice(regions, num_records),
        'category': np.random.choice(categories, num_records),
        'customer_segment': np.random.choice(customer_segments, num_records),
        'sales': np.random.lognormal(mean=5.0, sigma=1.0, size=num_records),
        'quantity': np.random.randint(1, 11, num_records),
        'discount': np.random.choice([0, 0.05, 0.1, 0.15, 0.2], num_records),
        'profit': np.random.normal(loc=100, scale=50, size=num_records)
    })
    
    # Add some correlations
    df['profit'] = df['sales'] * 0.3 - df['quantity'] * 5
    
    # Add some seasonal patterns
    for i, row in df.iterrows():
        month = row['order_date'].month
        # Higher sales in holiday season
        if month in [11, 12]:
            df.at[i, 'sales'] = row['sales'] * 1.5
            df.at[i, 'profit'] = row['profit'] * 1.3
        # Lower sales in summer
        elif month in [6, 7, 8]:
            df.at[i, 'sales'] = row['sales'] * 0.8
            df.at[i, 'profit'] = row['profit'] * 0.7
    
    # Round values for readability
    df['sales'] = df['sales'].round(2)
    df['profit'] = df['profit'].round(2)
    
    return df

def _generate_customer_data(num_records=200):
    """Generate a sample customer dataset."""
    # Implementation will go here
    # For now, return an empty DataFrame with appropriate columns
    return pd.DataFrame(columns=['customer_id', 'name', 'segment', 'age', 'income'])

def _generate_product_data(num_records=100):
    """Generate a sample product dataset."""
    # Implementation will go here
    # For now, return an empty DataFrame with appropriate columns
    return pd.DataFrame(columns=['product_id', 'name', 'category', 'price', 'cost'])