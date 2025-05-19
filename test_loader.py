"""
Script to test the data_loader module.
"""

import os
import sys
import pandas as pd

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_insights_engine.data_loader import get_sample_data

def main():
    print("Testing data_loader module...")
    
    # Generate sample sales data
    print("\nGenerating sample sales data...")
    sales_data = get_sample_data("sales")
    
    # Print information about the data
    print(f"Shape: {sales_data.shape}")
    print("\nColumns:")
    for col in sales_data.columns:
        print(f"  - {col} ({sales_data[col].dtype})")
    
    # Print a few rows
    print("\nSample data (first 5 rows):")
    print(sales_data.head())
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()