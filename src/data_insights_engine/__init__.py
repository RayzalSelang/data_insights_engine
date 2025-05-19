"""
Data Insights Engine - A comprehensive data analysis platform.
"""

from .data_loader import load_data, get_sample_data
from .data_processor import clean_data, preprocess_data
from .analyzer import generate_summary_stats, analyze_correlations, detect_outliers
from .visualizer import create_histogram, create_scatterplot, create_heatmap, create_barplot, create_timeseries

__version__ = "0.1.0"