"""
Streamlit web application for the Data Insights Engine.
"""

import streamlit as st
import pandas as pd
import os
import sys

# Add the src directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_insights_engine.data_loader import load_data, get_sample_data
from src.data_insights_engine.data_processor import clean_data, preprocess_data
from src.data_insights_engine.analyzer import generate_summary_stats, analyze_correlations, detect_outliers
from src.data_insights_engine.visualizer import create_histogram, create_scatterplot, create_heatmap, create_barplot, create_timeseries

# Set page title and layout
st.set_page_config(page_title="Data Insights Engine", page_icon="📊", layout="wide")

# Add header
st.title("📊 Data Insights Engine")
st.write("A comprehensive data analysis platform for deriving actionable insights from your data.")

# Sidebar for data loading options
st.sidebar.title("Data Options")

# Data source selection
data_source = st.sidebar.radio(
    "Select data source:",
    ("Upload your data", "Use sample data")
)

# Initialize session state for dataframe
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None

# Data loading
if data_source == "Upload your data":
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            # Save the file temporarily
            file_path = os.path.join("data", "raw", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load the data
            st.session_state.df = load_data(file_path)
            st.sidebar.success(f"Data loaded: {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"Error loading data: {e}")
else:
    sample_options = st.sidebar.selectbox(
        "Select sample dataset:",
        ("Sales Data", "Customer Data", "Product Data")
    )

    if st.sidebar.button("Load Sample Data"):
        dataset_name = sample_options.lower().split()[0]  # "Sales Data" -> "sales"
        st.session_state.df = get_sample_data(dataset_name)
        st.sidebar.success(f"Sample {sample_options} loaded")

# Process data if available
if st.session_state.df is not None:
    # Show data processing options
    st.sidebar.title("Data Processing")

    if st.sidebar.button("Clean and Preprocess Data"):
        # Clean data
        df_clean = clean_data(st.session_state.df)

        # Preprocess data
        st.session_state.df_processed = preprocess_data(df_clean)

        st.sidebar.success("Data cleaned and preprocessed")

    # Analysis options
    st.sidebar.title("Analysis Options")

    analysis_type = st.sidebar.selectbox(
        "Select analysis type:",
        ("Data Overview", "Distribution Analysis", "Correlation Analysis", "Time Series Analysis")
    )

    # Main content area
    if analysis_type == "Data Overview":
        st.header("Data Overview")

        # Show raw data sample
        st.subheader("Raw Data Sample")
        st.dataframe(st.session_state.df.head())

        # Show data info
        st.subheader("Data Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Number of rows: {st.session_state.df.shape[0]}")
            st.write(f"Number of columns: {st.session_state.df.shape[1]}")
        with col2:
            st.write(f"Missing values: {st.session_state.df.isna().sum().sum()}")
            st.write(f"Duplicate rows: {st.session_state.df.duplicated().sum()}")

        # Show processed data if available
        if st.session_state.df_processed is not None:
            st.subheader("Processed Data Sample")
            st.dataframe(st.session_state.df_processed.head())

            # Summary statistics
            st.subheader("Summary Statistics")
            summary_stats = generate_summary_stats(st.session_state.df_processed)
            st.dataframe(summary_stats)

    elif analysis_type == "Distribution Analysis" and st.session_state.df_processed is not None:
        st.header("Distribution Analysis")

        # Select column for analysis
        numeric_cols = st.session_state.df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_col = st.selectbox("Select column for analysis:", numeric_cols)

        # Show histogram
        st.subheader(f"Distribution of {selected_col}")
        fig = create_histogram(st.session_state.df_processed, selected_col)
        st.plotly_chart(fig, use_container_width=True)

        # Show statistics for the selected column
        st.subheader(f"Statistics for {selected_col}")
        stats = st.session_state.df_processed[selected_col].describe()
        st.write(stats)

        # Detect outliers
        outliers, outlier_stats = detect_outliers(st.session_state.df_processed[[selected_col]])
        st.subheader(f"Outlier Detection for {selected_col}")
        st.write(f"Number of outliers: {outlier_stats['outlier_count']}")
        st.write(f"Percentage of outliers: {outlier_stats['outlier_percentage']:.2f}%")

    elif analysis_type == "Correlation Analysis" and st.session_state.df_processed is not None:
        st.header("Correlation Analysis")

        # Calculate and show correlation matrix
        numeric_cols = st.session_state.df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        corr_matrix = st.session_state.df_processed[numeric_cols].corr()

        st.subheader("Correlation Heatmap")
        fig = create_heatmap(corr_matrix)
        st.plotly_chart(fig, use_container_width=True)

        # Show significant correlations
        st.subheader("Significant Correlations")
        threshold = st.slider("Correlation threshold:", 0.0, 1.0, 0.5, 0.05)

        # Analyze correlations with the selected threshold
        correlations = analyze_correlations(st.session_state.df_processed, threshold=threshold)

        if not correlations.empty:
            st.dataframe(correlations)

            # Allow user to select a correlation pair for visualization
            st.subheader("Visualize Correlation")
            correlation_pairs = [f"{row['Variable 1']} vs {row['Variable 2']}" for _, row in correlations.iterrows()]
            selected_pair = st.selectbox("Select correlation pair:", correlation_pairs)

            if selected_pair:
                var1, var2 = selected_pair.split(" vs ")

                # Create scatter plot
                fig = create_scatterplot(st.session_state.df_processed, var1, var2)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No correlations found with threshold >= {threshold}")

    elif analysis_type == "Time Series Analysis" and st.session_state.df_processed is not None:
        st.header("Time Series Analysis")

        # Find date columns
        date_cols = st.session_state.df_processed.select_dtypes(include=['datetime64']).columns.tolist()

        # Add columns that might be dates but not recognized as datetime
        for col in st.session_state.df_processed.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                if col not in date_cols:
                    try:
                        pd.to_datetime(st.session_state.df_processed[col])
                        date_cols.append(col)
                    except:
                        pass

        if date_cols:
            # Select date column
            selected_date_col = st.selectbox("Select date column:", date_cols)

            # Ensure column is datetime
            if st.session_state.df_processed[selected_date_col].dtype != 'datetime64[ns]':
                df_temp = st.session_state.df_processed.copy()
                df_temp[selected_date_col] = pd.to_datetime(df_temp[selected_date_col])
            else:
                df_temp = st.session_state.df_processed.copy()

            # Select value column
            numeric_cols = df_temp.select_dtypes(include=['int64', 'float64']).columns.tolist()
            selected_value_col = st.selectbox("Select value column:", numeric_cols)

            # Group by date
            group_options = ["Day", "Week", "Month", "Quarter", "Year"]
            group_by = st.selectbox("Group by:", group_options)

            freq_map = {
                "Day": "D",
                "Week": "W",
                "Month": "M",
                "Quarter": "Q",
                "Year": "Y"
            }

            # Group by time period
            df_grouped = df_temp.groupby(pd.Grouper(key=selected_date_col, freq=freq_map[group_by]))[selected_value_col].mean().reset_index()

            # Create time series plot
            st.subheader(f"{selected_value_col} over Time (by {group_by})")
            fig = create_timeseries(df_grouped, selected_date_col, selected_value_col)
            st.plotly_chart(fig, use_container_width=True)

            # Show statistics
            st.subheader("Time Series Statistics")
            st.write(f"Min date: {df_grouped[selected_date_col].min()}")
            st.write(f"Max date: {df_grouped[selected_date_col].max()}")
            st.write(f"Number of periods: {len(df_grouped)}")
            st.write(f"Average {selected_value_col}: {df_grouped[selected_value_col].mean():.2f}")

            # Show data table
            st.subheader("Time Series Data")
            st.dataframe(df_grouped)
        else:
            st.info("No date/time columns found in the dataset. Please ensure your data contains date or time information for time series analysis.")
else:
    # Display welcome message when no data is loaded
    st.write("👈 Please select a data source from the sidebar to get started.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("About Data Insights Engine")
        st.write("""
        Data Insights Engine is a comprehensive data analysis platform that helps you:

        * 🧹 **Clean and preprocess** your data automatically
        * 📊 **Visualize distributions** of your variables
        * 🔍 **Discover correlations** between different features
        * 📈 **Analyze time series** data for trends and patterns
        * 🔮 **Detect outliers** in your dataset

        Get started by uploading your own data or using one of our sample datasets.
        """)

    with col2:
        st.subheader("Sample Data Overview")
        st.write("""
        The Data Insights Engine comes with three sample datasets:

        1. **Sales Data** - Contains sales transactions with dates, regions, categories, and financial metrics
        2. **Customer Data** - Contains customer profiles with demographic and behavioral data
        3. **Product Data** - Contains product information with pricing, cost, and category data

        These sample datasets are perfect for exploring the capabilities of the Data Insights Engine.
        """)

# Add footer
st.markdown("---")
st.markdown("Created with ❤️ by [Your Name]")
