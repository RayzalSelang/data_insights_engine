"""
Visualization functions for the Data Insights Engine.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def create_histogram(df, column, bins=None, title=None):
    """
    Create a histogram visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to visualize
    column : str
        The column to plot
    bins : int, optional
        The number of bins to use
    title : str, optional
        The title of the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The histogram figure
    """
    if title is None:
        title = f'Distribution of {column}'
    
    fig = px.histogram(
        df, 
        x=column,
        nbins=bins,
        title=title,
        labels={column: column},
        opacity=0.7
    )
    
    # Add mean line
    mean_val = df[column].mean()
    fig.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                 annotation_text=f"Mean: {mean_val:.2f}", 
                 annotation_position="top right")
    
    # Add median line
    median_val = df[column].median()
    fig.add_vline(x=median_val, line_dash="dash", line_color="green", 
                 annotation_text=f"Median: {median_val:.2f}", 
                 annotation_position="top left")
    
    # Improve layout
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Frequency",
        template="plotly_white"
    )
    
    return fig

def create_scatterplot(df, x_col, y_col, color=None, size=None, title=None):
    """
    Create a scatter plot visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to visualize
    x_col : str
        The column to plot on the x-axis
    y_col : str
        The column to plot on the y-axis
    color : str, optional
        The column to use for color encoding
    size : str, optional
        The column to use for size encoding
    title : str, optional
        The title of the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The scatter plot figure
    """
    if title is None:
        title = f'{y_col} vs {x_col}'
    
    fig = px.scatter(
        df, 
        x=x_col, 
        y=y_col,
        color=color,
        size=size,
        title=title,
        labels={x_col: x_col, y_col: y_col},
        opacity=0.7
    )
    
    # Add trendline if no color grouping
    if color is None:
        fig.update_layout(
            shapes=[
                dict(
                    type='line',
                    xref='x', yref='y',
                    x0=df[x_col].min(), y0=df[y_col].min(),
                    x1=df[x_col].max(), y1=df[y_col].max(),
                    line=dict(color='red', dash='dash')
                )
            ]
        )
    
    # Improve layout
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white"
    )
    
    return fig

def create_heatmap(corr_matrix, title="Correlation Matrix"):
    """
    Create a heatmap visualization of a correlation matrix.
    
    Parameters:
    -----------
    corr_matrix : pandas.DataFrame
        The correlation matrix to visualize
    title : str, default="Correlation Matrix"
        The title of the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The heatmap figure
    """
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title=title,
        labels=dict(x="Variable", y="Variable", color="Correlation")
    )
    
    # Improve layout
    fig.update_layout(
        template="plotly_white",
        xaxis_title="",
        yaxis_title=""
    )
    
    return fig

def create_barplot(df, x_col, y_col, title=None):
    """
    Create a bar plot visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to visualize
    x_col : str
        The column for the x-axis (categories)
    y_col : str
        The column for the y-axis (values)
    title : str, optional
        The title of the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The bar plot figure
    """
    if title is None:
        title = f'{y_col} by {x_col}'
    
    fig = px.bar(
        df, 
        x=x_col, 
        y=y_col,
        title=title,
        labels={x_col: x_col, y_col: y_col}
    )
    
    # Improve layout
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white"
    )
    
    return fig

def create_timeseries(df, date_col, value_col, title=None):
    """
    Create a time series visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to visualize
    date_col : str
        The column containing the dates
    value_col : str
        The column containing the values
    title : str, optional
        The title of the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The time series figure
    """
    if title is None:
        title = f'{value_col} over Time'
    
    fig = px.line(
        df, 
        x=date_col, 
        y=value_col,
        title=title,
        labels={date_col: "Date", value_col: value_col}
    )
    
    # Add markers for data points
    fig.update_traces(mode='lines+markers')
    
    # Improve layout
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=value_col,
        template="plotly_white"
    )
    
    return fig

def create_boxplot(df, x_col, y_col, title=None):
    """
    Create a box plot visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to visualize
    x_col : str
        The column for the x-axis (categories)
    y_col : str
        The column for the y-axis (values)
    title : str, optional
        The title of the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The box plot figure
    """
    if title is None:
        title = f'Distribution of {y_col} by {x_col}'
    
    fig = px.box(
        df, 
        x=x_col, 
        y=y_col,
        title=title,
        labels={x_col: x_col, y_col: y_col}
    )
    
    # Improve layout
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white"
    )
    
    return fig

def create_piechart(df, values, names, title=None):
    """
    Create a pie chart visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to visualize
    values : str
        The column containing the values
    names : str
        The column containing the category names
    title : str, optional
        The title of the plot
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The pie chart figure
    """
    if title is None:
        title = f'Distribution of {values}'
    
    fig = px.pie(
        df, 
        values=values, 
        names=names,
        title=title
    )
    
    # Improve layout
    fig.update_layout(
        template="plotly_white"
    )
    
    return fig

def create_dashboard(df, numeric_cols=None, categorical_cols=None, date_col=None):
    """
    Create a comprehensive dashboard with multiple plots.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset to visualize
    numeric_cols : list, optional
        List of numeric columns to include
    categorical_cols : list, optional
        List of categorical columns to include
    date_col : str, optional
        Date column for time series
        
    Returns:
    --------
    dict
        Dictionary of plotly figures
    """
    dashboard = {}
    
    # Automatically detect column types if not provided
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if date_col is None:
        # Try to find a date column
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if date_cols:
            date_col = date_cols[0]
        else:
            # Try to convert string columns to dates
            for col in df.select_dtypes(include=['object']).columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        pd.to_datetime(df[col])
                        date_col = col
                        break
                    except:
                        pass
    
    # Create histograms for numeric columns
    if numeric_cols:
        for col in numeric_cols[:3]:  # Limit to first 3 columns
            dashboard[f'histogram_{col}'] = create_histogram(df, col)
    
    # Create bar plots for categorical columns
    if categorical_cols and numeric_cols:
        for cat_col in categorical_cols[:2]:  # Limit to first 2 categorical columns
            # Group by the categorical column and calculate the mean of the first numeric column
            value_col = numeric_cols[0]
            grouped_data = df.groupby(cat_col)[value_col].mean().reset_index()
            dashboard[f'barplot_{cat_col}_{value_col}'] = create_barplot(grouped_data, cat_col, value_col)
    
    # Create time series if date column exists
    if date_col and numeric_cols:
        # Prepare data
        df_time = df.copy()
        
        # Convert to datetime if needed
        if df_time[date_col].dtype != 'datetime64[ns]':
            try:
                df_time[date_col] = pd.to_datetime(df_time[date_col])
            except:
                # Skip time series if conversion fails
                pass
        
        # Create time series if conversion succeeded
        if df_time[date_col].dtype == 'datetime64[ns]':
            value_col = numeric_cols[0]
            # Group by month
            df_monthly = df_time.groupby(pd.Grouper(key=date_col, freq='M'))[value_col].mean().reset_index()
            dashboard[f'timeseries_{value_col}'] = create_timeseries(df_monthly, date_col, value_col)
    
    # Create correlation heatmap if multiple numeric columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        dashboard['heatmap'] = create_heatmap(corr_matrix)
    
    # Create scatter plot for top correlated variables if applicable
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        # Get upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Find highest absolute correlation
        max_corr = upper.abs().max().max()
        if max_corr > 0.3:  # Only if there's a meaningful correlation
            # Find the variables with the highest correlation
            max_idx = upper.abs().stack().idxmax()
            var1, var2 = max_idx[0], max_idx[1]
            dashboard[f'scatterplot_{var1}_{var2}'] = create_scatterplot(df, var1, var2)
    
    return dashboard