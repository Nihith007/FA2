import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Crypto Volatility Visualizer",
    page_icon="₿",
    layout="wide"
)

# Title and description
st.title("₿ Crypto Volatility Visualizer")
st.markdown("### Simulating Market Swings with Mathematics for AI and Python")
st.markdown("---")

# Sidebar controls
st.sidebar.header("Dashboard Controls")
st.sidebar.markdown("Adjust the parameters below to explore Bitcoin price volatility")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload Bitcoin CSV file", type=['csv'])

# Sample data function
@st.cache_data
def create_sample_data():
    """Create sample Bitcoin data if no file is uploaded"""
    dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='H')
    np.random.seed(42)
    
    base_price = 45000
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # Simulate price movement with volatility
        change = np.random.normal(0, 500)  # Random walk
        current_price += change
        prices.append(current_price)
    
    df = pd.DataFrame({
        'Timestamp': dates,
        'Open': prices,
        'High': [p + np.random.uniform(100, 500) for p in prices],
        'Low': [p - np.random.uniform(100, 500) for p in prices],
        'Close': prices,
        'Volume': np.random.uniform(1000, 10000, len(dates))
    })
    
    return df

# Load and prepare data
@st.cache_data
def load_data(file):
    """Load and prepare Bitcoin dataset"""
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = create_sample_data()
    
    # Convert timestamp to datetime
    if 'Timestamp' in df.columns:
        # Handle Unix timestamp or date string
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        except:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    elif 'Date' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Date'])
    
    # Rename for consistency
    df = df.rename(columns={'Close': 'Price'})
    
    # Handle missing values
    df = df.dropna()
    
    # Sort by timestamp
    df = df.sort_values('Timestamp')
    
    return df

# Load the data
df = load_data(uploaded_file)

# Sidebar filters
st.sidebar.subheader("Time Period Selection")
date_range = st.sidebar.slider(
    "Select number of days to display:",
    min_value=1,
    max_value=min(30, len(df) // 24),
    value=min(7, len(df) // 24)
)

# Filter data based on selection
df_filtered = df.tail(date_range * 24) if len(df) > date_range * 24 else df

# Calculate volatility metrics
def calculate_volatility(data):
    """Calculate volatility index"""
    returns = data['Price'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(len(returns))
    return volatility

def calculate_drift(data):
    """Calculate average price drift"""
    drift = (data['Price'].iloc[-1] - data['Price'].iloc[0]) / data['Price'].iloc[0] * 100
    return drift

# Sidebar pattern selection
st.sidebar.subheader("Volatility Analysis")
show_volatility_bands = st.sidebar.checkbox("Show Volatility Bands", value=True)
show_volume = st.sidebar.checkbox("Show Volume Analysis", value=True)

# Calculate metrics
volatility_index = calculate_volatility(df_filtered)
avg_drift = calculate_drift(df_filtered)

# Display key metrics
st.sidebar.markdown("---")
st.sidebar.subheader("Key Metrics")
st.sidebar.metric("Volatility Index", f"{volatility_index:.2f}%")
st.sidebar.metric("Average Drift", f"{avg_drift:.2f}%")
st.sidebar.metric("Current Price", f"${df_filtered['Price'].iloc[-1]:,.2f}")

# Main dashboard layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Bitcoin Price Over Time")
    
    # Create price chart
    fig_price = go.Figure()
    
    # Add price line
    fig_price.add_trace(go.Scatter(
        x=df_filtered['Timestamp'],
        y=df_filtered['Price'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add volatility bands if selected
    if show_volatility_bands:
        rolling_mean = df_filtered['Price'].rolling(window=24).mean()
        rolling_std = df_filtered['Price'].rolling(window=24).std()
        
        fig_price.add_trace(go.Scatter(
            x=df_filtered['Timestamp'],
            y=rolling_mean + 2*rolling_std,
            mode='lines',
            name='Upper Band',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig_price.add_trace(go.Scatter(
            x=df_filtered['Timestamp'],
            y=rolling_mean - 2*rolling_std,
            mode='lines',
            name='Lower Band',
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.2)',
            fill='tonexty',
            showlegend=True
        ))
    
    fig_price.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_price, use_container_width=True)

with col2:
    st.subheader("Price Statistics")
    
    # Display statistics
    stats_df = pd.DataFrame({
        'Metric': ['Min Price', 'Max Price', 'Average Price', 'Price Range'],
        'Value': [
            f"${df_filtered['Price'].min():,.2f}",
            f"${df_filtered['Price'].max():,.2f}",
            f"${df_filtered['Price'].mean():,.2f}",
            f"${df_filtered['Price'].max() - df_filtered['Price'].min():,.2f}"
        ]
    })
    
    st.dataframe(stats_df, hide_index=True, use_container_width=True)
    
    # Price distribution
    st.markdown("##### Price Distribution")
    fig_hist = px.histogram(
        df_filtered, 
        x='Price', 
        nbins=30,
        color_discrete_sequence=['#1f77b4']
    )
    fig_hist.update_layout(
        showlegend=False,
        height=250,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# High vs Low Comparison
st.subheader("High vs Low Price Comparison")
fig_highlow = go.Figure()

fig_highlow.add_trace(go.Scatter(
    x=df_filtered['Timestamp'],
    y=df_filtered['High'],
    mode='lines',
    name='High',
    line=dict(color='green', width=1.5)
))

fig_highlow.add_trace(go.Scatter(
    x=df_filtered['Timestamp'],
    y=df_filtered['Low'],
    mode='lines',
    name='Low',
    line=dict(color='red', width=1.5),
    fill='tonexty',
    fillcolor='rgba(0, 255, 0, 0.1)'
))

fig_highlow.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode='x unified',
    height=300
)

st.plotly_chart(fig_highlow, use_container_width=True)

# Volume Analysis
if show_volume:
    st.subheader("Volume Analysis")
    
    # Calculate daily volatility (High - Low)
    df_filtered['Daily_Volatility'] = df_filtered['High'] - df_filtered['Low']
    
    fig_volume = go.Figure()
    
    # Volume bars
    fig_volume.add_trace(go.Bar(
        x=df_filtered['Timestamp'],
        y=df_filtered['Volume'],
        name='Volume',
        marker_color='lightblue'
    ))
    
    fig_volume.update_layout(
        xaxis_title="Date",
        yaxis_title="Volume",
        hovermode='x unified',
        height=300
    )
    
    st.plotly_chart(fig_volume, use_container_width=True)

# Stable vs Volatile Periods
st.subheader("Stable vs Volatile Periods Analysis")

col3, col4 = st.columns(2)

with col3:
    # Calculate rolling volatility
    df_filtered['Rolling_Volatility'] = df_filtered['Price'].rolling(window=24).std()
    
    fig_vol_trend = go.Figure()
    fig_vol_trend.add_trace(go.Scatter(
        x=df_filtered['Timestamp'],
        y=df_filtered['Rolling_Volatility'],
        mode='lines',
        name='Volatility',
        line=dict(color='orange', width=2)
    ))
    
    fig_vol_trend.update_layout(
        title="Rolling Volatility (24-hour window)",
        xaxis_title="Date",
        yaxis_title="Standard Deviation",
        height=300
    )
    
    st.plotly_chart(fig_vol_trend, use_container_width=True)

with col4:
    # Identify stable vs volatile periods
    median_vol = df_filtered['Rolling_Volatility'].median()
    df_filtered['Period_Type'] = df_filtered['Rolling_Volatility'].apply(
        lambda x: 'Volatile' if x > median_vol else 'Stable'
    )
    
    period_counts = df_filtered['Period_Type'].value_counts()
    
    fig_periods = px.pie(
        values=period_counts.values,
        names=period_counts.index,
        title="Stable vs Volatile Periods Distribution",
        color=period_counts.index,
        color_discrete_map={'Stable': 'lightgreen', 'Volatile': 'salmon'}
    )
    fig_periods.update_layout(height=300)
    
    st.plotly_chart(fig_periods, use_container_width=True)

# Data preview section
with st.expander("📊 View Raw Data"):
    st.dataframe(df_filtered.head(50), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    **Crypto Volatility Visualizer** | Built with Streamlit & Python  
    *FinTechLab Pvt. Ltd. - Mathematics for AI Project*
""")
