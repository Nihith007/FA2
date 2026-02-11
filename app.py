import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
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
st.markdown("*Using sine, cosine, random noise, and integrals to model cryptocurrency volatility*")
st.markdown("---")

# Sidebar - Mode Selection
st.sidebar.header("🎯 Dashboard Mode")
mode = st.sidebar.radio(
    "Choose your analysis mode:",
    ["📊 Real Bitcoin Data", "🧮 Mathematical Simulation", "🔍 Compare Both"],
    help="Switch between real data analysis and mathematical simulations"
)

st.sidebar.markdown("---")

# =============================================================================
# REAL DATA MODE CONTROLS
# =============================================================================
if mode == "📊 Real Bitcoin Data":
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Bitcoin CSV file",
        type=['csv'],
        help="CSV should have: Timestamp, Open, High, Low, Close, Volume"
    )
    
    if uploaded_file is None:
        st.sidebar.info("💡 No file uploaded - using sample Bitcoin data")
    
    st.sidebar.subheader("⏱️ Time Range")
    if uploaded_file is not None:
        days_to_show = st.sidebar.slider("Days to display", 1, 30, 7)
    else:
        days_to_show = st.sidebar.slider("Days to display", 1, 30, 7)
    
    st.sidebar.subheader("📈 Display Options")
    show_volatility_bands = st.sidebar.checkbox("Show Volatility Bands", value=True)
    show_volume = st.sidebar.checkbox("Show Volume Analysis", value=True)

# =============================================================================
# MATHEMATICAL SIMULATION MODE CONTROLS
# =============================================================================
elif mode == "🧮 Mathematical Simulation":
    st.sidebar.header("🎛️ Mathematical Controls")
    
    # Pattern selection dropdown
    st.sidebar.subheader("📊 Pattern Type")
    pattern_type = st.sidebar.selectbox(
        "Choose price swing pattern:",
        ["Sine Wave (Smooth Cycles)", 
         "Cosine Wave (Smooth Cycles)", 
         "Random Noise (Chaotic Jumps)",
         "Sine + Noise (Realistic Market)",
         "Cosine + Noise (Realistic Market)",
         "Combined Waves (Complex Pattern)"]
    )
    
    # Mathematical parameter sliders
    st.sidebar.subheader("🔧 Wave Parameters")
    
    amplitude = st.sidebar.slider(
        "Amplitude (Swing Size)",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Controls how big the price swings are"
    )
    
    frequency = st.sidebar.slider(
        "Frequency (Swing Speed)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Controls oscillation speed"
    )
    
    drift = st.sidebar.slider(
        "Drift (Long-term Slope)",
        min_value=-50,
        max_value=50,
        value=10,
        step=5,
        help="Long-term trend using integrals"
    )
    
    noise_level = st.sidebar.slider(
        "Noise Level (Randomness)",
        min_value=0,
        max_value=500,
        value=100,
        step=10,
        help="Amount of random jumps"
    )
    
    st.sidebar.subheader("⏱️ Time Range")
    num_days = st.sidebar.slider("Number of Days", 1, 30, 7)

# =============================================================================
# COMPARISON MODE CONTROLS
# =============================================================================
else:  # Compare Both mode
    st.sidebar.header("📁 Real Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Bitcoin CSV",
        type=['csv']
    )
    
    st.sidebar.header("🧮 Simulation Settings")
    pattern_type = st.sidebar.selectbox(
        "Pattern:",
        ["Sine Wave (Smooth Cycles)", 
         "Cosine Wave (Smooth Cycles)", 
         "Sine + Noise (Realistic Market)"]
    )
    
    amplitude = st.sidebar.slider("Amplitude", 100, 5000, 1000, 100)
    frequency = st.sidebar.slider("Frequency", 0.1, 5.0, 1.0, 0.1)
    drift = st.sidebar.slider("Drift", -50, 50, 10, 5)
    noise_level = st.sidebar.slider("Noise", 0, 500, 100, 10)
    
    days_to_show = st.sidebar.slider("Days to display", 1, 30, 7)
    num_days = days_to_show

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data
def create_sample_bitcoin_data(days=30):
    """Create realistic sample Bitcoin data"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), end=datetime.now(), freq='H')
    np.random.seed(42)
    
    base_price = 45000
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        change = np.random.normal(0, 500)
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

@st.cache_data
def load_real_data(file):
    """Load and prepare real Bitcoin dataset"""
    if file is not None:
        df = pd.read_csv(file)
    else:
        df = create_sample_bitcoin_data()
    
    # Convert timestamp
    if 'Timestamp' in df.columns:
        try:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        except:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    elif 'Date' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Date'])
    
    # Ensure required columns exist
    if 'Close' in df.columns:
        df['Price'] = df['Close']
    
    # Handle missing values
    df = df.dropna()
    
    # Sort by timestamp
    df = df.sort_values('Timestamp')
    
    return df

@st.cache_data
def generate_mathematical_data(pattern, amp, freq, drift_val, noise, days):
    """Generate cryptocurrency price data using mathematical functions"""
    
    hours = days * 24
    time = np.linspace(0, days, hours)
    base_price = 45000
    prices = np.zeros(hours)
    
    # Generate pattern
    if pattern == "Sine Wave (Smooth Cycles)":
        prices = base_price + amp * np.sin(2 * np.pi * freq * time / days)
        
    elif pattern == "Cosine Wave (Smooth Cycles)":
        prices = base_price + amp * np.cos(2 * np.pi * freq * time / days)
        
    elif pattern == "Random Noise (Chaotic Jumps)":
        prices[0] = base_price
        for i in range(1, hours):
            prices[i] = prices[i-1] + np.random.normal(0, noise)
            
    elif pattern == "Sine + Noise (Realistic Market)":
        sine_component = amp * np.sin(2 * np.pi * freq * time / days)
        noise_component = np.cumsum(np.random.normal(0, noise, hours))
        prices = base_price + sine_component + noise_component
        
    elif pattern == "Cosine + Noise (Realistic Market)":
        cosine_component = amp * np.cos(2 * np.pi * freq * time / days)
        noise_component = np.cumsum(np.random.normal(0, noise, hours))
        prices = base_price + cosine_component + noise_component
        
    elif pattern == "Combined Waves (Complex Pattern)":
        wave1 = amp * np.sin(2 * np.pi * freq * time / days)
        wave2 = (amp/2) * np.cos(2 * np.pi * freq * 2 * time / days)
        wave3 = (amp/3) * np.sin(2 * np.pi * freq * 3 * time / days)
        prices = base_price + wave1 + wave2 + wave3
    
    # Add drift
    drift_component = drift_val * time / days
    prices = prices + drift_component
    prices = np.maximum(prices, 100)
    
    # Create timestamps
    start_date = datetime.now() - timedelta(days=days)
    timestamps = [start_date + timedelta(hours=i) for i in range(hours)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Timestamp': timestamps,
        'Time': time,
        'Price': prices,
        'Open': prices,
        'Close': prices,
        'High': prices + np.random.uniform(50, 200, hours),
        'Low': prices - np.random.uniform(50, 200, hours),
        'Volume': np.random.uniform(1000, 50000, hours)
    })
    
    return df

# =============================================================================
# METRIC CALCULATION FUNCTIONS
# =============================================================================

def calculate_volatility(data):
    """Calculate volatility index"""
    returns = data['Price'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(len(returns)) * 100
    return volatility

def calculate_drift_metric(data):
    """Calculate average price drift"""
    drift_pct = (data['Price'].iloc[-1] - data['Price'].iloc[0]) / data['Price'].iloc[0] * 100
    return drift_pct

# =============================================================================
# MAIN DASHBOARD RENDERING
# =============================================================================

if mode == "📊 Real Bitcoin Data":
    # =========================================================================
    # REAL DATA MODE
    # =========================================================================
    st.header("📊 Real Bitcoin Data Analysis")
    
    # Load data
    df_real = load_real_data(uploaded_file)
    
    # Filter by days
    df_filtered = df_real.tail(days_to_show * 24) if len(df_real) > days_to_show * 24 else df_real
    
    # Calculate metrics
    volatility = calculate_volatility(df_filtered)
    drift_metric = calculate_drift_metric(df_filtered)
    
    # Display metrics
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Key Metrics")
    st.sidebar.metric("Volatility Index", f"{volatility:.2f}%")
    st.sidebar.metric("Average Drift", f"{drift_metric:+.2f}%")
    st.sidebar.metric("Current Price", f"${df_filtered['Price'].iloc[-1]:,.2f}")
    
    # Main price chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Bitcoin Price Over Time")
        fig_price = go.Figure()
        
        fig_price.add_trace(go.Scatter(
            x=df_filtered['Timestamp'],
            y=df_filtered['Price'],
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ))
        
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
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        st.subheader("Price Statistics")
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
        
        st.markdown("##### Price Distribution")
        fig_hist = px.histogram(df_filtered, x='Price', nbins=30, color_discrete_sequence=['#1f77b4'])
        fig_hist.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # High vs Low
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
        height=300,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_highlow, use_container_width=True)
    
    # Volume
    if show_volume:
        st.subheader("Volume Analysis")
        fig_volume = go.Figure()
        
        fig_volume.add_trace(go.Bar(
            x=df_filtered['Timestamp'],
            y=df_filtered['Volume'],
            name='Volume',
            marker_color='lightblue'
        ))
        
        fig_volume.update_layout(
            xaxis_title="Date",
            yaxis_title="Volume",
            height=300,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # Stable vs Volatile
    st.subheader("Stable vs Volatile Periods")
    col3, col4 = st.columns(2)
    
    with col3:
        df_filtered['Rolling_Volatility'] = df_filtered['Price'].rolling(window=24).std()
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=df_filtered['Timestamp'],
            y=df_filtered['Rolling_Volatility'],
            mode='lines',
            line=dict(color='orange', width=2)
        ))
        
        fig_vol.update_layout(
            title="Rolling Volatility (24-hour)",
            xaxis_title="Date",
            yaxis_title="Std Dev",
            height=300,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col4:
        median_vol = df_filtered['Rolling_Volatility'].median()
        df_filtered['Period_Type'] = df_filtered['Rolling_Volatility'].apply(
            lambda x: 'Volatile' if x > median_vol else 'Stable'
        )
        
        period_counts = df_filtered['Period_Type'].value_counts()
        
        fig_pie = px.pie(
            values=period_counts.values,
            names=period_counts.index,
            title="Period Distribution",
            color=period_counts.index,
            color_discrete_map={'Stable': 'lightgreen', 'Volatile': 'salmon'}
        )
        fig_pie.update_layout(height=300)
        
        st.plotly_chart(fig_pie, use_container_width=True)

elif mode == "🧮 Mathematical Simulation":
    # =========================================================================
    # MATHEMATICAL SIMULATION MODE
    # =========================================================================
    st.header("🧮 Mathematical Simulation")
    
    # Generate data
    df_math = generate_mathematical_data(pattern_type, amplitude, frequency, drift, noise_level, num_days)
    
    # Calculate metrics
    volatility = calculate_volatility(df_math)
    drift_metric = calculate_drift_metric(df_math)
    
    # Display metrics
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Key Metrics")
    st.sidebar.metric("Volatility Index", f"{volatility:.2f}%")
    st.sidebar.metric("Average Drift", f"{drift_metric:+.2f}%")
    st.sidebar.metric("Current Price", f"${df_math['Price'].iloc[-1]:,.2f}")
    
    # Pattern info
    st.subheader(f"📊 Pattern: {pattern_type}")
    
    # Mathematical formula
    with st.expander("🧮 Mathematical Formula"):
        if "Sine" in pattern_type:
            st.latex(r"Price(t) = Base + Amplitude \times \sin(2\pi \times Frequency \times t) + Drift \times t + Noise")
        elif "Cosine" in pattern_type:
            st.latex(r"Price(t) = Base + Amplitude \times \cos(2\pi \times Frequency \times t) + Drift \times t + Noise")
        elif "Random" in pattern_type:
            st.latex(r"Price(t) = Price(t-1) + \mathcal{N}(0, \sigma)")
        
        st.markdown(f"""
        **Parameters:**
        - Base: $45,000
        - Amplitude: ${amplitude:,}
        - Frequency: {frequency}
        - Drift: ${drift}/day
        - Noise: ±${noise_level}
        """)
    
    # Main chart
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(
        x=df_math['Timestamp'],
        y=df_math['Price'],
        mode='lines',
        line=dict(color='#2E86DE', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 222, 0.1)'
    ))
    
    fig_main.update_layout(
        title=f"Simulated Price - {pattern_type}",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_main, use_container_width=True)
    
    # Additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("High vs Low")
        fig_hl = go.Figure()
        fig_hl.add_trace(go.Scatter(x=df_math['Timestamp'], y=df_math['High'], mode='lines', name='High', line=dict(color='green')))
        fig_hl.add_trace(go.Scatter(x=df_math['Timestamp'], y=df_math['Low'], mode='lines', name='Low', line=dict(color='red'), fill='tonexty'))
        fig_hl.update_layout(height=300, template='plotly_white')
        st.plotly_chart(fig_hl, use_container_width=True)
    
    with col2:
        st.subheader("Volume")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=df_math['Timestamp'], y=df_math['Volume'], marker_color='lightblue'))
        fig_vol.update_layout(height=300, template='plotly_white')
        st.plotly_chart(fig_vol, use_container_width=True)
    
    # Educational section
    st.markdown("---")
    st.markdown("### 🎓 Understanding the Mathematics")
    col_edu1, col_edu2, col_edu3 = st.columns(3)
    
    with col_edu1:
        st.markdown("""
        **🌊 Sine/Cosine**
        - Smooth oscillations
        - Amplitude = height
        - Frequency = speed
        """)
    
    with col_edu2:
        st.markdown("""
        **📈 Drift**
        - Long-term trend
        - Integral: ∫ drift dt
        - Cumulative effect
        """)
    
    with col_edu3:
        st.markdown("""
        **🎲 Noise**
        - Random jumps
        - N(0, σ)
        - Market uncertainty
        """)

else:
    # =========================================================================
    # COMPARISON MODE
    # =========================================================================
    st.header("🔍 Comparison: Real Data vs Mathematical Simulation")
    
    # Load/generate both datasets
    df_real = load_real_data(uploaded_file)
    df_real_filtered = df_real.tail(days_to_show * 24) if len(df_real) > days_to_show * 24 else df_real
    
    df_math = generate_mathematical_data(pattern_type, amplitude, frequency, drift, noise_level, num_days)
    
    # Side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Real Bitcoin Data")
        fig_real = go.Figure()
        fig_real.add_trace(go.Scatter(
            x=df_real_filtered['Timestamp'],
            y=df_real_filtered['Price'],
            mode='lines',
            line=dict(color='blue', width=2)
        ))
        fig_real.update_layout(height=350, template='plotly_white', xaxis_title="Time", yaxis_title="Price (USD)")
        st.plotly_chart(fig_real, use_container_width=True)
        
        vol_real = calculate_volatility(df_real_filtered)
        st.metric("Volatility", f"{vol_real:.2f}%")
    
    with col2:
        st.subheader("🧮 Mathematical Simulation")
        fig_math = go.Figure()
        fig_math.add_trace(go.Scatter(
            x=df_math['Timestamp'],
            y=df_math['Price'],
            mode='lines',
            line=dict(color='green', width=2)
        ))
        fig_math.update_layout(height=350, template='plotly_white', xaxis_title="Time", yaxis_title="Price (USD)")
        st.plotly_chart(fig_math, use_container_width=True)
        
        vol_math = calculate_volatility(df_math)
        st.metric("Volatility", f"{vol_math:.2f}%")
    
    # Overlay comparison
    st.subheader("📈 Overlay Comparison")
    fig_overlay = go.Figure()
    fig_overlay.add_trace(go.Scatter(x=df_real_filtered['Timestamp'], y=df_real_filtered['Price'], mode='lines', name='Real Data', line=dict(color='blue')))
    fig_overlay.add_trace(go.Scatter(x=df_math['Timestamp'], y=df_math['Price'], mode='lines', name='Simulation', line=dict(color='green')))
    fig_overlay.update_layout(height=400, template='plotly_white', xaxis_title="Time", yaxis_title="Price (USD)")
    st.plotly_chart(fig_overlay, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    **Crypto Volatility Visualizer** | Mathematics for AI-II - FA-2  
    *Built with Python, Streamlit, NumPy & Plotly* | **FinTechLab Pvt. Ltd.**
""")
