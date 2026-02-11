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

# Sidebar controls
st.sidebar.header("🎛️ Mathematical Controls")
st.sidebar.markdown("Adjust parameters to see how mathematical functions create price patterns")

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
    help="Controls how big the price swings are. Higher = more volatile!"
)

frequency = st.sidebar.slider(
    "Frequency (Swing Speed)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help="Controls how fast prices oscillate. Higher = more rapid changes!"
)

drift = st.sidebar.slider(
    "Drift (Long-term Slope)",
    min_value=-50,
    max_value=50,
    value=10,
    step=5,
    help="Long-term upward or downward trend using integrals"
)

noise_level = st.sidebar.slider(
    "Noise Level (Randomness)",
    min_value=0,
    max_value=500,
    value=100,
    step=10,
    help="Amount of random jumps added to the pattern"
)

# Time parameters
st.sidebar.subheader("⏱️ Time Range")
num_days = st.sidebar.slider(
    "Number of Days",
    min_value=1,
    max_value=30,
    value=7,
    help="How many days of price data to generate"
)

# Generate mathematical price data
@st.cache_data
def generate_price_data(pattern, amp, freq, drift_val, noise, days):
    """Generate cryptocurrency price data using mathematical functions"""
    
    # Create time array (hourly data)
    hours = days * 24
    time = np.linspace(0, days, hours)
    
    # Base price
    base_price = 45000
    
    # Initialize price array
    prices = np.zeros(hours)
    
    # Generate pattern based on selection
    if pattern == "Sine Wave (Smooth Cycles)":
        # Pure sine wave: amplitude * sin(2π * frequency * time)
        prices = base_price + amp * np.sin(2 * np.pi * freq * time / days)
        
    elif pattern == "Cosine Wave (Smooth Cycles)":
        # Pure cosine wave: amplitude * cos(2π * frequency * time)
        prices = base_price + amp * np.cos(2 * np.pi * freq * time / days)
        
    elif pattern == "Random Noise (Chaotic Jumps)":
        # Random walk with noise
        prices[0] = base_price
        for i in range(1, hours):
            prices[i] = prices[i-1] + np.random.normal(0, noise)
            
    elif pattern == "Sine + Noise (Realistic Market)":
        # Sine wave with random noise added
        sine_component = amp * np.sin(2 * np.pi * freq * time / days)
        noise_component = np.cumsum(np.random.normal(0, noise, hours))
        prices = base_price + sine_component + noise_component
        
    elif pattern == "Cosine + Noise (Realistic Market)":
        # Cosine wave with random noise added
        cosine_component = amp * np.cos(2 * np.pi * freq * time / days)
        noise_component = np.cumsum(np.random.normal(0, noise, hours))
        prices = base_price + cosine_component + noise_component
        
    elif pattern == "Combined Waves (Complex Pattern)":
        # Multiple waves combined (fundamental + harmonics)
        wave1 = amp * np.sin(2 * np.pi * freq * time / days)
        wave2 = (amp/2) * np.cos(2 * np.pi * freq * 2 * time / days)
        wave3 = (amp/3) * np.sin(2 * np.pi * freq * 3 * time / days)
        prices = base_price + wave1 + wave2 + wave3
    
    # Add drift (long-term trend using integral/cumulative sum)
    drift_component = drift_val * time / days
    prices = prices + drift_component
    
    # Ensure prices stay positive
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

# Generate data based on current parameters
df = generate_price_data(pattern_type, amplitude, frequency, drift, noise_level, num_days)

# Calculate metrics
def calculate_volatility(data):
    """Calculate volatility index"""
    returns = data['Price'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(len(returns)) * 100
    return volatility

def calculate_drift_metric(data):
    """Calculate average price drift"""
    drift_pct = (data['Price'].iloc[-1] - data['Price'].iloc[0]) / data['Price'].iloc[0] * 100
    return drift_pct

volatility_index = calculate_volatility(df)
avg_drift_metric = calculate_drift_metric(df)

# Display key metrics in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Key Metrics")
st.sidebar.metric("Volatility Index", f"{volatility_index:.2f}%", 
                  help="Measure of price variation")
st.sidebar.metric("Average Drift", f"{avg_drift_metric:+.2f}%",
                  help="Overall price trend")
st.sidebar.metric("Current Price", f"${df['Price'].iloc[-1]:,.2f}",
                  help="Latest simulated price")
st.sidebar.metric("Price Range", f"${df['Price'].max() - df['Price'].min():,.2f}",
                  help="Difference between highest and lowest price")

# Comparison mode
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Comparison Mode")
show_comparison = st.sidebar.checkbox("Compare Stable vs Volatile", value=False,
                                     help="Show two patterns side-by-side")

# Main content area
if not show_comparison:
    # Single pattern view
    st.subheader(f"📊 Price Pattern: {pattern_type}")
    
    # Mathematical explanation
    with st.expander("🧮 Mathematical Formula Used"):
        if "Sine" in pattern_type:
            st.latex(r"Price(t) = Base + Amplitude \times \sin(2\pi \times Frequency \times t) + Drift \times t + Noise")
            st.markdown(f"""
            **Current Parameters:**
            - Base Price: $45,000
            - Amplitude: ${amplitude:,} (swing size)
            - Frequency: {frequency} (cycles per period)
            - Drift: ${drift}/day (long-term slope)
            - Noise: ±${noise_level} (random variation)
            """)
        elif "Cosine" in pattern_type:
            st.latex(r"Price(t) = Base + Amplitude \times \cos(2\pi \times Frequency \times t) + Drift \times t + Noise")
            st.markdown(f"""
            **Current Parameters:**
            - Base Price: $45,000
            - Amplitude: ${amplitude:,} (swing size)
            - Frequency: {frequency} (cycles per period)
            - Drift: ${drift}/day (long-term slope)
            - Noise: ±${noise_level} (random variation)
            """)
        elif "Random" in pattern_type:
            st.latex(r"Price(t) = Price(t-1) + \mathcal{N}(0, \sigma_{noise})")
            st.markdown(f"""
            **Random Walk Model:**
            - Each step adds random noise from normal distribution
            - Noise Level (σ): ${noise_level}
            """)
    
    # Main price chart
    fig_main = go.Figure()
    
    fig_main.add_trace(go.Scatter(
        x=df['Timestamp'],
        y=df['Price'],
        mode='lines',
        name='Price',
        line=dict(color='#2E86DE', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 222, 0.1)'
    ))
    
    fig_main.update_layout(
        title=f"Simulated Cryptocurrency Price - {pattern_type}",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_main, use_container_width=True)
    
    # Two columns for additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 High vs Low Range")
        fig_highlow = go.Figure()
        
        fig_highlow.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df['High'],
            mode='lines',
            name='High',
            line=dict(color='green', width=1)
        ))
        
        fig_highlow.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df['Low'],
            mode='lines',
            name='Low',
            line=dict(color='red', width=1),
            fill='tonexty',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ))
        
        fig_highlow.update_layout(
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            height=300,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_highlow, use_container_width=True)
    
    with col2:
        st.subheader("📊 Volume Analysis")
        fig_volume = go.Figure()
        
        fig_volume.add_trace(go.Bar(
            x=df['Timestamp'],
            y=df['Volume'],
            name='Volume',
            marker_color='lightblue'
        ))
        
        fig_volume.update_layout(
            xaxis_title="Time",
            yaxis_title="Trading Volume",
            height=300,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # Volatility analysis
    st.subheader("📉 Volatility Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Rolling volatility
        window_size = max(24, len(df) // 10)
        df['Rolling_Volatility'] = df['Price'].rolling(window=window_size).std()
        
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=df['Timestamp'],
            y=df['Rolling_Volatility'],
            mode='lines',
            name='Volatility',
            line=dict(color='orange', width=2)
        ))
        
        fig_vol.update_layout(
            title=f"Rolling Volatility ({window_size}-hour window)",
            xaxis_title="Time",
            yaxis_title="Standard Deviation",
            height=300,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col4:
        # Price distribution
        fig_dist = px.histogram(
            df, 
            x='Price',
            nbins=30,
            title="Price Distribution",
            labels={'Price': 'Price (USD)', 'count': 'Frequency'}
        )
        fig_dist.update_layout(
            height=300,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)

else:
    # Comparison mode - Stable vs Volatile
    st.subheader("🔍 Comparison: Stable vs Volatile Money")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💚 Stable Pattern (Small Swings)")
        df_stable = generate_price_data(
            "Sine Wave (Smooth Cycles)",
            amplitude=200,  # Small amplitude
            freq=0.5,
            drift_val=5,
            noise=50,  # Low noise
            days=num_days
        )
        
        fig_stable = go.Figure()
        fig_stable.add_trace(go.Scatter(
            x=df_stable['Timestamp'],
            y=df_stable['Price'],
            mode='lines',
            name='Stable Price',
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ))
        
        fig_stable.update_layout(
            title="Low Volatility Pattern",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            height=300,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_stable, use_container_width=True)
        
        vol_stable = calculate_volatility(df_stable)
        st.metric("Volatility", f"{vol_stable:.2f}%", delta="-Low Risk", delta_color="normal")
        st.metric("Price Range", f"${df_stable['Price'].max() - df_stable['Price'].min():,.2f}")
    
    with col2:
        st.markdown("### 🔴 Volatile Pattern (Big Swings)")
        df_volatile = generate_price_data(
            "Sine + Noise (Realistic Market)",
            amplitude=2000,  # Large amplitude
            freq=2.0,
            drift_val=20,
            noise=300,  # High noise
            days=num_days
        )
        
        fig_volatile = go.Figure()
        fig_volatile.add_trace(go.Scatter(
            x=df_volatile['Timestamp'],
            y=df_volatile['Price'],
            mode='lines',
            name='Volatile Price',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)'
        ))
        
        fig_volatile.update_layout(
            title="High Volatility Pattern",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            height=300,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_volatile, use_container_width=True)
        
        vol_volatile = calculate_volatility(df_volatile)
        st.metric("Volatility", f"{vol_volatile:.2f}%", delta="+High Risk", delta_color="inverse")
        st.metric("Price Range", f"${df_volatile['Price'].max() - df_volatile['Price'].min():,.2f}")
    
    # Side-by-side comparison chart
    st.markdown("### 📊 Direct Comparison")
    fig_compare = go.Figure()
    
    fig_compare.add_trace(go.Scatter(
        x=df_stable['Timestamp'],
        y=df_stable['Price'],
        mode='lines',
        name='Stable (Low Volatility)',
        line=dict(color='green', width=2)
    ))
    
    fig_compare.add_trace(go.Scatter(
        x=df_volatile['Timestamp'],
        y=df_volatile['Price'],
        mode='lines',
        name='Volatile (High Volatility)',
        line=dict(color='red', width=2)
    ))
    
    fig_compare.update_layout(
        title="Stable vs Volatile Price Patterns",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_compare, use_container_width=True)

# Raw data view
with st.expander("📋 View Generated Data & Mathematical Values"):
    display_df = df[['Timestamp', 'Time', 'Price', 'High', 'Low', 'Volume']].copy()
    display_df['Time'] = display_df['Time'].round(2)
    display_df['Price'] = display_df['Price'].round(2)
    display_df['High'] = display_df['High'].round(2)
    display_df['Low'] = display_df['Low'].round(2)
    display_df['Volume'] = display_df['Volume'].round(0)
    
    st.dataframe(display_df, use_container_width=True, height=300)
    
    st.download_button(
        label="📥 Download CSV",
        data=display_df.to_csv(index=False),
        file_name="simulated_crypto_data.csv",
        mime="text/csv"
    )

# Educational section
st.markdown("---")
st.markdown("### 🎓 Understanding the Mathematics")

col_edu1, col_edu2, col_edu3 = st.columns(3)

with col_edu1:
    st.markdown("""
    **🌊 Sine/Cosine Waves**
    - Create smooth, periodic oscillations
    - Amplitude = swing height
    - Frequency = how often swings occur
    - Models natural market cycles
    """)

with col_edu2:
    st.markdown("""
    **📈 Drift (Integrals)**
    - Long-term upward/downward trend
    - Cumulative effect over time
    - Integral: ∫ drift dt = drift × time
    - Models market trends
    """)

with col_edu3:
    st.markdown("""
    **🎲 Random Noise**
    - Unpredictable jumps
    - Normal distribution: N(0, σ)
    - Adds realism to patterns
    - Models market uncertainty
    """)

# Footer
st.markdown("---")
st.markdown("""
    **Crypto Volatility Visualizer** | Mathematics for AI-II Project  
    *Built with Python, Streamlit, NumPy & Plotly*  
    **FinTechLab Pvt. Ltd.**
""")
