import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.optimize import minimize

# --- Helper Functions ---
def load_korean_tickers(file_path="stock_ko.txt"):
    ticker_map = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    name = parts[0]
                    ticker = parts[1]
                    ticker_map[ticker] = name
    except FileNotFoundError:
        pass # Ignore if file doesn't exist
    return ticker_map

def fetch_data(tickers, start_date, end_date):
    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if 'Adj Close' in raw_data:
            data = raw_data['Adj Close']
        elif 'Close' in raw_data:
            data = raw_data['Close']
        else:
            # Fallback for single ticker or flat structure
            data = raw_data
            
        if isinstance(data, pd.Series):
            data = data.to_frame()
            if len(tickers) == 1:
                data.columns = tickers
        
        # Ensure all tickers are present
        missing = [t for t in tickers if t not in data.columns]
        if missing:
            st.warning(f"Missing data for: {', '.join(missing)}")
            
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_metrics(daily_returns, benchmark_returns=None, risk_free_rate=0.02):
    total_return = (1 + daily_returns).prod() - 1
    days = len(daily_returns)
    years = days / 252
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe = (cagr - risk_free_rate) / volatility if volatility != 0 else 0
    
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    sortino = (cagr - risk_free_rate) / downside_std if downside_std != 0 else 0
    
    cumulative = (1 + daily_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    alpha = 0
    beta = 0
    correlation = 0
    if benchmark_returns is not None:
        common_index = daily_returns.index.intersection(benchmark_returns.index)
        y = daily_returns.loc[common_index]
        x = benchmark_returns.loc[common_index]
        if len(x) > 1:
            covariance = np.cov(y, x)[0][1]
            variance = np.var(x)
            beta = covariance / variance if variance != 0 else 0
            bench_cagr = (1 + x).prod() ** (252/len(x)) - 1
            alpha = cagr - (risk_free_rate + beta * (bench_cagr - risk_free_rate))
            correlation = y.corr(x)

    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown,
        "Alpha": alpha,
        "Beta": beta,
        "Benchmark Correlation": correlation,
        "Total Return": total_return
    }

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.02):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0.0, 1.0) for asset in range(num_assets))
    
    def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
        p_ret = np.sum(mean_returns * weights) * 252
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return -(p_ret - risk_free_rate) / p_vol

    result = minimize(neg_sharpe, num_assets * [1./num_assets,], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# --- Page Configuration ---
st.set_page_config(
    page_title="Pro Portfolio Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Theme Toggle
    theme_selection = st.radio("Theme", ["Dark", "Light"], horizontal=True)
    
    st.markdown("---")
    
    # Number of Portfolios
    num_portfolios = st.radio("Number of Portfolios", [1, 2, 3, 4, 5], horizontal=True, index=0)
    
    portfolios_config = []
    
    for i in range(num_portfolios):
        with st.expander(f"📁 Portfolio {i+1}", expanded=(i==0)):
            # Default values for demo
            default_tickers = "VOO, QQQ, SCHD, GLD, SGOV"
            if i == 1: default_tickers = "AAPL, MSFT, GOOGL, NVDA"
            if i == 2: default_tickers = "JEPI, JEPQ"
            if i == 3: default_tickers = "MAGS, GLD, BRK-B, SGOV"
            if i == 4: default_tickers = "069500, 005930, 000660, 005387"
            
            p_tickers_input = st.text_input(f"Tickers P{i+1}", default_tickers, key=f"t_{i}")
            p_strategy = st.radio(f"Strategy P{i+1}", ["Manual Weights", "Max Sharpe (Opt)"], key=f"s_{i}")
            
            p_weights_input = None
            if p_strategy == "Manual Weights":
                default_weights = "0.2, 0.2, 0.2, 0.2, 0.2"
                if i == 1: default_weights = "0.25, 0.25, 0.25, 0.25"
                if i == 2: default_weights = "0.5, 0.5"
                if i == 3: default_weights = "0.25, 0.25, 0.25, 0.25"
                if i == 4: default_weights = "0.25, 0.25, 0.25, 0.25"
                p_weights_input = st.text_input(f"Weights P{i+1}", default_weights, key=f"w_{i}")
            
            # Rebalancing Option
            p_rebalance = st.radio(f"Rebalancing P{i+1}", ["Yearly", "None (Buy & Hold)"], key=f"r_{i}")
            
            portfolios_config.append({
                "name": f"Portfolio {i+1}",
                "tickers_str": p_tickers_input,
                "strategy": p_strategy,
                "weights_str": p_weights_input,
                "rebalance": p_rebalance
            })
    
    st.markdown("---")
    st.subheader("Global Parameters")
    
    # Currency Selection
    currency_option = st.radio("Currency", ["USD ($)", "KRW (₩)"], horizontal=True)
    currency_symbol = "₩" if "KRW" in currency_option else "$"
    
    initial_capital = st.number_input(f"Initial Capital ({currency_symbol})", value=100000, step=10000)
    risk_free_rate = st.number_input("Risk Free Rate (%)", value=2.0, step=0.1) / 100
    
    col1, col2 = st.columns(2)
    with col1:
        # Allow back to 1980
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365*10), min_value=datetime(1980, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime.now())
        
    benchmark_ticker = st.text_input("Benchmark", "SPY")
    
    run_button = st.button("Run Analysis")

# --- Theme Configuration ---
is_dark = theme_selection == "Dark"
if is_dark:
    bg_color = "#0E1117"
    text_color = "#FAFAFA"
    card_bg = "#262730"
    input_bg = "#262730"
    input_text = "#FAFAFA"
    input_border = "1px solid #464B5C"
    chart_text_color = "#FAFAFA"
    chart_p_color = "#00CC96"
    chart_b_color = "#808080"
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
    plotly_template = "plotly_dark"
else:
    bg_color = "#FFFFFF"
    text_color = "#31333F"
    card_bg = "#F0F2F6"
    input_bg = "#FFFFFF"
    input_text = "#31333F"
    input_border = "1px solid #D6D6D6"
    chart_text_color = "#31333F" 
    chart_p_color = "#00CC96"
    chart_b_color = "#808080"
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
    plotly_template = "plotly_white"

# --- CSS Injection ---
st.markdown(f"""
    <style>
    /* Global Text Styling */
    .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp span, .stApp div {{
        color: {text_color} !important;
    }}
    .stApp {{
        background-color: {bg_color} !important;
    }}
    
    /* Header Styling */
    header[data-testid="stHeader"] {{
        background-color: {bg_color} !important;
    }}
    header[data-testid="stHeader"] * {{
        color: {text_color} !important;
    }}
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {bg_color} !important;
    }}
    section[data-testid="stSidebar"] > div {{
        background-color: {bg_color} !important;
    }}
    section[data-testid="stSidebar"] label {{
        color: {text_color} !important;
    }}
    
    /* Button Styling (Run Analysis) */
    .stButton button {{
        color: #FFFFFF !important;
        background-color: #FF4B4B !important;
        border: none !important;
    }}
    .stButton button:hover {{
        background-color: #FF2B2B !important;
        color: #FFFFFF !important;
    }}
    
    /* Expander Styling */
    div[data-testid="stExpander"] {{
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-radius: 10px;
        background-color: {card_bg};
        margin-bottom: 10px;
    }}
    div[data-testid="stExpander"] details summary {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border-radius: 5px;
    }}
    
    /* Input Styling */
    input, select, textarea {{
        background-color: {input_bg} !important;
        color: {input_text} !important;
        border: {input_border} !important;
        border-radius: 5px;
    }}
    div[data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        color: {input_text} !important;
        border: {input_border} !important;
        border-radius: 5px;
    }}
    
    /* Calendar / Date Picker Styling */
    div[data-baseweb="calendar"] {{
        background-color: {input_bg} !important;
        border: {input_border} !important;
    }}
    /* Calendar Header (Month/Year) */
    div[data-baseweb="calendar"] div[aria-live="polite"] {{
        color: {text_color} !important;
        background-color: transparent !important;
    }}
    /* Weekday Headers */
    div[data-baseweb="calendar"] div[role="grid"] div {{
        color: {text_color} !important;
        background-color: transparent !important;
    }}
    /* Day Numbers */
    div[data-baseweb="day"] {{
        color: {text_color} !important;
        background-color: transparent !important;
    }}
    /* Selected Day Text - Force White */
    div[data-baseweb="day"][aria-selected="true"] {{
        color: #FFFFFF !important;
        background-color: #FF4B4B !important;
    }}
    div[data-baseweb="day"]:hover {{
        color: #FFFFFF !important;
        background-color: #FF4B4B !important;
    }}
    /* Navigation Arrows & Header Container */
    div[data-baseweb="calendar"] > div:first-child {{
        background-color: transparent !important;
    }}
    div[data-baseweb="calendar"] button {{
        background-color: transparent !important;
    }}
    div[data-baseweb="calendar"] button svg {{
        fill: {text_color} !important;
        color: {text_color} !important;
    }}
    div[data-baseweb="calendar"] button svg path {{
        fill: {text_color} !important;
        color: {text_color} !important;
    }}
    
    /* Dropdown Menus */
    div[data-baseweb="menu"] {{
        background-color: {input_bg} !important;
    }}
    div[data-baseweb="menu"] div {{
        color: {input_text} !important;
    }}
    
    /* Table Styling */
    th {{
        font-weight: bold !important;
        color: {text_color} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# --- Main Content ---
st.title("📈 Pro Portfolio Analyzer")
st.markdown("Compare and analyze multiple portfolio strategies.")

if run_button:
    # 0. Load Korean Tickers
    ko_ticker_map = load_korean_tickers()
    
    # 1. Collect all unique tickers
    all_tickers = set()
    
    # Handle Benchmark Ticker
    bench_fetch_ticker = benchmark_ticker
    if benchmark_ticker in ko_ticker_map:
        bench_fetch_ticker = f"{benchmark_ticker}.KS"
    all_tickers.add(bench_fetch_ticker)
    
    parsed_portfolios = []
    
    for p_conf in portfolios_config:
        raw_ts = [t.strip().upper() for t in p_conf['tickers_str'].split(",")]
        
        processed_ts = []
        display_names = {}
        
        for t in raw_ts:
            if t in ko_ticker_map:
                fetch_t = f"{t}.KS"
                processed_ts.append(fetch_t)
                display_names[fetch_t] = ko_ticker_map[t]
            else:
                processed_ts.append(t)
                display_names[t] = t
                
        all_tickers.update(processed_ts)
        
        ws = []
        if p_conf['strategy'] == "Manual Weights":
            try:
                ws = [float(w.strip()) for w in p_conf['weights_str'].split(",")]
                if len(processed_ts) != len(ws):
                    st.error(f"Error in {p_conf['name']}: Ticker count ({len(processed_ts)}) != Weight count ({len(ws)})")
                    st.stop()
                if abs(sum(ws) - 1.0) > 0.01:
                    ws = [w / sum(ws) for w in ws] # Normalize
            except ValueError:
                st.error(f"Error in {p_conf['name']}: Invalid weights format")
                st.stop()
        
        parsed_portfolios.append({
            "name": p_conf['name'],
            "tickers": processed_ts,
            "display_names": display_names,
            "strategy": p_conf['strategy'],
            "weights": ws,
            "rebalance": p_conf['rebalance']
        })

    with st.spinner("Fetching market data..."):
        df_all = fetch_data(list(all_tickers), start_date, end_date)
        
        # Currency Conversion Logic
        fx_df = None
        if "KRW" in currency_option:
            fx_data = fetch_data(["KRW=X"], start_date, end_date)
            if fx_data is not None and not fx_data.empty:
                fx_df = fx_data["KRW=X"]
    
    if df_all is not None and not df_all.empty:
        # 1. Determine Common Analysis Period
        # Find the first valid index for each ticker
        ticker_start_dates = {}
        for col in df_all.columns:
            valid_idx = df_all[col].first_valid_index()
            if valid_idx is not None:
                ticker_start_dates[col] = valid_idx
        
        if not ticker_start_dates:
            st.error("No valid data found.")
            st.stop()
            
        # The analysis must start from the latest of these start dates to be fair
        global_start_date = max(ticker_start_dates.values())
        
        # If KRW, also consider FX data start date
        if fx_df is not None:
            fx_start = fx_df.first_valid_index()
            if fx_start > global_start_date:
                global_start_date = fx_start
        
        limiting_tickers = [t for t, d in ticker_start_dates.items() if d == global_start_date]
        
        # Filter data to common period and fill missing values (holidays)
        df_all = df_all.loc[global_start_date:].ffill().bfill()
        if fx_df is not None:
            fx_df = fx_df.loc[global_start_date:]
            # Align FX data to df_all (handle trading holidays)
            fx_df = fx_df.reindex(df_all.index).ffill()
            
            # Convert Prices to KRW
            # Price_KRW = Price_USD * Rate (Only for non-KRW assets)
            for col in df_all.columns:
                if not col.endswith(".KS"):
                    df_all[col] = df_all[col] * fx_df
        
        if df_all.empty:
            st.error("No overlapping data found for the selected tickers.")
            st.stop()

        # Display Analysis Period & Warning
        actual_start = df_all.index[0].strftime('%Y-%m-%d')
        actual_end = df_all.index[-1].strftime('%Y-%m-%d')
        
        st.markdown(f"### 📅 Analysis Period: {actual_start} ~ {actual_end}")
        
        # Show warning if the period is shorter than requested (approx check)
        requested_start_ts = pd.Timestamp(start_date)
        if global_start_date > requested_start_ts + timedelta(days=5): # 5 days buffer
            if len(limiting_tickers) <= 3:
                limit_str = ", ".join([f"**{t}**" for t in limiting_tickers])
            else:
                limit_str = f"**{limiting_tickers[0]}** and {len(limiting_tickers)-1} others"
            st.warning(f"⚠️ The analysis period was shortened because {limit_str} only has data starting from {actual_start}.")
        # Benchmark Data
        if bench_fetch_ticker not in df_all.columns:
            st.error(f"Benchmark {bench_fetch_ticker} data not found.")
            st.stop()
            
        bench_returns = df_all[bench_fetch_ticker].pct_change().dropna()
        # Calculate Benchmark Value using Price Normalization to match Portfolio calculation
        bench_price = df_all[bench_fetch_ticker]
        bench_value = (bench_price / bench_price.iloc[0]) * initial_capital
        bench_metrics = calculate_metrics(bench_returns, bench_returns, risk_free_rate)
        
        # Process Portfolios
        results = []
        
        for i, p in enumerate(parsed_portfolios):
            # Subset data (already filtered to common period)
            valid_tickers = [t for t in p['tickers'] if t in df_all.columns]
            if not valid_tickers:
                st.warning(f"{p['name']} has no valid data.")
                continue
                
            p_df = df_all[valid_tickers].dropna()
            
            if p_df.empty:
                st.warning(f"{p['name']} data is empty.")
                continue
                
            # Returns & Optimization
            daily_returns = p_df.pct_change().dropna()
            
            mean_returns = daily_returns.mean()
            cov_matrix = daily_returns.cov()
            
            weights = p['weights']
            if p['strategy'] == "Max Sharpe (Opt)":
                weights = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)
            
            # Calculate Value
            if p['rebalance'] == "Yearly":
                # Yearly Rebalancing Logic
                asset_rets = p_df.pct_change().fillna(0)
                years = sorted(asset_rets.index.year.unique())
                full_series = []
                current_capital = initial_capital
                
                for yr in years:
                    yr_rets = asset_rets[asset_rets.index.year == yr]
                    if yr_rets.empty: continue
                    yr_growth = (1 + yr_rets).cumprod()
                    yr_val_rel = (yr_growth * weights).sum(axis=1)
                    yr_val_abs = yr_val_rel * current_capital
                    full_series.append(yr_val_abs)
                    current_capital = yr_val_abs.iloc[-1]
                
                if full_series:
                    port_val = pd.concat(full_series)
                else:
                    port_val = pd.Series()
            else:
                # None (Buy & Hold)
                p_normalized = p_df / p_df.iloc[0]
                port_val = (p_normalized * weights).sum(axis=1) * initial_capital
            
            port_ret = port_val.pct_change().dropna()
            
            # Align benchmark returns for metrics calculation
            common_idx = port_ret.index.intersection(bench_returns.index)
            metrics_port_ret = port_ret.loc[common_idx]
            metrics_bench_ret = bench_returns.loc[common_idx]
            
            metrics = calculate_metrics(metrics_port_ret, metrics_bench_ret, risk_free_rate)
            
            # Use display names for tickers
            display_ticker_list = [p['display_names'].get(t, t) for t in valid_tickers]
            
            results.append({
                "name": p['name'],
                "value": port_val,
                "returns": port_ret,
                "metrics": metrics,
                "weights": weights,
                "tickers": display_ticker_list,
                "color": colors[i % len(colors)]
            })

        # --- Display Results ---
        
        # 1. Allocations
        st.subheader("Asset Allocations")
        cols = st.columns(len(results))
        for i, res in enumerate(results):
            with cols[i]:
                st.caption(f"**{res['name']}**")
                current_val = res['value'].iloc[-1]
                
                # Filter out small weights (< 0.5%)
                pie_df = pd.DataFrame({'Ticker': res['tickers'], 'Weight': res['weights']})
                pie_df = pie_df[pie_df['Weight'] >= 0.005]
                
                fig_pie = px.pie(pie_df, names='Ticker', values='Weight', hole=0.4,
                                 color_discrete_sequence=px.colors.sequential.Teal)
                fig_pie.update_traces(textinfo='label+percent', textposition='auto')
                fig_pie.update_layout(
                    template=plotly_template,
                    height=250,
                    margin=dict(l=0,r=0,t=0,b=0),
                    paper_bgcolor=card_bg,
                    plot_bgcolor=card_bg,
                    font=dict(color=chart_text_color),
                    showlegend=False,
                    annotations=[dict(text=f"{currency_symbol}{current_val:,.0f}", x=0.5, y=0.5, font_size=14, showarrow=False, font_color=chart_text_color)]
                )
                st.plotly_chart(fig_pie, width='stretch')

        # 2. Performance Summary Table
        st.subheader("Performance Summary")
        metric_names = ["CAGR", "Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Alpha", "Beta", "Benchmark Correlation"]
        
        summary_data = {"Metric": metric_names}
        
        # Add Portfolios
        for res in results:
            m = res['metrics']
            summary_data[res['name']] = [
                f"{m['CAGR']:.2%}", f"{m['Volatility']:.2%}", f"{m['Sharpe Ratio']:.2f}",
                f"{m['Sortino Ratio']:.2f}", f"{m['Max Drawdown']:.2%}", f"{m['Alpha']:.2%}", f"{m['Beta']:.2f}",
                f"{m['Benchmark Correlation']:.2f}"
            ]
            
        # Add Benchmark
        summary_data["Benchmark"] = [
            f"{bench_metrics['CAGR']:.2%}", f"{bench_metrics['Volatility']:.2%}", f"{bench_metrics['Sharpe Ratio']:.2f}",
            f"{bench_metrics['Sortino Ratio']:.2f}", f"{bench_metrics['Max Drawdown']:.2%}", f"{bench_metrics['Alpha']:.2%}", f"{bench_metrics['Beta']:.2f}",
            f"{bench_metrics['Benchmark Correlation']:.2f}"
        ]
        
        # Create styled DataFrame
        df_summary = pd.DataFrame(summary_data).set_index("Metric")
        styled_df = df_summary.style.set_table_styles([
            {'selector': 'th', 'props': [('font-weight', 'bold'), ('color', text_color)]},
            {'selector': 'td', 'props': [('color', text_color)]}
        ])
        
        st.table(styled_df)
        
        # 3. Growth Chart
        st.subheader("Portfolio Growth")
        fig = go.Figure()
        
        # Add Portfolios
        for res in results:
            fig.add_trace(go.Scatter(
                x=np.array(res['value'].index), y=res['value'],
                mode='lines', name=res['name'],
                line=dict(color=res['color'], width=2)
            ))
            
        # Add Benchmark
        if results:
            first_idx = results[0]['value'].index
            aligned_bench = bench_value.loc[first_idx]
            # Rebase benchmark to initial capital
            aligned_bench = aligned_bench / aligned_bench.iloc[0] * initial_capital
            
            fig.add_trace(go.Scatter(
                x=np.array(aligned_bench.index), y=aligned_bench,
                mode='lines', name=f"Benchmark ({benchmark_ticker})",
                line=dict(color=chart_b_color, width=2, dash='dash')
            ))
            
        fig.update_layout(
            template=plotly_template,
            hovermode="x unified",
            yaxis_title="Value ($)" if "USD" in currency_option else "Value (₩)",
            paper_bgcolor=card_bg,
            plot_bgcolor=card_bg,
            font=dict(color=chart_text_color),
            legend=dict(font=dict(color=chart_text_color), orientation="h", y=1.1),
            xaxis=dict(color=chart_text_color, showline=True, linecolor=chart_text_color, tickfont=dict(color=chart_text_color), title_font=dict(color=chart_text_color)),
            yaxis=dict(color=chart_text_color, showline=True, linecolor=chart_text_color, tickfont=dict(color=chart_text_color), title_font=dict(color=chart_text_color))
        )
        st.plotly_chart(fig, width='stretch')
        
        # 3.1 Exchange Rate Chart (if KRW)
        if fx_df is not None:
            st.subheader("💱 Exchange Rate (USD/KRW)")
            fig_fx = px.line(fx_df, x=np.array(fx_df.index), y=fx_df.values)
            fig_fx.update_traces(line_color='gray', line_width=1.5)
            fig_fx.update_layout(
                template=plotly_template,
                hovermode="x unified",
                yaxis_title="Exchange Rate (KRW)",
                paper_bgcolor=card_bg,
                plot_bgcolor=card_bg,
                font=dict(color=chart_text_color),
                xaxis=dict(color=chart_text_color, showline=True, linecolor=chart_text_color, tickfont=dict(color=chart_text_color), title_font=dict(color=chart_text_color)),
                yaxis=dict(color=chart_text_color, showline=True, linecolor=chart_text_color, tickfont=dict(color=chart_text_color), title_font=dict(color=chart_text_color))
            )
            st.plotly_chart(fig_fx, width='stretch')
        
        # 4. Drawdown Chart
        st.subheader("Drawdown Analysis")
        fig_dd = go.Figure()
        
        for res in results:
            # Calculate DD from value directly
            val = res['value']
            dd = (val - val.cummax()) / val.cummax()
            fig_dd.add_trace(go.Scatter(
                x=np.array(dd.index), y=dd,
                mode='lines', name=res['name'],
                line=dict(color=res['color'], width=1),
                fill='tozeroy'
            ))
            
        if results:
             # Benchmark DD
             b_val = bench_value
             b_dd = (b_val - b_val.cummax()) / b_val.cummax()
             fig_dd.add_trace(go.Scatter(
                 x=np.array(b_dd.index), y=b_dd,
                 mode='lines', name="Benchmark",
                 line=dict(color=chart_b_color, width=1, dash='dash')
             ))

        fig_dd.update_layout(
            template=plotly_template,
            hovermode="x unified",
            yaxis_title="Drawdown",
            paper_bgcolor=card_bg,
            plot_bgcolor=card_bg,
            font=dict(color=chart_text_color),
            legend=dict(font=dict(color=chart_text_color), orientation="h", y=1.1),
            xaxis=dict(color=chart_text_color, showline=True, linecolor=chart_text_color, tickfont=dict(color=chart_text_color), title_font=dict(color=chart_text_color)),
            yaxis=dict(color=chart_text_color, showline=True, linecolor=chart_text_color, tickfont=dict(color=chart_text_color), title_font=dict(color=chart_text_color))
        )
        st.plotly_chart(fig_dd, width='stretch')
        
        # 5. Efficient Frontier
        st.subheader("Efficient Frontier & Feasible Region")
        st.caption("Monte Carlo Simulation (Investable Universe)")
        
        if len(all_tickers) >= 2:
            # Generate random portfolios
            num_portfolios_sim = 2000
            results_list = []
            
            # Use daily returns of all tickers for simulation
            sim_returns = df_all.pct_change().dropna()
            mean_sim_returns = sim_returns.mean()
            cov_sim_matrix = sim_returns.cov()
            num_assets_sim = len(all_tickers)
            
            for _ in range(num_portfolios_sim):
                weights = np.random.random(num_assets_sim)
                weights /= np.sum(weights)
                
                p_ret = np.sum(mean_sim_returns * weights) * 252
                p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_sim_matrix, weights))) * np.sqrt(252)
                p_sharpe = (p_ret - risk_free_rate) / p_vol
                
                results_list.append({
                    "Return": p_ret,
                    "Volatility": p_vol,
                    "Sharpe": p_sharpe
                })
                
            sim_df = pd.DataFrame(results_list)
            
            # Plot
            fig_ef = go.Figure()
            
            fig_ef.add_trace(go.Scatter(
                x=sim_df["Volatility"], y=sim_df["Return"],
                mode='markers',
                name="",
                showlegend=False,
                marker=dict(
                    color=sim_df["Sharpe"],
                    colorscale='Viridis',
                    opacity=0.15,
                    showscale=True,
                    colorbar=dict(
                        title="Sharpe Ratio",
                        tickfont=dict(color=chart_text_color),
                        title_font=dict(color=chart_text_color)
                    )
                )
            ))
            
            # Overlay User Portfolios
            for res in results:
                m = res['metrics']
                fig_ef.add_trace(go.Scatter(
                    x=[m['Volatility']], y=[m['CAGR']],
                    mode='markers+text',
                    name=res['name'],
                    text=[res['name']],
                    textposition="top center",
                    marker=dict(color=res['color'], size=18, symbol='star', line=dict(width=2, color='white'))
                ))
                
            # Overlay Benchmark
            fig_ef.add_trace(go.Scatter(
                x=[bench_metrics['Volatility']], y=[bench_metrics['CAGR']],
                mode='markers+text',
                marker=dict(color=chart_b_color, size=12, symbol='diamond', line=dict(width=1, color='white')),
                name=f"Benchmark ({benchmark_ticker})",
                text=[benchmark_ticker],
                textposition="bottom center"
            ))
            
            fig_ef.update_layout(
                template=plotly_template,
                xaxis_title="Annualized Volatility (Risk)",
                yaxis_title="Annualized Return (CAGR)",
                paper_bgcolor=card_bg,
                plot_bgcolor=card_bg,
                font=dict(color=chart_text_color),
                title_font=dict(color=chart_text_color),

                legend_title_text=' ',
                legend=dict(font=dict(color=chart_text_color), orientation="h", y=1.1, title=dict(text=' ')),
                xaxis=dict(color=chart_text_color, showline=True, linecolor=chart_text_color, tickfont=dict(color=chart_text_color), title_font=dict(color=chart_text_color)),
                yaxis=dict(color=chart_text_color, showline=True, linecolor=chart_text_color, tickfont=dict(color=chart_text_color), title_font=dict(color=chart_text_color))
            )
            st.plotly_chart(fig_ef, width='stretch')
        else:
            st.info("Need at least 2 different assets to generate an Efficient Frontier.")

    else:
        st.warning("No data found.")
else:
    st.info("👈 Configure up to 5 portfolios in the sidebar and click 'Run Analysis'.")
