# 📈 Pro Portfolio Analyzer

[![Streamlit App](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://pythonppa.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated, interactive financial tool for backtesting, optimizing, and comparing multiple investment portfolios. Built with **Python** and **Streamlit**, this application provides professional-grade analytics for both individual and institutional investors.

## 🚀 Key Features

### 1. Multi-Portfolio Comparison
*   **Compare up to 5 portfolios** simultaneously against a chosen benchmark (e.g., SPY, QQQ).
*   Flexible asset selection including **US Stocks, ETFs, and Korean Stocks (KOSPI)**.

### 2. Advanced Optimization & Strategies
*   **Max Sharpe Ratio Optimization**: Automatically calculates the optimal asset weights to maximize risk-adjusted returns using the SLSQP algorithm.
*   **Manual Weighting**: Precision control over asset allocation.
*   **Rebalancing**: Choose between **Yearly Rebalancing** or a **Buy & Hold** strategy.

### 3. Professional Analytics
*   **Performance Metrics**: CAGR, Annualized Volatility, Sharpe Ratio, Sortino Ratio, Alpha, Beta, and Max Drawdown.
*   **Currency Conversion**: Real-time conversion between **USD ($)** and **KRW (₩)**, including automatic FX rate historical data integration.
*   **Risk Analysis**: Detailed drawdown charts and correlation analysis against the benchmark.

### 4. Visual Intelligence
*   **Portfolio Growth**: Interactive time-series charts of cumulative returns.
*   **Efficient Frontier**: Monte Carlo simulation (2,000+ iterations) to visualize the risk-return spectrum and find the optimal investable universe.
*   **Asset Allocation**: Dynamic donut charts representing the weight of each portfolio component.

---

## 🛠 Tech Stack

*   **Frontend/App Framework**: [Streamlit](https://streamlit.io/)
*   **Data Source**: [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance API)
*   **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Scientific Computing**: [SciPy](https://scipy.org/) (Optimization)
*   **Visualization**: [Plotly](https://plotly.com/python/) (Interactive Charts)
---

### 🔗 Live Demo
Access the application here:
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pythonppa.streamlit.app/)

---

## 💻 Local Installation

To run the project locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/Python_PPA.git
    cd Python_PPA
    ```

2.  **Create and activate a virtual environment** (optional but recommended):
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application**:
    ```bash
    streamlit run app.py
    ```

---

## 📝 Configuration

-   **Stock Symbols**: Enter US tickers directly (e.g., `AAPL`, `VOO`). For Korean stocks, enter the 6-digit code (the app automatically appends `.KS` or `.KQ` based on the internal mapping).
-   **Benchmark**: Default is `SPY`, but can be changed to any valid ticker.
-   **Ticker Shortcuts**: The app includes a pre-defined mapping for popular Korean stocks and ETFs (e.g., typing `Samsung` instead of `005930`).
-   **Analysis Period**: Supports historical data back to 1980 (depending on asset availability).

---

## ⚠️ Disclaimer

*This tool is for educational and informational purposes only. It does not constitute financial advice. Past performance is not indicative of future results. Always consult with a professional financial advisor before making investment decisions.*

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
