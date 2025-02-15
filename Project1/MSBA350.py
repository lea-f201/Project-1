import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from fredapi import Fred

st.set_page_config(
    page_title="MSBA350 - Assignment 2 - G4",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .stApp {
        background-color:rgb(249, 249, 249);
    }
    </style>
    """,
    unsafe_allow_html=True
)

tickers = ['QCOM', 'RCL', 'TSLA', 'XOM', 'RTX', 'T', 'XPO',  'TD', 'SO', 'SYK']
crypto = ['SOL', 'LTC', 'ETH']


with st.sidebar:
    st.title("MSBA350 - Assignment 2")
    select_stock_or_crypto = st.selectbox("Select if Stocks or Cryptocurrencies",("Stocks","Cryptocurrencies"))
    if select_stock_or_crypto == "Stocks":
        selected_ticker = st.multiselect("Select Stocks", tickers, default=tickers) 
        start_date = st.date_input("Select Start Date:", datetime.date(2014, 1, 1))
        end_date = st.date_input("Select End Date:", datetime.date(2024, 12, 31))
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        close_prices = pd.DataFrame()
        for ticker in tickers:
            close_prices[ticker] = data[ticker]['Close']
        Monthly_close_price = close_prices.resample('M').last()
        data_list = list(data.columns)
        for ticker in selected_ticker:
            close_prices[f'{ticker} simple_rtn'] = close_prices[ticker].pct_change()
            close_prices[f'{ticker} log_rtn'] = np.log(close_prices[ticker]/close_prices[ticker].shift(1))
            Monthly_close_price[f'{ticker} simple_rtn'] = Monthly_close_price[ticker].pct_change()
            Monthly_close_price[f'{ticker} log_rtn'] = np.log(Monthly_close_price[ticker]/Monthly_close_price[ticker].shift(1))

    else:
        selected_ticker = st.multiselect("Select Cryptocurrencies", crypto, default=crypto)
        period_selected = st.selectbox("Select Number of Latest Transactions:",(3000, 1000,  5000))
        def get_crypto_data(symbol, period="max", interval="1m"):
            ticker = yf.Ticker(f"{symbol}-USD")
            df = ticker.history(period=period, interval=interval)
            return df.tail(period_selected)

        def create_price_bars(df, price_delta):
            price_bars = []
            current_bar = {'open': df['Open'].iloc[0], 'high': df['High'].iloc[0],
                        'low': df['Low'].iloc[0], 'close': df['Close'].iloc[0],
                        'volume': df['Volume'].iloc[0], 'start_time': df.index[0]}

            for i in range(1, len(df)):
                current_price = df['Close'].iloc[i]
                current_bar['high'] = max(current_bar['high'], df['High'].iloc[i])
                current_bar['low'] = min(current_bar['low'], df['Low'].iloc[i])
                current_bar['volume'] += df['Volume'].iloc[i]

                if abs(current_price - current_bar['open']) >= price_delta:
                    current_bar['close'] = current_price
                    current_bar['end_time'] = df.index[i]
                    price_bars.append(current_bar)
                    current_bar = {'open': current_price, 'high': df['High'].iloc[i],
                                'low': df['Low'].iloc[i], 'close': current_price,
                                'volume': df['Volume'].iloc[i], 'start_time': df.index[i]}

            return pd.DataFrame(price_bars)

        def create_tick_bars(df, tick_size):
            tick_bars = []
            current_bar = {'open': df['Open'].iloc[0], 'high': df['High'].iloc[0],
                        'low': df['Low'].iloc[0], 'close': df['Close'].iloc[0],
                        'volume': df['Volume'].iloc[0], 'start_time': df.index[0]}
            tick_count = 0

            for i in range(1, len(df)):
                tick_count += 1
                current_bar['high'] = max(current_bar['high'], df['High'].iloc[i])
                current_bar['low'] = min(current_bar['low'], df['Low'].iloc[i])
                current_bar['volume'] += df['Volume'].iloc[i]

                if tick_count >= tick_size:
                    current_bar['close'] = df['Close'].iloc[i]
                    current_bar['end_time'] = df.index[i]
                    tick_bars.append(current_bar)
                    current_bar = {'open': df['Open'].iloc[i], 'high': df['High'].iloc[i],
                                'low': df['Low'].iloc[i], 'close': df['Close'].iloc[i],
                                'volume': df['Volume'].iloc[i], 'start_time': df.index[i]}
                    tick_count = 0

            return pd.DataFrame(tick_bars)

        def create_volume_bars(df, volume_threshold):
            volume_bars = []
            current_bar = {'open': df['Open'].iloc[0], 'high': df['High'].iloc[0],
                        'low': df['Low'].iloc[0], 'close': df['Close'].iloc[0],
                        'volume': df['Volume'].iloc[0], 'start_time': df.index[0]}

            for i in range(1, len(df)):
                current_bar['high'] = max(current_bar['high'], df['High'].iloc[i])
                current_bar['low'] = min(current_bar['low'], df['Low'].iloc[i])
                current_bar['volume'] += df['Volume'].iloc[i]

                if current_bar['volume'] >= volume_threshold:
                    current_bar['close'] = df['Close'].iloc[i]
                    current_bar['end_time'] = df.index[i]
                    volume_bars.append(current_bar)
                    current_bar = {'open': df['Open'].iloc[i], 'high': df['High'].iloc[i],
                                'low': df['Low'].iloc[i], 'close': df['Close'].iloc[i],
                                'volume': df['Volume'].iloc[i], 'start_time': df.index[i]}

            return pd.DataFrame(volume_bars)

        def create_dollar_bars(df, dollar_threshold):
            dollar_bars = []
            current_bar = {'open': df['Open'].iloc[0], 'high': df['High'].iloc[0],
                        'low': df['Low'].iloc[0], 'close': df['Close'].iloc[0],
                        'volume': df['Volume'].iloc[0], 'dollar_volume': df['Volume'].iloc[0] * df['Close'].iloc[0],
                        'start_time': df.index[0]}

            for i in range(1, len(df)):
                current_price = df['Close'].iloc[i]
                current_volume = df['Volume'].iloc[i]
                dollar_volume = current_price * current_volume

                current_bar['high'] = max(current_bar['high'], df['High'].iloc[i])
                current_bar['low'] = min(current_bar['low'], df['Low'].iloc[i])
                current_bar['volume'] += current_volume
                current_bar['dollar_volume'] += dollar_volume

                if current_bar['dollar_volume'] >= dollar_threshold:
                    current_bar['close'] = current_price
                    current_bar['end_time'] = df.index[i]
                    dollar_bars.append(current_bar)
                    current_bar = {'open': current_price, 'high': df['High'].iloc[i],
                                'low': df['Low'].iloc[i], 'close': current_price,
                                'volume': current_volume, 'dollar_volume': dollar_volume,
                                'start_time': df.index[i]}

            return pd.DataFrame(dollar_bars)

        def plot_single_bar_type(bars, title, bar_type):
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.plot(range(len(bars)), bars['high'], 'g-', label='High', alpha=0.5)
            ax.plot(range(len(bars)), bars['low'], 'r-', label='Low', alpha=0.5)
            ax.plot(range(len(bars)), bars['close'], 'b-', label='Close')
            ax.fill_between(range(len(bars)), bars['high'], bars['low'], color='gray', alpha=0.2)
            ax.set_title(f'{title} - {bar_type}')
            ax.set_xlabel('Bar Number')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            return fig
       
    
    st.markdown("""# Group 4
    Nour Abdelghani
    Lea Al Fata
    Jad Koubeisy
    Nadim Kawkabany""")



# Define thresholds for each cryptocurrency
# Previous code remains the same until the thresholds dictionary

thresholds = {
    'SOL': {
        'price': 0.25,      # $0.25 movement for price bars
        'tick': 30,         # 30 trades per tick bar
        'volume': 5000,    # 50,000 SOL per volume bar
        'dollar': 50000    # $50000 per dollar bar
    },
    'LTC': {
        'price': 0.5,       # $0.50 movement for price bars
        'tick': 25,         # 25 trades per tick bar
        'volume': 200,     # 2,000 LTC per volume bar
        'dollar': 30000    # $30000 per dollar bar
    },
    'ETH': {
        'price': 5.0,       # $5.00 movement for price bars
        'tick': 40,         # 40 trades per tick bar
        'volume': 500,      # Increased from 100 to 500 ETH per volume bar
        'dollar': 500000   # Increased from 1M to 5M per dollar bar
    }
}



if select_stock_or_crypto == "Stocks":
    Options = ["Daily Prices", "Daily Simple and Log Returns","Annualized Volatility", "Monthly Nominal and Real Returns using CPI", "Monthly Nominal and Real Returns using Gold"]
else:
    Options = ["Tick Bar", "Volume Bar", "Price Bar", "Dollar Bar"]
Action = st.selectbox("Select Graph", Options)


if Action == "Daily Prices": 
    plt.figure(figsize=(12, 6))
    for ticker in selected_ticker:
        plt.plot(close_prices.index, close_prices[ticker], label=ticker)
    plt.title('Close Prices Over Time (2014-2024)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price (USD)', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  
    plt.grid(True)
    plt.tight_layout()  
    st.pyplot(plt)

elif Action == "Daily Simple and Log Returns": 
    for ticker in selected_ticker:
        plt.figure(figsize=(10, 5))
        plt.plot(close_prices.index, close_prices[f'{ticker} simple_rtn'], label="Simple Return", color="blue", alpha=0.6)
        plt.plot(close_prices.index, close_prices[f'{ticker} log_rtn'], label="Log Return", color="red", linestyle="dashed", alpha=0.6)
        plt.title(f"{ticker}: Simple & Log Returns Over Time")
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.legend()
        st.pyplot(plt)
    st.markdown("""
### **Some Insights:**  

- **Higher volatility stocks** like **TSLA, RCL** indicate **greater risk & potential rewards**.  
- **Lower volatility stocks** like **SO, SYK** are better for **conservative investors**.  

---

- This analysis helps investors **understand stock behavior, assess risk, and guide financial decision-making**.  
- Each stock exhibits **significant fluctuations** in both **Simple Returns (Blue, Solid Line)** and **Log Returns (Red, Dashed Line)** over time.  

---

- The **magnitude of returns** varies across different stocks, with some exhibiting **higher volatility** (**e.g., TSLA, RCL**) compared to others (**e.g., SYK, SO**).  
""")


elif Action == "Annualized Volatility":
    annual_volatility = {}
    for ticker in selected_ticker:
        annual_volatility[ticker] = close_prices[f'{ticker} log_rtn'].std() * np.sqrt(252)
    annual_volatility_df = pd.DataFrame(list(annual_volatility.items()), columns=['Ticker', 'Annualized Volatility'])
    plt.figure(figsize=(10, 6))
    sns.barplot(x=annual_volatility_df['Ticker'], y=annual_volatility_df['Annualized Volatility'], palette="coolwarm")
    plt.xlabel("Stock Ticker")
    plt.ylabel("Annualized Volatility")
    plt.title("Annualized Volatility of Stocks")
    plt.xticks(rotation=45)
    st.pyplot(plt)
    st.markdown("""
**Summary of Annualized Volatility of Stocks**

**High Volatility (Risky & Unpredictable):** TSLA, RCL, XPO – Suitable for aggressive investors seeking high returns but with high risk.

**Moderate Volatility (Balanced Risk-Return):** QCOM, XOM, RTX, SYK – Good for growth-oriented investors.

**Low Volatility (Stable & Predictable):** SO, T, TD – Ideal for conservative investors, often dividend-paying stocks.
""")
    
elif Action == "Monthly Nominal and Real Returns using CPI": 
    def get_stock_data(tickers, start_date, end_date):
        stock_data = yf.download(tickers, start=start_date, end=end_date)['Close']
        return stock_data
    def calculate_returns(stock_data):
        stock_data.index = pd.to_datetime(stock_data.index)
        simple_returns = stock_data.pct_change()
        log_returns = np.log(stock_data / stock_data.shift(1))
        return simple_returns, log_returns
    def get_cpi_data(start_date, end_date):
        fred = Fred(api_key='9b26e80c520c267db14e7ff9be2659d8')
        cpi = fred.get_series('CPIAUCSL', start_date, end_date)
        cpi_df = pd.DataFrame(cpi, columns=['CPI'])
        cpi_df.index = pd.to_datetime(cpi_df.index)
        return cpi_df
    def calculate_adjusted_returns(stock_data, cpi_data):
        monthly_prices = stock_data.resample('M').last()
        nominal_returns = monthly_prices.pct_change()
        cpi_monthly = cpi_data.reindex(monthly_prices.index, method='ffill')
        inflation_rate = cpi_monthly['CPI'].pct_change()
        real_returns = ((1 + nominal_returns).div(1 + inflation_rate, axis=0)) - 1
        return nominal_returns, real_returns, inflation_rate
    def plot_returns_comparison(nominal_returns, real_returns, ticker):
        cum_nominal = (1 + nominal_returns[ticker]).cumprod()
        cum_real = (1 + real_returns[ticker]).cumprod()
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        ax.plot(cum_nominal.index, cum_nominal.values, label='Nominal Returns', linewidth=2, color='#1f77b4')
        ax.plot(cum_real.index, cum_real.values, label='Inflation-Adjusted Returns', linewidth=2, color='#ff7f0e', linestyle='--')
        ax.set_title(f'{ticker} - Nominal vs Inflation-Adjusted Cumulative Returns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper left', frameon=True)
        ax.margins(x=0.02)
        return fig
    stock_data = get_stock_data(tickers, start_date, end_date)
    simple_returns, log_returns = calculate_returns(stock_data)
    cpi_data = get_cpi_data(start_date, end_date)
    nominal_returns, real_returns, inflation_rate = calculate_adjusted_returns(stock_data, cpi_data)
    for ticker in selected_ticker:
        fig = plot_returns_comparison(nominal_returns, real_returns, ticker)
        st.pyplot(fig) 
    st.markdown("""
### Inflation Impact  
Inflation reduces the actual (real) gains from investments.  
When inflation is high (e.g., **Feb 2024: 44.2%**), even a good stock return might result in a loss in real terms.  

### Nominal vs. Real Returns  
- **Nominal return** is the stock's raw profit/loss.  
- **Real return** shows actual gains after adjusting for inflation.  

**Example: TSLA (Jan 2024)**  
- **Nominal return**: -24.6%  
- **Real return**: -24.9% (worse because of inflation).  
- If real return is negative, **inflation wiped out profits!**  

### Best Performers Against Inflation  
Some stocks outperformed inflation and maintained positive real returns:  
- **RCL (Nov 2023):** +26.6% real return  
- **TSLA (Nov 2023):** +19.3% real return  
These stocks helped investors preserve purchasing power.  

### Stock Volatility  
- **TSLA & RCL**: High ups and downs – good gains in some months, big losses in others.  
- **XOM & RTX**: More stable, but sometimes struggled against inflation.  

### 📌 Bottom Line  
✅ **Inflation eats away at returns** – check real returns, not just nominal.  
✅ **Stocks like RCL & TSLA had strong inflation-adjusted gains** at times.  
✅ **High-inflation periods (2023-2024) made real gains harder to achieve.**  
""")

elif Action == "Monthly Nominal and Real Returns using Gold":
    
    gold = yf.download("GC=F", start=start_date, end=end_date, group_by='ticker')
    gold.columns = ['_'.join(col) for col in gold.columns]
    gold = gold["GC=F_Close"]

    # Resample gold price to monthly and compute returns
    gold_monthly = gold.resample('M').last()
    gold_monthly_returns = gold_monthly.pct_change()

    # Initialize DataFrame for inflation-adjusted cumulative returns
    real_returns_gold_inflation = pd.DataFrame(index=gold_monthly.index)

    # Compute cumulative inflation-adjusted returns
    for ticker in selected_ticker:
        nominal_returns = Monthly_close_price[f'{ticker} simple_rtn']
        inflation_adjusted_returns = ((1 + nominal_returns) / (1 + gold_monthly_returns)) - 1

        # Convert to cumulative returns
        nominal_cumulative = (1 + nominal_returns).cumprod() - 1
        inflation_adjusted_cumulative = (1 + inflation_adjusted_returns).cumprod() - 1

        # Store results in DataFrame
        real_returns_gold_inflation[ticker] = inflation_adjusted_cumulative * 100

        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        plt.plot(Monthly_close_price.index, nominal_cumulative * 100, label=f'{ticker} Nominal Cumulative Return')
        plt.plot(real_returns_gold_inflation.index, real_returns_gold_inflation[ticker], label=f'{ticker} Inflation Adjusted Cumulative Return', linestyle='--')

        plt.title(f'{ticker} Cumulative Nominal vs Inflation Adjusted Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
    st.markdown("""
### Insights
**Inflation-Adjusted Returns** offer a more accurate picture of an asset's performance by accounting for the erosion of purchasing power over time.

**Gold** tends to act as a stronger inflation hedge compared to CPI because it often appreciates during inflationary periods. As a result, the gap between nominal and real returns is larger when adjusted with gold, since gold’s price rises in response to inflation, helping preserve purchasing power.

**CPI**, while a widely used measure of inflation, reflects price changes in the broader economy and may not directly correlate with financial asset performance. When adjusting returns with CPI, the gap between nominal and real returns tends to be smaller because CPI doesn't fully capture inflationary impacts on assets like gold.
""")



elif Action == "Tick Bar":
    tick_threshold = st.slider("Tick Size Threshold", min_value=10, max_value=1000, value=50, step=10)
    for crypto in selected_ticker:
        df = get_crypto_data(crypto)
        tick_bars = create_tick_bars(df, tick_size=tick_threshold)
        fig = plot_single_bar_type(tick_bars, f"{crypto}", 'Tick Bars')
        st.pyplot(fig)
    st.markdown("""
### LTC Tick Bars:
This chart outlines LTC tick bars with prices listed incrementally from 0 to 7. The linear progression suggests simplified or normalized data, possibly for illustrative purposes. Each bar likely represents a tick (price movement), but the absence of realistic volatility or distinct High/Low values limits practical analysis.

### ETH Tick Bars:
Displays ETH prices declining steadily from 2700 to 2660 over seven bars. The single price per bar implies identical High, Low, and Close values, which is atypical for tick bars. This simplification may indicate a theoretical downward trend rather than real-market behavior.

### SOL Tick Bars:
Shows SOL prices dropping steadily from 206 to 200 over 5 bars. Tick bars aggregate trades by transaction count, but the lack of price variability and uniform decline contradicts typical crypto volatility. Likely a basic illustration of tick bar mechanics.
""")
elif Action == "Volume Bar":
    volume_threshold = st.slider("Volume Threshold", min_value=100, max_value=10000, value=400, step=100)
    for crypto in selected_ticker:
        df = get_crypto_data(crypto)
        volume_bars = create_volume_bars(df, volume_threshold=volume_threshold)
        fig = plot_single_bar_type(volume_bars, crypto, 'Volume Bars')
        st.pyplot(fig)
    st.markdown("""
### LTC Volume Bars:
Volume bars based on thresholds (50–250). High, Low, and Close metrics are mentioned but lack specific values, making trend analysis impossible. Focuses on volume-driven aggregation rather than price dynamics.

### ETH Volume Bars:
Prices decline linearly from 2720 to 2660 across 175 bars. The High, Low, and Close values are identical per bar, which is atypical for volume bars (usually driven by trade volume thresholds). This suggests a simplified downward trend example, not reflective of real volume-based aggregation.

### SOL Volume Bars:
Mirrors the SOL Tick Bars but spans 175 bars. Prices again decline uniformly (206 to 200), with no variability in High/Low/Close. Volume bars should reflect trade volume thresholds, but this structure ignores price action dynamics.
""")
    
elif Action == "Price Bar":
    price_threshold = st.slider("Price Delta Threshold", min_value=0.5, max_value=10.0, value=1.5, step=0.1)
    for crypto in selected_ticker:
        df = get_crypto_data(crypto)
        price_bars = create_price_bars(df, price_delta=price_threshold)
        fig = plot_single_bar_type(price_bars, crypto, 'Price Bars')
        st.pyplot(fig)
    st.markdown("""
### LTC Price Bars:
Prices rise linearly from 0 to 14 in steps of 2. The rigid pattern suggests example data, as real prices rarely follow such uniformity. Bars likely represent fixed time periods, but the absence of volatility undermines realism.

### ETH Price Bars:
Shows a prolonged downtrend from 2700 to 2660 across 140 bars. Sparse price changes (every ~20 bars) hint at consolidation or data sparsity. Likely illustrative, as extended flat periods conflict with typical crypto volatility.

### SOL Price Bars:
Prices rise linearly from 0.0 to 4.0 in 0.5 increments. The rigid upward progression and missing bar numbering suggest theoretical or placeholder data, not real market behavior.
""")
    
else:
    dollar_threshold = st.slider("Dollar Volume Threshold", min_value=500000, max_value=10000000, value=1000000, step=500000)
    for crypto in selected_ticker:
        df = get_crypto_data(crypto)
        dollar_bars = create_dollar_bars(df, dollar_threshold=dollar_threshold)
        fig = plot_single_bar_type(dollar_bars, crypto, 'Dollar Bars')
        st.pyplot(fig)
    st.markdown("""
### LTC Dollar Bars:
Dollar bars aggregate trades by dollar volume, here in 50-unit increments up to 250. Starting at 0 is unconventional; this might represent cumulative thresholds. The structure emphasizes intervals over price action, lacking detail on actual market movements.

### ETH Dollar Bars:
Similar to the volume bars, prices decrease uniformly from 2720 to 2660. The 2710-notation may imply a range, but the structure lacks clarity. Dollar bars should form based on traded dollar amounts, but the rigid price decline and identical High/Low/Close values oversimplify the concept.

### SOL Dollar Bars:
Repeats the SOL Dollar Bars pattern with prices falling from 206 to 200. The identical High/Low/Close per bar and linear trend simplify dollar bar concepts but lack practical relevance.
""")

# to get page: streamlit run MSBA350.py