import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import threading
import asyncio
from sklearn.linear_model import LinearRegression
import requests
from streamlit_autorefresh import st_autorefresh

from websocket_client import WebsocketManager  # Your websocket client from separate file

# Static fee tiers for fallback or user selection
FEE_TIERS = {
    "Tier 1": {"maker_fee": 0.0010, "taker_fee": 0.0025},
    "Tier 2": {"maker_fee": 0.0008, "taker_fee": 0.0020},
    "Tier 3": {"maker_fee": 0.0005, "taker_fee": 0.0015},
}

# Configure Streamlit page layout and title
st.set_page_config(page_title="GoQuant Trade Simulator", layout="wide")
st.markdown("# 🚀 GoQuant Trade Simulator")

# Auto-refresh Streamlit app every 1 second to update live data
st_autorefresh(interval=1000, key="auto_refresh")


class AsyncRunner:
    """
    Runs the WebSocket client asynchronously in a separate thread to fetch live data.

    Attributes:
        client (WebsocketManager): Instance managing WebSocket connection.
        thread (threading.Thread): Background thread running the asyncio event loop.
    """

    def __init__(self, client: WebsocketManager):
        self.client = client
        self.thread = None

    def start(self):
        """Starts the background thread if not already running."""
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()

    def _run_loop(self):
        """Runs the async WebSocket connection coroutine."""
        asyncio.run(self.client.connect())

    def stop(self):
        """Stops the WebSocket client and waits for the thread to finish."""
        self.client.stop()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)


# Initialize WebSocket manager and runner in Streamlit session state
if "ws_manager" not in st.session_state:
    st.session_state.ws_manager = WebsocketManager(asset="BTC-USDT")
    st.session_state.runner = AsyncRunner(st.session_state.ws_manager)
    st.session_state.runner.start()

# Input parameters expander section for user inputs
with st.expander("🮮 Input Parameters", expanded=True):
    col1, col2 = st.columns([3, 3])
    with col1:
        # Currently only OKX supported, but selectable for future extension
        exchange = st.selectbox("Exchange", ["OKX"], index=0)

        # Favorite assets for quick selection
        favorites = ["BTC-USDT", "ETH-USDT", "OKB-USDT", "XRP-USDT", "SOL-USDT", "DOGE-USDT", "ADA-USDT", "SUI-USDT"]

        # Asset selector defaults to currently subscribed asset if in favorites
        asset = st.selectbox(
            "Spot Asset",
            favorites,
            index=favorites.index(st.session_state.ws_manager.asset) if st.session_state.ws_manager.asset in favorites else 0,
        )

        # Order type: Market or Limit order
        order_type = st.selectbox("Order Type", ["Market", "Limit"], index=0)

    with col2:
        # Trade quantity in USD
        quantity_usd = st.number_input("Quantity (USD)", value=100.0, min_value=1.0)

        # Estimated market volatility percentage, used in Almgren-Chriss model
        volatility = st.number_input("Volatility (%)", value=0.5, min_value=0.0)

        # Fee tier selection for static fee model
        fee_tier = st.selectbox("Fee Tier", list(FEE_TIERS.keys()), index=0)

        # Fetch fees from the selected static fee tier for display
        maker_fee_current = FEE_TIERS[fee_tier]["maker_fee"]
        taker_fee_current = FEE_TIERS[fee_tier]["taker_fee"]

        st.markdown(f"**Current Tier Fees:** Maker Fee = {maker_fee_current*100:.3f}%, Taker Fee = {taker_fee_current*100:.3f}%")

# If user changes the selected asset, restart WebSocket subscription accordingly
if asset != st.session_state.ws_manager.asset:
    st.session_state.runner.stop()
    st.session_state.ws_manager = WebsocketManager(asset=asset)
    st.session_state.runner = AsyncRunner(st.session_state.ws_manager)
    st.session_state.runner.start()

# Fetch latest snapshot of orderbook data from WebSocket manager
latest_snapshot = st.session_state.ws_manager.get_data()


def fetch_okx_fee_rates(inst_id: str):
    """
    Fetch maker and taker fees dynamically from OKX API for given instrument ID.

    Args:
        inst_id (str): Instrument ID, e.g., "BTC-USDT".

    Returns:
        Tuple[float, float]: Maker fee and taker fee as decimal fractions (e.g., 0.001),
                             or (None, None) on failure.
    """
    url = "https://www.okx.com/api/v5/account/trade-fee"
    params = {"instType": "SPOT", "instId": inst_id}
    try:
        response = requests.get(url, params=params, timeout=3)
        data = response.json()
        if data["code"] == "0" and data["data"]:
            fee_info = data["data"][0]
            # OKX returns fees as negative values, so negate them to positive
            maker_fee = -float(fee_info["maker"])
            taker_fee = -float(fee_info["taker"])
            return maker_fee, taker_fee
        else:
            return None, None
    except Exception:
        # In case of any error (network/API), return None to fallback to static fees
        return None, None


# Try to fetch dynamic fees from OKX; fallback to static fees if fails
maker_fee_dynamic, taker_fee_dynamic = fetch_okx_fee_rates(asset)
if maker_fee_dynamic is not None and taker_fee_dynamic is not None:
    current_maker_fee = maker_fee_dynamic
    current_taker_fee = taker_fee_dynamic
else:
    current_maker_fee = maker_fee_current
    current_taker_fee = taker_fee_current

# Output metrics expander showing real-time market data and calculated costs
with st.expander("📊 Output Metrics", expanded=True):
    bids = latest_snapshot.get("bids", [])
    asks = latest_snapshot.get("asks", [])

    # Measure start time for internal latency calculation
    start_time = time.perf_counter()

    if bids and asks:
        top_bid = float(bids[0][0])
        top_ask = float(asks[0][0])
        timestamp_val = latest_snapshot.get("timestamp", time.time())
        timestamp = pd.to_datetime(timestamp_val, utc=True)

        # Slippage calculation: relative difference between best ask and bid prices in percent
        slippage = (top_ask - top_bid) / top_bid * 100

        # Generate sample timestamps and slippage data to fit linear regression for prediction
        now = datetime.datetime.utcnow()
        timestamps = np.array([(now - datetime.timedelta(seconds=9 - i)).timestamp() for i in range(10)]).reshape(-1, 1)
        slippage_samples = np.random.uniform(low=0.1, high=1.0, size=10).reshape(-1, 1)

        # Linear regression model to predict slippage trend
        lr = LinearRegression()
        lr.fit(timestamps, slippage_samples)
        predicted_slippage = lr.predict(np.array([[now.timestamp()]]))[0][0]

        # Market impact is estimated as a fraction of predicted slippage times trade quantity
        market_impact = predicted_slippage * 0.01 * quantity_usd

        # Expected fees based on order type and current fee rates
        expected_fees = quantity_usd * (current_taker_fee if order_type == "Market" else current_maker_fee)

        # Net cost is sum of absolute slippage, fees, and market impact
        net_cost = abs(slippage) + expected_fees + market_impact
    else:
        st.warning("Waiting for live market data...")
        top_bid = 0.0
        top_ask = 0.0
        slippage = 0.0
        market_impact = 0.0
        expected_fees = 0.0
        net_cost = 0.0

    # Measure end time and compute latency in milliseconds
    end_time = time.perf_counter()
    internal_latency = round((end_time - start_time) * 1000, 2)  # ms

    # Display key metrics as Streamlit metric widgets in three rows of three columns
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Top Bid 💰", f"{top_bid:.2f}")
    mcol2.metric("Top Ask 💵", f"{top_ask:.2f}")
    mcol3.metric("Slippage 🔻", f"{slippage:.2f} %")

    mcol4, mcol5, mcol6 = st.columns(3)
    mcol4.metric("Expected Fees 💸", f"${expected_fees:.2f}")
    mcol5.metric("Market Impact 📉", f"${market_impact:.2f}")
    mcol6.metric("Net Cost 📊", f"${net_cost:.2f}")

    mcol7, mcol8, mcol9 = st.columns(3)
    mcol7.metric("Order Type 📟", order_type)
    mcol8.metric("Fee Tier 🏷️", fee_tier)
    mcol9.metric("Latency ⏱️", f"{internal_latency} ms")

# Market data visualization section
with st.expander("📈 Visualize Market Data", expanded=True):
    if bids and asks:
        top_bid = float(bids[0][0])
        top_ask = float(asks[0][0])
        timestamp_val = latest_snapshot.get("timestamp", time.time())
        timestamp = pd.to_datetime(timestamp_val, utc=True)

        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("#### 🔴 Top Bid vs 🟢 Top Ask")

            # Plot top bid and ask prices over time (single latest point here)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot([timestamp], [top_bid], 'ro-', label="Top Bid")
            ax.plot([timestamp], [top_ask], 'go-', label="Top Ask")
            ax.set_title(f"Bid/Ask Prices Over Time ({asset})")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
            fig.tight_layout()
            st.pyplot(fig)

        with right_col:
            st.markdown("#### 📉 Slippage Over Time (Sampled & Linear Regression)")

            # Prepare datetime objects for plotting
            timestamps_dt = [datetime.datetime.utcfromtimestamp(ts[0]) for ts in timestamps]

            # Plot sampled slippage points and linear regression fit line
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            ax2.plot(timestamps_dt, slippage_samples, 'o-', color='purple', label="Sampled Slippage")

            ts_line = np.linspace(timestamps.min(), timestamps.max(), 100).reshape(-1, 1)
            slippage_pred_line = lr.predict(ts_line)
            ts_line_dt = [datetime.datetime.utcfromtimestamp(t[0]) for t in ts_line]

            ax2.plot(ts_line_dt, slippage_pred_line, '-', color='orange', label="Linear Regression Fit")
            ax2.set_title("Slippage Over Time")
            ax2.set_xlabel("Timestamp")
            ax2.set_ylabel("Slippage (%)")
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
            fig2.tight_layout()
            st.pyplot(fig2)

# Almgren-Chriss optimal execution model section
with st.expander("🖐️ Almgren-Chriss Optimal Execution Model", expanded=False):
    st.markdown("This model calculates the optimal trading schedule to minimize execution cost and risk.")

    # User inputs for total shares, time intervals, and risk aversion parameter
    ac_col1, ac_col2, ac_col3 = st.columns(3)
    with ac_col1:
        total_quantity = st.number_input("Total Shares to Trade", value=10000)
    with ac_col2:
        num_intervals = st.number_input("Number of Time Intervals", value=10, min_value=1)
    with ac_col3:
        risk_aversion = st.number_input("Risk Aversion (\u03bb)", value=1e-6, format="%.1e")

    # Constants and parameters for Almgren-Chriss formula
    gamma = 2.5e-6  # Market impact coefficient
    sigma = volatility * 0.01  # Volatility converted from % to decimal
    T = 1.0  # Total time horizon
    delta_t = T / num_intervals

    # Compute sum of hyperbolic sines used in the formula denominator
    sinh_sum = np.sinh(np.sqrt(risk_aversion * gamma / sigma**2) * T)

    # Calculate optimal shares to trade per interval using Almgren-Chriss formula
    optimal_trades = [
        total_quantity * np.sinh(np.sqrt(risk_aversion * gamma / sigma**2) * (T - k * delta_t)) / sinh_sum
        for k in range(num_intervals)
    ]

    st.markdown("### 📟 Optimal Trade Schedule")

    # Display optimal trades as a DataFrame with intervals
    trade_df = pd.DataFrame({"Interval": list(range(1, num_intervals + 1)), "Shares to Trade": np.round(optimal_trades, 2)})
    st.dataframe(trade_df, use_container_width=True)

    st.markdown("### 📉 Execution Schedule Visualization")

    # Plot the optimal execution schedule over the intervals
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(trade_df["Interval"], trade_df["Shares to Trade"], marker='o', color='navy')
    ax3.set_title("Optimal Execution Schedule (Almgren-Chriss)")
    ax3.set_xlabel("Interval")
    ax3.set_ylabel("Shares")
    ax3.grid(True)
    st.pyplot(fig3)
