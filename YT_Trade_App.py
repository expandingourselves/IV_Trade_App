"""
DISCLAIMER:

This software is provided solely for educational and research purposes.
It is not intended to provide investment advice, and no investment recommendations are made herein.
The developers are not financial advisors and accept no responsibility for any financial decisions or losses resulting from the use of this software.
Always consult a professional financial advisor before making any investment decisions.
"""

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from typing import Optional, List
import altair as alt

st.set_page_config(page_title="Earnings Position Checker", page_icon="📈", layout="centered")

# ============================ Core utilities ============================

def badge(text: str, good: Optional[bool] = None) -> str:
    if good is True:
        color = "#16a34a"  # green
    elif good is False:
        color = "#b91c1c"  # red
    else:
        color = "#2563eb"  # blue (neutral)
    return f"<span style='background:{color};color:white;padding:2px 8px;border-radius:12px;font-size:0.85rem'>{text}</span>"

def filter_dates(dates: List[str]) -> List[str]:
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)

    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]
            break

    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr

    raise ValueError("No date 45 days or more in the future found.")

def yang_zhang(price_data: pd.DataFrame, window=30, trading_periods=252, return_last_only=True):
    # Expect price_data columns: Open, High, Low, Close
    log_ho = np.log(price_data['High'] / price_data['Open'])
    log_lo = np.log(price_data['Low'] / price_data['Open'])
    log_co = np.log(price_data['Close'] / price_data['Open'])

    log_oc = np.log(price_data['Open'] / price_data['Close'].shift(1))
    log_oc_sq = log_oc**2

    log_cc = np.log(price_data['Close'] / price_data['Close'].shift(1))
    log_cc_sq = log_cc**2

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_vol = log_cc_sq.rolling(window=window).sum() * (1.0 / (window - 1.0))
    open_vol  = log_oc_sq.rolling(window=window).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(window=window).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

    if return_last_only:
        return result.dropna().iloc[-1]
    else:
        return result.dropna()

def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)

    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]

    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]:
            return float(ivs[0])
        elif dte > days[-1]:
            return float(ivs[-1])
        else:
            return float(spline(dte))

    return term_spline

def get_current_price(ticker_obj: yf.Ticker) -> Optional[float]:
    # Try attribute then mapping access; fall back to 1d history
    try:
        px = getattr(ticker_obj.fast_info, "last_price", None)
        if px is None:
            fi = ticker_obj.fast_info
            try:
                px = fi["last_price"]
            except Exception:
                px = None
        if px and px > 0:
            return float(px)
    except Exception:
        pass
    todays = ticker_obj.history(period='1d')
    if len(todays) == 0:
        return None
    return float(todays['Close'].iloc[-1])

@st.cache_data(show_spinner=False, ttl=300)
def fetch_expirations_and_price(ticker: str):
    """Cache ONLY simple data (not the Ticker object)."""
    t = yf.Ticker(ticker)
    expirations = list(t.options or [])
    price = get_current_price(t)
    return expirations, price

def compute_recommendation(ticker: str):
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return "No stock symbol provided."

        expirations, underlying_price = fetch_expirations_and_price(ticker)
        if not expirations:
            return f"Error: No options found for stock symbol '{ticker}'."
        if not underlying_price:
            return "Error: Unable to retrieve underlying stock price."

        try:
            exp_dates = filter_dates(expirations)
        except Exception:
            return "Error: Not enough option data."

        # Recreate Ticker for chains (don’t use cached object)
        stock = yf.Ticker(ticker)

        options_chains = {}
        for exp_date in exp_dates:
            try:
                options_chains[exp_date] = stock.option_chain(exp_date)
            except Exception:
                continue

        atm_iv = {}
        straddle = None
        first_done = False

        for exp_date, chain in options_chains.items():
            calls = getattr(chain, "calls", pd.DataFrame())
            puts  = getattr(chain, "puts",  pd.DataFrame())
            if calls.empty or puts.empty:
                continue

            # ATM IV via nearest strikes
            call_diffs = (calls['strike'] - underlying_price).abs()
            put_diffs  = (puts['strike']  - underlying_price).abs()
            call_idx   = call_diffs.idxmin()
            put_idx    = put_diffs.idxmin()

            call_iv = float(calls.loc[call_idx, 'impliedVolatility'])
            put_iv  = float(puts.loc[put_idx,  'impliedVolatility'])
            atm_iv_value = (call_iv + put_iv) / 2.0
            atm_iv[exp_date] = atm_iv_value

            # First expiry: straddle mid for expected move
            if not first_done:
                call_bid = calls.loc[call_idx, 'bid']
                call_ask = calls.loc[call_idx, 'ask']
                put_bid  = puts.loc[put_idx,  'bid']
                put_ask  = puts.loc[put_idx,  'ask']

                call_mid = (call_bid + call_ask) / 2.0 if pd.notna(call_bid) and pd.notna(call_ask) else None
                put_mid  = (put_bid  + put_ask)  / 2.0 if pd.notna(put_bid)  and pd.notna(put_ask)  else None
                if call_mid is not None and put_mid is not None:
                    straddle = float(call_mid + put_mid)
                first_done = True

        if not atm_iv:
            return "Error: Could not determine ATM IV for any expiration dates."

        today = datetime.today().date()
        dtes, ivs = [], []
        for exp_date, iv in atm_iv.items():
            exp_date_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
            days_to_expiry = (exp_date_obj - today).days
            if days_to_expiry > 0:
                dtes.append(days_to_expiry)
                ivs.append(iv)

        if not dtes:
            return "Error: Unable to compute DTEs."

        term_spline = build_term_structure(dtes, ivs)

        # slope between nearest DTE and 45 days
        d0 = min(dtes)
        d45 = 45
        ts_slope_0_45 = 0.0 if d0 == d45 else (term_spline(d45) - term_spline(d0)) / (45 - d0)

        # Historical prices for Yang–Zhang
        price_history = stock.history(period='3mo')
        if price_history.empty:
            return "Error: Not enough historical price data."

        try:
            yz_vol_annual = yang_zhang(price_history)
        except Exception:
            return "Error: Failed to compute realized volatility."

        iv30_rv30 = term_spline(30) / yz_vol_annual

        # Liquidity proxy
        try:
            avg_volume = price_history['Volume'].rolling(30).mean().dropna().iloc[-1]
        except Exception:
            avg_volume = np.nan

        expected_move = f"{round(straddle / underlying_price * 100, 2)}%" if straddle else None

        return {
            'avg_volume': bool(avg_volume >= 1_500_000) if pd.notna(avg_volume) else False,
            'iv30_rv30': float(iv30_rv30) >= 1.25,
            'ts_slope_0_45': float(ts_slope_0_45) <= -0.00406,
            'expected_move': expected_move,
            'underlying_price': underlying_price,
            'atm_iv_points': pd.DataFrame({'DTE': dtes, 'ATM_IV': ivs}).sort_values('DTE').reset_index(drop=True),
        }
    except Exception:
        return "Error occurred processing."

# ============================ UI ============================

st.title("📈 Earnings Position Checker")
st.markdown(
    "<a href='https://stocktwits.com/sentiment/calendar' target='_blank'>"
    "🔗 Stocktwits Earnings Calendar ↗</a>",
    unsafe_allow_html=True,
)
st.caption("Evaluates pre-earnings criteria from options term structure (ATM IV) and Yang–Zhang realized volatility.")

with st.expander("Read the disclaimer"):
    st.write(__doc__)

ticker = st.text_input("Enter Stock Symbol", value="", placeholder="e.g., AAPL", help="US equity ticker with listed options")
run = st.button("Run Check", type="primary")

if run:
    if not ticker.strip():
        st.warning("Please enter a stock symbol.")
        st.stop()

    with st.spinner("Fetching data and computing metrics..."):
        result = compute_recommendation(ticker)

    if isinstance(result, str):
        st.error(result)
        st.stop()

    avg_volume_bool = result['avg_volume']
    iv30_rv30_bool  = result['iv30_rv30']
    ts_slope_bool   = result['ts_slope_0_45']
    expected_move   = result['expected_move']
    price           = result['underlying_price']
    atm_df          = result['atm_iv_points']

    # Overall verdict
    if avg_volume_bool and iv30_rv30_bool and ts_slope_bool:
        title = "Recommended"
        color = "#065f46"
    elif ts_slope_bool and ((avg_volume_bool and not iv30_rv30_bool) or (iv30_rv30_bool and not avg_volume_bool)):
        title = "Consider"
        color = "#92400e"
    else:
        title = "Avoid"
        color = "#7f1d1d"

    st.markdown(f"<h3 style='color:{color};margin-top:0.5rem'>{title}</h3>", unsafe_allow_html=True)
    st.markdown(
        f"**Ticker:** `{ticker.upper()}`  |  **Last Price:** ${price:,.2f}"
        + (f"  |  **Expected Move (nearest straddle):** {badge(expected_move)}" if expected_move else ""),
        unsafe_allow_html=True,
    )

    # Criteria table
    rows = [
        ("avg_volume ≥ 1.5M (30-day avg)", avg_volume_bool),
        ("IV30 / RV30 ≥ 1.25", iv30_rv30_bool),
        ("Term slope (d0→45d) ≤ −0.00406", ts_slope_bool),
    ]
    for label, ok in rows:
        st.markdown(f"- {label}: {badge('PASS' if ok else 'FAIL', ok)}", unsafe_allow_html=True)

    # ATM IV chart in %, with expected move overlay
    if isinstance(atm_df, pd.DataFrame) and not atm_df.empty:
        st.subheader("ATM IV vs DTE")

        chart_df = atm_df.copy()
        chart_df["ATM_IV_pct"] = chart_df["ATM_IV"] * 100  # convert to %

        base = (
            alt.Chart(chart_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("DTE:Q", title="Days to Expiration (DTE)"),
                y=alt.Y("ATM_IV_pct:Q", title="ATM IV (%)"),
                tooltip=[alt.Tooltip("DTE:Q", title="DTE"), alt.Tooltip("ATM_IV_pct:Q", title="ATM IV (%)", format=".2f")]
            )
            .properties(width=700, height=420)
        )

        overlays = []
        title_suffix = ""
        if expected_move:
            try:
                exp_val = float(expected_move.strip("%"))
                overlays.append(
                    alt.Chart(pd.DataFrame({"y": [exp_val]}))
                    .mark_rule(strokeDash=[5, 5])
                    .encode(y="y:Q")
                )
                title_suffix = f"  —  Expected Move ≈ {expected_move}"
            except Exception:
                pass

        chart = alt.layer(base, *overlays).properties(
            title=f"ATM Implied Volatility Term Structure{title_suffix}"
        )
        st.altair_chart(chart, use_container_width=True)

    st.info(
        "Notes:\n"
        "- IV30 is interpolated from the ATM IV term structure; RV30 is Yang–Zhang annualized realized volatility.\n"
        "- The slope check looks for a downward term structure from the nearest expiry to 45 days."
    )

# ============================ Footer ============================

st.caption("Data source: Yahoo Finance (via yfinance). Dates/vols are approximate and may be delayed.")
st.caption("Created by Shashank Agarwal ")