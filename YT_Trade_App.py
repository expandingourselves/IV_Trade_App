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
from typing import Optional, List, Dict, Any
import altair as alt
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Earnings Position Checker", 
    page_icon="📈", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================ Core utilities ============================

def badge(text: str, good: Optional[bool] = None) -> str:
    """Create colored badge for display"""
    if good is True:
        color = "#16a34a"  # green
    elif good is False:
        color = "#b91c1c"  # red
    else:
        color = "#2563eb"  # blue (neutral)
    return f"<span style='background:{color};color:white;padding:2px 8px;border-radius:12px;font-size:0.85rem'>{text}</span>"

def filter_dates(dates: List[str]) -> List[str]:
    """Filter dates to find appropriate expiration dates"""
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
    """Calculate Yang-Zhang volatility estimator"""
    try:
        # Ensure we have required columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        if not all(col in price_data.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Need: {required_cols}")
        
        # Remove any rows with NaN values
        price_data = price_data[required_cols].dropna()
        
        if len(price_data) < window + 1:
            raise ValueError(f"Insufficient data: need at least {window + 1} rows, got {len(price_data)}")
        
        log_ho = np.log(price_data['High'] / price_data['Open'])
        log_lo = np.log(price_data['Low'] / price_data['Open'])
        log_co = np.log(price_data['Close'] / price_data['Open'])

        log_oc = np.log(price_data['Open'] / price_data['Close'].shift(1))
        log_oc_sq = log_oc**2

        log_cc = np.log(price_data['Close'] / price_data['Close'].shift(1))
        log_cc_sq = log_cc**2

        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

        close_vol = log_cc_sq.rolling(window=window).sum() * (1.0 / (window - 1.0))
        open_vol = log_oc_sq.rolling(window=window).sum() * (1.0 / (window - 1.0))
        window_rs = rs.rolling(window=window).sum() * (1.0 / (window - 1.0))

        k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
        result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

        result_clean = result.dropna()
        if len(result_clean) == 0:
            raise ValueError("No valid Yang-Zhang volatility values computed")
            
        if return_last_only:
            return float(result_clean.iloc[-1])
        else:
            return result_clean
    except Exception as e:
        raise ValueError(f"Yang-Zhang calculation failed: {str(e)}")

def build_term_structure(days, ivs):
    """Build volatility term structure interpolator"""
    days = np.array(days)
    ivs = np.array(ivs)

    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]

    # Use cubic if we have enough points, otherwise linear
    kind = 'cubic' if len(days) >= 4 else 'linear'
    spline = interp1d(days, ivs, kind=kind, fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]:
            return float(ivs[0])
        elif dte > days[-1]:
            return float(ivs[-1])
        else:
            return float(spline(dte))

    return term_spline

def get_current_price(ticker_obj: yf.Ticker) -> Optional[float]:
    """Get current stock price with multiple fallback methods"""
    try:
        # Method 1: Fast info
        try:
            info = ticker_obj.fast_info
            if hasattr(info, 'last_price') and info.last_price:
                return float(info.last_price)
            if isinstance(info, dict) and 'last_price' in info and info['last_price']:
                return float(info['last_price'])
        except:
            pass
        
        # Method 2: Today's history
        try:
            today_data = ticker_obj.history(period='1d', interval='1m')
            if not today_data.empty:
                return float(today_data['Close'].iloc[-1])
        except:
            pass
            
        # Method 3: Recent history
        try:
            recent_data = ticker_obj.history(period='5d')
            if not recent_data.empty:
                return float(recent_data['Close'].iloc[-1])
        except:
            pass
            
        # Method 4: Info dict
        try:
            info = ticker_obj.info
            if 'currentPrice' in info and info['currentPrice']:
                return float(info['currentPrice'])
            if 'regularMarketPrice' in info and info['regularMarketPrice']:
                return float(info['regularMarketPrice'])
        except:
            pass
            
        return None
        
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=300)
def fetch_ticker_data(ticker: str) -> Dict[str, Any]:
    """Fetch and cache ticker data"""
    try:
        ticker_obj = yf.Ticker(ticker)
        
        # Get expirations with timeout
        expirations = []
        try:
            expirations = list(ticker_obj.options or [])
        except Exception:
            pass
        
        # Get current price
        price = get_current_price(ticker_obj)
        
        # Get historical data
        hist_data = None
        try:
            hist_data = ticker_obj.history(period='6mo')  # Increased period for better data
            if hist_data.empty:
                hist_data = ticker_obj.history(period='3mo')
        except Exception:
            pass
            
        return {
            'expirations': expirations,
            'price': price,
            'history': hist_data
        }
        
    except Exception as e:
        return {
            'expirations': [],
            'price': None,
            'history': None,
            'error': str(e)
        }

@st.cache_data(show_spinner=False, ttl=300)
def fetch_options_chain(ticker: str, exp_date: str) -> Dict[str, Any]:
    """Fetch and cache options chain for a specific expiration"""
    try:
        ticker_obj = yf.Ticker(ticker)
        chain = ticker_obj.option_chain(exp_date)
        
        calls = getattr(chain, 'calls', pd.DataFrame())
        puts = getattr(chain, 'puts', pd.DataFrame())
        
        return {
            'calls': calls,
            'puts': puts,
            'success': True
        }
    except Exception as e:
        return {
            'calls': pd.DataFrame(),
            'puts': pd.DataFrame(), 
            'success': False,
            'error': str(e)
        }

def compute_recommendation(ticker: str):
    """Main computation function with improved error handling"""
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return "No stock symbol provided."

        # Fetch basic ticker data
        ticker_data = fetch_ticker_data(ticker)
        
        if 'error' in ticker_data:
            return f"Error fetching data for {ticker}: {ticker_data['error']}"
            
        expirations = ticker_data['expirations']
        underlying_price = ticker_data['price']
        price_history = ticker_data['history']
        
        if not expirations:
            return f"No options found for {ticker}. Verify the ticker symbol."
        
        if not underlying_price or underlying_price <= 0:
            return f"Unable to retrieve current price for {ticker}."
            
        if price_history is None or price_history.empty:
            return f"Unable to retrieve price history for {ticker}."

        # Filter expiration dates
        try:
            exp_dates = filter_dates(expirations)
        except ValueError as e:
            return f"Options data issue: {str(e)}"

        # Limit to first 5 expirations for performance
        exp_dates = exp_dates[:5]
        
        # Fetch options chains with progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        options_data = {}
        for i, exp_date in enumerate(exp_dates):
            status_text.text(f"Fetching options chain {i+1}/{len(exp_dates)}...")
            progress_bar.progress((i + 1) / len(exp_dates))
            
            chain_data = fetch_options_chain(ticker, exp_date)
            if chain_data['success']:
                options_data[exp_date] = chain_data
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        progress_bar.empty()
        status_text.empty()

        if not options_data:
            return "Unable to fetch any options chains."

        # Calculate ATM IV for each expiration
        atm_iv = {}
        straddle = None
        first_done = False

        for exp_date, chain_data in options_data.items():
            calls = chain_data['calls']
            puts = chain_data['puts']
            
            if calls.empty or puts.empty:
                continue

            try:
                # Find ATM options
                call_diffs = (calls['strike'] - underlying_price).abs()
                put_diffs = (puts['strike'] - underlying_price).abs()
                
                call_idx = call_diffs.idxmin()
                put_idx = put_diffs.idxmin()

                call_iv = float(calls.loc[call_idx, 'impliedVolatility'])
                put_iv = float(puts.loc[put_idx, 'impliedVolatility'])
                
                # Basic sanity check
                if call_iv <= 0 or put_iv <= 0 or call_iv > 10 or put_iv > 10:
                    continue
                    
                atm_iv_value = (call_iv + put_iv) / 2.0
                atm_iv[exp_date] = atm_iv_value

                # Calculate straddle for first valid expiration
                if not first_done:
                    try:
                        call_bid = calls.loc[call_idx, 'bid']
                        call_ask = calls.loc[call_idx, 'ask']
                        put_bid = puts.loc[put_idx, 'bid']
                        put_ask = puts.loc[put_idx, 'ask']

                        if all(pd.notna([call_bid, call_ask, put_bid, put_ask])) and all(x > 0 for x in [call_bid, call_ask, put_bid, put_ask]):
                            call_mid = (call_bid + call_ask) / 2.0
                            put_mid = (put_bid + put_ask) / 2.0
                            straddle = float(call_mid + put_mid)
                            first_done = True
                    except:
                        pass
                        
            except Exception:
                continue

        if not atm_iv:
            return "Unable to calculate ATM IV for any expiration dates."

        # Build term structure
        today = datetime.today().date()
        dtes, ivs = [], []
        
        for exp_date, iv in atm_iv.items():
            try:
                exp_date_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
                days_to_expiry = (exp_date_obj - today).days
                if days_to_expiry > 0:
                    dtes.append(days_to_expiry)
                    ivs.append(iv)
            except:
                continue

        if len(dtes) < 2:
            return "Insufficient options data for analysis."

        term_spline = build_term_structure(dtes, ivs)

        # Calculate term structure slope
        d0 = min(dtes)
        d45 = 45
        ts_slope_0_45 = 0.0 if d0 == d45 else (term_spline(d45) - term_spline(d0)) / (45 - d0)

        # Calculate Yang-Zhang volatility
        try:
            yz_vol_annual = yang_zhang(price_history)
        except ValueError as e:
            return f"Volatility calculation error: {str(e)}"

        # IV/RV ratio
        iv30 = term_spline(30)
        iv30_rv30 = iv30 / yz_vol_annual

        # Volume analysis
        try:
            recent_volume = price_history['Volume'].tail(30)
            avg_volume = recent_volume.mean() if not recent_volume.empty else 0
        except:
            avg_volume = 0

        # Expected move calculation
        expected_move = None
        if straddle:
            expected_move = f"{round(straddle / underlying_price * 100, 2)}%"

        return {
            'success': True,
            'avg_volume': bool(avg_volume >= 1_500_000),
            'iv30_rv30': float(iv30_rv30) >= 1.25,
            'ts_slope_0_45': float(ts_slope_0_45) <= -0.00406,
            'expected_move': expected_move,
            'underlying_price': underlying_price,
            'atm_iv_points': pd.DataFrame({'DTE': dtes, 'ATM_IV': ivs}).sort_values('DTE').reset_index(drop=True),
            'iv30_rv30_value': iv30_rv30,
            'ts_slope_value': ts_slope_0_45,
            'avg_volume_value': avg_volume
        }
        
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# ============================ UI ============================

st.title("📈 Earnings Position Checker")
st.markdown(
    "[![Stocktwits Earnings Calendar](https://img.shields.io/badge/📅-Stocktwits_Earnings_Calendar-blue)](https://stocktwits.com/sentiment/calendar)",
    unsafe_allow_html=True,
)
st.caption("Evaluates pre-earnings criteria using options term structure (ATM IV) and Yang–Zhang realized volatility.")

# Instructions
with st.expander("📋 How to Use"):
    st.markdown("""
    1. **Enter a US stock ticker** (e.g., AAPL, MSFT, TSLA)
    2. **Click "Run Analysis"** to fetch data and compute metrics
    3. **Review the recommendation** based on three key criteria:
       - **Volume**: 30-day average ≥ 1.5M shares
       - **IV/RV Ratio**: 30-day implied/realized volatility ≥ 1.25
       - **Term Structure**: Downward sloping (slope ≤ -0.00406)
    
    **Recommendations:**
    - 🟢 **Recommended**: All 3 criteria met
    - 🟡 **Consider**: Term structure + 1 other criteria
    - 🔴 **Avoid**: Less than 2 criteria met
    """)

# Disclaimer
with st.expander("⚠️ Important Disclaimer"):
    st.warning("""
    **EDUCATIONAL PURPOSE ONLY**
    
    This tool is for educational and research purposes only. It does not provide investment advice. 
    The developers are not financial advisors and accept no responsibility for any financial decisions 
    or losses. Always consult a professional financial advisor before making investment decisions.
    """)

# Input section
col1, col2 = st.columns([3, 1])
with col1:
    ticker = st.text_input(
        "Stock Symbol", 
        value="", 
        placeholder="e.g., AAPL, MSFT, TSLA",
        help="Enter a US equity ticker symbol with listed options"
    )
with col2:
    st.write("")  # Spacing
    run = st.button("🔍 Run Analysis", type="primary", use_container_width=True)

if run:
    if not ticker.strip():
        st.warning("⚠️ Please enter a stock symbol.")
        st.stop()

    with st.spinner("🔄 Analyzing options data... This may take 30-60 seconds."):
        result = compute_recommendation(ticker)

    if isinstance(result, str) or not result.get('success', False):
        error_msg = result if isinstance(result, str) else result.get('error', 'Unknown error')
        st.error(f"❌ {error_msg}")
        
        # Suggestions for common issues
        st.info("""
        **Common issues:**
        - Verify the ticker symbol is correct (US equities only)
        - Some tickers may have limited options data
        - Try again in a few moments if rate-limited
        """)
        st.stop()

    # Extract results
    avg_volume_bool = result['avg_volume']
    iv30_rv30_bool = result['iv30_rv30']
    ts_slope_bool = result['ts_slope_0_45']
    expected_move = result['expected_move']
    price = result['underlying_price']
    atm_df = result['atm_iv_points']
    
    # Additional metrics for display
    iv30_rv30_value = result.get('iv30_rv30_value', 0)
    ts_slope_value = result.get('ts_slope_value', 0)
    avg_volume_value = result.get('avg_volume_value', 0)

    # Overall recommendation logic
    criteria_met = sum([avg_volume_bool, iv30_rv30_bool, ts_slope_bool])
    
    if criteria_met == 3:
        recommendation = "🟢 Recommended"
        color = "#16a34a"
        description = "All criteria met - favorable setup"
    elif criteria_met == 2 and ts_slope_bool:
        recommendation = "🟡 Consider"
        color = "#ea580c"
        description = "Term structure favorable with one additional criteria"
    else:
        recommendation = "🔴 Avoid"
        color = "#dc2626"
        description = "Insufficient criteria met"

    # Display results
    st.markdown(f"<h2 style='color:{color};margin-top:1rem'>{recommendation}</h2>", unsafe_allow_html=True)
    st.caption(description)

    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${price:,.2f}")
    with col2:
        if expected_move:
            st.metric("Expected Move", expected_move)
        else:
            st.metric("Expected Move", "N/A")
    with col3:
        st.metric("Criteria Met", f"{criteria_met}/3")

    # Detailed criteria
    st.subheader("📊 Criteria Analysis")
    
    criteria_data = [
        {
            "Criterion": "Volume (30d avg ≥ 1.5M)",
            "Status": "✅ PASS" if avg_volume_bool else "❌ FAIL",
            "Value": f"{avg_volume_value:,.0f}",
            "Target": "≥ 1,500,000"
        },
        {
            "Criterion": "IV30/RV30 Ratio ≥ 1.25",
            "Status": "✅ PASS" if iv30_rv30_bool else "❌ FAIL", 
            "Value": f"{iv30_rv30_value:.2f}",
            "Target": "≥ 1.25"
        },
        {
            "Criterion": "Term Slope ≤ -0.00406",
            "Status": "✅ PASS" if ts_slope_bool else "❌ FAIL",
            "Value": f"{ts_slope_value:.5f}",
            "Target": "≤ -0.00406"
        }
    ]
    
    criteria_df = pd.DataFrame(criteria_data)
    st.dataframe(criteria_df, use_container_width=True, hide_index=True)

    # Volatility term structure chart
    if not atm_df.empty:
        st.subheader("📈 ATM Implied Volatility Term Structure")
        
        chart_df = atm_df.copy()
        chart_df["ATM_IV_pct"] = chart_df["ATM_IV"] * 100

        # Create base chart
        base_chart = alt.Chart(chart_df).mark_line(
            point=alt.OverlayMarkDef(filled=True, size=100),
            strokeWidth=3,
            color="#2563eb"
        ).encode(
            x=alt.X("DTE:Q", title="Days to Expiration"),
            y=alt.Y("ATM_IV_pct:Q", title="ATM Implied Volatility (%)", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("DTE:Q", title="DTE"),
                alt.Tooltip("ATM_IV_pct:Q", title="ATM IV (%)", format=".2f")
            ]
        ).properties(
            width=700,
            height=400,
            title="Implied Volatility Term Structure"
        )

        # Add expected move line if available
        layers = [base_chart]
        if expected_move:
            try:
                exp_val = float(expected_move.strip("%"))
                rule_df = pd.DataFrame({"expected_move": [exp_val]})
                
                exp_move_line = alt.Chart(rule_df).mark_rule(
                    strokeDash=[5, 5],
                    strokeWidth=2,
                    color="#dc2626"
                ).encode(
                    y=alt.Y("expected_move:Q"),
                    tooltip=alt.value(f"Expected Move: {expected_move}")
                )
                layers.append(exp_move_line)
                
            except:
                pass

        final_chart = alt.layer(*layers).resolve_scale(y='independent')
        st.altair_chart(final_chart, use_container_width=True)

    # Additional information
    with st.expander("ℹ️ Methodology & Notes"):
        st.markdown("""
        **Data Sources:** Yahoo Finance (may have delays)
        
        **Calculations:**
        - **ATM IV**: Average of call and put implied volatilities at nearest-to-money strikes
        - **Yang-Zhang RV**: 30-day realized volatility using Yang-Zhang estimator  
        - **Term Structure Slope**: Linear slope from nearest expiry to 45-day interpolated IV
        - **Expected Move**: Straddle price as percentage of underlying (nearest expiry)
        
        **Limitations:**
        - Data may be delayed or incomplete
        - Options with low volume may have unreliable pricing
        - Analysis is point-in-time and market conditions change rapidly
        """)

    # Footer
    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not investment advice.")
    
    # Success message
    st.success(f"✅ Analysis completed for {ticker.upper()}")

else:
    # Show example when not running
    st.markdown("---")
    st.markdown("**💡 Try these popular tickers:** `AAPL`, `MSFT`, `GOOGL`, `TSLA`, `AMZN`, `NVDA`")
    
    # Sample output preview
    with st.expander("👀 Preview: Sample Analysis Results"):
        st.image("https://via.placeholder.com/600x300/f0f0f0/666666?text=Sample+Volatility+Chart", caption="Example: ATM IV Term Structure")
        
        sample_df = pd.DataFrame({
            "Criterion": ["Volume ≥ 1.5M", "IV/RV ≥ 1.25", "Term Slope ≤ -0.00406"],
            "Status": ["✅ PASS", "✅ PASS", "❌ FAIL"],
            "Value": ["2,450,000", "1.32", "0.00123"],
            "Target": ["≥ 1,500,000", "≥ 1.25", "≤ -0.00406"]
        })
        st.dataframe(sample_df, use_container_width=True, hide_index=True)
