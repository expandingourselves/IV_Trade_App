"""
DISCLAIMER:

This software is provided solely for educational and research purposes.
It is not intended to provide investment advice, and no investment recommendations are made herein.
The developers are not financial advisors and accept no responsibility for any financial decisions or losses resulting from the use of this software.
Always consult a professional financial advisor before making any investment decisions.
"""

from typing import Optional, List, Tuple, Dict
import time
from json import JSONDecodeError
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from scipy.interpolate import interp1d
import altair as alt
from yahooquery import Ticker as YQTicker
from requests.exceptions import RequestException, Timeout, ConnectionError as ReqConnectionError

st.set_page_config(page_title="Earnings Position Checker", page_icon="📈", layout="centered")

# ============================ Helpers & analytics ============================

def badge(text: str, good: Optional[bool] = None) -> str:
    if good is True:
        color = "#16a34a"  # green
    elif good is False:
        color = "#b91c1c"  # red
    else:
        color = "#2563eb"  # blue (neutral)
    return f"<span style='background:{color};color:white;padding:2px 8px;border-radius:12px;font-size:0.85rem'>{text}</span>"

def filter_dates(dates: List[str]) -> List[str]:
    today = datetime.now(timezone.utc).date()
    cutoff_date = today + timedelta(days=45)
    sorted_dates = sorted(datetime.strptime(d, "%Y-%m-%d").date() for d in dates)

    arr: List[str] = []
    for i, dt in enumerate(sorted_dates):
        if dt >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]
            break

    if arr:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr
    raise ValueError("No date 45 days or more in the future found.")

def yang_zhang(price_data: pd.DataFrame, window=30, trading_periods=252, return_last_only=True):
    log_ho = np.log(price_data['High'] / price_data['Open'])
    log_lo = np.log(price_data['Low'] / price_data['Open'])
    log_co = np.log(price_data['Close'] / price_data['Open'])
    log_oc = np.log(price_data['Open'] / price_data['Close'].shift(1))
    log_cc = np.log(price_data['Close'] / price_data['Close'].shift(1))

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_vol = (log_cc**2).rolling(window=window).sum() * (1.0 / (window - 1.0))
    open_vol  = (log_oc**2).rolling(window=window).sum() * (1.0 / (window - 1.0))
    window_rs = (rs**1).rolling(window=window).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

    return result.dropna().iloc[-1] if return_last_only else result.dropna()

def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)
    order = days.argsort()
    spline = interp1d(days[order], ivs[order], kind='linear', fill_value="extrapolate")

    def term_spline(dte: float):
        dte = float(dte)
        if dte <= days[order][0]:  return float(ivs[order][0])
        if dte >= days[order][-1]: return float(ivs[order][-1])
        return float(spline(dte))
    return term_spline

# ============================ HTTP sessions ============================

def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": "https://finance.yahoo.com/",
        "Cache-Control": "no-cache",
    })
    return s

@st.cache_resource(ttl=900)
def get_http_session() -> requests.Session:
    return _new_session()

# ============================ Data fetch: yfinance → query2 → yahooquery ============================

def get_current_price_yf(t: yf.Ticker) -> Optional[float]:
    try:
        px = getattr(t.fast_info, "last_price", None)
        if px is None:
            fi = t.fast_info
            try: px = fi["last_price"]
            except Exception: px = None
        if px and px > 0: return float(px)
    except Exception:
        pass
    todays = t.history(period='1d', interval='1d', auto_adjust=False, prepost=False)
    if len(todays) == 0: return None
    return float(todays['Close'].iloc[-1])

@st.cache_data(show_spinner=False, ttl=300)
def fetch_expirations_and_price(ticker: str) -> Tuple[str, List[str], Optional[float], Dict[str, int], Optional[str]]:
    """
    Returns (mode, expirations, price, exp_map_unix, error):
      - mode: "yf" or "direct" or "yq"
      - expirations: list of "YYYY-MM-DD"
      - price: underlying price
      - exp_map_unix: for direct/yq mode, map date->unix expiration (else {})
      - error: message if all methods fail
    """
    ticker = ticker.upper().replace(".", "-").strip()
    session = get_http_session()

    # 1) yfinance (cheapest)
    err = None
    for attempt in range(3):
        try:
            t = yf.Ticker(ticker, session=session if attempt == 0 else _new_session())
            expirations = list(t.options or [])
            price = get_current_price_yf(t)
            if expirations and price is not None:
                return "yf", expirations, price, {}, None
            raise ValueError("Empty options or price")
        except (JSONDecodeError, ReqConnectionError, Timeout, RequestException, ValueError) as e:
            err = f"{type(e).__name__}: {e}"
            time.sleep(0.6 * (attempt + 1))

    # 2) query2 finance API (no /validate)
    base = f"https://query2.finance.yahoo.com/v7/finance/options/{ticker}"
    exp_map_unix: Dict[str, int] = {}
    for attempt in range(3):
        try:
            r = session.get(base, timeout=8)
            if r.status_code == 429:
                raise ValueError("Rate-limited (429) on query2 base")
            if r.status_code != 200 or not r.text or not r.headers.get("content-type","").startswith("application/json"):
                raise ValueError(f"Bad response ({r.status_code})")
            data = r.json()
            result = data["optionChain"]["result"][0]
            exp_unix = result.get("expirationDates", [])
            expirations = [datetime.utcfromtimestamp(u).strftime("%Y-%m-%d") for u in exp_unix]
            exp_map_unix = {datetime.utcfromtimestamp(u).strftime("%Y-%m-%d"): int(u) for u in exp_unix}
            q = result.get("quote", {}) or {}
            price = q.get("regularMarketPrice") or q.get("postMarketPrice") or q.get("preMarketPrice")
            if expirations and price:
                return "direct", expirations, float(price), exp_map_unix, None
            raise ValueError("No expirations/price in direct API")
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            time.sleep(0.8 * (attempt + 1))

    # 3) yahooquery (disable validate to avoid /v6/finance/quote/validate 429s)
    try:
        yq = YQTicker(ticker, validate=False, formatted=False)
        price_map = yq.price or {}
        p = (price_map.get(ticker) or {}).get("regularMarketPrice")
        exp_unix = (yq.option_expiration_dates or {}).get(ticker, [])
        expirations = [datetime.utcfromtimestamp(u).strftime("%Y-%m-%d") for u in exp_unix]
        exp_map_unix = {datetime.utcfromtimestamp(u).strftime("%Y-%m-%d"): int(u) for u in exp_unix}
        if expirations and p:
            st.session_state.setdefault("_yq_clients", {})[ticker] = yq  # reuse cookies for chains
            return "yq", expirations, float(p), exp_map_unix, None
        raise ValueError("yahooquery returned no expirations/price")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    return "error", [], None, {}, f"Yahoo data request failed. ({err})"

def fetch_chain_direct(ticker: str, exp_unix: int, session: requests.Session) -> Tuple[pd.DataFrame, pd.DataFrame]:
    url = f"https://query2.finance.yahoo.com/v7/finance/options/{ticker}?date={exp_unix}"
    r = session.get(url, timeout=8)
    if r.status_code == 429:
        raise ValueError("Rate-limited (429) on query2 chain")
    if r.status_code != 200 or not r.headers.get("content-type","").startswith("application/json"):
        raise ValueError(f"Bad chain response ({r.status_code})")
    data = r.json()
    opt = data["optionChain"]["result"][0]["options"][0]
    calls = pd.DataFrame(opt.get("calls", []))
    puts  = pd.DataFrame(opt.get("puts",  []))
    return calls, puts

def fetch_chain_yq(ticker: str, exp_unix: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        yq = st.session_state.get("_yq_clients", {}).get(ticker)
        if yq is None:
            yq = YQTicker(ticker, validate=False, formatted=False)
        df = yq.option_chain(date=exp_unix)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(), pd.DataFrame()
        if "optionType" not in df.columns:
            return pd.DataFrame(), pd.DataFrame()
        calls = df[df["optionType"] == "calls"].copy()
        puts  = df[df["optionType"] == "puts"].copy()
        return calls, puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

# ============================ Main computation ============================

def compute_recommendation(ticker: str):
    try:
        ticker = ticker.strip().upper().replace(".", "-")
        if not ticker:
            return "No stock symbol provided."

        mode, expirations, underlying_price, exp_map_unix, err = fetch_expirations_and_price(ticker)
        if err:
            return err
        if not expirations:
            return f"Error: No options found for stock symbol '{ticker}'."
        if not underlying_price:
            return "Error: Unable to retrieve underlying stock price."

        # Choose expiries (nearest → include >=45d)
        try:
            exp_dates = filter_dates(expirations)
        except Exception:
            return "Error: Not enough option data."

        session = get_http_session()
        stock_yf = yf.Ticker(ticker, session=session) if mode == "yf" else None

        atm_iv: Dict[str, float] = {}
        straddle = None
        first_done = False

        for exp_date in exp_dates:
            # fetch chain
            if mode == "yf":
                chain = None
                for attempt in range(3):
                    try:
                        chain = stock_yf.option_chain(exp_date)
                        break
                    except Exception:
                        time.sleep(0.4 * (attempt + 1))
                if chain is None:
                    continue
                calls = getattr(chain, "calls", pd.DataFrame())
                puts  = getattr(chain, "puts",  pd.DataFrame())
            elif mode == "direct":
                exp_unix = exp_map_unix.get(exp_date)
                if not exp_unix: continue
                try:
                    calls, puts = fetch_chain_direct(ticker, exp_unix, session)
                except Exception:
                    continue
            else:  # yq
                exp_unix = exp_map_unix.get(exp_date)
                if not exp_unix: continue
                calls, puts = fetch_chain_yq(ticker, exp_unix)

            if calls.empty or puts.empty:
                continue
            if "strike" not in calls.columns or "strike" not in puts.columns:
                continue

            # nearest with valid IV
            def nearest_with_iv(df: pd.DataFrame):
                if "impliedVolatility" not in df.columns:
                    return None
                df2 = df[pd.notna(df["impliedVolatility"])].copy()
                if df2.empty:
                    return None
                df2["dist"] = (df2["strike"].astype(float) - float(underlying_price)).abs()
                return df2.sort_values("dist").iloc[0]

            call_row = nearest_with_iv(calls)
            put_row  = nearest_with_iv(puts)
            if call_row is None or put_row is None:
                continue

            call_iv = float(call_row["impliedVolatility"])
            put_iv  = float(put_row["impliedVolatility"])
            atm_iv_value = (call_iv + put_iv) / 2.0
            atm_iv[exp_date] = atm_iv_value

            # straddle mid on first expiry
            if not first_done:
                def mid(row):
                    b = row.get("bid"); a = row.get("ask")
                    if pd.notna(b) and pd.notna(a):
                        return (float(b) + float(a)) / 2.0
                    return None
                cm = mid(call_row); pm = mid(put_row)
                if cm is not None and pm is not None:
                    straddle = cm + pm
                first_done = True

        if not atm_iv:
            return "Error: Could not determine ATM IV for any expiration dates."

        today = datetime.now(timezone.utc).date()
        dtes, ivs = [], []
        for exp_date, iv in atm_iv.items():
            ed = datetime.strptime(exp_date, "%Y-%m-%d").date()
            dte = (ed - today).days
            if dte > 0:
                dtes.append(dte); ivs.append(iv)

        if not dtes:
            return "Error: Unable to compute DTEs."

        term_spline = build_term_structure(dtes, ivs)
        d0 = min(dtes); d45 = 45
        ts_slope_0_45 = 0.0 if d0 == d45 else (term_spline(d45) - term_spline(d0)) / (45 - d0)

        # History: compute YZ on latest ~60 trading days (fetch 6mo for safety)
        px = yf.Ticker(ticker, session=get_http_session()).history(period='6mo', interval='1d', auto_adjust=True, prepost=False)
        px = px.dropna(subset=['Open','High','Low','Close'])
        if len(px) < 35:
            return "Error: Not enough historical price data."

        yz_vol_annual = float(yang_zhang(px.tail(60), window=30))
        if not np.isfinite(yz_vol_annual) or yz_vol_annual <= 0:
            return "Error: Realized volatility computation invalid."

        iv30_rv30 = float(term_spline(30)) / yz_vol_annual

        try:
            avg_volume = px['Volume'].rolling(30).mean().dropna().iloc[-1]
        except Exception:
            avg_volume = np.nan

        expected_move = f"{round((straddle or 0) / underlying_price * 100, 2)}%" if straddle else None

        return {
            'avg_volume': bool(avg_volume >= 1_500_000) if pd.notna(avg_volume) else False,
            'iv30_rv30': float(iv30_rv30) >= 1.25,
            'ts_slope_0_45': float(ts_slope_0_45) <= -0.00406,
            'expected_move': expected_move,
            'underlying_price': underlying_price,
            'atm_iv_points': pd.DataFrame({'DTE': dtes, 'ATM_IV': ivs}).sort_values('DTE').reset_index(drop=True),
        }
    except Exception as e:
        return f"Error occurred processing. ({type(e).__name__}: {e})"

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

with st.sidebar:
    if st.button("🔄 Clear cache"):
        st.cache_data.clear()
        st.success("Cache cleared. Re-run the check.")

ticker = st.text_input("Enter Stock Symbol", value="", placeholder="e.g., AAPL")
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

    # Verdict
    if avg_volume_bool and iv30_rv30_bool and ts_slope_bool:
        title, color = "Recommended", "#065f46"
    elif ts_slope_bool and (avg_volume_bool ^ iv30_rv30_bool):
        title, color = "Consider", "#92400e"
    else:
        title, color = "Avoid", "#7f1d1d"

    st.markdown(f"<h3 style='color:{color};margin-top:0.5rem'>{title}</h3>", unsafe_allow_html=True)
    st.markdown(
        f"**Ticker:** `{ticker.strip().upper().replace('.', '-')}`  |  **Last Price:** ${price:,.2f}"
        + (f"  |  **Expected Move (nearest straddle):** {badge(expected_move)}" if expected_move else ""),
        unsafe_allow_html=True,
    )

    # Criteria bullets
    for label, ok in [
        ("avg_volume ≥ 1.5M (30-day avg)", avg_volume_bool),
        ("IV30 / RV30 ≥ 1.25", iv30_rv30_bool),
        ("Term slope (d0→45d) ≤ −0.00406", ts_slope_bool),
    ]:
        st.markdown(f"- {label}: {badge('PASS' if ok else 'FAIL', ok)}", unsafe_allow_html=True)

    # Chart
    if not atm_df.empty:
        st.subheader("ATM IV vs DTE")
        chart_df = atm_df.copy()
        chart_df["ATM_IV_pct"] = chart_df["ATM_IV"] * 100
        base = (
            alt.Chart(chart_df).mark_line(point=True).encode(
                x=alt.X("DTE:Q", title="Days to Expiration (DTE)"),
                y=alt.Y("ATM_IV_pct:Q", title="ATM IV (%)"),
                tooltip=[alt.Tooltip("DTE:Q", title="DTE"),
                         alt.Tooltip("ATM_IV_pct:Q", title="ATM IV (%)", format=".2f")]
            ).properties(width=700, height=420)
        )
        overlay = []
        title_suffix = ""
        if expected_move:
            try:
                overlay.append(
                    alt.Chart(pd.DataFrame({"y": [float(expected_move.strip('%'))]}))
                    .mark_rule(strokeDash=[5, 5]).encode(y="y:Q")
                )
                title_suffix = f"  —  Expected Move ≈ {expected_move}"
            except Exception:
                pass
        st.altair_chart(alt.layer(base, *overlay).properties(
            title=f"ATM Implied Volatility Term Structure{title_suffix}"
        ), use_container_width=True)

        st.download_button("Download ATM IV points (CSV)",
                           chart_df[["DTE", "ATM_IV"]].to_csv(index=False),
                           file_name="atm_iv_points.csv")

    st.info(
        "Notes:\n"
        "- IV30 is interpolated from the ATM IV term structure; RV30 is Yang–Zhang annualized realized volatility.\n"
        "- The slope check looks for a downward term structure from the nearest expiry to 45 days."
    )

# Footer
st.caption("Data source: Yahoo Finance (yfinance / query2 API / yahooquery). Data may be delayed.")
st.caption("Created by Shashank Agarwal")
