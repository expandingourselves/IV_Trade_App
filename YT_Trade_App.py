"""
DISCLAIMER:

This software is provided solely for educational and research purposes.
It is not intended to provide investment advice, and no investment recommendations are made herein.
The developers are not financial advisors and accept no responsibility for any financial decisions or losses resulting from the use of this software.
Always consult a professional financial advisor before making any investment decisions.
"""

from typing import Optional, List, Tuple, Dict
import time, random
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

# ============================ Small helpers ============================

def badge(text: str, good: Optional[bool] = None) -> str:
    color = "#2563eb"
    if good is True: color = "#16a34a"
    elif good is False: color = "#b91c1c"
    return f"<span style='background:{color};color:white;padding:2px 8px;border-radius:12px;font-size:0.85rem'>{text}</span>"

def filter_dates(dates: List[str]) -> List[str]:
    today = datetime.now(timezone.utc).date()
    cutoff_date = today + timedelta(days=45)
    sorted_dates = sorted(datetime.strptime(d, "%Y-%m-%d").date() for d in dates)

    out: List[str] = []
    for i, dt in enumerate(sorted_dates):
        if dt >= cutoff_date:
            out = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]
            break
    if out:
        if out[0] == today.strftime("%Y-%m-%d"):
            return out[1:]
        return out
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
    days = np.array(days); ivs = np.array(ivs)
    order = days.argsort()
    spline = interp1d(days[order], ivs[order], kind='linear', fill_value="extrapolate")
    def term_spline(dte: float):
        dte = float(dte)
        if dte <= days[order][0]:  return float(ivs[order][0])
        if dte >= days[order][-1]: return float(ivs[order][-1])
        return float(spline(dte))
    return term_spline

def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": "https://finance.yahoo.com/",
    })
    return s

@st.cache_resource(ttl=900)
def get_http_session() -> requests.Session:
    return _new_session()

def _epoch_from_date_str(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc, hour=0, minute=0, second=0, microsecond=0)
    return int(dt.timestamp())

# ============================ Robust data getters (price & expirations split) ============================

@st.cache_data(show_spinner=False, ttl=300)
def get_price_any(ticker: str) -> Tuple[Optional[float], Optional[str]]:
    ticker = ticker.upper().replace(".", "-").strip()
    sess = get_http_session()

    # yfinance
    for i in range(3):
        try:
            t = yf.Ticker(ticker, session=sess if i == 0 else _new_session())
            px = getattr(t.fast_info, "last_price", None)
            if px is None:
                fi = t.fast_info
                try: px = fi["last_price"]
                except Exception: px = None
            if px and px > 0:
                return float(px), "yf.fast_info"
            todays = t.history(period="1d", interval="1d", auto_adjust=False, prepost=False)
            if not todays.empty:
                return float(todays["Close"].iloc[-1]), "yf.history"
        except (JSONDecodeError, ReqConnectionError, Timeout, RequestException):
            time.sleep(0.4 + 0.4 * i + random.random()*0.3)

    # direct quote
    try:
        r = sess.get(
            "https://query2.finance.yahoo.com/v6/finance/quote",
            params={"symbols": ticker, "lang":"en-US", "region":"US"},
            timeout=8,
        )
        if r.status_code == 200 and r.headers.get("content-type","").startswith("application/json"):
            data = r.json()
            res = (data.get("quoteResponse", {}) or {}).get("result", [])
            if res:
                px = res[0].get("regularMarketPrice") or res[0].get("postMarketPrice") or res[0].get("preMarketPrice")
                if px:
                    return float(px), "direct.quote"
    except Exception:
        pass

    # yahooquery
    try:
        yq = YQTicker(ticker, validate=False, formatted=False)
        pm = yq.price or {}
        p = (pm.get(ticker) or {}).get("regularMarketPrice")
        if p:
            return float(p), "yq.price"
    except Exception:
        pass

    return None, None

@st.cache_data(show_spinner=False, ttl=300)
def get_expirations_any(ticker: str) -> Tuple[List[str], Dict[str,int], Optional[str]]:
    """
    Returns (expirations_yyyy_mm_dd, date_to_unix_map, source)
    """
    ticker = ticker.upper().replace(".", "-").strip()
    sess = get_http_session()

    # yfinance
    for i in range(3):
        try:
            t = yf.Ticker(ticker, session=sess if i == 0 else _new_session())
            opts = list(t.options or [])
            if opts:
                return opts, {}, "yf.options"
        except (JSONDecodeError, ReqConnectionError, Timeout, RequestException):
            time.sleep(0.4 + 0.4 * i + random.random()*0.3)

    # direct options root (gives expirationDates as unix)
    try:
        r = sess.get(f"https://query2.finance.yahoo.com/v7/finance/options/{ticker}", timeout=8)
        if r.status_code == 200 and r.headers.get("content-type","").startswith("application/json"):
            data = r.json()
            result = data["optionChain"]["result"][0]
            exp_unix = result.get("expirationDates", []) or []
            if exp_unix:
                dates = [datetime.utcfromtimestamp(u).strftime("%Y-%m-%d") for u in exp_unix]
                exp_map = {datetime.utcfromtimestamp(u).strftime("%Y-%m-%d"): int(u) for u in exp_unix}
                return dates, exp_map, "direct.root"
    except Exception:
        pass

    # yahooquery: attribute or derive from option_chain
    try:
        yq = YQTicker(ticker, validate=False, formatted=False)
        exp_unix = []
        try:
            exp_attr = getattr(yq, "option_expiration_dates", None)
            if isinstance(exp_attr, dict):
                exp_unix = exp_attr.get(ticker, []) or []
        except Exception:
            exp_unix = []

        if not exp_unix:
            # derive from option_chain
            df_all = yq.option_chain()
            if isinstance(df_all, pd.DataFrame) and not df_all.empty and "expirationDate" in df_all.columns:
                exp_unix = sorted({int(x) for x in df_all["expirationDate"].dropna().astype(int).tolist()})

        if exp_unix:
            dates = [datetime.utcfromtimestamp(u).strftime("%Y-%m-%d") for u in exp_unix]
            exp_map = {datetime.utcfromtimestamp(u).strftime("%Y-%m-%d"): int(u) for u in exp_unix}
            # cache client for later chains
            st.session_state.setdefault("_yq_clients", {})[ticker] = yq
            return dates, exp_map, "yq"
    except Exception:
        pass

    return [], {}, None

# ============================ Chain fetchers ============================

def fetch_chain_yf(ticker: str, exp_date: str, sess: requests.Session) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t = yf.Ticker(ticker, session=sess)
    ch = t.option_chain(exp_date)
    return getattr(ch, "calls", pd.DataFrame()), getattr(ch, "puts", pd.DataFrame())

def fetch_chain_direct(ticker: str, exp_unix: int, sess: requests.Session) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r = sess.get(f"https://query2.finance.yahoo.com/v7/finance/options/{ticker}", params={"date": exp_unix}, timeout=8)
    if r.status_code != 200 or not r.headers.get("content-type","").startswith("application/json"):
        raise ValueError(f"bad status {r.status_code}")
    data = r.json()
    opt = data["optionChain"]["result"][0]["options"][0]
    return pd.DataFrame(opt.get("calls", [])), pd.DataFrame(opt.get("puts", []))

def fetch_chain_yq(ticker: str, exp_unix: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    yq = st.session_state.get("_yq_clients", {}).get(ticker) or YQTicker(ticker, validate=False, formatted=False)
    df = None
    try:
        df = yq.option_chain(date=exp_unix)
    except Exception:
        # derive by filtering all
        try:
            df_all = yq.option_chain()
            if isinstance(df_all, pd.DataFrame) and not df_all.empty and "expirationDate" in df_all.columns:
                df = df_all[df_all["expirationDate"] == int(exp_unix)].copy()
        except Exception:
            df = None
    if not isinstance(df, pd.DataFrame) or df.empty or "optionType" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    return df[df["optionType"] == "calls"].copy(), df[df["optionType"] == "puts"].copy()

# ============================ Main computation ============================

def compute_recommendation(ticker: str):
    try:
        t = ticker.strip().upper().replace(".", "-")
        if not t:
            return "No stock symbol provided."

        # Get price and expirations independently
        price, price_src = get_price_any(t)
        exps, exp_map_unix, exp_src = get_expirations_any(t)

        if not exps:
            return "Yahoo data request failed. (No expirations found; providers rate-limited / unavailable)"
        if not price:
            return "Yahoo data request failed. (No price found; providers rate-limited / unavailable)"

        # pick expiries (nearest .. include >=45d)
        try:
            exp_dates = filter_dates(exps)
        except Exception:
            return "Error: Not enough option data."

        sess = get_http_session()

        atm_iv: Dict[str, float] = {}
        straddle = None
        first_done = False

        for exp_date in exp_dates:
            # compute unix for direct/yq even if not provided
            exp_unix = exp_map_unix.get(exp_date, _epoch_from_date_str(exp_date))

            calls = puts = pd.DataFrame()

            # Try yfinance → direct → yq for the chain
            for attempt in range(3):
                try:
                    calls, puts = fetch_chain_yf(t, exp_date, sess)
                    if not (calls.empty or puts.empty): break
                except Exception:
                    time.sleep(0.2 + 0.2*attempt + random.random()*0.2)

            if calls.empty or puts.empty:
                for attempt in range(3):
                    try:
                        calls, puts = fetch_chain_direct(t, exp_unix, sess)
                        if not (calls.empty or puts.empty): break
                    except Exception:
                        time.sleep(0.3 + 0.2*attempt + random.random()*0.2)

            if calls.empty or puts.empty:
                calls, puts = fetch_chain_yq(t, exp_unix)

            if calls.empty or puts.empty or "strike" not in calls.columns or "strike" not in puts.columns:
                continue

            def nearest_with_iv(df: pd.DataFrame):
                if "impliedVolatility" not in df.columns:
                    return None
                df2 = df[pd.notna(df["impliedVolatility"])].copy()
                if df2.empty:
                    return None
                df2["dist"] = (df2["strike"].astype(float) - float(price)).abs()
                return df2.sort_values("dist").iloc[0]

            call_row = nearest_with_iv(calls); put_row = nearest_with_iv(puts)
            if call_row is None or put_row is None:
                continue

            atm_iv[exp_date] = (float(call_row["impliedVolatility"]) + float(put_row["impliedVolatility"])) / 2.0

            if not first_done:
                def mid(row):
                    b = row.get("bid"); a = row.get("ask")
                    return (float(b) + float(a)) / 2.0 if pd.notna(b) and pd.notna(a) else None
                cm, pm = mid(call_row), mid(put_row)
                if cm is not None and pm is not None:
                    straddle = cm + pm
                first_done = True

        if not atm_iv:
            return "Error: Could not determine ATM IV for any expiration dates."

        # Build term structure
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

        # History for YZ (use 6mo for safety, 60 bars window)
        px = yf.Ticker(t, session=sess).history(period='6mo', interval='1d', auto_adjust=True, prepost=False)
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

        expected_move = f"{round((straddle or 0) / price * 100, 2)}%" if straddle else None

        return {
            'avg_volume': bool(avg_volume >= 1_500_000) if pd.notna(avg_volume) else False,
            'iv30_rv30': float(iv30_rv30) >= 1.25,
            'ts_slope_0_45': float(ts_slope_0_45) <= -0.00406,
            'expected_move': expected_move,
            'underlying_price': price,
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

    for label, ok in [
        ("avg_volume ≥ 1.5M (30-day avg)", avg_volume_bool),
        ("IV30 / RV30 ≥ 1.25", iv30_rv30_bool),
        ("Term slope (d0→45d) ≤ −0.00406", ts_slope_bool),
    ]:
        st.markdown(f"- {label}: {badge('PASS' if ok else 'FAIL', ok)}", unsafe_allow_html=True)

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
        if expected_move:
            try:
                overlay.append(
                    alt.Chart(pd.DataFrame({"y": [float(expected_move.strip('%'))]}))
                    .mark_rule(strokeDash=[5, 5]).encode(y="y:Q")
                )
            except Exception:
                pass
        st.altair_chart(alt.layer(base, *overlay).properties(
            title="ATM Implied Volatility Term Structure"
        ), use_container_width=True)
        st.download_button("Download ATM IV points (CSV)",
                           chart_df[["DTE", "ATM_IV"]].to_csv(index=False),
                           file_name="atm_iv_points.csv")

st.caption("Data source: Yahoo Finance (yfinance / query2 API / yahooquery). Data may be delayed.")
st.caption("Created by Shashank Agarwal")
