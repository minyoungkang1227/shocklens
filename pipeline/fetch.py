"""1단계: 원천 데이터 수집.

- ETF 12개 가격: yfinance (수정주가)
- 팩터 6개: FRED (API 키 없이 CSV로 받음)
결과: data/raw/prices.parquet, data/raw/fred.parquet
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from . import config

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def etf_tickers() -> list[str]:
    tickers = [b["ticker"] for b in config.portfolio()["baskets"].values()]
    # 팩터 계산에 필요한 ETF도 포함 (기술전쟁 = SOXX - SPY)
    for f in config.factors()["factors"].values():
        if f["source"] == "etf":
            tickers += [f["long"], f["short"]]
    return sorted(set(tickers))


def fetch_prices(start: str) -> pd.DataFrame:
    import yfinance as yf

    tickers = etf_tickers()
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    # yfinance 버전에 따라 컬럼이 MultiIndex(가격종류, 티커)로 온다
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    close = close.dropna(how="all").sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    missing = [t for t in tickers if t not in close.columns]
    if missing:
        raise RuntimeError(f"가격을 받지 못한 티커: {missing}")
    return close[tickers]


def parse_fred_csv(text: str, series: str) -> pd.Series:
    """FRED CSV 한 개를 날짜 인덱스 Series로 변환. 결측은 '.'으로 표시된다."""
    df = pd.read_csv(io.StringIO(text))
    date_col = df.columns[0]  # 'observation_date' 또는 'DATE' (시기에 따라 다름)
    s = pd.to_numeric(df[series], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    s.name = series
    return s.dropna()


def fetch_fred(series_list: list[str], start: str) -> pd.DataFrame:
    out = []
    for sid in series_list:
        r = requests.get(FRED_CSV.format(series=sid), timeout=30)
        r.raise_for_status()
        out.append(parse_fred_csv(r.text, sid))
    df = pd.concat(out, axis=1).sort_index()
    return df.loc[df.index >= pd.Timestamp(start)]


def run() -> None:
    config.ensure_dirs()
    start = config.portfolio()["start_date"]
    fred_ids = [f["series"] for f in config.factors()["factors"].values() if f["source"] == "fred"]

    prices = fetch_prices(start)
    prices.to_parquet(config.RAW_DIR / "prices.parquet")
    print(f"[fetch] ETF {prices.shape[1]}개, {prices.index.min().date()} ~ {prices.index.max().date()}, {len(prices)}일")

    fred = fetch_fred(fred_ids, start)
    fred.to_parquet(config.RAW_DIR / "fred.parquet")
    print(f"[fetch] FRED {fred.shape[1]}개 시리즈, {len(fred)}일")


if __name__ == "__main__":
    run()
