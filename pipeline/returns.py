"""2단계: 수익률과 팩터 변동 계산.

- 바스켓 수익률: 일별 로그수익률
- 팩터: kind에 따라 로그수익률(price) / 일별 차분(level) / ETF 수익률 차(spread_ret)
- 날짜 기준: ETF 거래일. FRED는 휴일이 달라 최대 5영업일까지만 직전 값으로 채운다.
결과: data/processed/returns.parquet, factor_returns.parquet, factor_levels.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

FFILL_LIMIT = 5


def basket_returns(prices: pd.DataFrame, baskets: dict) -> pd.DataFrame:
    tick_to_name = {b["ticker"]: name for name, b in baskets.items()}
    px = prices[list(tick_to_name)].rename(columns=tick_to_name)
    return np.log(px).diff().iloc[1:]


def factor_levels(fred: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """FRED 수준값을 ETF 거래일에 맞춘다."""
    return fred.reindex(fred.index.union(index)).ffill(limit=FFILL_LIMIT).reindex(index)


def factor_returns(prices: pd.DataFrame, levels: pd.DataFrame, factor_cfg: dict) -> pd.DataFrame:
    out = {}
    for name, f in factor_cfg.items():
        if f["kind"] == "price":
            # WTI는 2020-04-20에 음수(-37달러)를 기록했다. 로그를 취할 수 없으므로 그 날은 결측 처리.
            s = levels[f["series"]].where(lambda v: v > 0)
            out[name] = np.log(s).diff()
        elif f["kind"] == "level":
            out[name] = levels[f["series"]].diff()
        elif f["kind"] == "spread_ret":
            lr = np.log(prices[[f["long"], f["short"]]]).diff()
            out[name] = lr[f["long"]] - lr[f["short"]]
        else:
            raise ValueError(f"알 수 없는 팩터 종류: {f['kind']}")
    return pd.DataFrame(out).iloc[1:]


def build(prices: pd.DataFrame, fred: pd.DataFrame, baskets: dict, factor_cfg: dict):
    rets = basket_returns(prices, baskets)
    levels = factor_levels(fred, prices.index)
    frets = factor_returns(prices, levels, factor_cfg)
    # 바스켓과 팩터가 모두 있는 날만 남긴다
    common = rets.dropna().index.intersection(frets.dropna().index)
    return rets.loc[common], frets.loc[common], levels.loc[common]


def run() -> None:
    prices = pd.read_parquet(config.RAW_DIR / "prices.parquet")
    fred = pd.read_parquet(config.RAW_DIR / "fred.parquet")
    rets, frets, levels = build(prices, fred, config.portfolio()["baskets"], config.factors()["factors"])
    if len(rets) < 0.8 * len(prices):
        print(f"[returns] 경고: 공통 거래일 {len(rets)}일 / ETF {len(prices)}일 -> 데이터 구간이 크게 줄었습니다.")
        print("[returns] FRED 시리즈별 첫 데이터 날짜 (하나만 유독 늦으면 그 시리즈가 원인):")
        print(fred.apply(lambda s: s.first_valid_index()).to_string())
    rets.to_parquet(config.PROCESSED_DIR / "returns.parquet")
    frets.to_parquet(config.PROCESSED_DIR / "factor_returns.parquet")
    levels.to_parquet(config.PROCESSED_DIR / "factor_levels.parquet")
    print(f"[returns] 공통 거래일 {len(rets)}일, 바스켓 {rets.shape[1]}개, 팩터 {frets.shape[1]}개")


if __name__ == "__main__":
    run()
