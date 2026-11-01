"""5단계: 강도 -> 충격 크기 변환용 분위수 표.

팩터별로 '20거래일 누적 변동'의 분포를 만들고, 상승 쪽과 하락 쪽 분위수를 따로 저장한다.
- up[q]   = 20일 변동의 q 분위수        (예: q=0.99 -> 상위 1% 상승폭)
- down[q] = 20일 변동의 (1-q) 분위수    (예: q=0.99 -> 하위 1% 하락폭, 음수)
결과: data/model/factor_quantiles_up.csv, factor_quantiles_down.csv
"""
from __future__ import annotations

import pandas as pd

from . import config

LEVELS = [0.50, 0.80, 0.95, 0.99, 0.995]


def horizon_changes(frets: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return frets.rolling(horizon).sum().dropna()


def quantile_tables(frets: pd.DataFrame, horizon: int, levels=LEVELS):
    h = horizon_changes(frets, horizon)
    up = pd.DataFrame({q: h.quantile(q) for q in levels})
    down = pd.DataFrame({q: h.quantile(1 - q) for q in levels})
    return up, down


def run() -> None:
    horizon = config.thresholds()["risk"]["horizon_days"]
    frets = pd.read_parquet(config.PROCESSED_DIR / "factor_returns.parquet")
    up, down = quantile_tables(frets, horizon)
    up.to_csv(config.MODEL_DIR / "factor_quantiles_up.csv")
    down.to_csv(config.MODEL_DIR / "factor_quantiles_down.csv")
    print(f"[quantiles] {horizon}일 변동 분위수 표 저장 (팩터 {len(up)}개)")


if __name__ == "__main__":
    run()
