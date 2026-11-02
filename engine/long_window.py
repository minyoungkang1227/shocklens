"""법령·산업정책의 장기 업종 효과: 발표 후 20/60/120거래일 바스켓 초과수익과 그 백분위. LLM 없음.

단기 스트레스테스트(3거래일 반응, 20거래일 ES)와 질문이 다르다. 여기서는 "이 법 이후 몇 달 동안
어느 업종이 시장 대비 이기거나 졌나"를 본다.
  초과수익 = 바스켓 누적 로그수익 - 기준(기본 미국 주식) 누적 로그수익
  백분위   = 같은 길이의 모든 기간 초과수익 분포에서 이번 값의 위치 (무작위 창 대비, 50이면 평범)
주의: 창이 길수록 다른 사건이 많이 섞인다. 인과가 아니라 '그 이후 무슨 일이 있었나'의 기록이다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def start_index(idx: pd.DatetimeIndex, date: str) -> int | None:
    pos = idx.searchsorted(pd.Timestamp(date))
    return int(pos) if pos < len(idx) else None


def excess_table(rets: pd.DataFrame, date: str, horizons=(20, 60, 120), benchmark: str = "us_equity") -> pd.DataFrame:
    """행 = 바스켓, 열 = (h, 'excess'|'pct'). 데이터가 모자란 기간은 NaN."""
    rets = rets.dropna(how="all")
    i0 = start_index(rets.index, date)
    cols = pd.MultiIndex.from_product([horizons, ["excess", "pct"]])
    out = pd.DataFrame(index=rets.columns, columns=cols, dtype=float)
    if i0 is None or benchmark not in rets:
        return out
    ex_all = rets.sub(rets[benchmark], axis=0)
    for h in horizons:
        if i0 + h > len(rets):
            continue
        roll = ex_all.rolling(h).sum()                     # 끝점 기준 h일 누적 초과수익 (모든 창)
        obs = ex_all.iloc[i0:i0 + h].sum()
        for b in rets.columns:
            dist = roll[b].dropna().to_numpy()
            out.loc[b, (h, "excess")] = float(obs[b])
            out.loc[b, (h, "pct")] = float((dist < obs[b]).mean() * 100) if len(dist) and b != benchmark else np.nan
    return out
