"""국면(regime) 판단: 지금 시장이 평상시인가 위기인가.

이전 방식(severity): 사건 강도가 4 이상이면 위기 베타·위기 표본. -> 같은 뉴스라도 시장이 조용할 때와 불안할 때가 구분되지 않는다.
현재 방식(market):   최근 n거래일 VIX 평균이 전체 기간의 상위 q 분위수 안이면 위기. 사건 강도는 충격 크기에만 쓴다.
"""
from __future__ import annotations

import pandas as pd


def market_regime(vix: pd.Series, quantile: float = 0.8, lookback: int = 5, asof: str | None = None) -> dict:
    v = vix.dropna()
    if asof is not None:
        v = v.loc[: pd.Timestamp(asof)]
    if len(v) == 0:
        return {"regime": "normal", "vix_recent": None, "vix_threshold": None, "vix_percentile": None}
    recent = float(v.iloc[-lookback:].mean())
    thr = float(v.quantile(quantile))
    pct = float((v < recent).mean())
    return {"regime": "crisis" if recent > thr else "normal", "vix_recent": recent, "vix_threshold": thr,
            "vix_percentile": pct, "asof": str(v.index[-1].date())}


def decide(event: dict, vix: pd.Series, cfg: dict, severity_min: int, asof: str | None = None) -> dict:
    """thresholds.regime.mode 에 따라 국면을 정한다. 반환: {regime, reason, ...지표}"""
    mode = cfg.get("mode", "market")
    if mode == "severity":
        r = "crisis" if event.get("severity", 3) >= severity_min else "normal"
        return {"regime": r, "mode": mode, "reason": f"사건 강도 {event.get('severity')} (기준 {severity_min})"}
    m = market_regime(vix, cfg.get("crisis_vix_quantile", 0.8), cfg.get("lookback_days", 5), asof)
    m["mode"] = mode
    if m["vix_recent"] is None:
        m["reason"] = "VIX 데이터 없음 -> 평상시"
    else:
        m["reason"] = (f"최근 {cfg.get('lookback_days', 5)}거래일 VIX 평균 {m['vix_recent']:.1f} "
                       f"(전체 기간 백분위 {m['vix_percentile']:.0%}, 위기 기준 {m['vix_threshold']:.1f})")
    return m
