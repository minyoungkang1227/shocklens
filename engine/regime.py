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


def series_state(s: pd.Series, quantile: float, lookback: int, asof: str | None, ref_date: str | None,
                 max_stale_days: int) -> dict | None:
    """보조 지수 하나의 최근 평균·백분위·기준. 데이터가 없거나 기준일(VIX 날짜)보다 너무 오래됐으면 None 대신 stale 표시."""
    v = s.dropna()
    if asof is not None:
        v = v.loc[: pd.Timestamp(asof)]
    if len(v) < lookback + 20:
        return None
    m = market_regime(v, quantile, lookback)
    stale = bool(ref_date) and (pd.Timestamp(ref_date) - v.index[-1]).days > max_stale_days
    return {"recent": m["vix_recent"], "threshold": m["vix_threshold"], "percentile": m["vix_percentile"],
            "asof": m["asof"], "high": m["regime"] == "crisis", "stale": stale}


def decide(event: dict, vix: pd.Series, cfg: dict, severity_min: int, asof: str | None = None,
           stress: pd.DataFrame | None = None) -> dict:
    """thresholds.regime.mode 에 따라 국면을 정한다. 반환: {regime, reason, ...지표}
    stress: OFR_FSI·AI_GPR 열을 가진 보조 지수 표(pipeline.indicators). 없으면 VIX만으로 판단(이전과 같음)."""
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
    # (2026-11-13) 보조 지수: OFR 금융스트레스지수는 위기 판단에 OR로 더하고, AI-GPR은 기본은 보고만 한다
    scfg = cfg.get("stress") or {}
    if stress is not None and m.get("vix_recent") is not None:
        lb, stale_days = int(cfg.get("lookback_days", 5)), int(scfg.get("max_stale_days", 10))
        triggers = ["VIX"] if m["regime"] == "crisis" else []
        for col, key, label, use, q in (
                ("OFR_FSI", "ofr", "OFR 금융스트레스", scfg.get("ofr", True), scfg.get("ofr_quantile", 0.8)),
                ("AI_GPR", "gpr", "AI 지정학 리스크", scfg.get("gpr_trigger", False), scfg.get("gpr_quantile", 0.9))):
            if col not in stress.columns:
                continue
            st = series_state(stress[col], float(q), lb, asof, m.get("asof"), stale_days)
            if st is None:
                continue
            m[key] = st
            note = f"{label} {st['recent']:.2f} (백분위 {st['percentile']:.0%}, {st['asof']} 기준"
            note += ", 오래된 값이라 판단에서 제외)" if st["stale"] else ")"
            m["reason"] += f"; {note}"
            if use and st["high"] and not st["stale"]:
                triggers.append(label)
        if triggers and m["regime"] != "crisis":
            m["regime"] = "crisis"
        if triggers:
            m["triggers"] = triggers
            m["reason"] += f" -> 위기 판단 근거: {', '.join(triggers)}"
    return m
