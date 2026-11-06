"""주의도 지표: 사건 직전에 시장·언론이 이미 이 주제를 얼마나 다루고 있었나.

블랙록 BGRI처럼 '보도 빈도'로 주의도를 재되, 여기서는 GDELT 보도량 타임라인(전체 기사 중 비율)을 쓴다.
  최근 7일 평균 보도량이 그 전 기간(기준선) 일별 분포에서 몇 백분위인지 계산한다.
  높으면(기본 80 이상) 사건이 '이미 예상된' 면이 있어 시장 반응이 작을 수 있다 -> 강도 '의외성' 재검토 신호.
점수를 자동으로 바꾸지는 않는다(설정 attention.adjust_surprise). 네트워크가 막히면 측정을 생략한다.
"""
from __future__ import annotations

import numpy as np

INSTRUMENT_TERMS = {
    "tariff": "tariff", "tariff_pause": "tariff truce", "tech_export_control": '"export controls" chips',
    "entity_list": '"entity list"', "resource_export_control": '"rare earth" OR gallium OR germanium',
    "currency": "yuan currency", "financial_sanction": "sanctions", "chokepoint_disruption": "blockade OR strait",
    "industrial_subsidy": "subsidies tax credit", "investment_restriction": '"outbound investment"',
    "domestic_regulation": 'regulation OR antitrust OR "data center"', "election_shift": "election",
}
COUNTRY_TERMS = {"US": "United States", "CN": "China", "EU": "European Union", "JP": "Japan", "TW": "Taiwan",
                 "KR": "Korea", "RU": "Russia", "IN": "India", "IR": "Iran"}


def query_for(event: dict) -> str:
    inst = INSTRUMENT_TERMS.get(event.get("instrument"), "")
    who = [COUNTRY_TERMS[c] for c in [event.get("initiator"), *event.get("target", [])] if c in COUNTRY_TERMS][:2]
    q = f"({inst})" if " OR " in inst else inst
    return " ".join([q] + [f'"{w}"' if " " in w else w for w in who]).strip()


def measure(points: list[tuple[str, float]], recent_days: int = 7, min_baseline: int = 20) -> dict | None:
    """points = [(날짜, 보도량)] 시간순. 반환 {recent, baseline_median, ratio, percentile} 또는 None(표본 부족)."""
    vals = np.array([v for _, v in points], dtype=float)
    if len(vals) < recent_days + min_baseline:
        return None
    recent, base = vals[-recent_days:], vals[:-recent_days]
    r = float(recent.mean())
    rolling = np.convolve(base, np.ones(recent_days) / recent_days, mode="valid")     # 기준선도 7일 평균 분포로
    pct = float((rolling < r).mean() * 100) if len(rolling) else 50.0
    med = float(np.median(base))
    return {"recent": round(r, 5), "baseline_median": round(med, 5),
            "ratio": round(r / med, 2) if med > 0 else None, "percentile": round(pct, 1)}


def interpret(m: dict | None, high: float = 80.0) -> str | None:
    if not m:
        return None
    if m["percentile"] >= high:
        return (f"사건 직전 7일 보도량이 평소 분포의 {m['percentile']:.0f}백분위로 높았습니다. "
                "이미 예상된 면이 있어 시장 반응이 작을 수 있으니 강도의 '의외성'을 재검토하십시오.")
    if m["percentile"] <= 30:
        return "사건 직전 보도량이 평소보다 낮았습니다. 시장이 예상하지 못했을 가능성이 큽니다."
    return None
