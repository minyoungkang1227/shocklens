"""직접 노출 판단 (Exposure Analyst가 쓰는 순수 계산, LLM 없음).

사건의 국가·산업을 노출 매핑표(config/exposure_map.yaml)와 대조해 바스켓별 등급(none/mid/high)을 정한다.

규칙: 수출통제처럼 특정 산업을 겨냥하는 수단은 '국가' 일치만으로는 최대 mid로 제한한다.
      예) 중국이 미국 대상 소재 수출통제 -> 미국 주식 전체가 아니라 반도체 바스켓이 high.
      관세·금융제재처럼 국가 전체를 겨냥하는 수단은 국가 일치도 high까지 허용한다.
"""
from __future__ import annotations

RANK = {"none": 0, "mid": 1, "high": 2}
SECTOR_TARGETED = {"resource_export_control", "tech_export_control", "entity_list", "industrial_subsidy",
                   "investment_restriction", "domestic_regulation"}
# election_shift는 국가 전체 정책 기대가 바뀌는 사건이라 국가 일치도 high까지 허용한다.


DOMESTIC = {"domestic_regulation", "election_shift"}
DOWN = {"high": "mid", "mid": "none", "none": "none"}


def home_country(m: dict) -> str | None:
    """국가 하나만 high인 바스켓(미국 주식·중국 주식 등)의 그 국가. 섹터·자산군 바스켓은 None."""
    highs = [c for c, lvl in m.get("countries", {}).items() if lvl == "high"]
    return highs[0] if len(highs) == 1 else None


def exposure_for(event: dict, exposure_map: dict) -> dict[str, str]:
    countries = {event["initiator"], *event.get("target", [])}
    sectors = set(event.get("sectors", []))
    country_cap = "mid" if event.get("instrument") in SECTOR_TARGETED else "high"
    domestic = event.get("instrument") in DOMESTIC
    out = {}
    for basket, m in exposure_map.items():
        best = "none"
        for c, lvl in m.get("countries", {}).items():
            if c in countries:
                lvl = lvl if RANK[lvl] <= RANK[country_cap] else country_cap
                if RANK[lvl] > RANK[best]:
                    best = lvl
        home = home_country(m)
        for s, lvl in m.get("sectors", {}).items():
            if s not in sectors:
                continue
            if domestic and home and home not in countries:
                lvl = DOWN[lvl]     # 국내 규제는 그 나라 산업이 표적: 다른 나라 주식 바스켓의 같은 산업은 한 단계 낮춤
            if RANK[lvl] > RANK[best]:
                best = lvl
        out[basket] = best
    return out
