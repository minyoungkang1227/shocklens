"""Exposure Analyst: 사건이 어떤 바스켓을 '직접' 겨냥하는지 판단 (LLM 없음).

강화 포인트
- 등급(high/mid/none)과 함께 근거 문장을 남긴다: 어떤 국가·산업이 매핑표의 어느 줄과 맞았는지.
- 산업을 겨냥하는 수단(수출통제·엔티티 리스트)은 국가 일치만으로 high를 주지 않는다(mid로 제한).
- 국부펀드 자산이 직접 표적(sovereign_asset_risk)이면 모든 바스켓에 경고 사유를 붙인다.
"""
from __future__ import annotations

from functools import lru_cache

from engine.exposure import RANK, SECTOR_TARGETED, exposure_for

from .context import Context


@lru_cache(maxsize=1)
def supply_links() -> tuple:
    try:
        from pipeline import config
        return tuple((config.load_yaml("supply_chain.yaml") or {}).get("links", []))
    except FileNotFoundError:
        return ()


def apply_supply_chain(event: dict, grades: dict, reasons: dict, labels: dict, links=None) -> list[str]:
    """공급망 의존표와 맞으면 노출을 끌어올린다. 반환: 적용된 연결 이름 목록"""
    hit = []
    sectors, countries = set(event.get("sectors", [])), {event.get("initiator"), *event.get("target", [])}
    for ln in (links if links is not None else supply_links()):
        if ln.get("instruments") and event.get("instrument") not in ln["instruments"]:
            continue
        if not (sectors & set(ln.get("sectors_any") or []) or countries & set(ln.get("countries_any") or [])):
            continue
        hit.append(ln["label"])
        for b, lvl in ln.get("baskets", {}).items():
            if b not in grades:
                continue
            if RANK[lvl] > RANK[grades[b]]:
                grades[b] = lvl
            extra = f"공급망 {ln['label']}({lvl}: {ln.get('why', '')})"
            reasons[b] = (reasons[b] + ", " + extra) if b in reasons else f"{labels.get(b, b)}: {extra}"
    return hit


def analyze(event: dict, ctx: Context) -> dict:
    grades = exposure_for(event, ctx.exposure_map)
    countries = {event["initiator"], *event.get("target", [])}
    sectors = set(event.get("sectors", []))
    capped = event.get("instrument") in SECTOR_TARGETED
    reasons = {}
    for b, lvl in grades.items():
        if lvl == "none":
            continue
        m = ctx.exposure_map.get(b, {})
        why = [f"산업 {s}({g})" for s, g in m.get("sectors", {}).items() if s in sectors and g == lvl]
        why += [f"국가 {c}({g}{'→mid 제한' if capped and RANK[g] > RANK['mid'] else ''})"
                for c, g in m.get("countries", {}).items() if c in countries]
        reasons[b] = f"{ctx.labels.get(b, b)}: " + ", ".join(why)
    chains = apply_supply_chain(event, grades, reasons, ctx.labels)
    if chains:
        reasons["_supply_chain"] = "공급망 의존 경로: " + ", ".join(chains)
    if event.get("sovereign_asset_risk"):
        reasons["_sovereign"] = "국부펀드·중앙은행 해외자산이 직접 제재 표적: 보유 자산 동결·회수 위험 별도 점검 필요"
    return {"exposure": grades, "exposure_reasons": reasons}
