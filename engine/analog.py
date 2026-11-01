"""유사 사건 고르기 (Analog Finder가 쓰는 순수 계산, LLM 없음).

새 사건과 라이브러리 사건을 '구조'로 비교해 점수를 매기고, 점수가 높은 사건을 유사 사례로 쓴다.

규칙
1) 방향이 같은 사건만 비교한다. 확대(escalate) 사건에 완화(ease) 사건을 섞으면 충격 부호가 뒤집힌다.
2) clean 사건만 쓴다(기본값). 반응 창에 다른 큰 충격이 겹친 사건은 크기를 왜곡한다.
3) 구조 점수 = 수단 0.4 + 산업 0.3 + 국가 0.2 + 단계 0.1 (가중치는 thresholds.yaml)
   - 수단: 같으면 1, 같은 계열(예: 수출통제끼리)이면 0.5, 다르면 0
   - 산업, 국가: 두 집합이 겹치는 비율 (자카드 = 교집합 / 합집합)
   - 단계: 같으면 1, 둘 다 공세 단계(escalate/enforce/retaliate)면 0.5, 아니면 0
4) 점수 상위 top_k건 중 relaxed_min 이상만 쓴다.
   최고 점수 >= direct_min -> direct, >= relaxed_min -> relaxed, 아니면 synthetic(유사 사건 없이 강도로 산정).

임베딩(설명문 의미 유사도)은 아직 붙이지 않았다. 지금 점수는 구조 점수만이다.
"""
from __future__ import annotations

FAMILY = {
    "tariff": "trade",
    "tariff_pause": "trade",
    "tech_export_control": "control",
    "entity_list": "control",
    "resource_export_control": "control",
    "currency": "currency",
    "financial_sanction": "finance",
}
AGGRESSIVE = {"announce", "escalate", "enforce", "retaliate"}


def direction(ev: dict) -> str:
    return ev.get("direction", "escalate")


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _countries(ev: dict) -> set:
    return {ev["initiator"], *ev.get("target", [])}


def score_parts(a: dict, b: dict) -> dict[str, float]:
    ia, ib = a.get("instrument"), b.get("instrument")
    if ia == ib:
        inst = 1.0
    elif FAMILY.get(ia) is not None and FAMILY.get(ia) == FAMILY.get(ib):
        inst = 0.5
    else:
        inst = 0.0
    sa, sb = a.get("stage"), b.get("stage")
    if sa == sb:
        stage = 1.0
    elif sa in AGGRESSIVE and sb in AGGRESSIVE:
        stage = 0.5
    else:
        stage = 0.0
    return {
        "instrument": inst,
        "sector": _jaccard(set(a.get("sectors", [])), set(b.get("sectors", []))),
        "countries": _jaccard(_countries(a), _countries(b)),
        "stage": stage,
    }


def struct_score(a: dict, b: dict, weights: dict[str, float]) -> float:
    p = score_parts(a, b)
    return sum(weights[k] * p[k] for k in weights)


def rank(target: dict, library: list[dict], available: list[str], weights: dict[str, float],
         clean_only: bool = True, past_only: bool = False) -> list[dict]:
    """후보 전체를 점수순으로 돌려준다. 각 항목: id, score, parts, clean, same_direction."""
    out = []
    for ev in library:
        if ev["id"] == target.get("id") or ev["id"] not in available:
            continue
        if past_only and str(ev["date"]) >= str(target["date"]):
            continue
        if direction(ev) != direction(target):
            continue
        if clean_only and ev.get("clean") is not True:
            continue
        parts = score_parts(target, ev)
        out.append({"id": ev["id"], "score": sum(weights[k] * parts[k] for k in weights),
                    "parts": parts, "date": str(ev["date"])})
    return sorted(out, key=lambda r: (-r["score"], r["date"]))


def select(target: dict, library: list[dict], available: list[str], cfg: dict,
           clean_only: bool = True, past_only: bool = False) -> tuple[list[str], str, list[dict]]:
    """(유사 사건 id 목록, 경로 tier, 점수표) 를 돌려준다."""
    ranked = rank(target, library, available, cfg["weights"], clean_only, past_only)
    picked = [r for r in ranked if r["score"] >= cfg["relaxed_min"]][: cfg.get("top_k", 3)]
    if not picked:
        return [], "synthetic", ranked
    tier = "direct" if picked[0]["score"] >= cfg["direct_min"] else "relaxed"
    return [r["id"] for r in picked], tier, ranked
