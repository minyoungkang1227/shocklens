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

확장 (2026-11-02)
5) 문장 유사도: 기사와 사건 설명문의 단어 가중 유사도(textsim.py)를 구조 점수와 섞는다.
   점수 = (1 - text_weight) x 구조 + text_weight x 문장   (기사 텍스트가 없으면 구조 점수만)
6) 충격 크기 등급: 사건을 강도로 소(1~2)·중(3)·대(4~5)로 나누고 같은 등급끼리만 비교한다.
   작은 사건과 큰 사건을 섞으면 작은 사건의 잡음 같은 반응이 큰 사건의 충격을 희석하기 때문.
   같은 등급 후보가 min_same_class보다 적으면 인접 등급을 가중치 절반으로 보충한다.
7) 각 후보의 최종 가중치(weight) = 점수 x 등급 가중치. Shock Mapper가 가중 분포에 쓴다.
8) (2026-11-02) 강도 연속 점수가 있으면 등급 칸 대신 '가까울수록 높은' 부드러운 가중을 쓴다:
   weight = 점수 x closeness(강도 차이). 사건이 적을 때 칸이 비는 문제가 없고, 등급 경계에서 갑자기 끊기지 않는다.
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
    "chokepoint_disruption": "supply",
    "industrial_subsidy": "industrial",
    "investment_restriction": "control",
}
AGGRESSIVE = {"announce", "escalate", "enforce", "retaliate"}


def direction(ev: dict) -> str:
    return ev.get("direction", "escalate")


DEFAULT_CLASSES = {"small": [1, 2], "medium": [3], "large": [4, 5]}
CLASS_ORDER = ["small", "medium", "large"]


def size_class(severity: int, classes: dict | None = None) -> str:
    for name, levels in (classes or DEFAULT_CLASSES).items():
        if int(severity) in levels:
            return name
    return "medium"


def _adjacent(a: str, b: str) -> bool:
    return abs(CLASS_ORDER.index(a) - CLASS_ORDER.index(b)) == 1


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
         clean_only: bool = True, past_only: bool = False, text_sims: dict[str, float] | None = None,
         text_weight: float = 0.0, classes: dict | None = None) -> list[dict]:
    """후보 전체를 점수순으로 돌려준다. 각 항목: id, score, struct, text, parts, size_class, date."""
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
        struct = sum(weights[k] * parts[k] for k in weights)
        txt = (text_sims or {}).get(ev["id"])
        score = struct if txt is None else (1 - text_weight) * struct + text_weight * txt
        out.append({"id": ev["id"], "score": score, "struct": struct, "text": txt, "parts": parts,
                    "size_class": size_class(ev.get("severity", 3), classes), "date": str(ev["date"])})
    return sorted(out, key=lambda r: (-r["score"], r["date"]))


def select(target: dict, library: list[dict], available: list[str], cfg: dict,
           clean_only: bool = True, past_only: bool = False,
           text_sims: dict[str, float] | None = None) -> tuple[list[str], str, list[dict]]:
    """(유사 사건 id 목록, 경로 tier, 점수표) 를 돌려준다.

    cfg에 size_classes가 있으면 등급 분리를 적용하고, 점수표 각 항목에 weight(가중 분포용)를 채운다.
    """
    tw = cfg.get("text_weight", 0.0) if text_sims else 0.0
    classes = cfg.get("size_classes")
    ranked = rank(target, library, available, cfg["weights"], clean_only, past_only, text_sims, tw, classes)
    ok = [r for r in ranked if r["score"] >= cfg["relaxed_min"]]
    pts = {e["id"]: e.get("severity_points") for e in library}
    tp = target.get("severity_points")
    scale = cfg.get("closeness_scale", 20.0)   # thresholds.analog 또는 severity 어느 쪽에 있어도 됨
    if tp is not None and any(pts.get(r["id"]) is not None for r in ok):
        from .severity import closeness
        for r in ok:
            c = closeness(tp, pts[r["id"]], scale) if pts.get(r["id"]) is not None else 0.5
            r["closeness"] = round(c, 3)
            r["weight"] = r["score"] * c
        picked = sorted(ok, key=lambda r: -r["weight"])[: cfg.get("top_k", 3)]
    elif classes:
        my = size_class(target.get("severity", 3), classes)
        same = [r for r in ok if r["size_class"] == my]
        adj = [r for r in ok if _adjacent(r["size_class"], my)]
        for r in same:
            r["weight"] = r["score"]
        picked = same
        if len(same) < cfg.get("min_same_class", 2):
            for r in adj:
                r["weight"] = r["score"] * cfg.get("adjacent_class_weight", 0.5)
            picked = same + adj
        picked = sorted(picked, key=lambda r: -r["weight"])[: cfg.get("top_k", 3)]
    else:
        picked = ok[: cfg.get("top_k", 3)]
        for r in picked:
            r["weight"] = r["score"]
    if not picked:
        return [], "synthetic", ranked
    tier = "direct" if max(r["score"] for r in picked) >= cfg["direct_min"] else "relaxed"
    return [r["id"] for r in picked], tier, ranked
