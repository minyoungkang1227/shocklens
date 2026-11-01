"""Analog Finder: 사건과 구조가 비슷한 과거 사건을 고른다 (LLM 없음, engine.analog 사용).

강화 포인트
- 점수만 주지 않고 '왜 골랐는지' 한 줄 근거를 붙인다 (수단·산업·국가·단계 중 무엇이 겹쳤나).
- 같은 방향(확대/완화) + clean 사건만. 빠진 후보(clean: false)도 따로 돌려줘 브리핑에서 투명하게 밝힌다.
- 새 사건 날짜가 있으면 그보다 미래의 사건은 쓰지 않는다 (사후 정보 누수 방지).
"""
from __future__ import annotations

from engine import analog

from .context import Context

PART_KO = {"instrument": "수단", "sector": "산업", "countries": "국가", "stage": "단계"}


def _reason(parts: dict[str, float], ev_lib: dict) -> str:
    hit = [f"{PART_KO[k]} {'일치' if v >= 1 else '일부 일치'}" for k, v in parts.items() if v > 0]
    return f"{ev_lib.get('date', '')} {ev_lib.get('title', '')} — " + (", ".join(hit) if hit else "겹치는 요소 없음")


def find(event: dict, ctx: Context, use_unclean: bool = False) -> dict:
    cfg = ctx.thresholds["analog"]
    available = list(ctx.f_event.index)
    past_only = bool(event.get("date"))
    target = {**event, "date": event.get("date") or "9999-12-31"}
    ids, tier, ranked = analog.select(target, ctx.events, available, cfg,
                                      clean_only=not use_unclean, past_only=past_only)
    picks = []
    for r in ranked:
        if r["id"] in ids:
            picks.append({"id": r["id"], "score": round(r["score"], 3), "parts": r["parts"],
                          "reason": _reason(r["parts"], ctx.event_by_id(r["id"]) or {})})
    excluded = [e["id"] for e in ctx.events
                if e["id"] != event.get("id") and e.get("clean") is not True
                and analog.direction(e) == analog.direction(event) and not use_unclean]
    return {
        "analogs": picks,
        "analog_tier": tier,
        "analog_table": [{"id": r["id"], "score": round(r["score"], 3), "parts": r["parts"]} for r in ranked[:5]],
        "analog_excluded": excluded,
    }
