"""법령·산업정책 장기 효과 보기.

  python -m agents.law_effect                       # 라이브러리의 산업정책·투자제한 사건 전체
  python -m agents.law_effect --event us_chips_2022 # 한 사건
  python -m agents.law_effect --date 2022-08-09     # 라이브러리에 없는 날짜

출력: 바스켓별 20/60/120거래일 미국 주식 대비 초과수익과 무작위 창 대비 백분위.
"""
from __future__ import annotations

import argparse
import sys

from engine.long_window import excess_table

from .context import Context, load_context

LONG_TERM_INSTRUMENTS = {"industrial_subsidy", "investment_restriction"}


def for_event(ev: dict, ctx: Context, horizons=(20, 60, 120)):
    return excess_table(ctx.rets, ev.get("react_from") or ev["date"], horizons)


def reference_events(instrument: str, ctx: Context) -> list[dict]:
    return [e for e in sorted(ctx.events, key=lambda e: e["date"]) if e.get("instrument") == instrument]


def summary_rows(ev: dict, ctx: Context, top: int = 3) -> list[dict]:
    """화면·후속 질문용: 60거래일 초과수익 절댓값 상위 바스켓."""
    t = for_event(ev, ctx)
    if (60, "excess") not in t or t[(60, "excess")].isna().all():
        return []
    s = t[(60, "excess")].drop(labels=["us_equity"], errors="ignore").dropna()
    rows = []
    for b in s.abs().sort_values(ascending=False).index[:top]:
        rows.append({"event": ev["id"], "basket": ctx.labels.get(b, b), "excess_60d": float(t.loc[b, (60, "excess")]),
                     "pct_60d": float(t.loc[b, (60, "pct")])})
    return rows


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--event")
    ap.add_argument("--date")
    a = ap.parse_args()
    ctx = load_context()
    if a.date:
        evs = [{"id": f"date_{a.date}", "date": a.date, "title": "(지정 날짜)"}]
    elif a.event:
        e = ctx.event_by_id(a.event)
        if e is None:
            raise SystemExit(f"사건 없음: {a.event}")
        evs = [e]
    else:
        evs = [e for e in sorted(ctx.events, key=lambda e: e["date"]) if e.get("instrument") in LONG_TERM_INSTRUMENTS]
    for e in evs:
        t = for_event(e, ctx)
        print(f"\n=== {e['id']} {e['date']} {e.get('title', '')[:50]} ===")
        show = t.copy()
        for h in (20, 60, 120):
            if (h, "excess") in show:
                show[(h, "excess")] = show[(h, "excess")].map(lambda x: "" if x != x else f"{x:+.2%}")
                show[(h, "pct")] = show[(h, "pct")].map(lambda x: "" if x != x else f"{x:.0f}")
        show.index = [ctx.labels.get(b, b) for b in show.index]
        print(show.to_string())
    print("\n초과수익 = 미국 주식 대비 누적 로그수익 차이, pct = 같은 길이 모든 기간 대비 백분위(50 = 평범). 인과가 아닌 사후 기록입니다.")


if __name__ == "__main__":
    main()
