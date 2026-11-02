"""Analog Finder 평가: 코드 순위 vs Claude 재순위.

  python -m agents.analog_eval            # 사람 정답(data/library/analog_labels.yaml)과 상위 3건 비교 + 백테스트
  python -m agents.analog_eval --no-bt    # 일치율만

지표
- 정답 일치(hit@3): 상위 3건에 사람이 고른 사건이 하나라도 있는 비율
- 정밀도(precision@3): 상위 3건 중 정답 비율 / 재현율(recall@3): 정답 중 상위 3건에 든 비율
- 백테스트: 같은 사건으로 방향 적중·범위 적중·severe 커버·MAE (재순위가 실제 예측을 좋게 하는지)
정답이 빈 사건은 일치율에서 빼고, 백테스트는 실제 수익률이 있는 clean 사건 전체로 한다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from engine.textsim import event_text

from . import analog_finder
from .backtest import actual_return, predict
from .context import load_context
from .llm import LLM

LABELS = Path(__file__).resolve().parents[1] / "data" / "library" / "analog_labels.yaml"


def load_labels(path: Path = LABELS) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {k: list(v or []) for k, v in (d.get("labels") or {}).items()}


def top3(ev: dict, ctx, llm=None) -> list[str]:
    a = analog_finder.find(ev, ctx, event_text(ev), llm=llm)
    return [p["id"] for p in sorted(a["analogs"], key=lambda p: -p["weight"])[:3]]


def overlap(pred: list[str], truth: list[str]) -> dict:
    hit = len(set(pred) & set(truth))
    return {"hit@3": hit > 0, "precision@3": hit / max(len(pred), 1), "recall@3": hit / max(len(truth), 1)}


def run(ctx, llm, labels: dict, backtest: bool = True) -> pd.DataFrame:
    rows = []
    for ev in sorted(ctx.events, key=lambda e: e["date"]):
        if ev.get("clean") is not True:
            continue
        truth = labels.get(ev["id"], [])
        for method, l in (("code", None), ("rerank", llm)):
            if method == "rerank" and (l is None or not l.enabled):
                continue
            r = {"event": ev["id"], "method": method, "has_label": bool(truth)}
            p3 = top3(ev, ctx, l)
            r["top3"] = ",".join(p3)
            if truth:
                r.update(overlap(p3, truth))
            if backtest:
                act = actual_return(ev["id"], ctx)
                if act is not None:
                    p = predict(ev, ctx, True, llm=l)
                    lo, hi = min(p["mild"], p["severe"]), max(p["mild"], p["severe"])
                    ease = ev.get("direction", "escalate") == "ease"
                    r.update({"dir_hit": bool(np.sign(p["central"]) == np.sign(act)), "in_range": bool(lo <= act <= hi),
                              "severe_cover": bool(act <= p["severe"]) if ease else bool(act >= p["severe"]),
                              "abs_err": abs(p["central"] - act)})
            rows.append(r)
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for m, g in df.groupby("method", sort=False):
        lab = g[g.has_label]
        row = {"정답 있는 사건": len(lab)}
        if len(lab):
            row.update({"정답 일치": lab["hit@3"].mean(), "정밀도@3": lab["precision@3"].mean(),
                        "재현율@3": lab["recall@3"].mean()})
        if "dir_hit" in g:
            b = g.dropna(subset=["dir_hit"])
            row.update({"BT 사건": len(b), "BT 방향": b["dir_hit"].mean(), "BT 범위": b["in_range"].mean(),
                        "BT severe": b["severe_cover"].mean(), "BT MAE": b["abs_err"].mean()})
        out[m] = row
    return pd.DataFrame(out).T


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bt", action="store_true")
    a = ap.parse_args()
    ctx = load_context()
    llm = LLM()
    labels = load_labels()
    n_lab = sum(1 for v in labels.values() if v)
    print(f"정답 라벨 {n_lab}건 ({LABELS.name}), LLM {'있음 ' + llm.model if llm.enabled else '없음 -> 코드 순위만'}")
    df = run(ctx, llm, labels, not a.no_bt)
    print(df.drop(columns=[c for c in ("has_label",) if c in df]).to_string(index=False))
    print("\n=== 요약 ===")
    print(summary(df).to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"LLM 호출 {llm.calls}회")


if __name__ == "__main__":
    main()
