"""LLM 있음/없음 비교 실험: 같은 사건을 규칙 기반과 Claude로 각각 해석해 정확도와 백테스트 적중률을 비교한다.

  python -m agents.llm_compare            # clean 사건 전체 (LLM 키 필요, 없으면 규칙만)
  python -m agents.llm_compare --n 1      # 강도 자기일관성 끄기 (호출 수 1/3)

입력: 사건 라이브러리의 title + description 을 '뉴스'로 간주 (같은 입력을 두 방식에 넣는다)
정답: 라이브러리에 사람이 정리한 라벨 (수단·방향·발동국·대상국·산업·강도 6요소)
지표
- 사건 해석: 수단·방향·발동국 일치율, 대상국 겹침, 산업 Jaccard, 강도 등급 일치, 강도 점수 오차(점)
- 백테스트: 해석한 사건으로 유사 사건 검색→충격→3거래일 손실을 예측해 실제와 비교 (방향 적중·범위 적중·severe 커버·MAE)
  '라벨' 행은 정답 라벨을 그대로 넣었을 때 (해석이 완벽할 때의 상한)
결과: 화면 출력 + data/model/llm_compare.csv
"""
from __future__ import annotations

import argparse
import copy
import sys
import time

import numpy as np
import pandas as pd

from engine import severity as sev
from pipeline import config

from . import event_analyst
from .backtest import actual_return, predict
from .context import load_context
from .llm import LLM


def news_of(ev: dict) -> str:
    return f"{ev['title']}. {' '.join(str(ev.get('description', '')).split())}"


def jaccard(a, b) -> float:
    a, b = set(a or []), set(b or [])
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def score(pred: dict, truth: dict, sev_cfg: dict) -> dict:
    w = sev_cfg.get("component_weights")
    t_pts = sev.score(truth.get("severity_components"), w, truth.get("severity", 3))
    p_pts = pred.get("severity_points", 50.0)
    return {
        "instrument": pred.get("instrument") == truth["instrument"],
        "direction": pred.get("direction", "escalate") == truth.get("direction", "escalate"),
        "initiator": pred.get("initiator") == truth["initiator"],
        "target": bool(set(pred.get("target", [])) & set(truth.get("target", []))),
        "sectors_j": jaccard(pred.get("sectors"), truth.get("sectors")),
        "sev_level": sev.level(p_pts) == sev.level(t_pts),
        "sev_abs_pts": abs(p_pts - t_pts),
    }


def run_one(ev: dict, method: str, ctx, llm: LLM, sev_cfg: dict) -> dict:
    t0 = time.time()
    if method == "라벨":
        pred, ok, err = ev, True, ""
    else:
        out = event_analyst.analyze(news_of(ev), llm if method == "llm" else LLM(provider="none"), None, sev_cfg)
        pred, ok, err = out["event"], out["event_complete"], "; ".join(out["event_errors"])[:80]
        if method == "llm" and out.get("llm_error"):
            err = (err + " / " if err else "") + f"LLM 실패→규칙: {out['llm_error']}"
    row = {"event": ev["id"], "method": method, "complete": ok, "sec": round(time.time() - t0, 1), "error": err,
           "mode": out["mode"] if method != "라벨" else "label"}
    if not ok:
        return row
    row.update(score(pred, ev, sev_cfg))
    act = actual_return(ev["id"], ctx)
    if act is not None:
        e = {**pred, "id": ev["id"], "date": ev["date"], "clean": True}
        if ev.get("react_from"):
            e["react_from"] = ev["react_from"]
        p = predict(e, ctx, use_text=True)
        lo, hi = min(p["mild"], p["severe"]), max(p["mild"], p["severe"])
        ease = ev.get("direction", "escalate") == "ease"
        row.update({"central": p["central"], "actual": act,
                    "dir_hit": bool(np.sign(p["central"]) == np.sign(act)),
                    "in_range": bool(lo <= act <= hi),
                    "severe_cover": bool(act <= p["severe"]) if ease else bool(act >= p["severe"]),
                    "abs_err": abs(p["central"] - act)})
    return row


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df[df.complete].groupby("method", sort=False)
    s = pd.DataFrame({
        "구조화 성공": df.groupby("method", sort=False)["complete"].mean(),
        "수단": g["instrument"].mean(), "방향": g["direction"].mean(), "발동국": g["initiator"].mean(),
        "대상국": g["target"].mean(), "산업 J": g["sectors_j"].mean(),
        "강도 등급": g["sev_level"].mean(), "강도 오차(점)": g["sev_abs_pts"].mean(),
    })
    if "dir_hit" in df:
        b = df.dropna(subset=["dir_hit"]).groupby("method", sort=False)
        s["BT 사건 수"] = b.size()
        s["BT 방향 적중"] = b["dir_hit"].mean()
        s["BT 범위 적중"] = b["in_range"].mean()
        s["BT severe 커버"] = b["severe_cover"].mean()
        s["BT MAE"] = b["abs_err"].mean()
    return s


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="강도 채점 횟수 (기본: thresholds.yaml)")
    a = ap.parse_args()
    ctx = load_context()
    llm = LLM()
    sev_cfg = copy.deepcopy(ctx.thresholds.get("severity", {}))
    if a.n is not None:
        sev_cfg.setdefault("self_consistency", {})["n"] = a.n
    methods = ["rule"] + (["llm"] if llm.enabled else []) + ["라벨"]
    if not llm.enabled:
        print("LLM 키가 없어 규칙 기반과 라벨만 비교합니다 (.env의 LLM_API_KEY 확인)")
    events = [e for e in sorted(ctx.events, key=lambda e: e["date"]) if e.get("clean") is True]
    rows = []
    for ev in events:
        for m in methods:
            r = run_one(ev, m, ctx, llm, sev_cfg)
            rows.append(r)
            print(f"  {ev['id']:24s} {m:5s} [{r['mode']}] {'성공' if r['complete'] else '실패'} {r['error']} {r['sec']}s")
    n_fallback = sum(1 for r in rows if r["method"] == "llm" and r["mode"] == "rule")
    if n_fallback:
        print(f"\n!! llm 방식 {n_fallback}건이 Claude 호출 실패로 규칙 경로로 떨어졌습니다. 위 오류를 확인하세요 "
              f"(llm 행과 rule 행이 같으면 비교가 무의미).")
    df = pd.DataFrame(rows)
    s = summary(df)
    pct = "{:.0%}".format
    fm = {c: pct for c in s.columns if c not in ("강도 오차(점)", "BT 사건 수", "BT MAE")}
    fm.update({"강도 오차(점)": "{:.1f}".format, "BT MAE": "{:.2%}".format})
    print(f"\n=== 요약 (clean 사건 {len(events)}건, LLM 호출 {llm.calls}회) ===")
    print(s.to_string(formatters=fm))
    out = config.MODEL_DIR / "llm_compare.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n상세 결과: {out}")


if __name__ == "__main__":
    main()
