"""사건 강도 '블라인드' 재채점: python -m agents.relabel_severity [--n 3] [--events id1,id2]

왜 필요한가
  라이브러리의 강도 6요소 점수는 사람이 '결과(시장 반응)를 아는 상태'에서 매겼을 수 있다.
  그러면 '강도 점수 vs 실제 반응' 상관(위약 검정)이 낙관적으로 나온다(사후 확증).
  여기서는 LLM이 사건 발표 당일에 알 수 있는 정보(제목 + 설명문 중 발표 내용)만 보고 다시 채점한다.
  이후 전개·시장 반응을 적은 문장은 넣지 않는다.

출력
  data/library/severity_blind.yaml : 사건별 블라인드 6요소(중앙값), 표본 간 퍼짐, 점수·등급, 사람 라벨과의 차이
  화면: 사람 라벨 vs 블라인드 비교표, 등급이 바뀐 사건, (placebo.csv가 있으면) 두 라벨의 반응 백분위 상관

적용
  thresholds.yaml 의 severity.labels 를 'blind' 로 두면 load_events 가 블라인드 점수를 쓴다(기본 'human').
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd
import yaml

from engine import severity as sev
from pipeline import config, library

from . import prompts
from .event_analyst import consensus, sample_severity
from .llm import LLM

BLIND_FILE = config.DATA_DIR / "library" / "severity_blind.yaml"

from engine.textsim import announcement_only


def announcement_text(ev: dict) -> str:
    return f"{ev.get('date', '')} {ev.get('title', '')}. {announcement_only(ev.get('description', ''))}"


def score_blind(ev: dict, llm: LLM, n: int) -> tuple[dict, dict] | None:
    head = {k: ev.get(k) for k in ("title", "instrument", "direction", "initiator", "target", "sectors", "stage")}
    text = announcement_text(ev)
    news_block = (f"<news>\n[s1] {text}\n</news>\n\n"
                  "주의: 발표 당일에 알 수 있는 정보만 근거로 채점합니다. 이 사건이 나중에 시장에 미친 영향이나 "
                  "이후 전개를 알고 있더라도 반영하지 않습니다.")
    sc = dict((config.thresholds().get("severity", {}).get("self_consistency") or {}))
    sc["n"] = n
    first = None
    try:
        r = llm.complete_json(prompts.SEVERITY_SYSTEM, f"사건 구조: {head}\n\n{news_block}", max_tokens=600,
                              schema=prompts.SEVERITY_SCHEMA, name="emit")
        first = (r or {}).get("severity_components") or None
    except Exception as e:
        print(f"  [{ev['id']}] 채점 실패: {type(e).__name__}: {e}")
    if not first or not all(k in first for k in sev.COMPONENTS):
        return None
    samples = sample_severity({**head, "severity_components": first}, news_block, llm, sc)
    return consensus(samples)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="사건당 채점 횟수 (중앙값 사용)")
    ap.add_argument("--events", default="", help="쉼표로 구분한 사건 id (기본: 전체)")
    a = ap.parse_args()
    llm = LLM()
    if not getattr(llm, "enabled", False):
        print("LLM 키가 없습니다. .env 의 LLM_API_KEY 를 확인하세요.")
        return
    th = config.thresholds().get("severity", {})
    weights = th.get("component_weights")
    events = library.load_events()
    only = {x.strip() for x in a.events.split(",") if x.strip()}
    out, rows = {}, []
    for ev in events:
        if only and ev["id"] not in only:
            continue
        r = score_blind(ev, llm, a.n)
        if r is None:
            continue
        med, spread = r
        pts = sev.score(med, weights)
        lvl = sev.level(pts)
        human_pts = ev.get("severity_points")
        out[ev["id"]] = {"severity_components": med, "spread": spread, "severity_points": round(pts, 1),
                         "severity": lvl, "human_points": None if human_pts is None else round(human_pts, 1),
                         "human_level": ev.get("severity_label", ev.get("severity")),
                         "text_used": announcement_text(ev)}
        rows.append({"사건": ev["id"], "사람 등급": ev.get("severity_label", ev.get("severity")),
                     "블라인드 등급": lvl, "사람 점수": human_pts, "블라인드 점수": round(pts, 1),
                     "퍼짐(최대)": max(spread.values()) if spread else None})
        print(f"  {ev['id']:18s} 사람 {ev.get('severity_label', ev.get('severity'))} -> 블라인드 {lvl} "
              f"({pts:.0f}점, 표본 퍼짐 {max(spread.values()) if spread else '-'})")
    if not rows:
        print("채점된 사건이 없습니다.")
        return
    with open(BLIND_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump({"generated_by": getattr(llm, "model", "llm"), "n_samples": a.n, "events": out},
                       f, allow_unicode=True, sort_keys=False)
    df = pd.DataFrame(rows)
    changed = df[df["사람 등급"] != df["블라인드 등급"]]
    print(f"\n등급이 바뀐 사건 {len(changed)}/{len(df)}건")
    if len(changed):
        print(changed.to_string(index=False))
    both = df.dropna(subset=["사람 점수"])
    if len(both) >= 3:
        print(f"사람 점수 vs 블라인드 점수 상관: {np.corrcoef(both['사람 점수'], both['블라인드 점수'])[0, 1]:+.2f}")
    pl = config.MODEL_DIR / "placebo.csv"
    if pl.exists():
        p = pd.read_csv(pl)
        key = "event" if "event" in p.columns else p.columns[0]
        pct_col = next((c for c in p.columns if "portfolio_pct" in c), None)
        if pct_col:
            m = df.merge(p[[key, pct_col]], left_on="사건", right_on=key)
            m = m.dropna(subset=["사람 점수"])
            if len(m) >= 3:
                print(f"실제 반응 백분위와의 상관 — 사람 라벨 {np.corrcoef(m['사람 점수'], m[pct_col])[0, 1]:+.2f} / "
                      f"블라인드 라벨 {np.corrcoef(m['블라인드 점수'], m[pct_col])[0, 1]:+.2f}  ({len(m)}건)")
    print(f"\n저장: {BLIND_FILE}  (thresholds.yaml severity.labels: blind 로 적용)")


if __name__ == "__main__":
    main()
