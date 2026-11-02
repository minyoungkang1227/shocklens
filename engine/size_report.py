"""충격 크기 등급별 분석: python -m engine.size_report

작은 사건은 작은 사건끼리, 큰 사건은 큰 사건끼리 모아 발표 후 3거래일 반응을 따로 요약한다.
- 등급별 사건 수(clean만), 팩터별 '충격 방향 크기'(방향 x 변동)의 중앙값·최대값
- 등급별 포트폴리오 반응(평상시 베타로 근사)의 중앙값·최악값
강도 -> 충격 크기 보정 규칙을 정하거나, 등급 경계(소 1~2 / 중 3 / 대 4~5)가 적절한지 점검할 때 쓴다.
선행 조건: python -m pipeline.run_all
"""
from __future__ import annotations

import sys

import pandas as pd

from pipeline import config, library

from . import analog


def summarize(events: list[dict], f_event: pd.DataFrame, beta: pd.DataFrame, weights: pd.Series,
              classes: dict, clean_only: bool = True) -> dict[str, dict]:
    w = weights.reindex(beta.index).fillna(0.0)
    w = w / w.sum()
    out = {}
    for name in analog.CLASS_ORDER:
        evs = [e for e in events if analog.size_class(e.get("severity", 3), classes) == name
               and e["id"] in f_event.index and (e.get("clean") is True or not clean_only)]
        if not evs:
            out[name] = {"n": 0}
            continue
        rows, port = {}, {}
        for e in evs:
            mags = {f: d * f_event.loc[e["id"], f] for f, d in e["channels"].items()
                    if f in f_event.columns and pd.notna(f_event.loc[e["id"], f])}
            rows[e["id"]] = mags
            port[e["id"]] = float(w.to_numpy() @ (beta.reindex(columns=f_event.columns).to_numpy()
                                                  @ f_event.loc[e["id"]].fillna(0.0).to_numpy()))
        mag = pd.DataFrame(rows).T
        out[name] = {"n": len(evs), "ids": [e["id"] for e in evs],
                     "median_mag": mag.median().to_dict(), "max_mag": mag.max().to_dict(),
                     "port_median": float(pd.Series(port).median()), "port_worst": float(min(port.values()))}
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    th = config.thresholds()
    classes = th["analog"].get("size_classes", analog.DEFAULT_CLASSES)
    pf = config.portfolio()["baskets"]
    weights = pd.Series({b: v["weight"] for b, v in pf.items()}, dtype=float)
    f_event = pd.read_csv(config.MODEL_DIR / "event_factor_react.csv", index_col=0)
    beta = pd.read_csv(config.MODEL_DIR / "beta_normal.csv", index_col=0)
    events = library.load_events()
    for direction in ("escalate", "ease"):
        evs = [e for e in events if analog.direction(e) == direction]
        print(f"\n=== {direction} 사건, 등급별 발표 후 3거래일 반응 (clean 사건만) ===")
        res = summarize(evs, f_event, beta, weights, classes)
        for name, r in res.items():
            label = {"small": "소(강도 1~2)", "medium": "중(강도 3)", "large": "대(강도 4~5)"}[name]
            if not r["n"]:
                print(f"[{label}] 사건 없음 -> 이 등급의 새 사건은 강도 기반 분위수로 충격을 합성")
                continue
            print(f"[{label}] {r['n']}건 {r['ids']}")
            print(f"   포트폴리오 반응(평상시 베타 근사): 중앙 {r['port_median']:+.2%} / 최악 {r['port_worst']:+.2%}")
            mags = pd.DataFrame({"중앙값": r["median_mag"], "최대값": r["max_mag"]})
            print("   충격 방향 크기(방향 x 변동):")
            print("   " + mags.round(4).to_string().replace("\n", "\n   "))


if __name__ == "__main__":
    main()
