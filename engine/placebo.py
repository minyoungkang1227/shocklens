"""위약 검정(placebo test): python -m engine.placebo

질문: "지경학 사건 후 3거래일 반응이 평소 아무 날의 3거래일 변동과 정말 다른가?"
방법
  1. 사건이 없는 날(모든 사건의 반응 창 ±5일을 제외)에서 시작하는 3거래일 창을 수천 개 뽑는다 = 위약 분포.
  2. 사건마다, 그 사건의 영향 채널 팩터에 대해 '충격 방향 크기'(방향 x 변동)가 위약 분포에서 몇 번째 백분위인지 잰다.
     백분위 95%면 "평소 변동 중 상위 5%만큼 큰 반응"이라는 뜻.
  3. 포트폴리오 수준(평상시 베타로 근사한 3일 수익률)도 같은 방식으로 잰다.
  4. 사건 전체 요약: 채널 백분위의 중앙값, 80% 이상인 비율(사건 반응이 '우연'일 확률이 낮은 비율), 등급별 평균.
의미: 사건 표본이 20여 건이라도, 각 사건을 수천 개 위약과 비교하므로 "사건이 실제로 시장을 움직였는가"를
      통계적으로 말할 수 있다. 백분위가 낮게 나오는 사건은 clean 재검토나 강도 재조정의 근거가 된다.
결과: 화면 출력 + data/model/placebo.csv
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from pipeline import config, library

from . import analog
from .severity import level


def placebo_windows(frets: pd.DataFrame, event_dates: list[str], react_days: int, exclude_pad: int = 5,
                    n: int = 5000, seed: int = 20261112) -> pd.DataFrame:
    """사건 창을 피한 무작위 시작일에서 react_days 누적 팩터 변동. 행 = 표본, 열 = 팩터."""
    idx = frets.index
    bad = np.zeros(len(idx), dtype=bool)
    for d in event_dates:
        try:
            j = int(idx.searchsorted(pd.Timestamp(d)))
        except Exception:
            continue
        lo, hi = max(0, j - exclude_pad), min(len(idx), j + react_days + exclude_pad)
        bad[lo:hi] = True
    ok = np.flatnonzero(~bad)
    ok = ok[ok <= len(idx) - react_days]
    rng = np.random.default_rng(seed)
    starts = rng.choice(ok, size=min(n, len(ok)), replace=len(ok) < n)
    vals = np.stack([frets.iloc[s:s + react_days].sum(min_count=1).to_numpy() for s in starts])
    return pd.DataFrame(vals, columns=frets.columns)


def percentile_of(value: float, sample: np.ndarray) -> float:
    s = sample[~np.isnan(sample)]
    if len(s) == 0 or np.isnan(value):
        return np.nan
    return float((s < value).mean())


def run(events: list[dict], f_event: pd.DataFrame, frets_long: pd.DataFrame, beta: pd.DataFrame,
        weights: pd.Series, react_days: int, n: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame]:
    plc = placebo_windows(frets_long, [e.get("react_from") or e["date"] for e in events], react_days, n=n)
    w = weights.reindex(beta.index).fillna(0.0)
    w = w / w.sum()
    B = beta.reindex(columns=frets_long.columns).fillna(0.0).to_numpy()
    port_plc = (plc.fillna(0.0).to_numpy() @ B.T) @ w.to_numpy()          # 위약의 포트폴리오 근사 수익률
    rows = []
    for e in events:
        if e["id"] not in f_event.index:
            continue
        fp = f_event.loc[e["id"]]
        ch_pct = {}
        for f, d in e["channels"].items():
            if f in plc.columns:
                ch_pct[f] = percentile_of(d * fp[f], d * plc[f].to_numpy())   # 충격 방향으로 본 크기의 백분위
        port = float((B @ fp.fillna(0.0).to_numpy()) @ w.to_numpy())
        sign = -1 if e.get("direction", "escalate") == "escalate" else 1      # 확대 사건은 손실 방향이 큰 쪽
        port_pct = percentile_of(sign * port, sign * port_plc)
        rows.append({"event": e["id"], "date": e["date"], "direction": e.get("direction", "escalate"),
                     "severity": e.get("severity"), "points": round(e.get("severity_points", 0), 1),
                     "clean": e.get("clean") is True, "channel_pct_median": float(np.nanmedian(list(ch_pct.values()))),
                     "channel_pct_max": float(np.nanmax(list(ch_pct.values()))), "portfolio_pct": port_pct,
                     "portfolio_3d": port, **{f"pct_{k}": v for k, v in ch_pct.items()}})
    df = pd.DataFrame(rows)
    return df, plc


def summary(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[df["clean"]]
    out = {
        "전체": df, "clean만": clean,
        **{f"등급 {g}": clean[clean["severity"] == g] for g in sorted(clean["severity"].dropna().unique())},
    }
    rows = []
    for name, sub in out.items():
        if sub.empty:
            continue
        rows.append({"구분": name, "사건 수": len(sub),
                     "채널 백분위 중앙값": sub["channel_pct_median"].median(),
                     "채널 백분위≥80% 비율": (sub["channel_pct_median"] >= 0.8).mean(),
                     "포트폴리오 백분위 중앙값": sub["portfolio_pct"].median(),
                     "포트폴리오 백분위≥80% 비율": (sub["portfolio_pct"] >= 0.8).mean()})
    return pd.DataFrame(rows).set_index("구분")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    a = ap.parse_args()
    th = config.thresholds()
    pf = config.portfolio()["baskets"]
    weights = pd.Series({b: v["weight"] for b, v in pf.items()}, dtype=float)
    long_path = config.PROCESSED_DIR / "factor_returns_long.parquet"
    frets = pd.read_parquet(long_path if long_path.exists() else config.PROCESSED_DIR / "factor_returns.parquet")
    f_event = pd.read_csv(config.MODEL_DIR / "event_factor_react.csv", index_col=0)
    beta = pd.read_csv(config.MODEL_DIR / "beta_normal.csv", index_col=0)
    events = library.load_events()
    df, plc = run(events, f_event, frets, beta, weights, th["event_window"]["react_days"], a.n)
    pct = lambda x: f"{x:.0%}" if pd.notna(x) else "-"
    print(f"=== 위약 검정: 사건 {len(df)}건 vs 무작위 {len(plc)}개 3거래일 창 ===")
    show = df[["event", "severity", "points", "clean", "channel_pct_median", "channel_pct_max", "portfolio_pct", "portfolio_3d"]].copy()
    for c in ("channel_pct_median", "channel_pct_max", "portfolio_pct"):
        show[c] = show[c].map(pct)
    show["portfolio_3d"] = show["portfolio_3d"].map(lambda x: f"{x:+.2%}")
    show.columns = ["사건", "등급", "점수", "clean", "채널백분위(중앙)", "채널백분위(최대)", "포트폴리오백분위", "포트폴리오3일"]
    print(show.to_string(index=False))
    print("\n=== 요약 (백분위 80% 이상 = 평소 변동의 상위 20%보다 큰 반응) ===")
    print(summary(df).to_string(formatters={c: "{:.0%}".format for c in summary(df).columns if c != "사건 수"}))
    # 강도 점수와 실제 반응의 상관 (요소 가중치 보정의 출발점)
    c = df[df["clean"]]
    if len(c) >= 5:
        corr = c[["points", "portfolio_pct"]].corr().iloc[0, 1]
        print(f"\n강도 점수 vs 포트폴리오 반응 백분위 상관 (clean {len(c)}건): {corr:+.2f}  "
              f"(양수면 강도가 높을수록 실제 반응이 큼)")
    out = config.MODEL_DIR / "placebo.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n상세 결과: {out}")


if __name__ == "__main__":
    main()
