"""백테스트: 라이브러리 사건을 하나씩 '새 뉴스'로 가정하고 예측과 실제를 비교한다 (leave-one-out).

  python -m agents.backtest              # clean 사건만 대상 (기본)
  python -m agents.backtest --all        # clean: false 사건도 참고로 포함

규칙
- 예측할 때는 대상 사건보다 '먼저' 일어난 사건만 유사 사례로 쓴다 (사후 정보 누수 방지).
- 예측 = 발표 후 3거래일 포트폴리오 수익률의 mild / central / severe (결정론적 시나리오).
- 실제 = 그 사건 발표 후 3거래일 실제 포트폴리오 수익률.

지표
- 방향 적중: central과 실제의 부호가 같은 비율
- 범위 적중: 실제가 mild~severe 사이에 든 비율
- severe 커버: 실제 손실이 severe보다 나쁘지 않은 비율 (스트레스테스트로서 충분히 보수적인가)
- 평균 절대오차: |central - 실제|
설정 두 가지를 나란히 비교한다: v1(구조 점수만, 11/1 버전) vs v2(현재: 문장 유사도 + 등급 분리 + 가중 분포 + severe 하한선)
결과: 화면 출력 + data/model/backtest.csv
"""
from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

from engine import risk
from engine.textsim import event_text
from pipeline import config

from . import analog_finder, shock_mapper
from .context import Context, load_context
from .exposure_analyst import analyze as exposure_of

VARIANTS = {
    "v1_구조만": {"text_weight": 0.0, "size_classes": None, "severe_floor": False, "top_k": 3, "min_weighted": 99,
               "partial_unclean_weight": 0.0, "pool_tier": False},
    "v2_11.03": {"partial_unclean_weight": 0.0, "pool_tier": False},
    "v3_현재": {},        # + 겹친 사건 부분 사용(0.5) + 수단 풀 층
}


def variant_ctx(ctx: Context, overrides: dict) -> Context:
    th = copy.deepcopy(ctx.thresholds)
    for k, v in overrides.items():
        if v is None:
            th["analog"].pop(k, None)
        else:
            th["analog"][k] = v
    return replace(ctx, thresholds=th)


def asof_context(ctx: Context, date: str, min_days: int = 750) -> Context:
    """사건 날짜 '이전' 데이터만으로 베타·위기 베타·분위수 표를 다시 추정한 Context.
    (look-ahead 차단: 전체 기간으로 만든 모델 산출물을 과거 사건 테스트에 쓰지 않는다)
    이력이 min_days보다 짧으면 그대로 돌려주고 호출자가 건너뛴다."""
    from pipeline import betas as betas_mod, quantiles as q_mod
    cut = pd.Timestamp(date) - pd.Timedelta(days=1)
    rets = ctx.rets.loc[:cut]
    frets = ctx.frets.loc[:cut] if ctx.frets is not None else None
    if frets is None or len(rets) < min_days:
        return ctx
    frets = frets.reindex(rets.index).dropna()
    rets = rets.reindex(frets.index)
    th = ctx.thresholds
    b_norm, b_cris, _, _, _ = betas_mod.build(rets, frets, ctx.levels[ctx.vix_col].loc[:cut],
                                              th["betas"]["ridge_alpha"], th["betas"]["crisis_vix_quantile"])
    up, down = q_mod.quantile_tables(frets, th["event_window"]["react_days"])
    up.columns, down.columns = [str(c) for c in up.columns], [str(c) for c in down.columns]
    return replace(ctx, beta_normal=b_norm, beta_crisis=b_cris, q_up=up, q_down=down)


def predict(ev: dict, ctx: Context, use_text: bool, llm=None, asof: bool = True) -> dict:
    if asof and ev.get("date"):
        ctx = asof_context(ctx, ev["date"])
    a = analog_finder.find(ev, ctx, event_text(ev) if use_text else None, llm=llm)
    ids = [x["id"] for x in a["analogs"]]
    w = {x["id"]: x["weight"] for x in a["analogs"]}
    expo = exposure_of(ev, ctx)["exposure"]
    s = shock_mapper.map_shock(ev, ids, a["analog_tier"], expo, ctx, None, w, asof=ev["date"])
    beta = ctx.beta_crisis if s["beta_regime"] == "crisis" else ctx.beta_normal
    fs = shock_mapper.to_frame(s["factor_shock"])
    idio = shock_mapper.to_frame(s["idio_shock"])
    loss = risk.scenario_losses(risk.basket_shock(beta.reindex(ctx.rets.columns), fs, idio),
                                ctx.weights / ctx.weights.sum())
    nb = (ctx.thresholds.get("risk") or {}).get("noise_band") or {}
    half = 0.0
    if nb.get("enabled", True):
        ex = [(str(e.get("react_from") or e["date"]), str(e["date"])) for e in ctx.events if e.get("date")]
        half = risk.noise_band(ctx.rets, ctx.weights, ctx.thresholds["event_window"]["react_days"],
                               float(nb.get("q", 0.75)), ex, asof=ev.get("date") if asof else None)["half_width"]
    return {"analogs": ids, "tier": a["analog_tier"], "method": s["shock_method"], "band_half": half, **loss.to_dict()}


def actual_return(eid: str, ctx: Context) -> float | None:
    if eid not in ctx.b_event.index or ctx.b_event.loc[eid].isna().any():
        return None
    w = (ctx.weights / ctx.weights.sum()).reindex(ctx.b_event.columns).to_numpy()
    return float(risk.portfolio_return(ctx.b_event.loc[[eid]].to_numpy(), w)[0])


def run(ctx: Context, include_unclean: bool = False, asof: bool = True) -> pd.DataFrame:
    """asof=True: 사건 이전 데이터만으로 베타·분위수를 재추정(기본). False: 전체 기간 산출물 사용(옛 방식, 비교용)."""
    rows = []
    for vname, ov in VARIANTS.items():
        vctx = variant_ctx(ctx, ov)
        use_text = vname != "v1_구조만"
        for ev in sorted(ctx.events, key=lambda e: e["date"]):
            if not include_unclean and ev.get("clean") is not True:
                continue
            act = actual_return(ev["id"], ctx)
            if act is None:
                continue
            p = predict(ev, vctx, use_text, asof=asof)
            lo, hi = min(p["mild"], p["severe"]), max(p["mild"], p["severe"])
            ease = ev.get("direction", "escalate") == "ease"
            rows.append({
                "variant": vname, "event": ev["id"], "clean": ev.get("clean") is True,
                "direction": ev.get("direction", "escalate"), "tier": p["tier"], "method": p["method"],
                "analogs": ",".join(p["analogs"]), "mild": p["mild"], "central": p["central"], "severe": p["severe"],
                "actual": act,
                "dir_hit": bool(np.sign(p["central"]) == np.sign(act)),
                "in_range": bool(lo <= act <= hi),
                "band_half": p["band_half"],
                "in_band": bool(lo - p["band_half"] <= act <= hi + p["band_half"]),
                "band_width": (hi - lo) + 2 * p["band_half"],
                "severe_cover": bool(act <= p["severe"]) if ease else bool(act >= p["severe"]),
                "abs_err": abs(p["central"] - act),
            })
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("variant")
    return pd.DataFrame({
        "사건 수": g.size(),
        "방향 적중": g["dir_hit"].mean(),
        "범위 적중": g["in_range"].mean(),
        "예측구간 커버": g["in_band"].mean(),
        "구간 폭": g["band_width"].mean(),
        "severe 커버": g["severe_cover"].mean(),
        "평균 절대오차": g["abs_err"].mean(),
    })


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="clean: false 사건도 포함")
    ap.add_argument("--no-asof", action="store_true", help="전체 기간 베타·분위수 사용 (look-ahead 허용, 비교용)")
    a = ap.parse_args()
    ctx = load_context()
    print("베타·분위수: " + ("전체 기간 (look-ahead 허용)" if a.no_asof else "사건 이전 데이터만으로 재추정 (as-of)"))
    df = run(ctx, a.all, asof=not a.no_asof)
    if df.empty:
        print("비교할 사건이 없습니다 (실제 수익률이 있는 사건 필요)")
        return
    pct = lambda x: f"{x:+.2%}"
    for vname in VARIANTS:
        sub = df[df.variant == vname]
        print(f"\n=== {vname} ===")
        show = sub[["event", "tier", "method", "mild", "central", "severe", "band_half", "actual",
                    "in_range", "in_band", "severe_cover"]].copy()
        for c in ("mild", "central", "severe", "actual"):
            show[c] = show[c].map(pct)
        show["band_half"] = show["band_half"].map(lambda x: f"±{x:.2%}")
        print(show.to_string(index=False))
    s = summary(df)
    print("\n=== 요약 ===")
    print(s.to_string(formatters={"방향 적중": "{:.0%}".format, "범위 적중": "{:.0%}".format,
                                  "예측구간 커버": "{:.0%}".format, "구간 폭": "{:.2%}".format,
                                  "severe 커버": "{:.0%}".format, "평균 절대오차": "{:.2%}".format}))
    out = config.MODEL_DIR / "backtest.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n상세 결과: {out}")


if __name__ == "__main__":
    main()
