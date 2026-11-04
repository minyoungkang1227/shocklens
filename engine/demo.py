"""엔진 데모: python -m engine.demo [--event cn_ree_2025]

선택한 사건을 '새로 들어온 뉴스'라고 가정하고, 나머지 사건들을 유사 사례로 써서 스트레스테스트를 돌린다.
(사건 하나를 빼고 예측하는 leave-one-out 방식이라, 실제로 그 사건 때 일어난 일과 비교할 수 있다.)
선행 조건: python -m pipeline.run_all 을 먼저 실행해 data/model/ 이 채워져 있어야 한다.
"""
from __future__ import annotations

import argparse

import pandas as pd
import yaml

from pipeline import config, library
from . import analog, calclog, risk, shock
from .exposure import exposure_for


def load_model():
    m = config.MODEL_DIR
    p = config.PROCESSED_DIR
    return {
        "rets": pd.read_parquet(p / "returns.parquet"),
        "levels": pd.read_parquet(p / "factor_levels.parquet"),
        "beta_normal": pd.read_csv(m / "beta_normal.csv", index_col=0),
        "beta_crisis": pd.read_csv(m / "beta_crisis.csv", index_col=0),
        "f_post": pd.read_csv(m / "event_factor_post.csv", index_col=0),
        "b_post": pd.read_csv(m / "event_basket_post.csv", index_col=0),
        "resid_post": pd.read_csv(m / "event_resid_post.csv", index_col=0),
        "q_up_post": pd.read_csv(m / "factor_quantiles_up.csv", index_col=0),
        "q_down_post": pd.read_csv(m / "factor_quantiles_down.csv", index_col=0),
        "f_react": pd.read_csv(m / "event_factor_react.csv", index_col=0),
        "b_react": pd.read_csv(m / "event_basket_react.csv", index_col=0),
        "resid_react": pd.read_csv(m / "event_resid_react.csv", index_col=0),
        "q_up_react": pd.read_csv(m / "factor_quantiles_react_up.csv", index_col=0),
        "q_down_react": pd.read_csv(m / "factor_quantiles_react_down.csv", index_col=0),
    }


def main(event_id: str, window: str = "react", use_all: bool = False, past_only: bool = False) -> None:
    th = config.thresholds()
    pf = config.portfolio()["baskets"]
    weights = pd.Series({b: v["weight"] for b, v in pf.items()}, dtype=float)
    with open(config.CONFIG_DIR / "exposure_map.yaml", encoding="utf-8") as f:
        exposure_map = yaml.safe_load(f)

    md = load_model()
    F, B, R = md[f"f_{window}"], md[f"b_{window}"], md[f"resid_{window}"]
    QU, QD = md[f"q_up_{window}"], md[f"q_down_{window}"]
    win_days = th["event_window"]["react_days"] if window == "react" else th["event_window"]["post_days"]
    events = {e["id"]: e for e in library.load_events()}
    ev = events[event_id]
    ev_list = list(events.values())
    acfg = th["analog"]
    analogs, tier, ranked = analog.select(ev, ev_list, list(F.index), acfg,
                                          clean_only=not use_all, past_only=past_only)
    pw = float(acfg.get("partial_unclean_weight", 0.0))
    excluded = [e["id"] for e in ev_list if e["id"] != event_id and e["id"] in F.index
                and analog.direction(e) == analog.direction(ev) and not use_all
                and analog.clean_factor(e, True, pw) is None]

    regime = "crisis" if ev["severity"] >= th["severity"]["crisis_regime_min"] else "normal"
    beta = md["beta_crisis"] if regime == "crisis" else md["beta_normal"]
    flipped = []
    if analogs:
        F_ch, flipped = shock.channel_filter(F, ev["channels"], acfg.get("clip_wrong_sign", True))
        flipped = [x for x in flipped if x[0] in analogs]
        f_shock = shock.from_analogs(F_ch, analogs, beta, weights)
    else:
        f_shock = shock.from_severity(ev["channels"], ev["severity"], QU, QD,
                                      th["severity"]["quantile_map"], list(F.columns))

    exposure = exposure_for(ev, exposure_map)
    idio = shock.idio_shock(exposure, R, analogs, list(md["rets"].columns))

    vix = md["levels"][config.factors()["factors"]["risk_off"]["series"]]
    mask = vix >= vix.quantile(th["betas"]["crisis_vix_quantile"]) if regime == "crisis" else None
    r = th["risk"]
    out = risk.run_stress(md["rets"], weights, beta, f_shock, idio, mask, r["horizon_days"], r["block_len"],
                          r["n_paths"], r["var_level"], r["es_level"], r["seed"])
    out.pop("_base_regime_paths", None)        # 앙상블용 경로 배열은 장부에 저장하지 않는다
    calc_id = calclog.save(
        inputs={"event_id": event_id, "analogs": analogs, "excluded_unclean": excluded, "tier": tier, "regime": regime,
                "factor_shock": f_shock.to_dict(), "exposure": exposure, "channels": ev["channels"],
                "analog_scores": ranked[:5], "zeroed_wrong_sign": flipped},
        outputs=out, data_version=str(md["rets"].index.max().date()))

    # 실제로 그 사건 때 포트폴리오가 어땠는지 (leave-one-out 비교용)
    actual = None
    if event_id in B.index:
        w = weights / weights.sum()
        actual = float(risk.portfolio_return(B.loc[[event_id], w.index].to_numpy(), w.to_numpy())[0])

    pct = lambda x: f"{x * 100:+.2f}%"
    print(f"\n=== {ev['title']} ===")
    print(f"경로: {tier} | 유사 사건: {analogs} | 베타: {regime} | 충격 산정 창: 발표 후 {win_days}거래일")
    print(f"영향 채널(이 팩터만 충격에 사용): {ev['channels']}  | 방향: {analog.direction(ev)}")
    print("\n[유사 사건 구조 점수 상위 5] (수단 .4 / 산업 .3 / 국가 .2 / 단계 .1)")
    for r_ in ranked[:5]:
        p_ = r_["parts"]
        mark = "*" if r_["id"] in analogs else " "
        print(f"  {mark} {r_['id']:18s} {r_['score']:.2f}  (수단 {p_['instrument']:.1f} 산업 {p_['sector']:.2f} "
              f"국가 {p_['countries']:.2f} 단계 {p_['stage']:.1f})")
    if not ranked:
        print("  (같은 방향의 clean 사건 없음)")
    if flipped:
        print("  기대와 반대 방향이라 0으로 둔 값:", ", ".join(f"{i}.{f}={v:+.4f}" for i, f, v in flipped))
    print(f"\n[팩터 충격, 발표 후 {win_days}거래일]")
    print(f_shock.round(4).to_string())
    print("\n[직접 노출 high 바스켓]", [b for b, l in exposure.items() if l == "high"])
    print(f"\n[결정론적 시나리오 포트폴리오 수익률, {win_days}거래일 충격]")
    for k, v in out["scenario_loss"].items():
        print(f"  {k:8s} {pct(v)}")
    fo = out["scenario_loss_factor_only"]["central"]
    print(f"  (central 분해: 팩터 경로 {pct(fo)} + 직접 노출 고유충격 약 {pct(out['scenario_loss']['central'] - fo)})")
    d = out["decomposition"]
    print("\n[20일 손실 분포 - ES97.5 분해] (손실을 양수로 표시)")
    print(f"  ① 평상시 (전체 기간 표본)     VaR95 {out['baseline_normal']['VaR']:.2%}  ES97.5 {out['baseline_normal']['ES']:.2%}")
    reg_name = "위기 국면 (VIX 상위 20% 표본)" if regime == "crisis" else "평상시 국면 (①과 같음)  "
    print(f"  ② {reg_name}  VaR95 {out['baseline']['VaR']:.2%}  ES97.5 {out['baseline']['ES']:.2%}")
    print(f"  ③ ② + 이번 사건 충격          VaR95 {out['stressed']['VaR']:.2%}  ES97.5 {out['stressed']['ES']:.2%}")
    print(f"  -> 국면 효과(②-①) {d['regime_effect_ES']:+.2%}p | 사건 효과(③-②) {d['event_effect_ES']:+.2%}p | 합계(③-①) {d['total_change_ES']:+.2%}p")
    print(f"  정규분포 MC로 계산한 ③의 ES97.5 {out['normal_mc_stressed']['ES']:.2%}  (과소평가 {out['es_underestimation_normal']:.2%}p)")
    top = sorted(out["tail_contribution"].items(), key=lambda kv: -kv[1])[:3]
    print("\n[ES 구간 손실 기여 상위 3]", ", ".join(f"{b} {c:.0%}" for b, c in top))
    if actual is not None:
        print(f"\n[실제] 그 사건 후 {win_days}거래일 포트폴리오 수익률: {pct(actual)}  (central 시나리오 {pct(out['scenario_loss']['central'])})")
        if not ev.get("clean", False):
            print("  주의: 이 사건은 반응 창에 다른 큰 충격이 겹쳐 있어(clean: false), 실제 수익률에 사건 외 요인이 섞여 있습니다.")
            for c in ev.get("confounders", []):
                print(f"   - {c}")
    if excluded:
        print(f"\n[유사 사례 후보에서 뺀 사건 (같은 방향이지만 clean: false)] {excluded}  (--all-analogs 로 포함 가능)")
    print(f"\ncalc_log_id = {calc_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="cn_ree_2025")
    ap.add_argument("--window", choices=["react", "post"], default="react",
                    help="충격 크기 산정 창: react=발표 후 3거래일(기본), post=20거래일 누적")
    ap.add_argument("--all-analogs", action="store_true",
                    help="clean: false 사건도 유사 사례로 사용 (기본은 clean 사건만)")
    ap.add_argument("--past-only", action="store_true",
                    help="대상 사건보다 먼저 일어난 사건만 유사 사례로 사용 (실제 예측 상황과 같게)")
    ap.add_argument("--list", action="store_true", help="라이브러리 사건 목록만 출력")
    a = ap.parse_args()
    if a.list:
        for e in library.load_events():
            print(f"{e['id']:18s} {e['date']}  {analog.direction(e):8s} clean={e.get('clean')}  {e['title']}")
    else:
        main(a.event, a.window, a.all_analogs, a.past_only)
