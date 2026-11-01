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
from . import calclog, risk, shock


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
        "resid": pd.read_csv(m / "event_resid_post.csv", index_col=0),
        "q_up": pd.read_csv(m / "factor_quantiles_up.csv", index_col=0),
        "q_down": pd.read_csv(m / "factor_quantiles_down.csv", index_col=0),
    }


def exposure_for(event: dict, exposure_map: dict) -> dict[str, str]:
    """Exposure Analyst의 간이판: 사건의 국가·산업과 매핑표를 대조해 바스켓별 최고 등급을 고른다."""
    rank = {"none": 0, "mid": 1, "high": 2}
    keys_c = {event["initiator"], *event.get("target", [])}
    keys_s = set(event.get("sectors", []))
    out = {}
    for b, m in exposure_map.items():
        best = "none"
        for c, lvl in m.get("countries", {}).items():
            if c in keys_c and rank[lvl] > rank[best]:
                best = lvl
        for s, lvl in m.get("sectors", {}).items():
            if s in keys_s and rank[lvl] > rank[best]:
                best = lvl
        out[b] = best
    return out


def main(event_id: str) -> None:
    th = config.thresholds()
    pf = config.portfolio()["baskets"]
    weights = pd.Series({b: v["weight"] for b, v in pf.items()}, dtype=float)
    with open(config.CONFIG_DIR / "exposure_map.yaml", encoding="utf-8") as f:
        exposure_map = yaml.safe_load(f)

    md = load_model()
    events = {e["id"]: e for e in library.load_events()}
    ev = events[event_id]
    analogs = [i for i in md["f_post"].index if i != event_id]

    regime = "crisis" if ev["severity"] >= th["severity"]["crisis_regime_min"] else "normal"
    beta = md["beta_crisis"] if regime == "crisis" else md["beta_normal"]
    if analogs:
        tier = "direct(leave-one-out)"
        f_shock = shock.from_analogs(md["f_post"], analogs, beta, weights)
    else:
        tier = "synthetic"
        f_shock = shock.from_severity(ev["channels"], ev["severity"], md["q_up"], md["q_down"],
                                      th["severity"]["quantile_map"], list(md["f_post"].columns))

    exposure = exposure_for(ev, exposure_map)
    idio = shock.idio_shock(exposure, md["resid"], analogs, list(md["rets"].columns))

    vix = md["levels"][config.factors()["factors"]["risk_off"]["series"]]
    mask = vix >= vix.quantile(th["betas"]["crisis_vix_quantile"]) if regime == "crisis" else None
    r = th["risk"]
    out = risk.run_stress(md["rets"], weights, beta, f_shock, idio, mask, r["horizon_days"], r["block_len"],
                          r["n_paths"], r["var_level"], r["es_level"], r["seed"])
    calc_id = calclog.save(
        inputs={"event_id": event_id, "analogs": analogs, "tier": tier, "regime": regime,
                "factor_shock": f_shock.to_dict(), "exposure": exposure},
        outputs=out, data_version=str(md["rets"].index.max().date()))

    # 실제로 그 사건 때 포트폴리오가 어땠는지 (leave-one-out 비교용)
    actual = None
    if event_id in md["b_post"].index:
        w = weights / weights.sum()
        actual = float(risk.portfolio_return(md["b_post"].loc[[event_id], w.index].to_numpy(), w.to_numpy())[0])

    pct = lambda x: f"{x * 100:+.2f}%"
    print(f"\n=== {ev['title']} ===")
    print(f"경로: {tier} | 유사 사건: {analogs} | 베타: {regime}")
    print("\n[팩터 충격, 20일 누적]")
    print(f_shock.round(4).to_string())
    print("\n[직접 노출 high 바스켓]", [b for b, l in exposure.items() if l == "high"])
    print("\n[결정론적 시나리오 포트폴리오 수익률]")
    for k, v in out["scenario_loss"].items():
        print(f"  {k:8s} {pct(v)}")
    print("\n[20일 손실 분포] (손실을 양수로 표시)")
    print(f"  평상시   VaR95 {out['baseline']['VaR']:.2%}  ES97.5 {out['baseline']['ES']:.2%}")
    print(f"  충격 후  VaR95 {out['stressed']['VaR']:.2%}  ES97.5 {out['stressed']['ES']:.2%}")
    print(f"  정규MC   ES97.5 {out['normal_mc_stressed']['ES']:.2%}  (과소평가 {out['es_underestimation_normal']:.2%}p)")
    top = sorted(out["tail_contribution"].items(), key=lambda kv: -kv[1])[:3]
    print("\n[ES 구간 손실 기여 상위 3]", ", ".join(f"{b} {c:.0%}" for b, c in top))
    if actual is not None:
        print(f"\n[실제] 그 사건 후 20일 포트폴리오 수익률: {pct(actual)}  (central 시나리오 {pct(out['scenario_loss']['central'])})")
    print(f"\ncalc_log_id = {calc_id}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="cn_ree_2025")
    main(ap.parse_args().event)
