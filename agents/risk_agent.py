"""Risk Engine 에이전트: 충격 -> ES97.5/VaR95 분해, 계산 장부(calc_log_id) 기록 (LLM 없음).

브리핑에 들어갈 모든 숫자는 여기서 나오고, Reporter·Verifier는 이 결과만 참조한다.
"""
from __future__ import annotations

from engine import calclog, risk

from .context import Context
from .shock_mapper import to_frame


def run(state: dict, ctx: Context) -> dict:
    r = ctx.thresholds["risk"]
    f_shock = to_frame(state["factor_shock"])
    idio = to_frame(state["idio_shock"]) if state.get("idio_shock") else None
    regime = state["beta_regime"]
    beta = ctx.beta_crisis if regime == "crisis" else ctx.beta_normal
    mask = None
    if regime == "crisis":
        vix = ctx.levels[ctx.vix_col]
        mask = vix >= vix.quantile(ctx.thresholds["betas"]["crisis_vix_quantile"])
    seed = state.get("seed", r["seed"])
    out = risk.run_stress(ctx.rets, ctx.weights, beta, f_shock, idio, mask, r["horizon_days"], r["block_len"],
                          r["n_paths"], r["var_level"], r["es_level"], seed)
    base_paths = out.pop("_base_regime_paths", None)
    ecfg = r.get("ensemble") or {}
    if ecfg.get("enabled", True) and base_paths is not None:
        from engine import ensemble
        from engine.risk import basket_shock
        vix = ctx.levels[ctx.vix_col]
        crisis_state = vix >= vix.quantile(ctx.thresholds["betas"]["crisis_vix_quantile"])
        bsh = basket_shock(beta.reindex(ctx.rets.columns), f_shock, idio)
        out["ensemble"] = ensemble.run(ctx.rets, ctx.weights, bsh, base_paths, crisis_state, regime, ecfg,
                                       r["horizon_days"], ecfg.get("n_paths", r["n_paths"]), r["var_level"],
                                       r["es_level"], seed)
    top = sorted(out["tail_contribution"].items(), key=lambda kv: -kv[1])[:3]
    try:
        from engine.structure import structure
        from pipeline import config
        jcfg = config.load_yaml("jurisdictions.yaml") or {}
        out["structure"] = structure({k: float(v) for k, v in ctx.weights.items()}, out["tail_contribution"],
                                     state["event"], jcfg)
    except FileNotFoundError:
        out["structure"] = {}
    result = {
        "scenario_loss": out["scenario_loss"],
        "scenario_loss_factor_only": out["scenario_loss_factor_only"],
        "baseline_normal": out["baseline_normal"],
        "baseline": out["baseline"],
        "stressed": out["stressed"],
        "stressed_mild": out["stressed_mild"],
        "stressed_severe": out["stressed_severe"],
        "decomposition": out["decomposition"],
        "regime_info": state.get("regime_info", {}),
        "normal_mc_ES": out["normal_mc_stressed"]["ES"],
        "es_underestimation_normal": out["es_underestimation_normal"],
        "tail_top3": [[b, float(c)] for b, c in top],
        "regime": regime,
        "horizon_days": r["horizon_days"],
        "structure": out.get("structure", {}),
        "ensemble": out.get("ensemble", {}),
    }
    calc_id = calclog.save(
        inputs={"run_id": state.get("run_id"), "event": state["event"],
                "analogs": [a["id"] for a in state.get("analogs", [])], "tier": state.get("analog_tier"),
                "factor_shock": state["factor_shock"], "exposure": state.get("exposure"),
                "whatif_overrides": state.get("whatif_overrides"), "seed": seed},
        outputs=out, data_version=ctx.data_version)
    return {"risk_result": result, "calc_log_id": calc_id}
