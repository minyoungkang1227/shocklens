"""앙상블 Risk Engine과 역스트레스테스트 (네트워크 없음)."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from agents.hq import run_library_event, run_reverse
from engine import ensemble, reverse, risk

from .test_agents import NoLLM, ctx  # noqa: F401

FACTORS = ["risk_off", "dollar", "cny", "oil", "ust10y", "credit", "tech_war"]


# ------------------------------------------------------------------ 앙상블 구성 요소
def test_gpd_pwm_recovers_parameters():
    y = stats.genpareto.rvs(0.25, scale=1.0, size=20000, random_state=1)
    xi, sigma = ensemble.gpd_pwm(y)
    assert abs(xi - 0.25) < 0.05 and abs(sigma - 1.0) < 0.06


def test_weighted_es_matches_unweighted_and_mixture_is_conservative():
    rng = np.random.default_rng(0)
    p = rng.standard_normal(20000) * 0.03
    a = risk.var_es(p, 0.95, 0.975)
    b = ensemble.weighted_var_es(p, np.ones_like(p), 0.95, 0.975)
    assert abs(a["ES"] - b["ES"]) < 1e-9
    calm, wild = rng.standard_normal(20000) * 0.01, rng.standard_normal(20000) * 0.05
    mix = ensemble.weighted_var_es(np.concatenate([calm, wild]), np.ones(40000), 0.95, 0.975)["ES"]
    avg = (risk.var_es(calm, 0.95, 0.975)["ES"] + risk.var_es(wild, 0.95, 0.975)["ES"]) / 2
    assert mix > avg                                   # 경로 섞기는 숫자 평균보다 꼬리를 크게 본다


def test_semiparametric_marginal_tails_extend_beyond_data():
    z = stats.t.rvs(4, size=5000, random_state=3)
    m = ensemble.SemiParametricMarginal(z, 0.05)
    assert m.ppf(np.array([1e-5]))[0] < z.min() and m.ppf(np.array([1 - 1e-5]))[0] > z.max()
    mid = m.ppf(np.array([0.5]))[0]
    assert abs(mid - np.median(z)) < 0.05


def test_brief_uses_ensemble_headline(ctx):
    r = run_library_event("cn_ree_oct_2025", ctx, NoLLM())
    en = r["risk_result"]["ensemble"]
    assert set(en["models"]) == {"bootstrap", "fhs_evt_t", "regime_switch"}
    assert en["range"][0] <= en["ES"] <= en["range"][1] + 1e-9 and en["conservative"] == en["range"][1]
    assert r["verification"]["passed"] and "앙상블" in r["final_brief"]


# ------------------------------------------------------------------ 역스트레스테스트
def test_reverse_closed_form():
    b = pd.Series([0.5, -0.2], index=["a", "c"])
    S = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["a", "c"], columns=["a", "c"])
    out = reverse.solve(b, S, 0.05)
    x = np.array([out["shock"]["a"], out["shock"]["c"]])
    assert abs(b.to_numpy() @ x + 0.05) < 1e-12                 # 정확히 목표 손실
    # 같은 손실을 내는 다른 충격보다 마할라노비스 거리가 작아야 한다
    Si = np.linalg.inv(S.to_numpy())
    alt = np.array([-0.1, 0.0])
    assert x @ Si @ x <= alt @ Si @ alt + 1e-12
    assert abs(out["distance"] - np.sqrt(x @ Si @ x)) < 1e-9


def test_run_reverse_and_execute(ctx):
    rng = np.random.default_rng(5)
    fr = pd.DataFrame(rng.normal(0, 0.01, (len(ctx.rets), 7)), ctx.rets.index, FACTORS)
    fr["risk_off"] = rng.normal(0, 1.0, len(fr))
    c = replace(ctx, frets=fr)
    out = run_reverse(c, loss=0.05, factors=["tech_war", "cny", "risk_off"])
    rv = out["reverse"]
    assert rv["factors"] == ["tech_war", "cny", "risk_off"] and abs(sum(rv["contribution"].values()) + 0.05) < 1e-9
    assert rv["nearest_events"] and 0 <= rv["normal_prob"] <= 1
    ex = run_reverse(c, loss=0.03, execute=True, llm=NoLLM())
    assert ex["run"]["status"] == "ok" and ex["run"]["verification"]["passed"]
    with pytest.raises(ValueError):
        run_reverse(c, target_es=0.0001)
