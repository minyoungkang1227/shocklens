"""계산 엔진 단위 테스트 (합성 데이터, 네트워크 없음)."""
import numpy as np
import pandas as pd
import pytest

from engine import calclog, risk, shock
from pipeline import library, quantiles

FACTORS = ["risk_off", "oil", "tech_war"]
BASKETS = ["eq", "bond"]


def test_var_es_on_known_distribution():
    port = np.linspace(-0.10, 0.10, 2001)       # 균등분포
    m = risk.var_es(port, var_level=0.95, es_level=0.975)
    assert m["VaR"] == pytest.approx(0.09, abs=1e-3)       # 하위 5% 경계
    assert m["ES"] == pytest.approx(0.0975, abs=1e-3)      # 하위 2.5% 평균
    assert m["ES"] > m["VaR"]


def test_block_bootstrap_is_reproducible_and_respects_mask():
    rets = np.arange(100, dtype=float)[:, None] * np.array([[1.0, 0.0]])
    a = risk.block_bootstrap(rets, np.arange(100), 10, 5, 50, np.random.default_rng(1))
    b = risk.block_bootstrap(rets, np.arange(100), 10, 5, 50, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    # 시작일을 0~9로 제한하면 사용되는 날은 0~13 사이 -> 10일 합의 최댓값은 2*(9+10+11+12+13)
    c = risk.block_bootstrap(rets, np.arange(10), 10, 5, 200, np.random.default_rng(2))
    assert c[:, 0].max() <= 2 * (9 + 10 + 11 + 12 + 13)


def test_from_analogs_mild_central_severe():
    f_post = pd.DataFrame({"oil": [0.05, -0.02, 0.10], "tech_war": [-0.03, -0.01, -0.06]},
                          index=["e1", "e2", "e3"])
    s = shock.from_analogs(f_post, ["e1", "e2", "e3"])
    assert s.loc["oil", "mild"] == pytest.approx(-0.02)
    assert s.loc["oil", "central"] == pytest.approx(0.05)
    assert s.loc["oil", "severe"] == pytest.approx(0.10)
    assert s.loc["tech_war", "severe"] == pytest.approx(-0.06)


def test_from_analogs_event_level_keeps_joint_moves():
    # e1: 유가만 크게 오름(손실 작음), e2: 기술전쟁 급락(손실 큼)
    f_post = pd.DataFrame({"oil": [0.10, 0.01], "tech_war": [-0.01, -0.08]}, index=["e1", "e2"])
    beta = pd.DataFrame([[-0.1, 1.0]], index=["eq"], columns=["oil", "tech_war"])
    s = shock.from_analogs(f_post, ["e1", "e2"], beta, pd.Series({"eq": 1.0}))
    # severe는 e2의 팩터 변동 '전체', mild는 e1의 팩터 변동 '전체'
    assert list(s["severe"]) == [0.01, -0.08]
    assert list(s["mild"]) == [0.10, -0.01]


def test_from_severity_uses_direction_and_level():
    q_up = pd.DataFrame({"0.5": [1.0, 0.0, 0.0], "0.8": [2.0, 0.0, 0.0], "0.95": [3.0, 0.0, 0.0],
                         "0.99": [4.0, 0.0, 0.0], "0.995": [5.0, 0.0, 0.0]}, index=FACTORS)
    q_down = -q_up
    qmap = {1: 0.50, 2: 0.80, 3: 0.95, 4: 0.99, 5: 0.995}
    s = shock.from_severity({"risk_off": 1, "oil": -1}, 3, q_up, q_down, qmap, FACTORS)
    assert list(s.loc["risk_off"]) == [2.0, 3.0, 4.0]       # 강도 2,3,4
    assert list(s.loc["oil"]) == [0.0, 0.0, 0.0]            # q_down의 oil 행은 0
    assert (s.loc["tech_war"] == 0).all()                   # 채널에 없는 팩터는 0
    s5 = shock.from_severity({"risk_off": 1}, 5, q_up, q_down, qmap, FACTORS)
    assert s5.loc["risk_off", "severe"] == 5.0              # 5를 넘지 않음


def test_apply_overrides():
    s = pd.DataFrame(1.0, index=FACTORS, columns=shock.SCENARIOS)
    o = shock.apply_overrides(s, {"oil": {"mult": 2}, "tech_war": {"abs": -0.5}})
    assert (o.loc["oil"] == 2.0).all() and (o.loc["tech_war"] == -0.5).all()
    assert (o.loc["risk_off"] == 1.0).all()
    with pytest.raises(KeyError):
        shock.apply_overrides(s, {"gold": {"mult": 2}})


def test_negative_shock_raises_es_and_contributions_sum_to_one():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    rets = pd.DataFrame(rng.normal(0, [0.01, 0.003], size=(1500, 2)), index=idx, columns=BASKETS)
    beta = pd.DataFrame([[-0.01, 0.0, 1.0], [0.002, 0.0, 0.0]], index=BASKETS, columns=FACTORS)
    f = pd.DataFrame({"mild": [5, 0, -0.02], "central": [10, 0, -0.05], "severe": [20, 0, -0.10]}, index=FACTORS)
    w = pd.Series({"eq": 60, "bond": 40})
    out = risk.run_stress(rets, w, beta, f, None, None, 20, 5, 4000, 0.95, 0.975, 7)
    assert out["stressed"]["ES"] > out["baseline"]["ES"]
    assert sum(out["tail_contribution"].values()) == pytest.approx(1.0, abs=1e-4)
    s = out["scenario_loss"]
    assert s["severe"] < s["central"] < s["mild"] < 0       # 충격이 클수록 손실이 큼
    # 같은 시드면 같은 결과
    out2 = risk.run_stress(rets, w, beta, f, None, None, 20, 5, 4000, 0.95, 0.975, 7)
    assert out2["stressed"] == out["stressed"]


def test_event_windows_detects_jump():
    idx = pd.bdate_range("2024-01-01", periods=60)
    frets = pd.DataFrame(0.0, index=idx, columns=["oil"])
    rets = pd.DataFrame(0.0, index=idx, columns=["eq"])
    frets.iloc[30] = 0.1                     # 사건일에 유가 +0.1
    rets.iloc[30] = -0.02                    # 주식 -2%, 그중 베타로 설명되는 부분은 -0.5 * 0.1 = -0.05
    beta = pd.DataFrame([[-0.5]], index=["eq"], columns=["oil"])
    events = [{"id": "x", "date": str(idx[30].date())}]
    fp, fpre, bp, resid = library.event_windows(events, rets, frets, beta, 5, 20)
    assert fp.loc["x", "oil"] == pytest.approx(0.1)
    assert bp.loc["x", "eq"] == pytest.approx(-0.02)
    assert resid.loc["x", "eq"] == pytest.approx(-0.02 - (-0.05))
    assert fpre.loc["x", "oil"] == pytest.approx(0.0)


def test_quantile_tables_shapes_and_order():
    rng = np.random.default_rng(3)
    frets = pd.DataFrame(rng.normal(size=(1000, 2)), columns=["a", "b"])
    up, down = quantiles.quantile_tables(frets, 20)
    assert (up[0.995] > up[0.95]).all() and (down[0.995] < down[0.95]).all()


def test_calclog_roundtrip(tmp_path):
    cid = calclog.save({"x": 1}, {"ES": np.float64(0.031), "nested": {"v": [1, 2]}}, "2026-10-30", tmp_path)
    rec = calclog.load(cid, tmp_path)
    assert rec["outputs"]["ES"] == pytest.approx(0.031)
    assert set(calclog.numbers(rec["outputs"])) == {0.031, 1.0, 2.0}
