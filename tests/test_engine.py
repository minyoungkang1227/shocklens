"""계산 엔진 단위 테스트 (합성 데이터, 네트워크 없음)."""
import numpy as np
import pandas as pd
import pytest

from engine import calclog, risk, shock
from engine.exposure import exposure_for
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


def test_react_window_sums_only_first_days():
    idx = pd.bdate_range("2024-01-01", periods=60)
    frets = pd.DataFrame(0.0, index=idx, columns=["oil"])
    rets = pd.DataFrame(0.0, index=idx, columns=["eq"])
    frets.iloc[30:33, 0] = [0.05, 0.03, 0.02]   # 발표 후 3일
    frets.iloc[33:50, 0] = 0.01                 # 그 뒤의 일반 흐름
    beta = pd.DataFrame([[0.0]], index=["eq"], columns=["oil"])
    events = [{"id": "x", "date": str(idx[30].date())}]
    f3, _, _, _ = library.event_windows(events, rets, frets, beta, 5, 3)
    f20, _, _, _ = library.event_windows(events, rets, frets, beta, 5, 20)
    assert f3.loc["x", "oil"] == pytest.approx(0.10)
    assert f20.loc["x", "oil"] == pytest.approx(0.10 + 0.01 * 17)


def test_decomposition_regime_and_event_effects():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2015-01-01", periods=2000)
    vol = np.where(np.arange(2000) % 10 == 0, 0.04, 0.008)          # 10일에 하루는 고변동
    rets = pd.DataFrame(rng.normal(0, 1, size=(2000, 2)) * vol[:, None], index=idx, columns=BASKETS)
    beta = pd.DataFrame([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], index=BASKETS, columns=FACTORS)
    f = pd.DataFrame({"mild": [0, 0, -0.01], "central": [0, 0, -0.03], "severe": [0, 0, -0.06]}, index=FACTORS)
    w = pd.Series({"eq": 60, "bond": 40})
    # 국면 표본 없음 -> 국면 효과 0
    o1 = risk.run_stress(rets, w, beta, f, None, None, 20, 5, 3000, 0.95, 0.975, 1)
    assert o1["decomposition"]["regime_effect_ES"] == 0.0
    assert o1["decomposition"]["event_effect_ES"] > 0
    # 고변동 날짜만 국면 표본으로 -> 국면 효과 양수
    mask = pd.Series(vol > 0.01, index=idx)
    o2 = risk.run_stress(rets, w, beta, f, None, mask, 20, 5, 3000, 0.95, 0.975, 1)
    d = o2["decomposition"]
    assert d["regime_effect_ES"] > 0
    assert d["total_change_ES"] == pytest.approx(d["regime_effect_ES"] + d["event_effect_ES"])


def test_select_analogs_excludes_target_and_unclean():
    events = [{"id": "a", "clean": True}, {"id": "b", "clean": False}, {"id": "c", "clean": True}, {"id": "d"}]
    avail = ["a", "b", "c", "d"]
    assert library.select_analogs(avail, events, "a") == ["c"]
    assert library.select_analogs(avail, events, "a", clean_only=False) == ["b", "c", "d"]


def test_idio_shock_removes_common_market_move():
    # 사건 e1에서 모든 바스켓이 +2% 같이 올랐고, 반도체만 추가로 -3% (= 상대적으로 더 맞음)
    resid = pd.DataFrame({"a": [0.02], "b": [0.02], "c": [0.02], "semis": [-0.01]}, index=["e1"])
    out = shock.idio_shock({"semis": "high", "a": "none"}, resid, ["e1"], ["a", "b", "c", "semis"])
    assert out.loc["semis", "central"] == pytest.approx(-0.03)   # 공통 +2% 제거
    assert (out.loc["a"] == 0).all()


def test_exposure_caps_country_for_sector_targeted_controls():
    emap = {"us_equity": {"countries": {"US": "high"}, "sectors": {"semiconductor": "mid"}},
            "semis": {"countries": {"US": "mid"}, "sectors": {"semiconductor": "high"}},
            "bonds": {"countries": {"US": "mid"}, "sectors": {}}}
    ctrl = {"initiator": "CN", "target": ["US"], "sectors": ["semiconductor"], "instrument": "resource_export_control"}
    tariff = {"initiator": "US", "target": ["CN"], "sectors": [], "instrument": "tariff"}
    e1 = exposure_for(ctrl, emap)
    assert e1 == {"us_equity": "mid", "semis": "high", "bonds": "mid"}
    e2 = exposure_for(tariff, emap)
    assert e2["us_equity"] == "high"


def test_channel_filter_zeroes_non_channel_and_wrong_sign():
    f = pd.DataFrame({"risk_off": [2.0, -1.0], "oil": [0.05, 0.03], "tech_war": [-0.02, -0.04]},
                     index=["e1", "e2"])
    out, flipped = shock.channel_filter(f, {"risk_off": 1, "tech_war": -1})
    assert (out["oil"] == 0).all()                         # 채널 밖 팩터는 0
    assert out.loc["e2", "risk_off"] == 0                  # 기대(+)와 반대로 움직인 값은 0
    assert out.loc["e1", "risk_off"] == 2.0
    assert flipped == [("e2", "risk_off", -1.0)]
    keep, _ = shock.channel_filter(f, {"risk_off": 1}, clip_wrong_sign=False)
    assert keep.loc["e2", "risk_off"] == -1.0


def _ev(i, inst, sectors, init, target, stage, clean=True, direction="escalate", date="2020-01-01"):
    return {"id": i, "instrument": inst, "sectors": sectors, "initiator": init, "target": target,
            "stage": stage, "clean": clean, "direction": direction, "date": date}


def test_analog_scores_and_filters():
    from engine import analog
    cfg = {"weights": {"instrument": 0.4, "sector": 0.3, "countries": 0.2, "stage": 0.1},
           "direct_min": 0.7, "relaxed_min": 0.4, "top_k": 3}
    tgt = _ev("t", "resource_export_control", ["semiconductor"], "CN", ["US"], "announce", date="2023-07-03")
    lib = [tgt,
           _ev("same", "resource_export_control", ["semiconductor"], "CN", ["US"], "announce"),     # 1.0
           _ev("family", "tech_export_control", ["semiconductor"], "US", ["CN"], "enforce"),         # .2+.3+.2+.05
           _ev("dirty", "resource_export_control", ["semiconductor"], "CN", ["US"], "announce", clean=False),
           _ev("ease", "resource_export_control", ["semiconductor"], "CN", ["US"], "announce", direction="ease"),
           _ev("far", "financial_sanction", ["banks"], "US", ["RU"], "enforce")]
    ids, tier, ranked = analog.select(tgt, lib, [e["id"] for e in lib], cfg)
    assert ranked[0]["id"] == "same" and ranked[0]["score"] == pytest.approx(1.0)
    assert ranked[1]["score"] == pytest.approx(0.75)
    assert "dirty" not in ids and "ease" not in ids and "t" not in ids
    assert "far" not in ids                                 # 점수 0.05 < relaxed_min
    assert ids == ["same", "family"] and tier == "direct"
    # 미래 사건 제외
    ids2, _, _ = analog.select(tgt, lib, [e["id"] for e in lib], cfg, past_only=True)
    assert ids2 == ["same", "family"]                       # 둘 다 2020년이라 그대로
    ids3, tier3, _ = analog.select({**tgt, "date": "2019-01-01"}, lib, [e["id"] for e in lib], cfg, past_only=True)
    assert ids3 == [] and tier3 == "synthetic"


def test_library_file_is_consistent():
    events = library.load_events()
    ids = [e["id"] for e in events]
    assert len(ids) == len(set(ids)) and len(ids) >= 10
    from engine.analog import FAMILY
    for e in events:
        assert e["instrument"] in FAMILY, e["id"]
        assert e.get("direction", "escalate") in {"escalate", "ease"}
        assert isinstance(e.get("clean"), bool), e["id"]
        assert set(e["channels"]) <= {"risk_off", "dollar", "cny", "oil", "ust10y", "credit", "tech_war"}
        assert len(e["sources"]) >= 1
