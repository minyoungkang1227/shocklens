"""보조 위기 지수(OFR 금융스트레스·AI-GPR) 수집과 국면 판단 테스트 (네트워크 없음, 2026-11-13)."""
import numpy as np
import pandas as pd

from engine import regime
from pipeline import indicators

from .test_agents import ctx  # noqa: F401  (fixture)

DATES = pd.bdate_range("2015-01-01", "2026-11-12")


def _csv(cols: dict) -> bytes:
    return pd.DataFrame(cols).to_csv(index=False).encode()


def test_parse_ofr_with_components_and_unknown_headers():
    raw = _csv({"Date": DATES.strftime("%Y-%m-%d"), "OFR FSI": np.linspace(-2, 1, len(DATES)),
                "Credit": 0.1, "Volatility": 0.2, "United States": 0.3})
    df, memo = indicators.parse_csv(raw, "OFR_FSI", ["^ofr ?fsi$", "^fsi$"], keep_all_numeric=True)
    assert memo["value_column"] == "OFR FSI" and "OFR_FSI_CREDIT" in df and "OFR_FSI_UNITED_STATES" in df
    assert df.index[0] == pd.Timestamp("2015-01-01")


def test_parse_gpr_falls_back_to_first_numeric_and_keeps_matching_components():
    raw = _csv({"day": DATES.strftime("%m/%d/%Y"), "index_value": 100.0, "threat_component": 50.0, "noise": 1.0})
    df, memo = indicators.parse_csv(raw, "AI_GPR", ["^ai[_ ]?gpr$"], ["threat", "act"])
    assert memo["value_column"] == "index_value"                         # 이름이 안 맞으면 첫 숫자 열
    assert list(df.columns) == ["AI_GPR", "AI_GPR_THREAT_COMPONENT"]      # 하위 지수 패턴만 같이 저장


def test_run_survives_one_failed_source(tmp_path, monkeypatch):
    monkeypatch.setattr(indicators, "load_cfg", lambda: {
        "ofr_fsi": {"urls": ["bad", "ok_ofr"], "value_patterns": ["fsi"]},
        "ai_gpr": {"urls": ["broken"], "value_patterns": ["gpr"]}})

    def getter(u):
        if u == "ok_ofr":
            return _csv({"date": DATES.strftime("%Y-%m-%d"), "fsi": 0.0})
        raise ConnectionError(u)
    rep = indicators.run(getter, tmp_path / "s.parquet")
    assert rep["ofr_fsi"]["url"] == "ok_ofr" and "error" in rep["ai_gpr"]
    assert list(pd.read_parquet(tmp_path / "s.parquet").columns) == ["OFR_FSI"]


def _vix(calm=True):
    v = pd.Series(15.0 + np.sin(np.arange(len(DATES))) , index=DATES)
    if not calm:
        v.iloc[-10:] = 60.0
    return v


def test_ofr_alone_can_trigger_crisis_and_gpr_only_reports():
    stress = pd.DataFrame({"OFR_FSI": np.r_[np.zeros(len(DATES) - 10), np.full(10, 5.0)],
                           "AI_GPR": np.r_[np.full(len(DATES) - 10, 100.0), np.full(10, 400.0)]}, index=DATES)
    cfg = {"mode": "market", "crisis_vix_quantile": 0.8, "lookback_days": 5,
           "stress": {"ofr": True, "ofr_quantile": 0.8, "gpr_trigger": False, "gpr_quantile": 0.9}}
    base = regime.decide({}, _vix(), cfg, 4)
    assert base["regime"] == "normal" and "triggers" not in base
    r = regime.decide({}, _vix(), cfg, 4, stress=stress)
    assert r["regime"] == "crisis" and r["triggers"] == ["OFR 금융스트레스"]      # VIX 조용해도 신용·자금시장 위기는 잡음
    assert r["gpr"]["high"] and "AI 지정학 리스크" not in r["triggers"]            # GPR은 기본 보고만
    cfg["stress"]["gpr_trigger"] = True
    assert "AI 지정학 리스크" in regime.decide({}, _vix(), cfg, 4, stress=stress)["triggers"]


def test_stale_indicator_is_excluded_and_no_stress_keeps_old_behaviour():
    old = pd.DataFrame({"OFR_FSI": np.r_[np.zeros(len(DATES) - 40), np.full(10, 5.0)]},
                       index=DATES[:-30])                                     # VIX보다 30영업일 이상 오래됨
    cfg = {"mode": "market", "crisis_vix_quantile": 0.8, "lookback_days": 5, "stress": {"max_stale_days": 10}}
    r = regime.decide({}, _vix(), cfg, 4, stress=old)
    assert r["regime"] == "normal" and r["ofr"]["stale"] and "판단에서 제외" in r["reason"]
    a = regime.decide({}, _vix(calm=False), cfg, 4)
    b = regime.decide({}, _vix(calm=False), cfg, 4, stress=None)
    assert a == b and a["regime"] == "crisis"


def test_asof_ignores_future_stress():
    stress = pd.DataFrame({"OFR_FSI": np.r_[np.zeros(len(DATES) - 10), np.full(10, 5.0)]}, index=DATES)
    cfg = {"mode": "market", "crisis_vix_quantile": 0.8, "lookback_days": 5, "stress": {}}
    r = regime.decide({}, _vix(), cfg, 4, asof="2026-06-01", stress=stress)        # 백테스트: 그 날짜까지만 본다
    assert r["regime"] == "normal" and not r["ofr"]["high"]


def test_end_to_end_brief_with_stress_passes_verifier(ctx):
    from agents.hq import run_news
    from .test_agents import REE_NEWS, NoLLM
    idx = ctx.levels.index
    ctx.stress = pd.DataFrame({"OFR_FSI": np.r_[np.zeros(len(idx) - 10), np.full(10, 5.0)],
                               "AI_GPR": 100.0 + np.arange(len(idx)) * 0.0}, index=idx)
    st = run_news(REE_NEWS, ctx, NoLLM())
    assert st["status"] == "ok" and st["verification"]["passed"]
    assert st["regime_info"]["regime"] == "crisis" and "OFR 금융스트레스" in st["regime_info"]["reason"]
    assert "OFR 금융스트레스" in st["final_brief"]
