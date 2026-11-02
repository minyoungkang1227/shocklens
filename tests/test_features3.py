"""11/2 3차: 확장 지경학 분석 (수단 확장, 시나리오, 주의도, 공급망, 구조 점검, 장기 효과) — 네트워크 없음."""
import numpy as np
import pandas as pd

from agents import event_analyst, followup
from agents.exposure_analyst import apply_supply_chain
from agents.hq import parse_whatif, run_scenario, run_whatif
from engine import attention, structure
from engine.long_window import excess_table
from pipeline import config

from .test_agents import NoLLM, ctx  # noqa: F401


# ------------------------------------------------------------------ 수단 확장
def test_new_instruments_rule_extraction():
    cases = {
        "중국이 대만 해협 봉쇄를 선언했다. 미국 반도체 기업 공급망이 위협받는다.": "chokepoint_disruption",
        "미국 정부는 중국 첨단기술 기업에 대한 해외 투자 제한 규칙을 발표했다.": "investment_restriction",
        "미국 의회가 반도체 보조금과 세액 공제를 담은 법안을 통과시켜 중국 공장 증설을 제한했다.": "industrial_subsidy",
    }
    for text, inst in cases.items():
        assert event_analyst.analyze(text, NoLLM())["event"]["instrument"] == inst, text


def test_library_has_law_events():
    from pipeline import library
    ids = {e["id"]: e for e in library.load_events()}
    assert ids["us_chips_2022"]["instrument"] == "industrial_subsidy" and ids["us_chips_2022"]["clean"] is False
    assert ids["us_outbound_2023"]["instrument"] == "investment_restriction"


# ------------------------------------------------------------------ 시나리오 모드
def test_scenarios_config_valid():
    scs = config.scenarios()
    assert len(scs) >= 6 and len({s["id"] for s in scs}) == len(scs)


def test_run_scenario_forced_regime_and_warning(ctx):
    r = run_scenario("sc_taiwan_blockade", ctx, NoLLM())
    assert r["status"] == "ok" and r["verification"]["passed"]
    assert r["beta_regime"] == "crisis" and "가상 시나리오" in r["final_brief"]
    assert r["event"]["severity"] == 5
    w = run_whatif(r, "평상시 국면이라면?", ctx, NoLLM())
    assert w["beta_regime"] == "normal" and w["status"] == "ok"


def test_custom_scenario_dict(ctx):
    sc = {"id": "my", "title": "(가정) 테스트", "description": "테스트 시나리오", "instrument": "tariff",
          "initiator": "US", "target": ["CN"], "sectors": ["autos"],
          "severity_components": {k: 2 for k in ("scope", "immediacy", "substitut", "escalation", "parties", "surprise")},
          "channels": {"risk_off": 1}}
    r = run_scenario(sc, ctx, NoLLM(), regime="normal")
    assert r["status"] == "ok" and r["beta_regime"] == "normal" and r["event"]["severity"] <= 2


def test_whatif_regime_parse():
    assert parse_whatif("위기 국면이라면?") == {"_regime": "crisis"}
    assert parse_whatif("위기 국면에서 유가 2배") == {"_regime": "crisis", "oil": {"mult": 2.0}}


# ------------------------------------------------------------------ 주의도
def test_attention_measure_and_interpret():
    pts = [(f"d{i}", 1.0) for i in range(60)] + [(f"r{i}", 5.0) for i in range(7)]
    m = attention.measure(pts)
    assert m["percentile"] == 100.0 and m["ratio"] == 5.0
    note = attention.interpret(m)
    assert "의외성" in note and "%" not in note             # 브리핑 숫자 검증과 충돌하지 않게 % 없음
    assert attention.measure(pts[:10]) is None
    q = attention.query_for({"instrument": "resource_export_control", "initiator": "CN", "target": ["US"]})
    assert "rare earth" in q and "China" in q


# ------------------------------------------------------------------ 공급망·구조 점검
def test_supply_chain_raises_exposure():
    grades = {"semis": "none", "jp_equity": "none", "us_equity": "none", "eu_equity": "none", "tw_equity": "none"}
    reasons = {}
    ev = {"instrument": "resource_export_control", "initiator": "CN", "target": ["US"], "sectors": ["critical_minerals"]}
    hit = apply_supply_chain(ev, grades, reasons, {})
    assert "희토류·영구자석" in hit and grades["semis"] == "high" and grades["jp_equity"] == "mid"
    assert "공급망" in reasons["semis"]


def test_structure_blocs_and_adverse_share():
    jcfg = config.load_yaml("jurisdictions.yaml")
    w = {b: v["weight"] for b, v in config.portfolio()["baskets"].items()}
    st = structure.structure(w, {"us_equity": 0.6, "cn_equity": 0.4}, {"initiator": "US", "target": ["CN"]}, jcfg)
    assert abs(sum(st["bloc_shares"].values()) - 1) < 1e-6
    assert abs(st["adverse_share"] - st["country_shares"]["CN"]) < 1e-9 and st["adverse_countries"] == ["CN"]
    assert st["effective_countries"] > 1 and abs(sum(st["tail_by_bloc"].values()) - 1) < 1e-6


def test_structure_in_brief_and_followup(ctx):
    r = run_scenario("sc_cn_financial_sanction", ctx, NoLLM())
    assert r["verification"]["passed"]
    assert "구조 점검(분절화)" in r["final_brief"] and "상대 블록 관할(중국)" in r["final_brief"]
    a = followup.answer("블록별 비중은?", r, ctx, NoLLM())
    assert a["tool_calls"][0]["name"] == "get_structure"
    assert followup.check_numbers(a["answer"], [c["output"] for c in a["tool_calls"]]) == []


# ------------------------------------------------------------------ 장기 효과
def test_long_window_excess_table():
    idx = pd.bdate_range("2020-01-01", periods=400)
    rets = pd.DataFrame({"us_equity": np.zeros(400), "semis": np.zeros(400)}, idx)
    rets.iloc[200:260, 1] = 0.002                          # 이벤트 후 60일 동안만 초과수익
    t = excess_table(rets, str(idx[200].date()), (20, 60, 120))
    assert abs(t.loc["semis", (60, "excess")] - 0.12) < 1e-9
    assert t.loc["semis", (60, "pct")] > 90
    assert np.isnan(excess_table(rets, "2030-01-01").loc["semis", (20, "excess")])
