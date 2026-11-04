import pytest
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


def test_scenario_from_text_and_run(ctx):
    from agents.hq import scenario_from_text
    sc = scenario_from_text("만약 중국이 대만 해협을 봉쇄해 반도체 공급이 끊긴다면?", ctx, NoLLM())
    assert sc["instrument"] == "chokepoint_disruption" and sc["initiator"] == "CN" and sc["title"].startswith("(가정)")
    sc["target"] = ["TW", "US"]
    sc["channels"] = {}                                   # 화면에서 수단을 바꾼 경우: 기본 채널로 채워져야 함
    r = run_scenario(sc, ctx, NoLLM(), regime="crisis")
    assert r["status"] == "ok" and r["verification"]["passed"] and r["event"]["channels"]
    import pytest
    with pytest.raises(ValueError):
        scenario_from_text("내일 날씨가 좋으면 좋겠다", ctx, NoLLM())


def test_carry_calc_tags_fixes_paragraph_with_tag_only_at_end():
    from agents.reporter import carry_calc_tags
    from agents.verifier import _sentences, TAG
    draft = ("기준 모형 분해에서 ① {{ES_NORMAL}}, ② {{ES_BASE}}이고, 사건 효과는 {{EVENT_EFFECT}}입니다. "
             "사건 효과가 양수이므로 위험이 커진 방향입니다 [calc].\n"
             "근거 경로는 완화된 유사 사례입니다. relaxed는 강도가 다른 사건을 낮은 가중치로 쓴 경우입니다 [calc].\n"
             "중국이 수출통제를 확대했습니다. 미국이 추가 관세를 위협했습니다 [src:s1].")
    out, n = carry_calc_tags(draft)
    lines = out.splitlines()
    assert all(TAG.search(s) for s in _sentences("\n".join(lines[:2])))
    assert n == 2
    # 뉴스 출처 태그는 물려주지 않는다
    assert lines[2].startswith("중국이 수출통제를 확대했습니다. ")


def _lib():
    base = dict(direction="escalate", initiator="CN", target=["US"], sectors=["semiconductor"], stage="escalate",
                severity=4, severity_points=70)
    return [
        {**base, "id": "a_clean", "date": "2023-01-01", "instrument": "resource_export_control", "clean": True},
        {**base, "id": "b_macro", "date": "2023-02-01", "instrument": "resource_export_control", "clean": False,
         "confounder_type": "macro"},
        {**base, "id": "c_other", "date": "2023-03-01", "instrument": "resource_export_control", "clean": False,
         "confounder_type": "other"},
        {**base, "id": "d_family", "date": "2023-04-01", "instrument": "entity_list", "sectors": ["autos"],
         "initiator": "US", "target": ["CN"], "stage": "announce", "clean": True},
    ]


def test_partial_unclean_weight_and_other_excluded():
    from engine import analog
    cfg = {"direct_min": 0.7, "relaxed_min": 0.4, "top_k": 5, "partial_unclean_weight": 0.5,
           "weights": {"instrument": 0.4, "sector": 0.3, "countries": 0.2, "stage": 0.1}}
    target = {"id": "new", "date": "2025-01-01", "instrument": "resource_export_control", "direction": "escalate",
              "initiator": "CN", "target": ["US"], "sectors": ["semiconductor"], "stage": "escalate",
              "severity": 4, "severity_points": 70}
    lib = _lib()
    ids, tier, ranked = analog.select(target, lib, [e["id"] for e in lib], cfg)
    w = {r["id"]: r for r in ranked if r["id"] in ids}
    assert "a_clean" in ids and "b_macro" in ids and "c_other" not in ids
    assert w["b_macro"]["weight"] == pytest.approx(w["a_clean"]["weight"] * 0.5)
    assert tier == "direct"
    # 계수 0이면 예전처럼 clean만
    ids0, _, _ = analog.select(target, lib, [e["id"] for e in lib], {**cfg, "partial_unclean_weight": 0.0})
    assert "b_macro" not in ids0


def test_pool_tier_before_synthetic():
    from engine import analog
    cfg = {"direct_min": 0.7, "relaxed_min": 0.4, "top_k": 5, "partial_unclean_weight": 0.5, "pool_tier": True,
           "weights": {"instrument": 0.4, "sector": 0.3, "countries": 0.2, "stage": 0.1}}
    # 구조가 거의 안 겹치는 새 사건: 같은 계열(control)이지만 산업·국가·단계가 다름
    target = {"id": "new", "date": "2025-01-01", "instrument": "investment_restriction", "direction": "escalate",
              "initiator": "EU", "target": ["RU"], "sectors": ["energy"], "stage": "enforce",
              "severity": 3, "severity_points": 50}
    lib = _lib()
    ids, tier, ranked = analog.select(target, lib, [e["id"] for e in lib], cfg)
    assert tier == "pool" and set(ids) == {"a_clean", "b_macro", "d_family"}
    ids2, tier2, _ = analog.select(target, lib, [e["id"] for e in lib], {**cfg, "pool_tier": False})
    assert tier2 == "synthetic" and ids2 == []


def test_blind_text_drops_aftermath_and_asof_context_uses_past_only():
    from agents.relabel_severity import announcement_text
    ev = {"date": "2025-10-09", "title": "중국, 희토류 수출통제 대폭 확대",
          "description": "중국산 희토류가 0.1% 이상 들어간 제품까지 허가 대상으로 넓힘. 다음 날(10/10) 트럼프가 100% 추가 관세를 위협. 11/7 중국이 시행을 유예."}
    t = announcement_text(ev)
    assert "허가 대상" in t and "트럼프" not in t and "유예" not in t


def test_asof_context_reestimates_from_past_data():
    import numpy as np, pandas as pd
    from agents.backtest import asof_context
    from agents.context import Context
    idx = pd.bdate_range("2012-01-02", periods=1600)
    rng = np.random.default_rng(0)
    frets = pd.DataFrame(rng.normal(0, 0.01, (len(idx), 2)), index=idx, columns=["risk_off", "dollar"])
    rets = pd.DataFrame({"a": frets["risk_off"] * 1.0 + rng.normal(0, 0.002, len(idx)),
                         "b": frets["dollar"] * -0.5 + rng.normal(0, 0.002, len(idx))}, index=idx)
    # 후반부에 베타를 바꿔 놓으면, as-of 추정은 전반부 베타에 가까워야 한다
    rets.iloc[1000:, 0] = frets["risk_off"].iloc[1000:] * 3.0
    levels = pd.DataFrame({"VIXCLS": rng.uniform(10, 40, len(idx))}, index=idx)
    ctx = Context(thresholds={"betas": {"ridge_alpha": 1.0, "crisis_vix_quantile": 0.8},
                              "event_window": {"react_days": 3}},
                  weights=pd.Series({"a": 0.5, "b": 0.5}), exposure_map={}, events=[], rets=rets, levels=levels,
                  beta_normal=pd.DataFrame(), beta_crisis=pd.DataFrame(), f_event=pd.DataFrame(),
                  b_event=pd.DataFrame(), resid_event=pd.DataFrame(), q_up=pd.DataFrame(), q_down=pd.DataFrame(),
                  vix_col="VIXCLS", labels={}, frets=frets)
    c = asof_context(ctx, str(idx[1000].date()))
    full = asof_context(ctx, str(idx[-1].date() + pd.Timedelta(days=2)))
    # 전체 기간 베타는 후반부(3.0)가 섞여 커지고, as-of 베타는 전반부(1.0)만 반영한다 (릿지 축소는 둘 다 같은 비율)
    assert c.beta_normal.loc["a", "risk_off"] < 0.7 * full.beta_normal.loc["a", "risk_off"]
    assert list(c.q_up.columns) == ["0.5", "0.8", "0.95", "0.99", "0.995"]


def test_noise_band_excludes_event_windows_and_respects_asof():
    import numpy as np, pandas as pd
    from engine.risk import noise_band
    idx = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(1)
    rets = pd.DataFrame(rng.normal(0, 0.005, (400, 2)), index=idx, columns=["a", "b"])
    # 한 구간에 큰 사건 충격을 심어두면, 그 창을 빼고 잰 띠는 더 좁아야 한다
    rets.iloc[200:203] = -0.05
    w = pd.Series({"a": 0.5, "b": 0.5})
    full = noise_band(rets, w, days=3, q=0.999)
    ex = noise_band(rets, w, days=3, q=0.999, exclude=[(str(idx[200].date()), str(idx[200].date()))])
    assert ex["half_width"] < full["half_width"] and ex["n_days"] < full["n_days"]
    early = noise_band(rets, w, days=3, q=0.75, asof=str(idx[150].date()))
    assert early["n_days"] < 150 and early["half_width"] > 0


def test_choose_alpha_cv_prefers_weak_shrinkage_on_clean_linear_data():
    import numpy as np, pandas as pd
    from pipeline.betas import choose_alpha_cv, resolve_alpha
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(0, 1, (1500, 3)), columns=list("xyz"))
    y = pd.DataFrame({"p": X["x"] * 2 - X["y"] + rng.normal(0, 0.1, 1500)})
    a, sc = choose_alpha_cv(y, X, grid=(0.001, 0.1, 1.0, 3.0))
    assert a <= 0.1 and sc.loc[1.0] > sc.loc[0.001]
    assert resolve_alpha("cv", y, X) == a and resolve_alpha(0.3, y, X) == 0.3


def test_verifier_rejects_truncated_llm_brief_when_structure_required():
    from agents import verifier
    facts = {"ES_STRESSED": "8.67%"}
    rr = {"decomposition": {"event_effect_ES": 0.001}, "es_underestimation_normal": 0.01}
    truncated = "## 요약\n중국이 희토류 수출통제를 역외 적용 [src:s1]\n\n### 점검 목록 (자동)\n- 계산 기록: calc_x [calc]."
    v = verifier.check(truncated, facts, {"s1": "중국이 희토류 수출통제를 역외 적용"}, rr, {"uncertain_fields": []},
                       require_structure=True)
    assert not v["passed"] and any(i["kind"] == "incomplete" for i in v["issues"]) and v["route"] == "reporter"
    # 템플릿(규칙) 경로나 단위 시험처럼 구조 검사를 끄면 예전과 같다
    assert not any(i["kind"] == "incomplete" for i in
                   verifier.check(truncated, facts, {"s1": "중국이 희토류 수출통제를 역외 적용"}, rr, {"uncertain_fields": []})["issues"])


def test_adaptive_severity_sampling_adds_small_model_only_when_spread_is_wide():
    from agents import event_analyst
    from tests.test_agents import FakeLLM
    from tests.test_features import COMPS, DRAFT, REE_NEWS

    class TwoModelLLM(FakeLLM):
        model = "big"

        def __init__(self, jsons):
            super().__init__(jsons=jsons)
            self.small_calls = 0

        def derive(self, model=None, small=False):
            parent = self

            class Small(FakeLLM):
                model = "small"

                def complete_json(s2, *a, **k):
                    parent.small_calls += 1
                    return parent.complete_json(*a, **k)
            return Small()

    cfg = {"self_consistency": {"n": 3, "small_share": 1, "spread_warn": 2, "extra_n": 2}}
    # 1) 세 표본이 일치 -> 추가 채점 없음 (호출 3회), 작은 모델 1회 포함
    same = {"severity_components": dict(COMPS)}
    llm = TwoModelLLM([DRAFT, same, same, same, same])
    out = event_analyst.analyze(REE_NEWS, llm, severity_cfg=cfg)
    assert out["event"]["severity_samples"] == 3 and llm.small_calls == 1 and llm.calls == 3
    # 2) 흩어지면 2회 더 (총 5), 추가분에도 작은 모델이 섞임
    wide = {"severity_components": {**COMPS, "surprise": 1, "scope": 5}}
    llm2 = TwoModelLLM([DRAFT, wide, same, same, same])
    out2 = event_analyst.analyze(REE_NEWS, llm2, severity_cfg=cfg)
    assert out2["event"]["severity_samples"] == 5 and llm2.small_calls == 2
    assert "추가 채점" in " ".join(out2["event"]["notes"])
