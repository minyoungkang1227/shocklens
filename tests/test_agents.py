"""에이전트·헤드쿼터 테스트 (합성 데이터 + 가짜 LLM, 네트워크 없음)."""
import numpy as np
import pandas as pd
import pytest
import yaml

from agents import event_analyst, verifier
from agents.context import Context
from agents.hq import parse_whatif, run_library_event, run_news, run_whatif
from engine import calclog
from pipeline import config, library

FACTORS = ["risk_off", "dollar", "cny", "oil", "ust10y", "credit", "tech_war"]


class FakeLLM:
    """정해진 응답을 순서대로 돌려주는 가짜 LLM."""
    provider, model, enabled = "fake", "fake", True

    def __init__(self, jsons=None, texts=None):
        self.jsons, self.texts, self.calls = list(jsons or []), list(texts or []), 0

    def complete_json(self, system, user, max_tokens=2000, retries=1, **kw):
        self.calls += 1
        return self.jsons.pop(0) if self.jsons else None

    def complete_text(self, system, user, max_tokens=2000):
        self.calls += 1
        return self.texts.pop(0) if self.texts else None


class NoLLM(FakeLLM):
    enabled = False


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setattr(calclog, "LOG_DIR", tmp_path)
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", "2026-10-30")
    pf = config.portfolio()["baskets"]
    bk = list(pf)
    rets = pd.DataFrame(rng.normal(0, 0.008, (len(idx), len(bk))), idx, bk)
    levels = pd.DataFrame({"VIXCLS": 15 + np.abs(rng.normal(0, 5, len(idx)))}, idx)
    beta = pd.DataFrame(rng.normal(0, 0.003, (len(bk), 7)), bk, FACTORS)
    beta["risk_off"] = -0.003
    beta["tech_war"] = 0.6
    events = library.load_events()
    ids = [e["id"] for e in events]
    f_ev = pd.DataFrame(0.0, ids, FACTORS)
    for e in events:                                     # 채널 방향대로 움직였다고 가정
        for f, d in e["channels"].items():
            f_ev.loc[e["id"], f] = d * {"risk_off": 2.0, "oil": 0.03, "credit": 0.005}.get(f, 0.01)
    resid = pd.DataFrame(rng.normal(0, 0.005, (len(ids), len(bk))), ids, bk)
    qcols = ["0.5", "0.8", "0.95", "0.99", "0.995"]
    q_up = pd.DataFrame([[1, 2, 3, 4, 5]] * 7, FACTORS, qcols, dtype=float) * 0.01
    with open(config.CONFIG_DIR / "exposure_map.yaml", encoding="utf-8") as f:
        emap = yaml.safe_load(f)
    th = config.thresholds()
    th["risk"]["n_paths"] = 500
    th.setdefault("attention", {})["enabled"] = False          # 테스트는 네트워크를 쓰지 않는다
    return Context(thresholds=th, weights=pd.Series({b: v["weight"] for b, v in pf.items()}, dtype=float),
                   exposure_map=emap, events=events, rets=rets, levels=levels, beta_normal=beta,
                   beta_crisis=beta * 1.2, f_event=f_ev, b_event=f_ev @ beta.T, resid_event=resid,
                   q_up=q_up, q_down=-q_up, labels={b: v["label"] for b, v in pf.items()})


SANCTION_NEWS = ("미국과 EU는 26일 러시아 주요 은행을 SWIFT에서 배제하고 러시아 중앙은행의 외환보유액 사용을 제한하기로 합의했다. "
                 "에너지 수출 대금 결제에도 차질이 예상된다.")
REE_NEWS = ("중국 상무부는 9일 희토류 관련 기술까지 수출 허가 대상으로 확대한다고 발표했다. "
            "업계는 미국 반도체·방산 공급망에 타격이 불가피하다고 본다.")


# ------------------------------------------------------------------ Event Analyst
def test_rule_extract_sanction_roles_and_sovereign_flag():
    out = event_analyst.analyze(SANCTION_NEWS, NoLLM())
    ev = out["event"]
    assert out["event_complete"] and out["mode"] == "rule"
    assert ev["instrument"] == "financial_sanction" and ev["initiator"] == "US" and ev["target"] == ["RU"]
    assert ev["direction"] == "escalate" and ev["stage"] != "ease"
    assert ev["sovereign_asset_risk"] is True
    assert ev["channels"] == {"oil": 1, "risk_off": 1, "credit": -1}      # 수단별 기본 채널


def test_guard_fixes_vocab_evidence_and_channel_sign():
    sents = event_analyst.split_sentences(REE_NEWS)
    draft = {"instrument": "resource_export_control", "direction": "escalate", "initiator": "cn",
             "target": ["US", "CN", "MARS"], "sectors": ["semiconductor", "space"], "stage": "boom",
             "severity": 9, "channels": {"risk_off": -1, "tech_war": -1, "gold": 1},
             "evidence": {"instrument": ["s1", "s99"], "initiator": ["s1"]}}
    ev, errors = event_analyst.guard(draft, sents, REE_NEWS)
    assert ev is not None and not errors
    assert ev.initiator == "CN" and ev.target == ["US"]                # 소문자 보정, 발동국·어휘 밖 제거
    assert ev.sectors == ["semiconductor"] and ev.severity == 5 and ev.stage == "announce"
    assert ev.channels == {"risk_off": 1, "tech_war": -1}               # 어휘 밖 제거 + 방향 모순 보정
    assert ev.evidence["instrument"] == ["s1"]                           # 없는 문장 id 제거
    assert "target" in ev.uncertain_fields and "sectors" in ev.uncertain_fields
    assert any("risk_off" in n for n in ev.notes)


def test_guard_rejects_unknown_instrument():
    sents = event_analyst.split_sentences(REE_NEWS)
    ev, errors = event_analyst.guard({"instrument": "blockade", "initiator": "CN", "target": ["US"]}, sents, REE_NEWS)
    assert ev is None and any("blockade" in e for e in errors)


# ------------------------------------------------------------------ Verifier
def test_verifier_catches_number_tag_source_and_direction():
    facts = {"ES_STRESSED": "8.73%", "EVENT_EFFECT": "+0.16%p"}
    sources = {"s1": "중국이 100% 관세를 예고했다"}
    rr = {"decomposition": {"event_effect_ES": 0.0016}}
    ev = {"uncertain_fields": []}
    ok = "ES97.5는 8.73%입니다 [calc]. 중국은 100% 관세를 예고했습니다 [src:s1]."
    assert verifier.check(ok, facts, sources, rr, ev)["passed"]
    bad = ("ES97.5는 9.10%입니다 [calc]. 사건 효과는 +0.16%p입니다. 관련 보도가 있었습니다 [src:s7]. "
           "따라서 위험이 감소했습니다 [calc].")
    v = verifier.check(bad, facts, sources, rr, ev)
    kinds = {i["kind"] for i in v["issues"]}
    assert {"number_mismatch", "untagged", "unknown_source", "direction"} <= kinds
    assert v["route"] == "reporter"
    v2 = verifier.check(ok, facts, sources, rr, {"uncertain_fields": ["initiator"]})
    assert v2["route"] == "event_analyst"


# ------------------------------------------------------------------ 헤드쿼터
def test_library_event_runs_parallel_nodes_and_passes(ctx):
    res = run_library_event("cn_ree_oct_2025", ctx, NoLLM())
    nodes = [t["node"] for t in res["trace"]]
    assert res["status"] == "ok" and res["verification"]["passed"]
    assert "analog_finder" in nodes and "exposure_analyst" in nodes
    assert nodes.index("shock_mapper") > max(nodes.index("analog_finder"), nodes.index("exposure_analyst"))
    assert "cn_ree_oct_2025" not in [a["id"] for a in res["analogs"]]      # 자기 자신 제외
    # 겹친 사건(same_dispute)은 가중치를 줄여 포함, other(전쟁 등)는 제외
    by = {a["id"]: a for a in res["analogs"]}
    if "cn_ree_2025" in by:
        assert by["cn_ree_2025"]["clean_factor"] < 1.0
    assert "ru_swift_2022" not in by
    assert res["calc_log_id"].startswith("calc_")
    assert "{{" not in res["final_brief"]


def test_event_analyst_retry_with_feedback_then_success(ctx):
    bad = {"instrument": "blockade", "initiator": "CN", "target": ["US"]}
    good = {"title": "중국 희토류 통제 확대", "instrument": "resource_export_control", "direction": "escalate",
            "initiator": "CN", "target": ["US"], "sectors": ["semiconductor", "defense"], "stage": "escalate",
            "severity": 4, "channels": {"tech_war": -1, "risk_off": 1, "cny": 1},
            "evidence": {k: ["s1"] for k in ("instrument", "initiator", "target", "sectors", "stage", "severity")}}
    res = run_news(REE_NEWS, ctx, FakeLLM(jsons=[bad, good]))
    ev_runs = [t for t in res["trace"] if t["node"] == "event_analyst"]
    assert len(ev_runs) == 2 and res["event"]["severity"] == 4 and res["event_mode"] == "llm"
    assert res["status"] == "ok"


def test_reporter_hallucination_is_rejected_then_rewritten(ctx):
    hallucinated = "## 요약\n포트폴리오 ES는 12.34%로 급등합니다 [calc]."
    good = ("## 요약\n사건 반영 후 ES97.5는 {{ES_STRESSED}}입니다 [calc].\n"
            "## 포트폴리오 영향\n사건 자체 효과는 {{EVENT_EFFECT}}입니다 [calc].")
    llm = FakeLLM(texts=[hallucinated, good])
    res = run_library_event("cn_ban_us_2024", ctx, llm)
    reps = [t for t in res["trace"] if t["node"] == "reporter"]
    assert len(reps) == 2 and res["status"] == "ok"
    assert "12.34" not in res["final_brief"]


def test_partial_end_after_max_retries(ctx):
    llm = FakeLLM(texts=["ES는 99.99%입니다 [calc]."] * 5)
    res = run_library_event("cn_ban_us_2024", ctx, llm)
    assert res["status"] == "partial" and res["final_brief"].startswith("## 판단 보류")
    assert len([t for t in res["trace"] if t["node"] == "reporter"]) == 3   # 최초 1 + 재시도 2


def test_whatif_reruns_from_shock_mapper(ctx):
    base = run_library_event("cn_ban_us_2024", ctx, NoLLM())
    assert parse_whatif("반도체 충격 3배, 유가 +10%") == {"oil": {"abs": 0.1}, "tech_war": {"mult": 3.0}}
    res = run_whatif(base, "반도체 충격 3배", ctx, NoLLM())
    assert [t["node"] for t in res["trace"]][0] == "shock_mapper"
    assert res["risk_result"]["stressed"]["ES"] > base["risk_result"]["stressed"]["ES"]
    with pytest.raises(ValueError):
        parse_whatif("그냥 좀 세게")


def test_event_analyst_corrects_direction_from_words():
    text = "미중 관세 휴전 유예가 종료되면서 미국은 중국산 제품 관세를 다시 인상한다고 발표했다. 중국은 보복 관세로 맞대응하겠다고 밝혔다."
    ev = event_analyst.analyze(text, NoLLM())["event"]
    assert ev["direction"] == "escalate" and ev["instrument"] == "tariff"
    assert "direction" in ev["uncertain_fields"]


def test_news_run_uses_text_similarity_and_size_class(ctx):
    res = run_news(REE_NEWS, ctx, NoLLM())
    assert res["status"] == "ok" and res["size_class"] in ("small", "medium", "large")
    assert all(a["text"] is not None for a in res["analogs"])          # 기사 유사도가 점수에 반영됨
    assert res["shock_method"].split("(")[0] in ("weighted", "event_level", "synthetic")


def test_event_guard_computes_severity_from_components():
    sents = event_analyst.split_sentences(REE_NEWS)
    draft = {"instrument": "resource_export_control", "initiator": "CN", "target": ["US"],
             "severity_components": {"scope": 5, "immediacy": 5, "substitut": 5, "escalation": 4, "parties": 5, "surprise": "4"},
             "evidence": {"instrument": ["s1"], "initiator": ["s1"], "target": ["s2"], "scope": ["s1"]}}
    ev, errors = event_analyst.guard(draft, sents, REE_NEWS)
    assert not errors and ev.severity == 5 and ev.severity_points > 80 and ev.severity_components["surprise"] == 4
    ev2, _ = event_analyst.guard({"instrument": "tariff", "initiator": "US", "target": ["CN"], "severity": 2}, sents, REE_NEWS)
    assert ev2.severity == 2 and ev2.severity_points == 25.0 and "severity" in ev2.uncertain_fields


def test_run_uses_market_regime_and_es_range(ctx):
    ctx.thresholds["regime"] = {"mode": "market", "crisis_vix_quantile": 0.8, "lookback_days": 5}
    res = run_library_event("cn_ban_us_2024", ctx, NoLLM())
    assert res["regime_info"]["mode"] == "market" and "VIX" in res["regime_info"]["reason"]
    r = res["risk_result"]
    assert r["stressed_severe"]["ES"] >= r["stressed"]["ES"]
    assert "최악 시나리오" in res["final_brief"]


def test_anthropic_call_uses_only_sdk_supported_arguments():
    """실제 설치된 anthropic SDK의 messages.create 시그니처와 우리가 넘기는 인자를 대조한다."""
    import inspect
    anthropic = pytest.importorskip("anthropic")
    from agents.llm import LLM
    allowed = set(inspect.signature(anthropic.Anthropic(api_key="x").messages.create).parameters)
    kw = LLM(provider="anthropic", api_key="x").anthropic_kwargs("sys", "hi", 100)
    assert set(kw) <= allowed, f"SDK가 받지 않는 인자: {set(kw) - allowed}"


def test_verifier_catches_horizon_and_normal_mc_mislabels():
    facts = {"LOSS_CENTRAL": "-0.11%", "NORMAL_MC_ES": "6.91%", "ES_STRESSED": "8.69%"}
    rr = {"decomposition": {"event_effect_ES": 0.001}}
    bad = ("중심 시나리오의 20거래일 손실은 -0.11%입니다 [calc]. "
           "평상시 몬테카를로 ES 6.91%는 스트레스 ES보다 낮습니다 [calc].")
    kinds = [i["kind"] for i in verifier.check(bad, facts, {}, rr, {"uncertain_fields": []})["issues"]]
    assert kinds.count("label") == 2
    ok = ("발표 후 3거래일 중앙 시나리오 수익률은 -0.11%입니다 [calc]. "
          "정규분포를 가정하면 ES가 6.91%로 낮게 나옵니다 [calc].")
    assert verifier.check(ok, facts, {}, rr, {"uncertain_fields": []})["passed"]
