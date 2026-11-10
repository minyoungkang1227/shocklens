"""국내 정책 사건(domestic_regulation·election_shift)과 데일리 워치 테스트 (네트워크 없음)."""
import datetime as dt

import yaml

from agents import daily_watch as dw
from agents import event_analyst
from agents.hq import run_news
from engine.analog import FAMILY
from engine.exposure import exposure_for
from engine.textsim import text_direction
from pipeline import config

from .test_agents import FakeLLM, NoLLM, ctx  # noqa: F401  (ctx는 fixture)

DC_NEWS = ("미국 중간선거에서 민주당이 상·하원을 모두 장악했다. 이후 인공지능(AI) 데이터센터 건설 규제 강화 우려가 커지면서 "
           "반도체주가 급락했다. 버니 샌더스 상원의원은 데이터센터 건설 유예를 촉구했다.")


# ------------------------------------------------------------------ 국내 정책 사건
def test_rule_extract_domestic_regulation_targets_own_country():
    out = event_analyst.analyze(DC_NEWS, NoLLM())
    ev = out["event"]
    assert out["event_complete"]
    # 선거가 원인이고 규제는 '우려·촉구' 단계 -> election_shift (prompts 정의·us_midterm_2026 라벨과 같은 기준)
    assert ev["instrument"] == "election_shift" and ev["initiator"] == "US" and ev["target"] == ["US"]
    assert "ai_infra" in ev["sectors"] and ev["direction"] == "escalate"
    assert ev["channels"] == {"tech_war": -1, "risk_off": 1}


def test_guard_fills_domestic_target_and_keeps_same_country():
    sents = event_analyst.split_sentences(DC_NEWS)
    ev, err = event_analyst.guard({"instrument": "domestic_regulation", "initiator": "US", "target": [],
                                   "evidence": {"initiator": ["s1"]}}, sents, DC_NEWS)
    assert ev is not None and not err and ev.target == ["US"]
    ev2, _ = event_analyst.guard({"instrument": "election_shift", "initiator": "US", "target": ["US"]}, sents, DC_NEWS)
    assert ev2.target == ["US"]
    ev3, _ = event_analyst.guard({"instrument": "tariff", "initiator": "US", "target": ["US", "CN"]}, sents, DC_NEWS)
    assert ev3.target == ["CN"]                      # 지경학 수단은 여전히 발동국=대상국 제거


def test_construction_moratorium_reads_as_escalation():
    assert text_direction("데이터센터 건설 유예를 촉구했다")["label"] == "escalate"
    assert text_direction("Sanders calls for a moratorium on new data centers")["label"] == "escalate"
    assert text_direction("관세 유예에 합의했다")["label"] == "ease"


def test_domestic_exposure_hits_home_industry_not_foreign_baskets():
    with open(config.CONFIG_DIR / "exposure_map.yaml", encoding="utf-8") as f:
        emap = yaml.safe_load(f)
    us = exposure_for({"instrument": "domestic_regulation", "initiator": "US", "target": ["US"],
                       "sectors": ["semiconductor", "ai_infra"]}, emap)
    assert us["semis"] == "high" and us["us_equity"] == "mid" and us["cn_equity"] == "none"
    assert us["tw_equity"] == "mid"                  # 대만 반도체는 간접(한 단계 낮춤)
    cn = exposure_for({"instrument": "domestic_regulation", "initiator": "CN", "target": ["CN"],
                       "sectors": ["tech_platform"]}, emap)
    assert cn["cn_equity"] == "high" and cn["us_equity"] == "none"


def test_domestic_family_and_library_events():
    assert FAMILY["domestic_regulation"] == FAMILY["election_shift"] == "domestic"
    from pipeline.library import load_events
    ev = {e["id"]: e for e in load_events()}
    for i in ("cn_alibaba_antitrust_2020", "cn_didi_cac_2021", "cn_edu_ban_2021", "us_ga_runoff_2021"):
        assert ev[i]["target"] == [ev[i]["initiator"]]
    assert ev["us_midterm_2018"]["direction"] == "ease"
    m = ev["us_midterm_2026"]                          # 2026 중간선거: 발표일 11/3, 반응은 11/4부터, 고용지표 겹침
    assert m["instrument"] == "election_shift" and m["direction"] == "escalate"
    assert m["react_from"] == "2026-11-04" and m["confounder_type"] == "macro" and not m["clean"]


def test_domestic_news_runs_end_to_end(ctx):
    st = run_news(DC_NEWS, ctx, NoLLM())
    assert st["status"] == "ok" and st["event"]["instrument"] in ("domestic_regulation", "election_shift")
    assert st["analog_tier"] in ("direct", "relaxed", "pool", "synthetic")
    assert all(FAMILY.get(ctx.event_by_id(a["id"])["instrument"]) == "domestic" for a in st["analogs"]) \
        or st["analog_tier"] == "synthetic"
    assert "국내 정책 사건" in st["final_brief"] and "자국 산업" in st["final_brief"]


# ------------------------------------------------------------------ 데일리 워치
def _rss(items, outlet):
    x = "".join(f"<item><title>{t} - {outlet}</title><link>https://ex.com/{i}</link><source>{outlet}</source></item>"
                for i, t in enumerate(items))
    return f"<rss><channel>{x}</channel></rss>".encode()


FEEDS = {
    "a": _rss(["美 데이터센터 규제 우려에 반도체주 급락", "데이터센터 규제 공포에 외국인 순매도", "삼성전자 실적 발표 예정"], "연합"),
    "b": _rss(["美 데이터센터 규제 우려에 반도체주 급락", "서울 아파트값 상승세 둔화"], "한경"),
}


def _cfg():
    cfg = dw.load_cfg()
    cfg["regions"] = {"KR": {"label": "국내", "feeds": [{"name": "A", "url": "a"}, {"name": "B", "url": "b"},
                                                         {"name": "막힘", "url": "zzz"}]}}
    cfg["fetch_article"] = False
    return cfg


def test_collect_dedupes_and_counts_outlets():
    heads, errs = dw.collect(_cfg()["regions"]["KR"], fetcher=lambda u: FEEDS[u])
    assert len(heads) == 4 and errs and "막힘" in errs[0]
    assert heads[0]["title"] == "美 데이터센터 규제 우려에 반도체주 급락"      # 매체명 꼬리 제거
    assert heads[0]["outlets"] == {"연합", "한경"}


def test_rule_topics_group_and_rank():
    heads, _ = dw.collect(_cfg()["regions"]["KR"], fetcher=lambda u: FEEDS[u])
    topics = dw.cluster_rule(heads, 3)
    assert topics[0]["headline_ids"] == [1, 2] and topics[0]["policy_event"] and topics[0]["n_outlets"] == 2


def test_llm_topics_drop_unknown_ids_and_numbers():
    heads, _ = dw.collect(_cfg()["regions"]["KR"], fetcher=lambda u: FEEDS[u])
    llm = FakeLLM(jsons=[{"topics": [{"title": "데이터센터 규제 공포", "headline_ids": [1, 2, 99], "policy_event": True,
                                      "importance": 5, "market_path": "반도체가 -12% 빠지며 대만으로 번진다"}]}])
    t = dw.pick_topics(heads, "국내", 3, llm)
    assert t[0]["headline_ids"] == [1, 2] and t[0]["mode"] == "llm"
    assert "12" not in t[0]["market_path"]                 # LLM 문장에 숫자를 남기지 않는다


def test_daily_watch_report_end_to_end(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    rep = dw.run(cfg=_cfg(), ctx=ctx, llm=NoLLM(), fetcher=lambda u: FEEDS[u], today=dt.date(2026, 11, 6))
    t0 = rep["regions"][0]["topics"][0]
    assert t0["analysis"]["status"] == "ok" and t0["analysis"]["calc_log_id"]
    assert t0["analysis"]["facts"]["ENS_ES"].endswith("%")
    md = rep["markdown"]
    assert "국내 핫토픽" in md and "20거래일 ES97.5" in md and "막힌 피드" in md
    assert rep["path"].endswith("2026-11-06.md")
    rep2 = dw.run(cfg=_cfg(), ctx=ctx, llm=NoLLM(), fetcher=lambda u: FEEDS[u], today=dt.date(2026, 11, 6))
    assert rep2["path"].endswith("2026-11-06_v2.md")       # 덮어쓰지 않는다
    assert t0["analysis"]["forecast"] and "실전 성적표" in md
    assert (tmp_path / "data" / "forecasts" / "forecasts.jsonl").exists()


def test_staleness_note():
    assert dw.staleness("2026-10-30", dt.date(2026, 11, 6), 4)
    assert dw.staleness("2026-11-05", dt.date(2026, 11, 6), 4) is None


# ------------------------------------------------------------------ 2026-11-09: 채널 차용·국내 사건 반도체 채널
def test_guard_adds_tech_war_for_domestic_semis_event():
    sents = event_analyst.split_sentences(DC_NEWS)
    ids = [x["id"] for x in sents]
    draft = {"instrument": "election_shift", "direction": "escalate", "initiator": "US", "target": ["US"],
             "sectors": ["semiconductor", "ai_infra"], "stage": "announce", "severity": 3,
             "channels": {"risk_off": 1}, "evidence": {"instrument": ids[:1], "initiator": ids[:1], "target": ids[:1]}}
    ev, errors = event_analyst.guard(draft, sents, DC_NEWS)
    assert ev is not None and ev.channels == {"risk_off": 1, "tech_war": -1}
    draft2 = {**draft, "sectors": ["tech_platform"]}                   # 반도체·AI 인프라가 아니면 붙이지 않음
    ev2, _ = event_analyst.guard(draft2, sents, DC_NEWS)
    assert ev2.channels == {"risk_off": 1}


def test_borrow_channels_fills_missing_semis_channel():
    import pandas as pd
    from engine import shock
    lib = [
        {"id": "cn_a", "date": "2021-01-01", "instrument": "domestic_regulation", "sectors": ["tech_platform"],
         "channels": {"risk_off": 1, "cny": 1}, "clean": True, "severity_points": 50.0},
        {"id": "cn_b", "date": "2021-02-01", "instrument": "domestic_regulation", "sectors": ["tech_platform"],
         "channels": {"risk_off": 1, "cny": 1}, "clean": True, "severity_points": 50.0},
        {"id": "chip_x", "date": "2022-10-07", "instrument": "tech_export_control", "sectors": ["semiconductor"],
         "channels": {"tech_war": -1, "risk_off": 1}, "clean": True, "severity_points": 50.0},
        {"id": "chip_y", "date": "2023-10-17", "instrument": "tech_export_control", "sectors": ["semiconductor"],
         "channels": {"tech_war": -1, "risk_off": 1}, "clean": False, "confounder_type": "macro",
         "severity_points": 50.0},
        {"id": "chip_future", "date": "2030-01-01", "instrument": "tech_export_control", "sectors": ["semiconductor"],
         "channels": {"tech_war": -1}, "clean": True, "severity_points": 50.0},
        {"id": "chip_ease", "date": "2022-01-01", "instrument": "tariff_pause", "direction": "ease",
         "sectors": ["semiconductor"], "channels": {"tech_war": -1}, "clean": True, "severity_points": 50.0},
    ]
    F = pd.DataFrame({"risk_off": [2.0, 1.0, 3.0, 2.0, 9.0, 1.0], "tech_war": [0.0, 0.0, -0.05, -0.03, -0.5, -0.4],
                      "cny": [0.01, 0.01, 0, 0, 0, 0]}, index=[e["id"] for e in lib])
    ev = {"id": "new", "date": "2026-11-03", "direction": "escalate", "sectors": ["semiconductor", "ai_infra"],
          "channels": {"risk_off": 1, "tech_war": -1}, "severity": 3}
    base = shock.from_analogs_weighted(F, {"cn_a": 1, "cn_b": 1}, ev["channels"])
    assert base.loc["tech_war"].abs().max() == 0                       # 차용 전: 반도체 채널 0
    out, used = shock.borrow_channels(ev, ["cn_a", "cn_b"], {"cn_a": 1, "cn_b": 1}, lib, F, base, {"weight": 0.5})
    assert used == {"tech_war": ["chip_x", "chip_y"]}                  # 미래·반대 방향 사건은 기증하지 않음
    assert out.loc["tech_war", "severe"] < out.loc["tech_war", "central"] < 0
    assert (out.loc["risk_off"] == base.loc["risk_off"]).all()         # 이미 근거가 있는 채널은 그대로
    off, used0 = shock.borrow_channels(ev, ["cn_a", "cn_b"], {"cn_a": 1, "cn_b": 1}, lib, F, base, {"enabled": False})
    assert used0 == {} and off.equals(base)


# ------------------------------------------------------------------ 2026-11-10 판정 점검에서 나온 보정
ENACTED_NEWS = "미국 의회가 데이터센터 건설 규제 법안을 통과시켰다. 중간선거 이후 민주당이 주도했다."
KR_MARGIN_NEWS = ("금융당국은 9일부터 내년 1월 31일까지 증권사의 신용융자 담보비율 140% 유지 의무를 한시 면제한다. "
                  "반대매매로 인한 시장 불안을 줄이기 위한 조치다.")


def test_enacted_domestic_regulation_stays_domestic():
    ev = event_analyst.analyze(ENACTED_NEWS, NoLLM())["event"]
    assert ev["instrument"] == "domestic_regulation" and ev["direction"] == "escalate"


def test_korean_deregulation_is_policy_ease_without_tech_channel():
    from agents import triage
    t = triage.rule_triage(KR_MARGIN_NEWS)
    assert t["relevant"] and t["importance"] >= 2
    out = event_analyst.analyze(KR_MARGIN_NEWS, NoLLM())
    ev = out["event"]
    assert out["event_complete"] and ev["instrument"] == "domestic_regulation"
    assert ev["initiator"] == "KR" and ev["target"] == ["KR"] and ev["direction"] == "ease"
    assert "tech_war" not in ev["channels"] and ev["channels"].get("risk_off") == -1


def test_market_events_are_not_policy():
    from agents import triage
    for text in ("코스피가 미국 반도체주 급등에 5%대 상승하며 매수 사이드카가 발동됐다.",
                 "엔비디아 이사회가 젠슨 황 CEO에 대한 조사에 착수했다. 주가는 4.2% 하락했다.",
                 "10월 고용 지표가 강하게 나오면서 12월 금리 인상 확률이 80%를 넘었다."):
        assert not triage.rule_triage(text)["relevant"]


def test_kr_margin_relief_library_event():
    from pipeline.library import load_events
    e = {x["id"]: x for x in load_events()}["kr_margin_relief_2026"]
    assert e["instrument"] == "domestic_regulation" and e["direction"] == "ease" and e["target"] == ["KR"]
    assert e["confounder_type"] == "other" and not e["clean"]
