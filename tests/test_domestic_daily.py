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
    assert ev["instrument"] == "domestic_regulation" and ev["initiator"] == "US" and ev["target"] == ["US"]
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


def test_domestic_news_runs_end_to_end(ctx):
    st = run_news(DC_NEWS, ctx, NoLLM())
    assert st["status"] == "ok" and st["event"]["instrument"] == "domestic_regulation"
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
