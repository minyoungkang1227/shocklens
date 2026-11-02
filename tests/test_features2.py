"""11/2 2차 기능 테스트 (네트워크 없음): 뉴스 입력 도구, 중요도 분류, What-if tool use, 재순위, Verifier 분리,
캐싱, 인용, 다국어, 라이브러리 대량 구축."""
import json
from types import SimpleNamespace

from agents import analog_finder, event_analyst, news_input, reporter, triage, verifier
from agents.hq import parse_whatif, run_news
from agents.llm import LLM
from pipeline import library_builder
from tools import docs, news

from .test_agents import REE_NEWS, FakeLLM, NoLLM, ctx  # noqa: F401

ARTICLE = {"url": "https://ex.com/a", "title": "China widens rare earth curbs",
           "text": "China's commerce ministry expanded export controls on rare earth technology. "
                   "US chipmakers face supply risks.", "chars": 100, "error": None}


# ------------------------------------------------------------------ 6. 뉴스 입력 도구
def test_input_kind():
    assert news.input_kind("https://example.com/x") == "url"
    assert news.input_kind("China expands rare earth export controls") == "headline"
    assert news.input_kind(REE_NEWS) == "article"


def test_extract_text_paragraphs():
    h = "<html><title>T</title><script>x</script><p>짧음</p><p>" + "중국 상무부가 희토류 수출통제를 확대했다. " * 3 + "</p></html>"
    title, body = news.extract_text(h)
    assert title == "T" and "희토류" in body and "짧음" not in body


def test_gather_url_and_headline_rule_path(monkeypatch):
    monkeypatch.setattr(news, "fetch_article", lambda url, **k: ARTICLE)
    monkeypatch.setattr(news, "search_news", lambda q, **k: {"results": [{"title": "t", "url": ARTICLE["url"]}],
                                                              "engine": "gdelt", "error": None})
    out = news_input.gather("https://ex.com/a", None)
    assert out["news_meta"]["kind"] == "url" and "rare earth" in out["text"]
    out = news_input.gather("China expands rare earth export controls", None)
    m = out["news_meta"]
    assert m["kind"] == "headline" and m["urls"] == [ARTICLE["url"]] and m["note"]
    assert out["text"].startswith("China expands")


def test_gather_headline_llm_tool_path(monkeypatch):
    monkeypatch.setattr(news, "fetch_article", lambda url, **k: ARTICLE)

    class L:
        enabled, provider = True, "anthropic"

        def chat_tools(self, system, msgs, tools, handler, **kw):
            o = handler("fetch_article", {"url": ARTICLE["url"]})
            return ARTICLE["url"], [{"name": "fetch_article", "input": {"url": ARTICLE["url"]}, "output": o}]

    out = news_input.gather("China expands rare earth export controls", L())
    assert out["news_meta"]["urls"] == [ARTICLE["url"]] and "commerce ministry" in out["text"]


def test_gather_network_failure_keeps_headline(monkeypatch):
    monkeypatch.setattr(news, "search_news", lambda q, **k: {"results": [], "engine": None, "error": "gdelt: Timeout"})
    out = news_input.gather("China expands rare earth export controls", None)
    assert out["text"] == "China expands rare earth export controls" and "검색 실패" in out["news_meta"]["error"]


# ------------------------------------------------------------------ 12. 다국어·긴 문서
def test_chinese_sentences_and_rules():
    out = event_analyst.analyze("中国商务部宣布对稀土相关技术实施出口管制。美国半导体企业将受到影响！", NoLLM())
    assert len(out["sentences"]) == 2
    ev = out["event"]
    assert ev["instrument"] == "resource_export_control" and ev["initiator"] == "CN" and ev["target"] == ["US"]


def test_key_sentences_and_lang():
    filler = "날씨가 맑았다. " * 60
    t = "중국이 발표했다. 리드 둘. 리드 셋. " + filler + "미국은 반도체 수출통제를 확대했다. " + filler
    k = docs.key_sentences(t, max_sentences=10)
    assert "수출통제" in k and k.count("날씨") < 10 and k.startswith("중국이")
    assert docs.detect_lang("中国商务部宣布") == "zh" and docs.detect_lang("tariff on China") == "en"


# ------------------------------------------------------------------ 13. 중요도 분류
def test_triage_rule():
    assert triage.rule_triage("오늘 서울 날씨는 맑겠습니다.")["relevant"] is False
    t = triage.rule_triage(REE_NEWS)
    assert t["relevant"] and t["importance"] >= 2


def test_low_importance_news_is_skipped_unless_forced(ctx):
    txt = "오늘 서울은 대체로 맑고 오후에 구름이 많겠습니다. 미세먼지는 보통 수준입니다."
    res = run_news(txt, ctx, NoLLM())
    assert res["status"] == "skipped" and "분석 생략" in res["final_brief"]
    assert run_news(REE_NEWS, ctx, NoLLM())["status"] == "ok"


def test_triage_llm_schema():
    t = triage.classify(REE_NEWS, FakeLLM(jsons=[{"relevant": True, "importance": 4, "reason": "희토류"}]))
    assert t["mode"] == "llm" and t["importance"] == 4


# ------------------------------------------------------------------ 7. What-if
def test_whatif_llm_schema_and_rule_fallback():
    r = parse_whatif("반도체 충격 3배", FakeLLM(jsons=[{"tech_war": {"mult": 3}, "bogus": {"mult": 2}}]))
    assert r == {"tech_war": {"mult": 3.0}}
    assert parse_whatif("반도체 충격 3배, 유가 +10%", FakeLLM(jsons=[{}])) == {"oil": {"abs": 0.1}, "tech_war": {"mult": 3.0}}


# ------------------------------------------------------------------ 8. Analog 재순위
def test_analog_rerank_reweights(ctx):
    ev = dict(ctx.event_by_id("cn_ree_oct_2025"))
    base = analog_finder.find(ev, ctx, REE_NEWS)
    ids = [p["id"] for p in base["analogs"]]
    assert ids
    rank = [{"id": ids[0], "relevance": 1, "why": "다름"}] + [{"id": i, "relevance": 5, "why": "같음"} for i in ids[1:]]
    if len(ids) == 1:
        rank[0]["relevance"] = 2
    fake = FakeLLM(jsons=[{"ranking": rank}])
    r = analog_finder.find(ev, ctx, REE_NEWS, llm=fake)
    p0 = next(p for p in r["analogs"] if p["id"] == ids[0])
    rel = rank[0]["relevance"]
    assert p0["llm_relevance"] == rel and abs(p0["weight"] - p0["code_weight"] * rel / 5) < 1e-3
    assert any("재순위" in n for n in r["analog_notes"])


# ------------------------------------------------------------------ 9. Verifier 분리·캐싱
def test_verifier_llm_warn_mode_does_not_reject():
    facts = {"ES_STRESSED": "8.80%"}
    rr = {"decomposition": {"event_effect_ES": 0.001}}
    brief = "## 요약\n사건 반영 ES는 8.80%입니다 [calc]."
    fake = FakeLLM(jsons=[{"unsupported": ["사건 반영 ES는"]}])
    fake.model = "claude-haiku-5-5"
    v = verifier.check(brief, facts, {}, rr, {"uncertain_fields": []}, fake, True, "warn")
    assert v["passed"] and v["warnings"] and v["checker_model"] == "claude-haiku-5-5"
    fake2 = FakeLLM(jsons=[{"unsupported": ["사건 반영 ES는"]}])
    assert not verifier.check(brief, facts, {}, rr, {"uncertain_fields": []}, fake2, True, "reject")["passed"]


def test_derive_and_system_cache(monkeypatch):
    llm = LLM(provider="anthropic", api_key="x")
    small = llm.derive(small=True)
    assert small.model == "claude-haiku-5-5" and small.provider == "anthropic"
    llm.cache = True
    llm._client = SimpleNamespace(messages=SimpleNamespace(count_tokens=lambda **k: SimpleNamespace(input_tokens=900)))
    sp = llm.system_param("긴 시스템 프롬프트")
    assert isinstance(sp, list) and sp[0]["cache_control"] == {"type": "ephemeral"}
    llm._client = SimpleNamespace(messages=SimpleNamespace(count_tokens=lambda **k: SimpleNamespace(input_tokens=100)))
    assert llm.system_param("짧은 프롬프트") == "짧은 프롬프트"           # 512토큰 미만은 캐시 안 함


# ------------------------------------------------------------------ 10. 인용
def test_complete_cited_parses_block_citations():
    cite = SimpleNamespace(type="content_block_location", start_block_index=1, end_block_index=2)
    resp = SimpleNamespace(content=[SimpleNamespace(type="text", text="중국이 통제를 확대했다", citations=None),
                                    SimpleNamespace(type="text", text="공급망 타격이 예상된다", citations=[cite])],
                           stop_reason="end_turn")
    llm = LLM(provider="anthropic", api_key="x")
    seen = {}
    llm._client = SimpleNamespace(messages=SimpleNamespace(create=lambda **k: seen.update(k) or resp))
    out = llm.complete_cited("sys", "u", ["문장1", "문장2"])
    assert out[1]["cites"] == [1]
    doc = seen["messages"][0]["content"][0]
    assert doc["citations"] == {"enabled": True} and doc["source"]["type"] == "content"


def test_apply_citations_adds_tag_at_sentence_end():
    blocks = [{"text": "## 무엇이 일어났나\n중국이 희토류 통제를 ", "cites": []},
              {"text": "확대했습니다", "cites": [0]}, {"text": ". 다음 문장입니다 [calc].", "cites": []}]
    text, st = reporter.apply_citations(blocks, ["s1", "s2"])
    assert "확대했습니다 [src:s1]." in text and st["added_tags"] == 1
    blocks = [{"text": "업계 타격입니다 [src:s1]", "cites": [1]}, {"text": ".", "cites": []}]
    _, st = reporter.apply_citations(blocks, ["s1", "s2"])
    assert st["mismatch"]


# ------------------------------------------------------------------ 11. 라이브러리 대량 구축
def test_library_builder_params_and_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(library_builder, "ARTICLES", tmp_path / "a.jsonl")
    rec = {"id": "a123456789", "url": "https://ex.com/a", "title": "Japan curbs chip tools", "seen": "20230331",
           "text": "일본 정부는 31일 반도체 장비 23개 품목을 수출 규제 대상에 추가한다고 발표했다. 중국을 겨냥한 조치로 풀이된다.",
           "keep": True, "triage": {"importance": 3}}
    (tmp_path / "a.jsonl").write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    p = library_builder.label_params(rec, "claude-sonnet-5-5")
    assert p["tool_choice"] == {"type": "auto"} and "<news>" in p["messages"][0]["content"]
    assert library_builder._cid("u").startswith("a") and len(library_builder._cid("u")) <= 64
    draft = {"title": "일본 반도체 장비 수출규제", "date": "2023-03-31", "instrument": "tech_export_control",
             "direction": "escalate", "initiator": "JP", "target": ["CN"], "sectors": ["semiconductor"],
             "stage": "announce", "severity_components": {"scope": 2, "immediacy": 2, "substitut": 4, "escalation": 2,
                                                          "parties": 3, "surprise": 3},
             "channels": {"tech_war": -1, "risk_off": 1}, "evidence": {k: ["s1"] for k in ("instrument", "initiator", "target")}}
    c = library_builder.to_candidates({"a123456789": draft}, {})
    assert len(c) == 1 and c[0]["clean"] is None and c[0]["sources"] == [rec["url"]]
    assert "검수" in c[0]["review"]


def test_verifier_label_check_ignores_pp_values():
    facts = {"LOSS_MILD": "+0.00%", "REGIME_EFFECT": "+0.00%p", "ES_REGIME": "8.60%"}
    rr = {"decomposition": {"event_effect_ES": 0.001}}
    ok = "20거래일 ES는 국면 표본 8.60%이며 국면 효과 +0.00%p입니다 [calc]."
    assert not [i for i in verifier.check(ok, facts, {}, rr, {"uncertain_fields": []})["issues"] if i["kind"] == "label"]
    bad = "20거래일 시나리오 수익률은 +0.00%입니다 [calc]."
    assert [i for i in verifier.check(bad, facts, {}, rr, {"uncertain_fields": []})["issues"] if i["kind"] == "label"]


def test_llm_failure_is_recorded_not_swallowed():
    class Boom(FakeLLM):
        def complete_json(self, *a, **k):
            raise RuntimeError("400 bad schema")

    out = event_analyst.analyze(REE_NEWS, Boom())
    assert out["mode"] == "rule" and "400 bad schema" in out["llm_error"]
    assert any("LLM 호출 실패" in n for n in out["event"]["notes"])
