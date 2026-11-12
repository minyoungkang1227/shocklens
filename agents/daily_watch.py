"""매일 핫토픽 감시 (Daily Watch): 국내·미국 뉴스에서 오늘 가장 큰 토픽을 고르고 파급효과를 브리핑한다.

    python -m agents.daily_watch                 # 오늘 보고서 (data/daily/YYYY-MM-DD.md)
    python -m agents.daily_watch --refresh       # 먼저 가격·팩터 데이터를 갱신 (국면 판단이 오늘 기준이 되도록)
    python -m agents.daily_watch --regions US --analyze 2 --force

흐름 (지역마다)
1. 수집: config/daily_watch.yaml의 RSS 피드에서 헤드라인을 모은다. 막힌 피드는 건너뛰고 기록한다.
2. 토픽 고르기: LLM이 있으면 헤드라인 번호 목록을 주고 토픽으로 묶어 상위 N개와 '정책 사건 여부'·정성 파급 경로를
   받는다(tool use, 헤드라인 번호는 enum처럼 범위 검사). 없으면 규칙: 문장 유사도로 묶고 기사 수·매체 수로 순위.
3. 분석: 상위 토픽 중 정책 사건(지경학 + 국내 규제·선거)을 ShockLens 전체 흐름(hq.run_news)에 태운다.
   대표 기사 본문을 가져오고, 실패하면 같은 토픽 헤드라인을 묶어 넣는다. 숫자는 전부 Risk Engine 결과(facts)에서 온다.
   정책 사건이 아닌 토픽(실적·지표 등)은 'ShockLens 범위 밖'으로 표시하고 정성 메모만 남긴다.
4. 저장: data/daily/YYYY-MM-DD.md 와 .json. 같은 날 다시 돌리면 _v2, _v3 으로 저장(덮어쓰지 않음).

원칙은 본 시스템과 같다: LLM은 토픽을 고르고 경로를 말로 설명할 뿐, 손실 숫자는 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from engine import textsim
from pipeline import config
from tools.news import get_capped, parse_xml

from .injection import SYSTEM_RULE, sanitize, suspicious
from .llm import LLM
from .triage import rule_triage

TOPIC_SCHEMA = {"type": "object", "properties": {"topics": {"type": "array", "items": {
    "type": "object", "properties": {
        "title": {"type": "string"},
        "headline_ids": {"type": "array", "items": {"type": "integer"}},
        "policy_event": {"type": "boolean"},
        "importance": {"type": "integer", "minimum": 1, "maximum": 5},
        "market_path": {"type": "string"}},
    "required": ["title", "headline_ids", "policy_event", "importance", "market_path"]}}},
    "required": ["topics"]}

TOPIC_SYSTEM = f"""당신은 국부펀드 리스크관리팀의 아침 뉴스 담당자입니다. 오늘 {{region}} 헤드라인 목록을 보고
금융시장에 가장 큰 영향을 주는 토픽을 최대 {{n}}개 고릅니다. 많은 매체가 같은 일을 다룰수록, 시장 전체에 영향이 클수록 위입니다.
- headline_ids: 그 토픽에 속하는 헤드라인 번호 (목록에 있는 번호만)
- title: 토픽 이름 (한국어, 25자 이내)
- policy_event: 정부·의회·선거가 만든 정책 사건이면 true (관세·수출통제·제재·환율 조치·봉쇄·산업정책 법령·투자 제한,
  자국 산업 규제·반독점·AI 데이터센터 규제, 선거·정권 교체로 인한 정책 전환). 기업 실적·경제지표·금리 결정 자체는 false
- importance: 1 작은 이슈 ~ 5 시장 전체를 흔드는 이슈
- market_path: 이 토픽이 해외 분산 포트폴리오(미국·유럽·일본·중국·대만·인도 주식, 반도체, 미 국채, 해외 국채,
  회사채, 인프라·부동산, 상장 사모)에 어떤 경로로 번지는지 한두 문장. 숫자(수익률·%)는 쓰지 않습니다.
emit 도구로 제출합니다.

{SYSTEM_RULE.replace('injection_suspect에 문장 id를 적습니다', '판단에서 무시합니다')}"""


# ---------------------------------------------------------------- 1. 수집
def load_cfg(path: Path | None = None) -> dict:
    with open(path or config.CONFIG_DIR / "daily_watch.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_feed(content: bytes, source: str, limit: int = 40) -> list[dict]:
    """RSS 2.0 / Atom 모두 처리. [{title, url, source, published}]"""
    root = parse_xml(content)                 # DTD 엔티티 폭탄 차단 (tools.news.parse_xml)
    out = []
    items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
    for it in items[:limit]:
        title = it.findtext("title") or it.findtext("{http://www.w3.org/2005/Atom}title") or ""
        link = it.findtext("link") or ""
        if not link:
            a = it.find("{http://www.w3.org/2005/Atom}link")
            link = a.get("href", "") if a is not None else ""
        pub = it.findtext("pubDate") or it.findtext("{http://www.w3.org/2005/Atom}updated") or ""
        outlet = it.findtext("source") or ""
        title = html.unescape(re.sub(r"<[^>]+>", " ", title)).strip()
        if outlet and title.endswith(" - " + outlet):          # Google 뉴스는 제목 끝에 매체명을 붙인다
            title = title[: -len(outlet) - 3].strip()
        if len(title) >= 8:
            out.append({"title": title, "url": link.strip(), "source": source, "outlet": outlet or source,
                        "published": pub})
    return out


def collect(region: dict, max_per_feed: int = 40, timeout: float = 10.0, fetcher=None) -> tuple[list[dict], list[str]]:
    """(중복 제거한 헤드라인, 실패한 피드 메모). fetcher(url) -> bytes 로 테스트에서 바꿔 끼운다."""
    def _get(url: str) -> bytes:
        # 설정에 적힌 고정 피드지만 응답 본문은 외부가 정한다 -> 상한을 두고 읽는다
        return get_capped(url, timeout)
    fetcher = fetcher or _get
    heads, errors, seen = [], [], {}
    for fd in region.get("feeds", []):
        try:
            items = parse_feed(fetcher(fd["url"]), fd["name"], max_per_feed)
        except Exception as e:
            errors.append(f"{fd['name']}: {type(e).__name__}")
            continue
        for h in items:
            key = re.sub(r"\W+", "", h["title"].lower())[:60]
            if key in seen:
                seen[key]["outlets"].add(h["outlet"])        # 같은 기사를 다른 피드·매체도 다룸 -> 매체 수에 반영
                continue
            h["outlets"] = {h["outlet"]}
            seen[key] = h
            heads.append(h)
    for i, h in enumerate(heads):
        h["id"] = i + 1
    return heads, errors


# ---------------------------------------------------------------- 2. 토픽 고르기
def _clean_feats(text: str, idf: dict) -> dict:
    f = textsim.featurize(text, idf)
    return {k: v for k, v in f.items() if not k.startswith("k_")}     # 국가 단어만 겹치는 헤드라인은 같은 토픽이 아니다


def cluster_rule(heads: list[dict], n: int, threshold: float = 0.35) -> list[dict]:
    """규칙 경로: 문장 유사도로 묶고 (매체 수, 기사 수) 순으로 정렬."""
    if not heads:
        return []
    idf = textsim.build_idf([h["title"] for h in heads])
    feats = [_clean_feats(h["title"], idf) for h in heads]
    clusters: list[list[int]] = []
    for i, f in enumerate(feats):
        best, best_s = None, 0.0
        for ci, members in enumerate(clusters):
            s = max(textsim.cosine(f, feats[j]) for j in members[:5])
            if s > best_s:
                best, best_s = ci, s
        if best is not None and best_s >= threshold:
            clusters[best].append(i)
        else:
            clusters.append([i])
    topics = []
    for members in clusters:
        hs = [heads[j] for j in members]
        text = " ".join(h["title"] for h in hs)
        tri = rule_triage(text)
        # 대표 헤드라인: 묶음 안에서 다른 헤드라인과 가장 많이 겹치는 것
        rep = max(members, key=lambda j: sum(textsim.cosine(feats[j], feats[k]) for k in members))
        topics.append({"title": heads[rep]["title"][:60], "headline_ids": [h["id"] for h in hs],
                       "policy_event": bool(tri["relevant"]), "importance": int(tri["importance"]),
                       "market_path": "", "n_outlets": len(set().union(*(h["outlets"] for h in hs))), "mode": "rule"})
    topics.sort(key=lambda t: (t["n_outlets"], len(t["headline_ids"]), t["importance"]), reverse=True)
    return topics[:n]


def _listing_line(h: dict) -> str:
    """헤드라인 한 줄. 제목은 외부 피드 값이므로 구획 태그를 지우고 줄바꿈을 접는다.

    parse_feed가 태그를 지우지만 두 경로가 남는다(2026-11-12 확인).
      · 이중 이스케이프: XML 파서가 &amp;lt;/news&amp;gt;를 &lt;/news&gt;로 풀면 지울 태그가
        없고, 그 뒤 html.unescape가 평문 </news>로 되돌린다 -> 구획이 닫힌다.
      · 줄바꿈 엔티티(&#10;): 제목 안에 줄이 생겨 "[999] (Fake) ..." 같은 가짜 항목을 만들 수
        있다. 목록이 줄 단위라서 모델이 없는 헤드라인을 있는 것으로 읽는다.
    """
    title = re.sub(r"\s+", " ", sanitize(h["title"])).strip()
    return f"[{h['id']}] ({h['outlet']}) {title}"


def pick_topics(heads: list[dict], region_label: str, n: int, llm: LLM | None, threshold: float = 0.35) -> list[dict]:
    if llm is not None and llm.enabled and heads:
        listing = "\n".join(_listing_line(h) for h in heads[:150])
        flagged = [h["id"] for h in heads[:150] if suspicious(h["title"])]
        if flagged:
            print(f"[daily_watch] 헤드라인 {len(flagged)}건에 지시문 의심 문구: id {flagged[:10]}")
        try:
            r = llm.complete_json(TOPIC_SYSTEM.format(region=region_label, n=n), f"<news>\n{listing}\n</news>",
                                  max_tokens=1500, schema=TOPIC_SCHEMA, name="emit")
            valid = {h["id"] for h in heads}
            out = []
            for t in (r or {}).get("topics", [])[:n]:
                ids = [int(i) for i in t.get("headline_ids", []) if int(i) in valid]     # 목록 밖 번호는 버린다
                if not ids:
                    continue
                out.append({"title": str(t.get("title", ""))[:40], "headline_ids": ids,
                            "policy_event": bool(t.get("policy_event")),
                            "importance": min(max(int(t.get("importance", 3)), 1), 5),
                            "market_path": re.sub(r"[-+]?\d+(\.\d+)?\s*%p?", "(수치 생략)", str(t.get("market_path", "")))[:300],
                            "n_outlets": len(set().union(*(h["outlets"] for h in heads if h["id"] in ids))), "mode": "llm"})
            if out:
                return out
        except Exception as e:                                              # LLM 실패 -> 규칙 경로
            print(f"[daily_watch] LLM 토픽 선택 실패({type(e).__name__}) -> 규칙 경로")
    return cluster_rule(heads, n, threshold)


# ---------------------------------------------------------------- 3. 분석
def topic_text(topic: dict, heads: list[dict], fetch: bool = True, fetcher=None) -> tuple[str, dict]:
    """분석에 넣을 텍스트와 출처 메모. 대표 기사 본문 -> 실패하면 헤드라인 묶음."""
    by_id = {h["id"]: h for h in heads}
    hs = [by_id[i] for i in topic["headline_ids"] if i in by_id]
    meta = {"headlines": [h["title"] for h in hs[:8]], "url": None, "mode": "headlines"}
    if fetch:
        from tools.news import fetch_article
        get = fetcher or fetch_article
        for h in hs[:3]:
            if not h.get("url"):
                continue
            art = get(h["url"])
            if art.get("text") and len(art["text"]) >= 300:
                meta.update(url=h["url"], mode="article")
                return f"{h['title']}.\n{art['text'][:6000]}", meta
    return "\n".join(f"{t}." for t in meta["headlines"]), meta


def summarize_state(st: dict) -> dict:
    """브리핑 State에서 보고서용 숫자(이미 코드가 포맷한 문자열)와 구조만 꺼낸다."""
    f, ev = st.get("facts") or {}, st.get("event") or {}
    keys = ["ENS_ES", "ENS_LO", "ENS_HI", "ENS_MAX", "ES_NORMAL", "ES_STRESSED", "EVENT_EFFECT", "EVENT_EFFECT_SEVERE",
            "REGIME", "LOSS_CENTRAL", "LOSS_SEVERE", "LOSS_BAND_LO", "LOSS_BAND_HI", "TOP1", "TOP1_SHARE", "TOP2",
            "TOP2_SHARE", "TOP3", "TOP3_SHARE", "TIER"]
    return {"status": st.get("status"), "calc_log_id": st.get("calc_log_id"),
            "event": {k: ev.get(k) for k in ("title", "instrument", "direction", "initiator", "target", "sectors",
                                              "stage", "severity", "channels", "uncertain_fields")},
            "event_mode": st.get("event_mode"), "analog_tier": st.get("analog_tier"),
            "analogs": [a.get("id") for a in st.get("analogs", [])][:5],
            "high_exposure": [b for b, g in (st.get("exposure") or {}).items() if g == "high"],
            "facts": {k: f[k] for k in keys if k in f}, "brief": st.get("final_brief") or st.get("draft_brief") or "",
            "triage": st.get("triage"), "data_version": st.get("data_version")}


def news_date_et(topic: dict, heads: list[dict]) -> dt.date | None:
    """토픽 헤드라인 중 가장 이른 발행 시각(미국 동부 기준 날짜). 반응 창 시작일 계산에 쓴다."""
    from email.utils import parsedate_to_datetime
    from .track_record import ET
    ids, ds = set(topic["headline_ids"]), []
    for h in heads:
        if h["id"] in ids and h.get("published"):
            try:
                d = parsedate_to_datetime(h["published"])
                ds.append((d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)).astimezone(ET).date())
            except (TypeError, ValueError):
                continue
    return min(ds) if ds else None


def topic_match(topic: dict, heads: list[dict], st: dict) -> tuple[float, str]:
    """(2026-11-11) 토픽(제목·헤드라인)과 사건의 '수단 근거 문장'이 같은 이야기인지. 기사 뒷부분의 다른 정책 문장에
    끌려간 해석을 거른다. 반환: (최대 문장 유사도, 근거 문장)"""
    ev = st.get("event") or {}
    ids = set((ev.get("evidence") or {}).get("instrument") or [])
    ev_text = " ".join(s["text"] for s in st.get("sentences") or [] if s["id"] in ids)
    if not ev_text:
        return 0.0, ""
    by_id = {h["id"]: h["title"] for h in heads}
    refs = [topic["title"]] + [by_id[i] for i in topic["headline_ids"] if i in by_id][:8]
    idf = textsim.build_idf(refs + [ev_text])
    fe = _clean_feats(ev_text, idf)
    return max(textsim.cosine(_clean_feats(r, idf), fe) for r in refs), ev_text[:120]


def analyze(topic: dict, heads: list[dict], ctx, llm: LLM, cfg: dict, force: bool = False, fetcher=None,
            record: bool = True, now: dt.datetime | None = None) -> dict:
    from . import hq, track_record
    text, meta = topic_text(topic, heads, cfg.get("fetch_article", True), fetcher)
    st = hq.run_news(text, ctx, llm, force=force or topic.get("policy_event", False))
    out = summarize_state(st)
    out["input"] = meta
    if st.get("status") == "ok":
        sim, ev_text = topic_match(topic, heads, st)
        out["topic_match"] = round(sim, 3)
        if sim < cfg.get("topic_match_min", 0.1):
            out["topic_mismatch"] = ev_text
            record = False                                # 토픽과 다른 사건을 계산한 결과는 성적표에 넣지 않는다
    if st.get("status") == "not_policy":
        record = False
    if record:                                      # 실전 성적표: 예측을 기록해 두고 3거래일 뒤 자동 채점
        rec = track_record.make_record(st, "daily_watch", news_date_et(topic, heads), now, topic["title"])
        out["forecast"] = {k: rec[k] for k in ("window_start", "pure")} if rec else None
        track_record.append(rec)
    return out


# ---------------------------------------------------------------- 4. 보고서
CH_KO = {"risk_off": "위험회피(VIX)", "dollar": "달러", "cny": "위안화", "oil": "유가", "ust10y": "미 10년 금리",
         "credit": "신용 스프레드", "tech_war": "반도체 상대수익"}


def render(report: dict, labels: dict) -> str:
    from .reporter import COUNTRY_KO, INST_KO, SECTOR_KO
    L = [f"# ShockLens 데일리 워치 {report['date']}", "",
         f"생성 {report['generated_at']} · 가격 데이터 기준일 {report['data_version']} · 토픽 선택 {report['topic_mode']}"]
    if report.get("stale_note"):
        L += ["", f"> ⚠ {report['stale_note']}"]
    for reg in report["regions"]:
        L += ["", f"## {reg['label']} 핫토픽", ""]
        if reg["feed_errors"]:
            L.append(f"_막힌 피드: {', '.join(reg['feed_errors'])}_")
            L.append("")
        if not reg["topics"]:
            L.append("수집된 헤드라인이 없습니다 (피드가 모두 막힘).")
            continue
        L += ["| 순위 | 토픽 | 기사·매체 | 정책 사건 | 중요도 |", "|---|---|---|---|---|"]
        for i, t in enumerate(reg["topics"], 1):
            L.append(f"| {i} | {t['title']} | {len(t['headline_ids'])}건·{t['n_outlets']}곳 | "
                     f"{'예' if t['policy_event'] else '아니오'} | {t['importance']}/5 |")
        for i, t in enumerate(reg["topics"], 1):
            L += ["", f"### {i}. {t['title']}"]
            if t.get("market_path"):
                L.append(f"- 파급 경로(정성, LLM): {t['market_path']}")
            a = t.get("analysis")
            if not a:
                L.append("- ShockLens 정량 분석: " + (t.get("skip_reason") or "대상 아님"))
                continue
            if a["status"] == "not_policy":
                L.append(f"- ShockLens 정량 분석: 정책 사건 아님으로 종료 ({a.get('brief', '').splitlines()[1][2:] if a.get('brief') else ''})")
                continue
            if a.get("topic_mismatch") is not None:
                L.append(f"- ⚠ 토픽과 사건 근거가 다른 이야기로 보여 정량 결과를 보류합니다 (유사도 {a.get('topic_match')}). "
                         f"사건 해석의 근거 문장: \"{a['topic_mismatch']}\" · 계산 기록 {a['calc_log_id']}")
                continue
            if a["status"] == "skipped":
                L.append(f"- ShockLens 정량 분석: 1차 분류에서 제외 ({(a.get('triage') or {}).get('reason', '')})")
                continue
            ev, f = a["event"], a["facts"]
            if not ev.get("instrument"):
                L.append(f"- ShockLens 정량 분석: 사건으로 구조화하지 못함(판단 보류, 상태 {a['status']})")
                continue
            tg = ", ".join(COUNTRY_KO.get(c, c) for c in ev.get("target") or [])
            secs = ", ".join(SECTOR_KO.get(s, s) for s in ev.get("sectors") or []) or "-"
            chans = ", ".join(f"{CH_KO.get(k, k)} {'↑' if v > 0 else '↓'}" for k, v in (ev.get("channels") or {}).items())
            L.append(f"- 사건 해석({a['event_mode']}): {COUNTRY_KO.get(ev['initiator'], ev['initiator'])} → {tg}, "
                     f"{INST_KO.get(ev['instrument'], ev['instrument'])}, 산업 {secs}, 강도 {ev.get('severity')}/5")
            L.append(f"- 영향 채널: {chans or '-'} · 직접 노출 high: "
                     f"{', '.join(labels.get(b, b) for b in a['high_exposure']) or '없음'}")
            if f.get("ENS_ES"):
                L.append(f"- 20거래일 ES97.5: 앙상블 {f['ENS_ES']} (모형 범위 {f.get('ENS_LO')}~{f.get('ENS_HI')}), "
                         f"사건 효과 {f.get('EVENT_EFFECT')} (최악 {f.get('EVENT_EFFECT_SEVERE')}), 국면 {f.get('REGIME')}")
            if f.get("LOSS_CENTRAL"):
                L.append(f"- 발표 후 3거래일: 중앙 {f['LOSS_CENTRAL']}, 최악 시나리오 {f.get('LOSS_SEVERE')}, "
                         f"예측구간 {f.get('LOSS_BAND_LO')} ~ {f.get('LOSS_BAND_HI')}")
            if f.get("TOP1"):
                L.append(f"- 꼬리손실 기여: {f['TOP1']} {f.get('TOP1_SHARE')}, {f.get('TOP2')} {f.get('TOP2_SHARE')}, "
                         f"{f.get('TOP3')} {f.get('TOP3_SHARE')}")
            fc = a.get("forecast")
            if fc:
                L.append(f"- 성적표 기록: 반응 창 {fc['window_start']}부터 3거래일 "
                         f"({'순수 예측' if fc['pure'] else '사후 해석: 기록 시점에 이미 장이 열림'}), 창이 끝나면 자동 채점")
            L.append(f"- 근거: {f.get('TIER', a.get('analog_tier'))} {a['analogs']} · 입력 {a['input']['mode']}"
                     + (f" ({a['input']['url']})" if a['input'].get('url') else "") + f" · 계산 기록 {a['calc_log_id']}")
            L += ["", "<details><summary>전체 브리핑</summary>", "", a["brief"], "", "</details>"]
    L += ["", "## 실전 성적표", "", report.get("track_record_md", "")]
    L += ["", "---", "숫자는 모두 ShockLens Risk Engine 계산 결과(계산 기록 번호로 재현 가능)이며, 파급 경로 문장은 LLM의 정성 설명입니다. "
          "투자 권유가 아닙니다."]
    return "\n".join(L)


def out_path(base: Path, date: str) -> Path:
    p = base / f"{date}.md"
    v = 2
    while p.exists():
        p = base / f"{date}_v{v}.md"
        v += 1
    return p


def staleness(data_version: str, today: dt.date, max_days: int) -> str | None:
    try:
        d = dt.date.fromisoformat(data_version)
    except ValueError:
        return None
    import numpy as np
    gap = int(np.busday_count(d, today))
    if gap > max_days:
        return (f"가격 데이터가 {data_version} 기준으로 영업일 {gap}일 지났습니다. 국면(VIX) 판단이 오늘 시장을 반영하지 못합니다. "
                "`--refresh`로 갱신하세요.")
    return None


def run(regions: list[str] | None = None, analyze_n: int | None = None, force: bool = False, refresh: bool = False,
        cfg: dict | None = None, ctx=None, llm: LLM | None = None, fetcher=None, article_fetcher=None,
        today: dt.date | None = None, save: bool = True, now: dt.datetime | None = None) -> dict:
    cfg = cfg or load_cfg()
    if refresh:
        from pipeline import run_all
        run_all.main()
    if ctx is None:
        from .context import load_context
        ctx = load_context()
    llm = llm or LLM()
    today = today or dt.date.today()
    n_top, n_an = cfg.get("topics_per_region", 3), cfg.get("analyze_per_region", 1) if analyze_n is None else analyze_n
    from . import track_record
    scored = track_record.score_all(ctx)                # 지난 예측 중 반응 창이 끝난 것을 먼저 채점
    record_md = track_record.report_md(scored, len(track_record.load()) - len(scored))
    report = {"date": today.isoformat(), "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
              "data_version": ctx.data_version, "topic_mode": "llm" if llm.enabled else "rule", "regions": [],
              "stale_note": staleness(ctx.data_version, today, cfg.get("stale_data_days", 4)),
              "track_record_md": record_md}
    for code in (regions or list(cfg["regions"])):
        rc = cfg["regions"][code]
        heads, errs = collect(rc, cfg.get("max_headlines_per_feed", 40), fetcher=fetcher)
        topics = pick_topics(heads, rc["label"], n_top, llm, cfg.get("cluster_threshold", 0.35))
        done = 0
        for t in topics:
            if done >= n_an:
                t["skip_reason"] = f"분석 개수 한도({n_an}) 밖"
                continue
            if not (t["policy_event"] or force):
                t["skip_reason"] = "정책 사건이 아님 (지경학·국내 정책 사건만 정량 분석)"
                continue
            print(f"[daily_watch] {rc['label']} 분석: {t['title']}")
            t["analysis"] = analyze(t, heads, ctx, llm, cfg, force, article_fetcher, now=now)
            done += 1
        report["regions"].append({"code": code, "label": rc["label"], "n_headlines": len(heads),
                                  "feed_errors": errs, "topics": topics,
                                  "headlines": [{"id": h["id"], "title": h["title"], "outlets": sorted(h["outlets"]), "url": h["url"]}
                                                for h in heads]})
    md = render(report, ctx.labels)
    report["markdown"] = md
    if save:
        base = config.ROOT / cfg.get("output_dir", "data/daily")
        base.mkdir(parents=True, exist_ok=True)
        p = out_path(base, report["date"])
        p.write_text(md, encoding="utf-8")
        p.with_suffix(".json").write_text(json.dumps({k: v for k, v in report.items() if k != "markdown"},
                                                     ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        report["path"] = str(p)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="ShockLens 데일리 워치: 국내·미국 핫토픽과 파급효과")
    ap.add_argument("--regions", nargs="*", help="KR US (기본: 설정 파일의 모든 지역)")
    ap.add_argument("--analyze", type=int, help="지역별 정량 분석 토픽 수 (기본: 설정값)")
    ap.add_argument("--force", action="store_true", help="정책 사건이 아니어도 정량 분석")
    ap.add_argument("--refresh", action="store_true", help="먼저 가격·팩터 데이터 갱신 (pipeline.run_all)")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    rep = run(a.regions, a.analyze, a.force, a.refresh, save=not a.no_save)
    print(rep["markdown"])
    if rep.get("path"):
        print(f"\n[daily_watch] 저장: {rep['path']}")


if __name__ == "__main__":
    main()
