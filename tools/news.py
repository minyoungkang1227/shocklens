"""뉴스 입력 도구: 헤드라인 검색과 기사 본문 추출.

  search_news("China rare earth export control")  -> [{title, url, source, date}]
  fetch_article("https://...")                     -> {url, title, text, chars}
  input_kind(text)                                 -> "url" | "headline" | "article"

검색: GDELT DOC API(무료, 실제 기사 URL) -> 실패하면 Google News RSS(제목·출처만).
본문: trafilatura가 설치돼 있으면 사용, 없으면 <p> 태그 텍스트를 모은다.
네트워크가 막히면 예외 대신 빈 결과와 error를 돌려준다 (호출한 쪽이 헤드라인만으로 계속 진행).
"""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import requests

UA = {"User-Agent": "Mozilla/5.0 (ShockLens research demo)"}
URL_RX = re.compile(r"^\s*https?://\S+\s*$")


def input_kind(text: str, headline_max_chars: int = 140) -> str:
    t = text.strip()
    if URL_RX.match(t):
        return "url"
    n_sent = len([p for p in re.split(r"(?<=[.!?。！？])\s*", t) if len(p.strip()) >= 4])
    if len(t) <= headline_max_chars and n_sent <= 1:
        return "headline"
    return "article"


GDELT_TIMEOUT = 4.0          # (2026-11-11) GDELT가 느리면 오래 기다리지 않고 Google News RSS로 넘어간다
GDELT_COOLDOWN = 600         # 타임아웃·오류가 나면 이 시간(초) 동안 GDELT를 건너뛴다
_gdelt_down_until = {"t": 0.0}


def search_news(query: str, max_results: int = 5, timeout: float = 8.0,
                start: str | None = None, end: str | None = None) -> dict:
    """{results: [...], engine, error}. start/end = 'YYYY-MM-DD' (GDELT 기간 검색, 2017년 이후)"""
    import time
    err = []
    if time.time() < _gdelt_down_until["t"] and not start:
        err.append("gdelt: 직전 실패로 잠시 건너뜀")
    params = {"query": query, "mode": "ArtList", "format": "json", "maxrecords": max_results, "sort": "HybridRel"}
    if start:
        params["startdatetime"] = start.replace("-", "") + "000000"
    if end:
        params["enddatetime"] = end.replace("-", "") + "235959"
    try:
        if err:
            raise RuntimeError("skip")
        r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, headers=UA,
                         timeout=min(timeout, GDELT_TIMEOUT))
        r.raise_for_status()
        arts = (r.json() or {}).get("articles", [])
        res = [{"title": a.get("title", ""), "url": a.get("url", ""), "source": a.get("domain", ""),
                "date": a.get("seendate", "")[:8], "lang": a.get("language", "")} for a in arts[:max_results]]
        if res or start:
            return {"results": res, "engine": "gdelt", "error": None}
    except Exception as e:
        if not err:
            err.append(f"gdelt: {type(e).__name__}")
            _gdelt_down_until["t"] = time.time() + GDELT_COOLDOWN
    try:
        hl = "ko" if re.search(r"[가-힣]", query) else "en"
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl={hl}"
        r = requests.get(url, headers=UA, timeout=timeout)
        r.raise_for_status()
        items = ET.fromstring(r.content).findall(".//item")[:max_results]
        res = [{"title": html.unescape(i.findtext("title", "")), "url": i.findtext("link", ""),
                "source": i.findtext("source", ""), "date": i.findtext("pubDate", "")} for i in items]
        return {"results": res, "engine": "google_news_rss", "error": "; ".join(err) or None}
    except Exception as e:
        err.append(f"google_rss: {type(e).__name__}")
    return {"results": [], "engine": None, "error": "; ".join(err)}


def _strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", s)).strip()


def extract_text(raw_html: str) -> tuple[str, str]:
    """(제목, 본문). trafilatura가 있으면 그것을, 없으면 <p> 문단을 모은다."""
    m = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', raw_html, re.I) or \
        re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.I | re.S)
    title = _strip_tags(m.group(1)) if m else ""
    try:
        import trafilatura
        body = trafilatura.extract(raw_html) or ""
        if body:
            return title, body
    except ImportError:
        pass
    clean = re.sub(r"<(script|style|noscript|nav|footer|header|aside)[^>]*>.*?</\1>", " ", raw_html, flags=re.I | re.S)
    paras = [_strip_tags(p) for p in re.findall(r"<p[^>]*>(.*?)</p>", clean, re.I | re.S)]
    paras = [re.sub(r"\s+", " ", p) for p in paras if len(p) >= 30]
    return title, "\n".join(paras)


def fetch_article(url: str, timeout: float = 10.0, max_chars: int = 20000) -> dict:
    try:
        r = requests.get(url.strip(), headers=UA, timeout=timeout)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or r.encoding
        title, text = extract_text(r.text)
        text = text[:max_chars]
        return {"url": url, "title": title, "text": text, "chars": len(text),
                "error": None if text else "본문을 찾지 못함"}
    except Exception as e:
        return {"url": url, "title": "", "text": "", "chars": 0, "error": f"{type(e).__name__}: {e}"[:120]}


def timeline_volume(query: str, start: str | None = None, end: str | None = None, timeout: float = 10.0) -> dict:
    """GDELT 보도량 타임라인 (전체 기사 중 해당 검색어 기사 비율, 일별). {points: [(YYYY-MM-DD, value)], error}"""
    params = {"query": query, "mode": "TimelineVol", "format": "json"}
    if start:
        params["startdatetime"] = start.replace("-", "") + "000000"
    if end:
        params["enddatetime"] = end.replace("-", "") + "235959"
    if not start and not end:
        params["timespan"] = "3months"
    try:
        r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, headers=UA, timeout=timeout)
        r.raise_for_status()
        series = (r.json() or {}).get("timeline", [])
        data = series[0].get("data", []) if series else []
        pts = [(str(d.get("date", ""))[:8], float(d.get("value", 0) or 0)) for d in data]
        pts = [(f"{d[:4]}-{d[4:6]}-{d[6:8]}", v) for d, v in pts if len(d) == 8]
        return {"points": pts, "error": None if pts else "빈 타임라인"}
    except Exception as e:
        return {"points": [], "error": f"{type(e).__name__}"}
