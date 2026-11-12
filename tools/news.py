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
import ipaddress
import re
import socket
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus, urljoin, urlsplit

import requests

UA = {"User-Agent": "Mozilla/5.0 (ShockLens research demo)"}
URL_RX = re.compile(r"^\s*https?://\S+\s*$")

# ---------------------------------------------------------------- URL 안전 검사 (SSRF)
# fetch_article에 들어오는 URL은 사용자 입력이거나 LLM이 기사 본문을 보고 고른 값이다.
# 즉 뉴스 본문에 주입된 지시가 그대로 '이 서버가 보내는 요청'이 될 수 있다. 내부망·클라우드
# 메타데이터(169.254.169.254)·localhost로 보내지 않도록 요청 전에 막는다.
ALLOWED_SCHEMES = ("http", "https")
MAX_REDIRECTS = 5
MAX_BYTES = 3_000_000        # 본문 상한. 무제한 응답으로 메모리를 채우는 것을 막는다


class BlockedURL(ValueError):
    """내부망·비표준 스킴을 가리켜 요청을 보내지 않았다."""


def _effective_ip(ip):
    """IPv6에 감싸인 IPv4를 꺼낸다(::ffff:127.0.0.1 등으로 우회하는 것을 막는다)."""
    for attr in ("ipv4_mapped", "sixtofour"):
        v = getattr(ip, attr, None)
        if v is not None:
            return v
    t = getattr(ip, "teredo", None)
    if t:
        return t[1]
    return ip


def assert_public_host(host: str) -> list[str]:
    """호스트를 해석해 모든 주소가 공개 주소인지 확인한다. 하나라도 아니면 BlockedURL."""
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError as e:
        raise BlockedURL(f"호스트 해석 실패: {host} ({type(e).__name__})") from e
    ips = []
    for info in infos:
        ip = _effective_ip(ipaddress.ip_address(info[4][0]))
        # is_global은 사설(10/8·172.16/12·192.168/16)·루프백·링크로컬(169.254/16)·예약 대역을 모두 걸러낸다
        if not ip.is_global or ip.is_multicast:
            raise BlockedURL(f"내부·예약 주소 차단: {host} -> {ip}")
        ips.append(str(ip))
    if not ips:
        raise BlockedURL(f"호스트 해석 실패: {host} (주소 없음)")
    return ips


def assert_fetchable(url: str) -> str:
    """스킴과 호스트를 검사하고 정리한 URL을 돌려준다."""
    u = (url or "").strip()
    parts = urlsplit(u)
    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise BlockedURL(f"허용하지 않는 스킴: {parts.scheme or '(없음)'}")
    if not parts.hostname:
        raise BlockedURL("호스트가 없는 URL")
    assert_public_host(parts.hostname)
    return u


def _read_capped(r, max_bytes: int) -> bytes:
    buf, total = [], 0
    for chunk in r.iter_content(65536):
        if not chunk:
            continue
        buf.append(chunk)
        total += len(chunk)
        if total >= max_bytes:
            break
    return b"".join(buf)[:max_bytes]


def _decode(r, raw: bytes) -> str:
    """헤더 charset -> 문서 meta charset -> 자동 추정 -> utf-8 순으로 디코딩한다."""
    enc = None
    m = re.search(r"charset=([\w-]+)", r.headers.get("Content-Type", ""), re.I)
    if m:
        enc = m.group(1)
    if not enc:
        m = re.search(rb'charset=["\']?([\w-]+)', raw[:4096], re.I)
        if m:
            enc = m.group(1).decode("ascii", "ignore")
    if not enc:
        try:                                            # requests가 들고 오는 charset_normalizer
            from charset_normalizer import from_bytes
            best = from_bytes(raw[:65536]).best()
            enc = best.encoding if best else None
        except Exception:
            enc = None
    try:
        return raw.decode(enc or "utf-8", errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _get_checked(url: str, timeout: float, max_bytes: int = MAX_BYTES,
                 max_redirects: int = MAX_REDIRECTS) -> tuple:
    """리다이렉트를 직접 따라가며 홉마다 주소를 다시 검사한다. (응답, 최종 URL, 본문 bytes)"""
    cur = assert_fetchable(url)
    for _ in range(max_redirects + 1):
        r = requests.get(cur, headers=UA, timeout=timeout, allow_redirects=False, stream=True)
        if r.is_redirect or r.is_permanent_redirect:
            loc = r.headers.get("Location") or ""
            r.close()
            if not loc:
                raise BlockedURL("Location이 없는 리다이렉트")
            cur = assert_fetchable(urljoin(cur, loc))       # 공개 -> 내부로 꺾는 리다이렉트를 막는다
            continue
        try:
            r.raise_for_status()
            return r, cur, _read_capped(r, max_bytes)
        except Exception:
            r.close()
            raise
    raise BlockedURL(f"리다이렉트가 {max_redirects}회를 넘었습니다")


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


def fetch_article(url: str, timeout: float = 10.0, max_chars: int = 20000,
                  max_bytes: int = MAX_BYTES) -> dict:
    """URL은 사용자·LLM이 준 값이므로 요청 전에 내부망 여부를 검사한다(assert_fetchable)."""
    r = None
    try:
        r, _final, raw = _get_checked(url, timeout, max_bytes)
        title, text = extract_text(_decode(r, raw))
        text = text[:max_chars]
        return {"url": url, "title": title, "text": text, "chars": len(text),
                "error": None if text else "본문을 찾지 못함"}
    except Exception as e:
        return {"url": url, "title": "", "text": "", "chars": 0, "error": f"{type(e).__name__}: {e}"[:120]}
    finally:
        if r is not None:
            r.close()


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
