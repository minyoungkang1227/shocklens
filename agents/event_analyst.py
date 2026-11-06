"""Event Analyst: 뉴스 텍스트 -> 구조화된 사건(Event).

LLM이 초안을 만들고, 코드 가드레일이 검사·보정한다. LLM이 없으면 키워드 규칙으로 초안을 만든다.

가드레일 (코드)
1. 어휘 검사: 수단·단계·국가·산업·팩터가 정해진 값인지. 틀리면 오류를 알려 1회 재요청, 그래도 틀리면 보정·기록.
2. 근거 검사: evidence의 문장 id가 실제로 있는지. 없는 id는 지우고, 근거 없는 필드는 uncertain_fields로.
3. 채널 보정: 비었거나 어휘 밖이면 수단별 기본 채널로 채움. 방향과 위험회피 부호가 모순이면 고침.
4. 동시발생 충격: LLM 목록 + 거시 키워드(금리 결정·고용·물가 등) 스캔을 합침.
5. 국부펀드 표적: 금융제재 + 중앙은행·국부펀드·외환보유액 언급이면 표시.
6. 필수 필드(수단·발동국·대상국)가 비면 event_complete=False -> 헤드쿼터가 재실행 또는 판단 보류.
"""
from __future__ import annotations

import re

from pydantic import ValidationError

from engine import severity as sev
from engine.textsim import text_direction

import statistics

from . import injection, prompts
from .llm import LLM
from .state import (COUNTRIES, DOMESTIC_INSTRUMENTS, FACTORS, INSTRUMENTS, SECTORS, STAGES, Event, default_channels)

REQUIRED = ["instrument", "initiator", "target"]

# ------------------------------------------------------------- 문장 나누기
def split_sentences(text: str) -> list[dict]:
    parts = re.split(r"(?<=[.!?])\s+|(?<=[。！？])\s*|\n+", text.strip())      # 중국어·일본어는 문장부호 뒤 공백이 없다
    sents = [p.strip() for p in parts if len(p.strip()) >= 4]
    return [{"id": f"s{i + 1}", "text": s} for i, s in enumerate(sents)]


def numbered(sentences: list[dict]) -> str:
    return "\n".join(f"[{s['id']}] {s['text']}" for s in sentences)


# ------------------------------------------------------------- 규칙 기반 대체 경로 (LLM 없을 때)
KW_INSTRUMENT = [   # 위에서부터 먼저 걸리는 것 (구체적인 수단 먼저)
    ("chokepoint_disruption", r"해협 ?봉쇄|봉쇄|해상 ?봉쇄|운하|항로 ?차단|blockade|strait|chokepoint|canal|封锁|海峡"),
    ("investment_restriction", r"해외 ?투자 ?(제한|심사|금지)|투자 ?제한|outbound investment|investment (ban|restriction|screening)|CFIUS|对外投资"),
    # 국내 규제: '수출 규제'와 헷갈리지 않게 대상(반독점·데이터센터·플랫폼·AI)이 붙은 표현만
    ("domestic_regulation", r"반독점|독점 ?금지|antitrust|데이터 ?센터[^.]{0,25}(규제|유예|중단|제한|모라토리엄|반대)|"
                            r"(AI|인공지능|플랫폼|빅테크)[^.]{0,10}(규제|단속)|moratorium|data ?cent(er|re)s?[^.]{0,30}(regulat|curb|ban|pause|restrict)|"
                            r"(AI|tech|platform) (regulation|crackdown)|crackdown|反垄断|监管|整顿"),
    ("industrial_subsidy", r"보조금|세액 ?공제|인플레이션 ?감축법|반도체법|CHIPS|IRA(?![A-Za-z])|subsid|tax credit|현지화|local content|补贴"),
    ("financial_sanction", r"SWIFT|자산 ?동결|금융제재|외환보유액|sanction|freeze|制裁|冻结"),
    ("entity_list", r"엔티티 ?리스트|entity list|거래 ?제한 ?명단|实体清单|不可靠实体"),
    ("tariff_pause", r"관세[^.]{0,20}(유예|인하|철회|합의)|유예|휴전|pause|truce|rollback|공동성명|joint statement|暂停|休战|联合声明"),
    ("resource_export_control", r"희토류|갈륨|게르마늄|안티몬|흑연|광물|rare earth|gallium|germanium|graphite|稀土|镓|锗|锑|石墨"),
    ("tech_export_control", r"반도체[^.]{0,20}(수출|통제)|수출통제|export control|칩 ?통제|出口管制|芯片"),
    ("currency", r"위안화|환율조작|절하|devalu|currency manipulator|汇率|贬值"),
    ("tariff", r"관세|tariff|301조|关税|加征"),
    ("election_shift", r"중간선거|대선|총선|선거 ?결과|정권 ?교체|election|midterm|选举|大选"),
]
KW_COUNTRY = {
    "US": r"미국|미 정부|백악관|트럼프|바이든|상무부 BIS|BIS|USTR|U\.S\.|United States|Washington|美国|美方|"
          r"美(?!元)|(?<![가-힣])미 (?=[가-힣])|뉴욕증시|월가|나스닥|연준|Wall Street|Nasdaq|Dow(?![a-z])|Congress|Senate|Democrats|Republicans",
    "CN": r"중국|베이징|시진핑|China|Beijing|中国|中方|商务部",
    "EU": r"유럽|(?<![A-Za-z])EU(?![A-Za-z])|European",
    "JP": r"일본|Japan|日本",
    "TW": r"대만|Taiwan|TSMC|台湾",
    "KR": r"한국|Korea|코스피|코스닥|KOSPI",
    "RU": r"러시아|Russia|俄罗斯",
    "IN": r"인도(?!네시아)|India",
}
KW_SECTOR = {
    "semiconductor": r"반도체|칩|chip|semiconductor|TSMC|엔비디아|Nvidia|半导体|芯片",
    "defense": r"방산|군사|defen[cs]e",
    "autos": r"자동차|전기차|(?<![A-Za-z])autos?(?![A-Za-z])|(?<![A-Za-z])EVs?(?![A-Za-z])",
    "energy": r"원유|석유|가스|에너지|(?<![A-Za-z])oil(?![A-Za-z])|(?<![A-Za-z])gas(?![A-Za-z])|energy",
    "banks": r"은행|금융기관|(?<![A-Za-z])banks?(?![A-Za-z])",
    "telecom": r"통신|화웨이|Huawei|5G",
    "agriculture": r"농산물|대두|agricultur|soybean",
    "critical_minerals": r"희토류|광물|갈륨|게르마늄|rare earth|mineral",
    "battery": r"배터리|리튬|흑연|battery|lithium",
    "industrials": r"기계|산업재|철강|알루미늄|steel|aluminum|machinery",
    "tech_platform": r"기술주|tech (stocks|shares)|플랫폼|빅테크|인터넷|클라우드|소프트웨어|platform|big tech|internet|cloud|software|알리바바|텐센트|Alibaba|Tencent|平台|互联网",
    "ai_infra": r"데이터 ?센터|AI ?인프라|인공지능 ?인프라|하이퍼스케일|data ?cent(er|re)|AI infrastructure|hyperscal|코어위브|CoreWeave|数据中心",
    "utilities": r"전력|전기 ?요금|유틸리티|송전|utilit|power grid|electricity",
}
KW_STAGE = [
    ("ease", r"유예|인하|합의|완화|철회|pause|truce"),
    ("retaliate", r"보복|맞대응|대응 조치|retaliat"),
    ("enforce", r"시행|발효|즉시|effective|금지"),
    ("escalate", r"추가|확대|인상|강화|escalat"),
]
MACRO_SHOCK = r"(FOMC|기준금리|금리 ?(인상|인하|결정)|고용 ?지표|고용보고서|비농업|ADP|CPI|소비자물가|물가 ?지표|실적 ?(쇼크|발표)|침공|전쟁|셧다운)"
SOVEREIGN = r"(중앙은행|외환보유액|국부펀드|sovereign|central bank|reserves)"


def _first_sentence_ids(sentences: list[dict], pattern: str) -> list[str]:
    return [s["id"] for s in sentences if re.search(pattern, s["text"], re.I)]


def rule_extract(sentences: list[dict]) -> dict:
    text = " ".join(s["text"] for s in sentences)
    ev: dict = {"evidence": {}, "uncertain_fields": []}
    for inst, pat in KW_INSTRUMENT:
        ids = _first_sentence_ids(sentences, pat)
        if ids:
            ev["instrument"], ev["evidence"]["instrument"] = inst, ids[:2]
            break
    ev["direction"] = "ease" if ev.get("instrument") == "tariff_pause" else "escalate"
    hits = {c: _first_sentence_ids(sentences, p) for c, p in KW_COUNTRY.items()}
    hits = {c: ids for c, ids in hits.items() if ids}
    order = sorted(hits, key=lambda c: (min(int(i[1:]) for i in hits[c]), -len(hits[c])))
    # 조사로 역할 추정: '러시아를', '중국에 대한', '중국산' -> 대상국 / '미국과', '미국은' -> 발동국 쪽
    tgt_mark = re.compile(r"^\s*(을|를|에 대한|에 대해|산|(?:의\s*)?(?:주요|중앙은행|기업|은행|반도체|정부))")
    init_mark = re.compile(r"^\s*(과|와|은|는|이|가)(?:\s|$)")
    targets_hint, init_hint = set(), set()
    for c in order:
        for m in re.finditer(KW_COUNTRY[c], text, re.I):
            nxt = text[m.end():m.end() + 10]
            if tgt_mark.match(nxt):
                targets_hint.add(c)
            elif init_mark.match(nxt):
                init_hint.add(c)
    if order:
        initiators = [c for c in order if c not in targets_hint] or order[:1]
        co_initiators = [c for c in initiators[1:] if c in init_hint and c not in targets_hint]
        ev["initiator"] = initiators[0]
        ev["evidence"]["initiator"] = hits[initiators[0]][:1]
        tg = [c for c in order if c in targets_hint and c != initiators[0]]
        if not tg:
            tg = [c for c in order if c != initiators[0] and c not in co_initiators][:2]
        if co_initiators:
            ev.setdefault("notes", []).append(f"공동 발동국으로 보이는 국가: {co_initiators}")
        if ev.get("instrument") in DOMESTIC_INSTRUMENTS:
            tg = [initiators[0]]                       # 국내 정책: 자국 산업이 대상
        ev["target"] = tg[:3]
        ev["evidence"]["target"] = sum((hits[c][:1] for c in ev["target"]), [])
    ev["sectors"] = [s for s, p in KW_SECTOR.items() if re.search(p, text, re.I)]
    if ev["sectors"]:
        ev["evidence"]["sectors"] = sum((_first_sentence_ids(sentences, KW_SECTOR[s])[:1] for s in ev["sectors"]), [])
    for st, pat in KW_STAGE:
        if st == "ease" and ev["direction"] != "ease":
            continue
        ids = _first_sentence_ids(sentences, pat)
        if ids:
            ev["stage"], ev["evidence"]["stage"] = st, ids[:1]
            break
    imm = 5 if re.search(r"즉시|즉각|immediately|effective today", text, re.I) else (
        2 if re.search(r"다음 ?달|개월|유예|검토|예고|consider|plan", text, re.I) else 3)
    scope = 5 if re.search(r"전면|전체|모든|all |across", text, re.I) else (
        2 if re.search(r"품목|기업 \d|몇 ?곳|일부", text, re.I) else 3)
    subst = 5 if re.search(r"희토류|갈륨|게르마늄|첨단 ?(장비|칩)|rare earth|독점", text, re.I) else (
        1 if re.search(r"철강|타이어|농산물|steel|tire", text, re.I) else 3)
    esc = 4 if re.search(r"보복|맞대응|retaliat", text, re.I) else (3 if re.search(r"추가|확대|강화", text) else 1)
    parties = 5 if ("US" in [ev.get("initiator")] + ev.get("target", []) and "CN" in [ev.get("initiator")] + ev.get("target", [])) else 3
    surprise = 4 if re.search(r"기습|전격|깜짝|예상 ?밖|surprise|unexpected", text, re.I) else (
        1 if re.search(r"예고|예정대로|알려진|as expected|scheduled", text, re.I) else 3)
    ev["severity_components"] = {"scope": scope, "immediacy": imm, "substitut": subst, "escalation": esc,
                                 "parties": parties, "surprise": surprise}
    first = sentences[0]["text"] if sentences else ""
    ev["title"] = first if len(first) <= 45 else ""          # 길면 Reporter가 구조 필드로 제목을 만든다
    ev["uncertain_fields"].append("severity")          # 규칙 경로의 강도는 항상 불확실 표시
    return ev


# ------------------------------------------------------------- 가드레일
def guard(draft: dict, sentences: list[dict], raw_text: str,
          severity_cfg: dict | None = None, exclude_ids: set[str] | None = None) -> tuple[Event | None, list[str]]:
    """초안 dict를 검사·보정해 Event로 만든다. (Event, 남은 오류 목록)
    exclude_ids: 지시문 의심 문장. 근거로 인정하지 않는다."""
    notes, errors = [], []
    d = {k: v for k, v in (draft or {}).items() if v not in (None, "")}
    valid_ids = {s["id"] for s in sentences} - set(exclude_ids or ())
    if exclude_ids and any(i in exclude_ids for ids in (d.get("evidence") or {}).values() for i in (ids or [])):
        notes.append(f"지시문 의심 문장 {sorted(exclude_ids)} 을 근거에서 제외")

    # 1. 어휘
    if d.get("instrument") not in INSTRUMENTS:
        errors.append(f"instrument '{d.get('instrument')}' 는 어휘 밖")
        d.pop("instrument", None)
    if d.get("stage") not in STAGES:
        if "stage" in d:
            notes.append(f"stage '{d['stage']}' -> announce 로 보정")
        d["stage"] = "ease" if d.get("direction") == "ease" else "announce"
    if d.get("direction") not in ("escalate", "ease"):
        d["direction"] = "ease" if d.get("instrument") == "tariff_pause" or d.get("stage") == "ease" else "escalate"
    # 기사 단어 방향과 대조: 완화로 읽었는데 확대어가 압도적이면(예: '유예 종료', '재인상') 확대로 고친다
    td = text_direction(raw_text)
    if d["direction"] == "ease" and td["score"] >= 2 and not td["ease_words"]:
        notes.append(f"기사 단어가 확대 방향({td['escalate_words'][:3]})이라 ease -> escalate 로 보정")
        d["direction"] = "escalate"
        if d.get("instrument") == "tariff_pause":
            d["instrument"] = "tariff"
        d["stage"] = "escalate"
        unc_dir = True
    elif d["direction"] == "escalate" and td["score"] <= -2 and not td["escalate_words"]:
        notes.append(f"기사 단어가 완화 방향({td['ease_words'][:3]})이라 escalate -> ease 로 보정")
        d["direction"] = "ease"
        d["stage"] = "ease"
        if d.get("instrument") == "tariff":
            d["instrument"] = "tariff_pause"
        unc_dir = True
    else:
        unc_dir = False
    if d["direction"] == "escalate" and d["stage"] == "ease":
        notes.append("확대 사건인데 stage가 ease -> enforce 로 보정")
        d["stage"] = "enforce"
    if d["direction"] == "ease" and d["stage"] != "ease":
        d["stage"] = "ease"
    if d.get("instrument") == "tariff_pause" and d["direction"] != "ease":
        if td["label"] == "escalate":                # 기사도 확대 -> 수단을 관세로
            notes.append("tariff_pause 인데 기사·방향 모두 확대 -> instrument를 tariff 로 보정")
            d["instrument"] = "tariff"
        else:                                         # 기사가 완화거나 불분명 -> 방향을 완화로
            notes.append("tariff_pause 인데 direction이 escalate -> ease 로 보정")
            d["direction"], d["stage"] = "ease", "ease"
    init = str(d.get("initiator", "")).upper()
    if init not in COUNTRIES:
        if init:
            errors.append(f"initiator '{init}' 는 어휘 밖")
        d.pop("initiator", None)
    else:
        d["initiator"] = init
    domestic = d.get("instrument") in DOMESTIC_INSTRUMENTS
    tgt = [str(c).upper() for c in d.get("target", [])
           if str(c).upper() in COUNTRIES and (domestic or str(c).upper() != init)]
    if len(tgt) != len(d.get("target", [])):
        notes.append("어휘 밖이거나 발동국과 같은 대상국 제거")
    if domestic and not tgt and init in COUNTRIES:
        tgt = [init]                                   # 국내 정책 사건은 자국 산업이 대상
        notes.append(f"국내 정책 사건이라 대상국을 발동국({init})으로 채움")
        if "initiator" in (d.get("evidence") or {}):
            d["evidence"]["target"] = list(d["evidence"]["initiator"])
    d["target"] = tgt
    secs = [s for s in d.get("sectors", []) if s in SECTORS]
    if len(secs) != len(d.get("sectors", [])):
        notes.append(f"어휘 밖 산업 제거: {sorted(set(d.get('sectors', [])) - set(secs))}")
    d["sectors"] = secs
    # 강도: 6요소 점수 -> 연속 점수 -> 등급. 요소가 없으면 severity 숫자(또는 3)로 환산
    raw_c = d.get("severity_components") or {}
    comps = {k: sev.clamp(raw_c[k]) for k in sev.COMPONENTS if k in raw_c}
    missing_c = [k for k in sev.COMPONENTS if k not in comps]
    if comps and missing_c:
        notes.append(f"강도 요소 누락 {missing_c} -> 3으로 채움")
        comps.update({k: 3 for k in missing_c})
    weights = (severity_cfg or {}).get("component_weights")
    if comps:
        d["severity_components"] = comps
        d["severity_points"] = sev.score(comps, weights)
    else:
        try:
            lvl = min(max(int(d.get("severity", 3)), 1), 5)
        except (TypeError, ValueError):
            lvl = 3
            notes.append("severity 해석 실패 -> 3")
        d["severity_points"] = sev.score(None, weights, lvl)
        notes.append("강도 요소 점수 없음 -> 단일 강도 값으로 환산")
    d["severity"] = sev.level(d["severity_points"])

    # 2. 근거
    ev_map = {}
    for field, ids in (d.get("evidence") or {}).items():
        ids = [i for i in (ids or []) if i in valid_ids]
        if ids:
            ev_map[field] = ids
    d["evidence"] = ev_map
    unc = set(d.get("uncertain_fields") or [])
    if unc_dir:
        unc.add("direction")
    for f in ["instrument", "initiator", "target", "sectors", "stage"]:
        if f not in ev_map:
            unc.add(f)
    if comps and not any(k in ev_map for k in sev.COMPONENTS):
        unc.add("severity")
    if not comps:
        unc.add("severity")

    # 3. 채널
    if d.get("instrument"):
        raw_ch = d.get("channels") or {}
        ch = {}
        for f, v in raw_ch.items():
            try:
                v = int(v)
            except (TypeError, ValueError):
                continue
            if f in FACTORS and v in (-1, 1):
                ch[f] = v
        if not ch:
            ch = default_channels(d["instrument"], d["direction"])
            notes.append("영향 채널을 수단별 기본값으로 채움")
            unc.add("channels")
        want = 1 if d["direction"] == "escalate" else -1
        if ch.get("risk_off", want) != want:
            notes.append(f"방향({d['direction']})과 risk_off 부호 모순 -> {want:+d} 로 보정")
            ch["risk_off"] = want
        d["channels"] = ch

    # 4. 동시발생 충격
    conc = list(d.get("concurrent_shocks") or [])
    for s in sentences:
        if s["id"] in (exclude_ids or ()):
            continue
        m = re.search(MACRO_SHOCK, s["text"], re.I)
        if m and not any(m.group(0) in c for c in conc):
            conc.append(f"{m.group(0)} 언급 [{s['id']}]")
    d["concurrent_shocks"] = conc

    # 5. 국부펀드 표적
    if d.get("instrument") == "financial_sanction" and re.search(SOVEREIGN, raw_text, re.I):
        d["sovereign_asset_risk"] = True

    d["uncertain_fields"] = sorted(unc)
    d["notes"] = notes + list(d.get("notes") or [])
    missing = [f for f in REQUIRED if not d.get(f)]
    if missing:
        return None, errors + [f"필수 필드 없음: {missing}"]
    keep = set(Event.model_fields)
    try:
        return Event(**{k: v for k, v in d.items() if k in keep}), errors
    except ValidationError as e:
        return None, errors + [str(e)]


# ------------------------------------------------------------- 강도 자기일관성
def _median_int(xs: list[int]) -> int:
    return int(statistics.median_low(sorted(xs))) if len(xs) % 2 == 0 else int(statistics.median(xs))


def consensus(samples: list[dict]) -> tuple[dict, dict]:
    """강도 표본 여러 개 -> (요소별 중앙값, 요소별 최대-최소)"""
    keys = [k for k in sev.COMPONENTS if all(k in s for s in samples)]
    med = {k: _median_int([sev.clamp(s[k]) for s in samples]) for k in keys}
    spread = {k: max(sev.clamp(s[k]) for s in samples) - min(sev.clamp(s[k]) for s in samples) for k in keys}
    return med, spread


def _score_once(head: dict, news_block: str, llm: LLM) -> dict | None:
    user = f"사건 구조: {head}\n\n{news_block}"
    try:
        r = llm.complete_json(prompts.SEVERITY_SYSTEM, user, max_tokens=600,
                              schema=prompts.SEVERITY_SCHEMA, name="emit")
        c = (r or {}).get("severity_components") or {}
        return c if all(k in c for k in sev.COMPONENTS) else None
    except Exception:
        return None


def resample_severity(draft: dict, news_block: str, llm: LLM, n: int, small_share: int = 0) -> list[dict]:
    """같은 뉴스로 강도만 n-1번 더 채점한다. small_share 번은 작은 모델(Haiku)로 채점해 같은 모델이 같은 실수를
    반복하는 것을 막는다(모델 다양성). 실패한 표본은 버린다. 각 표본에 '_model' 키로 출처를 남긴다."""
    head = {k: draft.get(k) for k in ("title", "instrument", "direction", "initiator", "target", "sectors", "stage")}
    out = []
    extra = max(n - 1, 0)
    small = llm.derive(small=True) if small_share > 0 else None
    for i in range(extra):
        use_small = small is not None and i < min(small_share, extra) and small.model != llm.model
        c = _score_once(head, news_block, small if use_small else llm)
        if c:
            out.append({**c, "_model": (small.model if use_small else llm.model)})
    return out


def sample_severity(draft: dict, news_block: str, llm: LLM, sc: dict) -> list[dict]:
    """자기일관성 + 적응형: 기본 n회(그중 small_share회는 Haiku). 요소 퍼짐이 spread_warn 이상이면 extra_n회 더 채점.
    확실한 사건엔 비용을 안 쓰고 애매한 사건에만 더 쓴다."""
    first = dict(draft.get("severity_components") or {})
    first["_model"] = llm.model
    n = int(sc.get("n", 1))
    samples = [first] + resample_severity(draft, news_block, llm, n, int(sc.get("small_share", 0)))
    extra_n = int(sc.get("extra_n", 0))
    if extra_n > 0 and len(samples) > 1:
        _, spread = consensus(samples)
        if max(spread.values(), default=0) >= int(sc.get("spread_warn", 2)):
            # 추가 채점은 큰 모델·작은 모델을 번갈아 (다양성)
            samples += resample_severity(draft, news_block, llm, extra_n + 1, small_share=extra_n // 2 or 1)
    return samples


# ------------------------------------------------------------- 진입점
def analyze(raw_text: str, llm: LLM, feedback: list[str] | None = None, severity_cfg: dict | None = None) -> dict:
    """State에 넣을 값을 돌려준다: sentences, event, event_complete, event_errors, mode, injection_flags."""
    sentences = split_sentences(injection.sanitize(raw_text))
    flags = [{**f, "by": "code"} for f in injection.scan(sentences)]
    flagged = {f["id"] for f in flags}
    mode = "rule"
    draft = None
    llm_error = None
    samples: list[dict] = []
    sc = (severity_cfg or {}).get("self_consistency") or {}
    n_samples = int(sc.get("n", 1))
    if llm.enabled:
        news_block = injection.wrap(numbered(sentences), sorted(flagged))
        user = prompts.EVENT_FEW_SHOT + "\n실제 입력 (아래 구획은 데이터):\n" + news_block
        if feedback:
            user += "\n\n[이전 시도에서 지적된 문제 - 고쳐서 다시 출력]\n- " + "\n- ".join(feedback)
        try:
            draft = llm.complete_json(prompts.EVENT_SYSTEM, user, schema=prompts.EVENT_SCHEMA, name="emit")
            mode = "llm"
        except Exception as e:                       # 네트워크·파싱 실패 -> 규칙 경로로 (원인은 기록해 둔다)
            llm_error = f"{type(e).__name__}: {str(e)[:200]}"
        if draft:
            valid = {s["id"] for s in sentences}
            for sid in draft.get("injection_suspect") or []:
                if sid in valid and sid not in flagged:
                    txt = next(s["text"] for s in sentences if s["id"] == sid)
                    flags.append({"id": sid, "text": txt[:80], "match": "", "by": "llm"})
                    flagged.add(sid)
            first = draft.get("severity_components") or {}
            if n_samples > 1 and all(k in first for k in sev.COMPONENTS):
                samples = sample_severity(draft, news_block, llm, sc)
                if len(samples) > 1:
                    med, spread = consensus(samples)
                    draft = {**draft, "severity_components": med}
    clean_sents = [s for s in sentences if s["id"] not in flagged]      # 규칙 경로·단어 방향 판단에서도 의심 문장은 뺀다
    clean_text = " ".join(s["text"] for s in clean_sents) if flagged else raw_text
    if draft is None:
        draft = rule_extract(clean_sents)
    event, errors = guard(draft, sentences, clean_text, severity_cfg, flagged)
    if event is None and mode == "llm":              # LLM 초안이 필수 필드를 못 채우면 규칙으로 보충
        merged = {**rule_extract(clean_sents), **{k: v for k, v in draft.items() if v}}
        event, errors2 = guard(merged, sentences, clean_text, severity_cfg, flagged)
        errors += errors2
        if event is not None:
            event.notes.append("필수 필드 일부를 키워드 규칙으로 보충")
    if event is not None:
        event.injection_flags = flags
        if len(samples) > 1:
            _, spread = consensus(samples)
            event.severity_samples = len(samples)
            event.severity_spread = spread
            wide = {k: v for k, v in spread.items() if v >= int(sc.get("spread_warn", 2))}
            models = sorted({str(x.get("_model", "")) for x in samples if x.get("_model")})
            who = f", 모델 {len(models)}종" if len(models) > 1 else ""
            adaptive = " (흩어져서 추가 채점)" if len(samples) > int(sc.get("n", 1)) else ""
            if wide:
                if "severity" not in event.uncertain_fields:
                    event.uncertain_fields = sorted(set(event.uncertain_fields) | {"severity"})
                event.notes.append(f"강도 {len(samples)}회 채점{who}{adaptive}이 흩어짐 {wide} -> 중앙값 사용, 강도 불확실")
            else:
                event.notes.append(f"강도 {len(samples)}회 채점{who}{adaptive} 일치 (최대 흩어짐 {max(spread.values(), default=0)})")
        if flags:
            event.notes.append(f"뉴스 안 지시문 의심 문장 {sorted(flagged)} -> 근거·출처에서 제외")
        if llm_error:
            event.notes.append(f"LLM 호출 실패로 규칙 경로 사용: {llm_error}")
    return {
        "sentences": sentences,
        "event": event.model_dump() if event else {},
        "event_complete": event is not None,
        "event_errors": errors,
        "mode": mode,
        "llm_error": llm_error,
        "injection_flags": flags,
    }
