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

from . import prompts
from .llm import LLM
from .state import (COUNTRIES, FACTORS, INSTRUMENTS, SECTORS, STAGES, Event, default_channels)

REQUIRED = ["instrument", "initiator", "target"]

# ------------------------------------------------------------- 문장 나누기
def split_sentences(text: str) -> list[dict]:
    parts = re.split(r"(?<=[.!?。])\s+|\n+", text.strip())
    sents = [p.strip() for p in parts if len(p.strip()) >= 4]
    return [{"id": f"s{i + 1}", "text": s} for i, s in enumerate(sents)]


def numbered(sentences: list[dict]) -> str:
    return "\n".join(f"[{s['id']}] {s['text']}" for s in sentences)


# ------------------------------------------------------------- 규칙 기반 대체 경로 (LLM 없을 때)
KW_INSTRUMENT = [   # 위에서부터 먼저 걸리는 것 (구체적인 수단 먼저)
    ("financial_sanction", r"SWIFT|자산 ?동결|금융제재|외환보유액|sanction|freeze"),
    ("entity_list", r"엔티티 ?리스트|entity list|거래 ?제한 ?명단"),
    ("tariff_pause", r"관세[^.]{0,20}(유예|인하|철회|합의)|유예|휴전|pause|truce|rollback|공동성명|joint statement"),
    ("resource_export_control", r"희토류|갈륨|게르마늄|안티몬|흑연|광물|rare earth|gallium|germanium|graphite"),
    ("tech_export_control", r"반도체[^.]{0,20}(수출|통제)|수출통제|export control|칩 ?통제"),
    ("currency", r"위안화|환율조작|절하|devalu|currency manipulator"),
    ("tariff", r"관세|tariff|301조"),
]
KW_COUNTRY = {
    "US": r"미국|미 정부|백악관|트럼프|바이든|상무부 BIS|BIS|USTR|U\.S\.|United States|Washington",
    "CN": r"중국|베이징|시진핑|China|Beijing|中国",
    "EU": r"유럽|(?<![A-Za-z])EU(?![A-Za-z])|European",
    "JP": r"일본|Japan",
    "TW": r"대만|Taiwan|TSMC",
    "KR": r"한국|Korea",
    "RU": r"러시아|Russia",
    "IN": r"인도(?!네시아)|India",
}
KW_SECTOR = {
    "semiconductor": r"반도체|칩|chip|semiconductor|TSMC|엔비디아|Nvidia",
    "defense": r"방산|군사|defen[cs]e",
    "autos": r"자동차|전기차|(?<![A-Za-z])autos?(?![A-Za-z])|(?<![A-Za-z])EVs?(?![A-Za-z])",
    "energy": r"원유|석유|가스|에너지|(?<![A-Za-z])oil(?![A-Za-z])|(?<![A-Za-z])gas(?![A-Za-z])|energy",
    "banks": r"은행|금융기관|(?<![A-Za-z])banks?(?![A-Za-z])",
    "telecom": r"통신|화웨이|Huawei|5G",
    "agriculture": r"농산물|대두|agricultur|soybean",
    "critical_minerals": r"희토류|광물|갈륨|게르마늄|rare earth|mineral",
    "battery": r"배터리|리튬|흑연|battery|lithium",
    "industrials": r"기계|산업재|철강|알루미늄|steel|aluminum|machinery",
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
    ev["severity"] = 4 if re.search(r"즉시|전면|100%|금지|대폭|ban", text, re.I) else 3
    first = sentences[0]["text"] if sentences else ""
    ev["title"] = first if len(first) <= 45 else ""          # 길면 Reporter가 구조 필드로 제목을 만든다
    ev["uncertain_fields"].append("severity")          # 규칙 경로의 강도는 항상 불확실 표시
    return ev


# ------------------------------------------------------------- 가드레일
def guard(draft: dict, sentences: list[dict], raw_text: str) -> tuple[Event | None, list[str]]:
    """초안 dict를 검사·보정해 Event로 만든다. (Event, 남은 오류 목록)"""
    notes, errors = [], []
    d = {k: v for k, v in (draft or {}).items() if v not in (None, "")}
    valid_ids = {s["id"] for s in sentences}

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
    if d["direction"] == "escalate" and d["stage"] == "ease":
        notes.append("확대 사건인데 stage가 ease -> enforce 로 보정")
        d["stage"] = "enforce"
    if d["direction"] == "ease" and d["stage"] != "ease":
        d["stage"] = "ease"
    if d.get("instrument") == "tariff_pause" and d["direction"] != "ease":
        notes.append("tariff_pause 인데 direction이 escalate -> ease 로 보정")
        d["direction"] = "ease"
    init = str(d.get("initiator", "")).upper()
    if init not in COUNTRIES:
        if init:
            errors.append(f"initiator '{init}' 는 어휘 밖")
        d.pop("initiator", None)
    else:
        d["initiator"] = init
    tgt = [str(c).upper() for c in d.get("target", []) if str(c).upper() in COUNTRIES and str(c).upper() != init]
    if len(tgt) != len(d.get("target", [])):
        notes.append("어휘 밖이거나 발동국과 같은 대상국 제거")
    d["target"] = tgt
    secs = [s for s in d.get("sectors", []) if s in SECTORS]
    if len(secs) != len(d.get("sectors", [])):
        notes.append(f"어휘 밖 산업 제거: {sorted(set(d.get('sectors', [])) - set(secs))}")
    d["sectors"] = secs
    try:
        d["severity"] = min(max(int(d.get("severity", 3)), 1), 5)
    except (TypeError, ValueError):
        d["severity"] = 3
        notes.append("severity 해석 실패 -> 3")

    # 2. 근거
    ev_map = {}
    for field, ids in (d.get("evidence") or {}).items():
        ids = [i for i in (ids or []) if i in valid_ids]
        if ids:
            ev_map[field] = ids
    d["evidence"] = ev_map
    unc = set(d.get("uncertain_fields") or [])
    for f in ["instrument", "initiator", "target", "sectors", "stage", "severity"]:
        if f not in ev_map:
            unc.add(f)

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


# ------------------------------------------------------------- 진입점
def analyze(raw_text: str, llm: LLM, feedback: list[str] | None = None) -> dict:
    """State에 넣을 값을 돌려준다: sentences, event, event_complete, event_errors, mode."""
    sentences = split_sentences(raw_text)
    mode = "rule"
    draft = None
    if llm.enabled:
        user = prompts.EVENT_FEW_SHOT + "\n실제 입력:\n" + numbered(sentences)
        if feedback:
            user += "\n\n[이전 시도에서 지적된 문제 - 고쳐서 다시 출력]\n- " + "\n- ".join(feedback)
        try:
            draft = llm.complete_json(prompts.EVENT_SYSTEM, user)
            mode = "llm"
        except Exception as e:                       # 네트워크·파싱 실패 -> 규칙 경로로
            feedback = (feedback or []) + [f"LLM 호출 실패: {e}"]
    if draft is None:
        draft = rule_extract(sentences)
    event, errors = guard(draft, sentences, raw_text)
    if event is None and mode == "llm":              # LLM 초안이 필수 필드를 못 채우면 규칙으로 보충
        merged = {**rule_extract(sentences), **{k: v for k, v in draft.items() if v}}
        event, errors2 = guard(merged, sentences, raw_text)
        errors += errors2
        if event is not None:
            event.notes.append("필수 필드 일부를 키워드 규칙으로 보충")
    return {
        "sentences": sentences,
        "event": event.model_dump() if event else {},
        "event_complete": event is not None,
        "event_errors": errors,
        "mode": mode,
    }
