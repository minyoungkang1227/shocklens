"""기사 문장 유사도 (단어별 가중치 + 방향성). LLM·외부 모델 없이 결정론적으로 계산한다.

왜 단순 단어 겹침이 아닌가
  "관세 유예"와 "관세 부과"는 단어가 거의 같지만 시장 방향은 정반대다.
  그래서 (1) 단어를 '개념'으로 묶고 종류별로 가중치를 다르게 주고, (2) 방향 단어는 따로 세서
  확대/완화 방향을 판정한다. 방향이 다른 사건은 유사도가 높아도 비교 대상에서 뺀다(analog.py).

특징(feature)과 가중치
  수단 개념 (관세, 수출통제, 희토류, 제재 ...)   x3.0  -> 무엇을 했나가 가장 중요
  산업 개념 (반도체, 방산, 에너지 ...)           x2.0
  국가 개념 (미국, 중국 ...)                     x1.5
  방향 개념 (확대어 / 완화어)                    x1.0
  일반 단어 (한글 2글자 조각, 영어 단어)          x0.3 x IDF  -> 흔한 단어일수록 낮게
  한국어·영어 동의어는 같은 개념으로 합친다 (희토류 = rare earth).
"""
from __future__ import annotations

import math
import re
from collections import Counter

# 개념 사전: 이름 -> (종류, 가중치, 정규식)
CONCEPTS: dict[str, tuple[str, float, str]] = {
    # 수단
    "c_tariff": ("instrument", 3.0, r"관세|tariff|301조"),
    "c_export_control": ("instrument", 3.0, r"수출 ?통제|수출 ?허가|수출 ?금지|export control|export ban"),
    "c_entity": ("instrument", 3.0, r"엔티티 ?리스트|entity list|거래 ?제한"),
    "c_ree": ("instrument", 3.0, r"희토류|rare earth"),
    "c_minerals": ("instrument", 3.0, r"갈륨|게르마늄|안티몬|흑연|핵심 ?광물|gallium|germanium|antimony|graphite|critical mineral"),
    "c_currency": ("instrument", 3.0, r"위안화|환율|절하|환율조작|yuan|renminbi|devalu|currency"),
    "c_sanction": ("instrument", 3.0, r"제재|sanction|SWIFT|자산 ?동결|외환보유액|freeze"),
    "c_blockade": ("instrument", 3.0, r"봉쇄|해협|운하|blockade|strait|chokepoint|canal"),
    "c_subsidy": ("instrument", 3.0, r"보조금|세액 ?공제|subsid|tax credit|CHIPS|인플레이션 ?감축법"),
    "c_invest": ("instrument", 3.0, r"투자 ?제한|해외 ?투자|outbound investment|investment restriction"),
    "c_chip_rule": ("instrument", 3.0, r"첨단 ?(칩|반도체)|반도체 ?장비|advanced (chip|computing)|chipmaking equipment"),
    # 산업
    "s_semis": ("sector", 2.0, r"반도체|칩|chip|semiconductor|파운드리|foundry"),
    "s_defense": ("sector", 2.0, r"방산|군사|국방|defen[cs]e|military"),
    "s_autos": ("sector", 2.0, r"자동차|전기차|(?<![A-Za-z])EVs?(?![A-Za-z])|automaker"),
    "s_energy": ("sector", 2.0, r"원유|석유|가스|에너지|(?<![A-Za-z])oil(?![A-Za-z])|energy|crude"),
    "s_banks": ("sector", 2.0, r"은행|금융기관|(?<![A-Za-z])banks?(?![A-Za-z])"),
    "s_telecom": ("sector", 2.0, r"통신|화웨이|huawei|5G"),
    "s_agri": ("sector", 2.0, r"농산물|대두|soybean|agricultur"),
    "s_battery": ("sector", 2.0, r"배터리|리튬|battery|lithium"),
    # 국가
    "k_US": ("country", 1.5, r"미국|백악관|트럼프|바이든|상무부 BIS|USTR|U\.S\.|United States|Washington"),
    "k_CN": ("country", 1.5, r"중국|베이징|China|Beijing|Chinese"),
    "k_EU": ("country", 1.5, r"유럽|(?<![A-Za-z])EU(?![A-Za-z])|Europe"),
    "k_JP": ("country", 1.5, r"일본|Japan"),
    "k_TW": ("country", 1.5, r"대만|Taiwan|TSMC"),
    "k_RU": ("country", 1.5, r"러시아|Russia"),
    "k_IN": ("country", 1.5, r"인도(?!네시아)|India"),
    "k_KR": ("country", 1.5, r"한국|Korea"),
}

ESCALATE = r"인상|확대|금지|보복|강화|위협|부과|발동|중단|추가 ?조치|맞대응|escalat|retaliat|impose|raise|ban|threat|tighten"
EASE = r"유예|인하|철회|합의|완화|휴전|해제|면제|pause|truce|rollback|lift|eas(?:e|ing)|deal|exempt"
EASE_ENDED = r"(유예|휴전|면제|truce|pause)[^.]{0,8}(종료|만료|끝|철회|expire|end)"   # '유예 종료' = 확대 신호
GENERIC_W = 0.3


def _concept_hits(text: str) -> Counter:
    c = Counter()
    for name, (_, _, pat) in CONCEPTS.items():
        n = len(re.findall(pat, text, re.I))
        if n:
            c[name] += n
    return c


def text_direction(text: str) -> dict:
    """기사 단어로 본 방향. score > 0 확대, < 0 완화. hits는 근거 단어."""
    ended = [m.group(0) for m in re.finditer(EASE_ENDED, text, re.I)]
    clean = re.sub(EASE_ENDED, " ", text, flags=re.I)
    e_hits = [m.group(0) for m in re.finditer(ESCALATE, clean, re.I)] + ended
    a_hits = [m.group(0) for m in re.finditer(EASE, clean, re.I)]
    score = len(e_hits) - len(a_hits)
    label = "escalate" if score > 0 else ("ease" if score < 0 else "unclear")
    return {"score": score, "label": label, "escalate_words": e_hits, "ease_words": a_hits}


def _generic_tokens(text: str) -> list[str]:
    toks = []
    for w in re.findall(r"[가-힣]{2,}", text):
        toks += [w[i:i + 2] for i in range(len(w) - 1)]          # 한글은 2글자 조각 (조사가 붙어도 겹치게)
    toks += [w.lower() for w in re.findall(r"[A-Za-z]{3,}", text)]
    return toks


def featurize(text: str, idf: dict[str, float] | None = None) -> dict[str, float]:
    feats: dict[str, float] = {}
    for name, n in _concept_hits(text).items():
        feats[name] = CONCEPTS[name][1] * (1 + math.log(n))
    d = text_direction(text)
    if d["escalate_words"]:
        feats["d_escalate"] = 1.0 * (1 + math.log(len(d["escalate_words"])))
    if d["ease_words"]:
        feats["d_ease"] = 1.0 * (1 + math.log(len(d["ease_words"])))
    for tok, n in Counter(_generic_tokens(text)).items():
        w = (idf or {}).get(tok, 1.0)
        feats["g_" + tok] = GENERIC_W * w * (1 + math.log(n))
    return feats


def build_idf(texts: list[str]) -> dict[str, float]:
    df = Counter()
    for t in texts:
        df.update(set(_generic_tokens(t)))
    n = len(texts)
    return {tok: math.log((n + 1) / (c + 1)) + 1 for tok, c in df.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def top_overlap(a: dict[str, float], b: dict[str, float], k: int = 4) -> list[str]:
    """두 문장에서 유사도에 가장 크게 기여한 개념(설명용). 일반 단어 조각은 제외."""
    shared = [(a[x] * b[x], x) for x in a if x in b and not x.startswith("g_")]
    return [x for _, x in sorted(shared, reverse=True)[:k]]


# 발표 이후 전개·시장 반응을 적은 문장 (유사 사건 찾기·블라인드 채점 모두 '발표 당일 정보'만 써야 결과를 알고 고르는 일이 없다)
AFTER = re.compile(r"(다음\s?날|이후|이튿날|직후 시장|시장은|급락|급등|반등|유예[됐했]|철회|결국|뒤에|나중에|\d{1,2}/\d{1,2}\s)")


def announcement_only(description: str) -> str:
    """설명문에서 발표 내용 문장만 남긴다. 첫 문장은 발표 자체이므로 항상 유지."""
    sents = re.split(r"(?<=[.。])\s+", " ".join(str(description or "").split()))
    keep = []
    for x in sents:
        if not x:
            continue
        if keep and AFTER.search(x):
            break
        keep.append(x)
    return " ".join(keep)


def event_text(ev: dict) -> str:
    return f"{ev.get('title', '')}. {announcement_only(ev.get('description', ''))}"


class TextIndex:
    """라이브러리 사건 문장을 미리 특징으로 바꿔 두고, 새 기사와의 유사도를 돌려준다."""

    def __init__(self, events: list[dict]):
        self.texts = {e["id"]: event_text(e) for e in events}
        self.idf = build_idf(list(self.texts.values()))
        self.vecs = {i: featurize(t, self.idf) for i, t in self.texts.items()}

    def query(self, text: str) -> tuple[dict[str, float], dict[str, list[str]], dict[str, float]]:
        q = featurize(text, self.idf)
        sims = {i: cosine(q, v) for i, v in self.vecs.items()}
        why = {i: top_overlap(q, v) for i, v in self.vecs.items()}
        return sims, why, q
