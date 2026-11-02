"""긴 입력(보고서 PDF, 장문 기사) 처리.

  read_pdf(path_or_bytes)              -> 텍스트 (pypdf)
  key_sentences(text, max_sentences)   -> 사건 판단에 쓸 핵심 문장만 원래 순서대로 고른 텍스트
  detect_lang(text)                    -> "ko" | "zh" | "ja" | "en"

핵심 문장 선별: 앞 3문장(리드)은 항상 넣고, 나머지는 지경학 단어(관세·수출통제·제재·희토류·국가명 등, 한·영·중)
점수가 높은 문장을 골라 원래 순서로 되돌린다. LLM 비용과 주입 위험을 함께 줄인다.
"""
from __future__ import annotations

import io
import re

GEO_WORDS = [
    # 한국어
    "관세", "수출통제", "수출 통제", "제재", "희토류", "반도체", "엔티티", "보복", "유예", "발표", "시행", "금지",
    "미국", "중국", "러시아", "대만", "일본", "유럽", "상무부", "백악관", "환율", "외환보유액", "광물",
    # English
    "tariff", "export control", "sanction", "rare earth", "semiconductor", "entity list", "retaliat",
    "ban", "commerce department", "white house", "beijing", "washington", "currency", "swift", "mineral",
    # 中文
    "关税", "出口管制", "制裁", "稀土", "半导体", "芯片", "实体清单", "反制", "商务部", "美国", "中国", "禁止",
]
_GEO_RX = re.compile("|".join(re.escape(w) for w in GEO_WORDS), re.I)
SPLIT_RX = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])\s*|\n+")


def detect_lang(text: str) -> str:
    ko = len(re.findall(r"[가-힣]", text))
    ja = len(re.findall(r"[぀-ヿ]", text))
    zh = len(re.findall(r"[一-鿿]", text))
    la = len(re.findall(r"[A-Za-z]", text))
    best = max([("ko", ko), ("ja", ja * 3), ("zh", zh), ("en", la / 4)], key=lambda x: x[1])
    return best[0]


def read_pdf(src, max_pages: int = 30) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src)
    pages = []
    for p in reader.pages[:max_pages]:
        t = p.extract_text() or ""
        t = re.sub(r"(?<=[가-힣a-z,])\n(?=[가-힣a-z])", " ", t)     # 줄바꿈으로 끊긴 문장 잇기
        pages.append(t)
    return "\n".join(pages)


def key_sentences(text: str, max_sentences: int = 40, lead: int = 3) -> str:
    sents = [s.strip() for s in SPLIT_RX.split(text) if s and len(s.strip()) >= 4]
    if len(sents) <= max_sentences:
        return text
    scored = [(len(_GEO_RX.findall(s)), i) for i, s in enumerate(sents)]
    keep = set(range(min(lead, len(sents))))
    for sc, i in sorted(scored, key=lambda x: (-x[0], x[1])):
        if len(keep) >= max_sentences:
            break
        if sc > 0:
            keep.add(i)
    return "\n".join(sents[i] for i in sorted(keep))
