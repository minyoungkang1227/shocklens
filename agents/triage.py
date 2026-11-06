"""뉴스 중요도 1차 분류: 작은 모델(기본 claude-haiku-5-5)로 먼저 거르고, 큰 건만 전체 파이프라인에 태운다.

판정: relevant(지경학 정책 사건인가) + importance 1~5 + 이유 한 줄
  importance < min_importance 이거나 relevant=false 이면 헤드쿼터가 전체 분석을 생략하고 'skipped'로 끝낸다.
  사용자가 강제 실행(force)을 고르면 분류와 상관없이 진행한다.
LLM이 없으면 키워드 규칙으로 판정한다 (정책 수단 단어 + 국가 단어 수).
사건 라이브러리 대량 구축(library_builder)에서도 같은 함수로 기사를 먼저 거른다.
"""
from __future__ import annotations

import re

from .injection import SYSTEM_RULE
from .llm import LLM

SCHEMA = {"type": "object", "properties": {
    "relevant": {"type": "boolean"},
    "importance": {"type": "integer", "minimum": 1, "maximum": 5},
    "reason": {"type": "string"}}, "required": ["relevant", "importance", "reason"]}

SYSTEM = f"""당신은 국부펀드 리스크관리팀의 뉴스 1차 분류자입니다. 뉴스가 '지경학 정책 사건'인지와 중요도만 판단합니다.
지경학 정책 사건: 관세·관세 유예, 수출통제, 거래제한 명단, 자원(희토류 등) 수출통제, 환율 조치, 금융제재,
해협·운하 봉쇄와 핵심 공급 차단, 보조금·세액공제·현지화 산업정책 법령, 해외투자·인수 제한,
그리고 국내 정책 사건: 주요국의 자국 산업 규제·반독점·AI 데이터센터 건설 제한, 선거·정권 교체로 인한 정책 방향 전환.
importance: 1 무관하거나 소문·논평 / 2 작은 조치(기업 몇 곳·품목 몇 개) / 3 한 산업 조치 / 4 여러 산업·주요국 간 / 5 전면 대결·국가 단위.
한·영·중 어느 언어든 판단합니다. reason은 한국어 한 줄. emit 도구로 제출합니다.

{SYSTEM_RULE.replace('injection_suspect에 문장 id를 적습니다', '판단에서 무시합니다')}"""

POLICY = r"봉쇄|blockade|封锁|보조금|subsid|补贴|투자 ?제한|outbound investment|관세|tariff|关税|수출 ?통제|export control|出口管制|엔티티|entity list|实体清单|희토류|rare earth|稀土|" \
         r"제재|sanction|制裁|SWIFT|환율조작|devalu|汇率|수출 ?금지|export ban|보복|retaliat|反制|유예|pause|truce|" \
         r"반독점|antitrust|反垄断|규제|regulat|crackdown|监管|모라토리엄|moratorium|중간선거|대선|election|midterm|选举"
BIG = r"전면|모든|전체|all |全面|즉시|immediately|立即|보복|retaliat|反制"
COUNTRY = r"미국|중국|러시아|대만|일본|EU|유럽|U\.S\.|United States|China|Russia|Taiwan|Japan|美国|中国|俄罗斯"


def rule_triage(text: str) -> dict:
    pol = len(set(m.group(0).lower() for m in re.finditer(POLICY, text, re.I)))
    cty = len(set(m.group(0) for m in re.finditer(COUNTRY, text, re.I)))
    big = bool(re.search(BIG, text, re.I))
    if pol == 0:
        return {"relevant": False, "importance": 1, "reason": "정책 수단 단어가 없음 (규칙 판정)"}
    imp = min(5, 2 + (cty >= 2) + big + (pol >= 3))
    return {"relevant": True, "importance": imp,
            "reason": f"정책 단어 {pol}종·국가 {cty}개{' ·확대 표현' if big else ''} (규칙 판정)"}


def classify(text: str, llm: LLM | None = None) -> dict:
    """반환: {relevant, importance, reason, mode(llm/rule), model}"""
    if llm is not None and llm.enabled:
        try:
            r = llm.complete_json(SYSTEM, f"<news>\n{text[:4000]}\n</news>", max_tokens=300, schema=SCHEMA, name="emit")
            if r and "importance" in r:
                imp = min(max(int(r["importance"]), 1), 5)
                return {"relevant": bool(r.get("relevant", True)), "importance": imp,
                        "reason": str(r.get("reason", ""))[:120], "mode": "llm", "model": getattr(llm, "model", "")}
        except Exception as e:
            out = rule_triage(text)
            out["reason"] += f" / LLM 실패: {type(e).__name__}"
            return {**out, "mode": "rule", "model": None}
    return {**rule_triage(text), "mode": "rule", "model": None}


def passes(t: dict, min_importance: int) -> bool:
    return bool(t["relevant"]) and t["importance"] >= min_importance
