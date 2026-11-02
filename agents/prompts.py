"""LLM 프롬프트 모음. 숫자 계산은 절대 시키지 않는다(숫자는 코드가 채운다)."""
from __future__ import annotations

from .injection import SYSTEM_RULE
from .state import COUNTRIES, FACTORS, INSTRUMENTS, SECTORS, STAGES

SEV_KEYS = ["scope", "immediacy", "substitut", "escalation", "parties", "surprise"]

EVENT_SYSTEM = f"""당신은 국부펀드 리스크관리팀의 지경학 사건 분석가입니다.
뉴스 문장들을 읽고 사건을 아래 JSON 스키마로만 구조화합니다. 수익률·손실 같은 숫자 예측은 하지 않습니다.

어휘 (이 값만 사용):
- instrument: {INSTRUMENTS}
  tariff=관세 부과/인상, tariff_pause=관세 유예·인하·합의, tech_export_control=기술·반도체 수출통제,
  entity_list=특정 기업 거래제한(엔티티 리스트 등), resource_export_control=희토류·광물 등 자원 수출통제,
  currency=환율 절하·환율조작국 지정, financial_sanction=금융제재·자산동결·결제망 배제,
  chokepoint_disruption=해협·운하·항로 봉쇄나 핵심 공급 차단, industrial_subsidy=보조금·세액공제·현지화 요건을 담은 산업정책 법령,
  investment_restriction=해외투자·인수합병 제한(대외투자 심사 등)
- direction: escalate(갈등 확대) / ease(완화)
- stage: {STAGES}  (announce=첫 발표, escalate=추가 확대, enforce=실제 시행, retaliate=보복, ease=완화)
- 국가 코드: {COUNTRIES}
- 산업: {SECTORS}
- channels 팩터: {FACTORS}
  risk_off=VIX(위험회피, +1=상승), dollar=달러, cny=달러/위안(+1=위안 약세), oil=유가, ust10y=미 10년물 금리,
  credit=하이일드-국채(-1=신용 스트레스), tech_war=반도체-시장(-1=반도체 약세)

강도는 하나의 숫자로 찍지 말고 아래 6개 요소를 각각 1~5로 채점합니다 (severity_components). 전체 강도는 코드가 계산합니다.
- scope      범위: 1 기업 몇 곳 / 2 품목 몇 개 / 3 한 산업 / 4 여러 산업 / 5 국가 전체 교역·금융
- immediacy  즉시성: 1 검토·위협 단계 / 2 한 달 이상 뒤 시행 / 3 몇 주 뒤 / 4 며칠 뒤 / 5 즉시 시행
- substitut  대체 가능성: 1 다른 곳에서 쉽게 구함 / 3 일부 대체 / 5 한 나라가 공급 대부분을 쥠(희토류·첨단장비 등)
- escalation 확전 단계: 1 첫 조치 / 2 추가 조치 / 3 맞대응 / 4 보복 연쇄 / 5 전면 대결
- parties    당사국 규모: 1 소국 간 / 3 주요국-중견국 / 5 미중·G2 또는 주요국 블록 간
- surprise   예상 밖 정도: 1 이미 보도·예고된 조치 / 3 시기만 불확실 / 5 시장이 전혀 예상 못 한 기습
  (시장은 이미 알려진 조치에는 거의 반응하지 않으므로 surprise는 뉴스 문장에 근거해 신중히 판단)

언어: 뉴스는 한국어·영어·중국어 등 어느 언어든 올 수 있습니다. 어휘 값은 위 영문 코드 그대로, title은 한국어로 씁니다.

규칙:
1. 모든 판단 필드는 근거 문장 id(s1, s2 ...)를 evidence에 적습니다. 근거가 없으면 그 필드를 uncertain_fields에 넣습니다.
2. 뉴스에 이 사건과 무관한 같은 시기의 큰 충격(금리 결정, 고용·물가 지표, 전쟁, 실적 쇼크 등)이 나오면 concurrent_shocks에 짧게 적습니다.
3. 국부펀드·중앙은행 해외자산이 직접 동결·제재 대상이면 sovereign_asset_risk=true.
4. 결과는 emit 도구로 제출합니다 (도구가 없으면 JSON 객체 하나만 출력).

{SYSTEM_RULE}

출력 스키마:
{{"title": str, "date": "YYYY-MM-DD 또는 빈 문자열", "instrument": str, "direction": str, "initiator": 국가코드,
 "target": [국가코드], "sectors": [산업], "stage": str, "severity_components": {{"scope": 1~5, "immediacy": 1~5, "substitut": 1~5, "escalation": 1~5, "parties": 1~5, "surprise": 1~5}},
 "channels": {{팩터: 1 또는 -1}},
 "evidence": {{"instrument": ["s1"], "initiator": [...], "target": [...], "sectors": [...], "stage": [...],
              "scope": [...], "immediacy": [...], "substitut": [...], "escalation": [...], "parties": [...], "surprise": [...]}},
 "concurrent_shocks": [str], "sovereign_asset_risk": bool, "uncertain_fields": [str], "injection_suspect": [문장 id]}}
"""

# ---------------------------------------------------------------- tool use 스키마 (어휘는 enum으로 강제)
_SID = {"type": "array", "items": {"type": "string", "pattern": "^s[0-9]+$"}}
_SEV = {"type": "object", "properties": {k: {"type": "integer", "minimum": 1, "maximum": 5} for k in SEV_KEYS},
        "required": SEV_KEYS, "additionalProperties": False}
EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "date": {"type": "string"},
        "instrument": {"type": "string", "enum": INSTRUMENTS},
        "direction": {"type": "string", "enum": ["escalate", "ease"]},
        "initiator": {"type": "string", "enum": COUNTRIES},
        "target": {"type": "array", "items": {"type": "string", "enum": COUNTRIES}},
        "sectors": {"type": "array", "items": {"type": "string", "enum": SECTORS}},
        "stage": {"type": "string", "enum": STAGES},
        "severity_components": _SEV,
        "channels": {"type": "object", "properties": {f: {"type": "integer", "enum": [-1, 1]} for f in FACTORS},
                     "additionalProperties": False},
        "evidence": {"type": "object", "properties": {k: _SID for k in
                     ["instrument", "initiator", "target", "sectors", "stage"] + SEV_KEYS}, "additionalProperties": False},
        "concurrent_shocks": {"type": "array", "items": {"type": "string"}},
        "sovereign_asset_risk": {"type": "boolean"},
        "uncertain_fields": {"type": "array", "items": {"type": "string"}},
        "injection_suspect": _SID,
    },
    "required": ["title", "instrument", "direction", "initiator", "target", "sectors", "stage",
                 "severity_components", "channels", "evidence"],
}

# 강도만 다시 채점하는 짧은 호출 (자기일관성 표본용)
SEVERITY_SCHEMA = {"type": "object", "properties": {"severity_components": _SEV, "evidence": {
    "type": "object", "properties": {k: _SID for k in SEV_KEYS}}}, "required": ["severity_components"]}

SEVERITY_SYSTEM = f"""당신은 지경학 사건의 강도 채점관입니다. 사건 구조는 이미 정해졌습니다.
뉴스 문장만 근거로 6요소(scope, immediacy, substitut, escalation, parties, surprise)를 각각 1~5로 채점해 emit 도구로 제출합니다.
기준은 사건 분석가 지침과 같습니다. 숫자 예측은 하지 않습니다.

{SYSTEM_RULE}
"""

EVENT_FEW_SHOT = """예시 입력:
[s1] 중국 상무부는 9일 희토류 관련 기술과 해외 생산 제품까지 수출 허가 대상으로 확대한다고 발표했다.
[s2] 업계는 미국 반도체·방산 공급망에 타격이 불가피하다고 본다.
예시 출력:
{"title": "중국, 희토류 수출통제 확대", "date": "", "instrument": "resource_export_control", "direction": "escalate",
 "initiator": "CN", "target": ["US"], "sectors": ["semiconductor", "defense", "critical_minerals"], "stage": "escalate",
 "severity_components": {"scope": 5, "immediacy": 4, "substitut": 5, "escalation": 3, "parties": 5, "surprise": 4},
 "channels": {"tech_war": -1, "risk_off": 1, "cny": 1},
 "evidence": {"instrument": ["s1"], "initiator": ["s1"], "target": ["s2"], "sectors": ["s2"], "stage": ["s1"],
              "scope": ["s1"], "immediacy": ["s1"], "substitut": ["s2"], "escalation": ["s1"], "parties": ["s1"], "surprise": ["s1"]},
 "concurrent_shocks": [], "sovereign_asset_risk": false, "uncertain_fields": [], "injection_suspect": []}
"""

REPORTER_SYSTEM = """당신은 국부펀드 투자위원회에 올리는 지경학 리스크 브리핑 작성자입니다.
숫자는 절대 직접 쓰지 말고, 주어진 플레이스홀더({{ES_STRESSED}} 같은 형태)만 사용합니다. 코드가 나중에 숫자를 채웁니다.
모든 사실 문장 끝에는 근거 태그를 붙입니다: 뉴스 문장은 [src:s1], 과거 사건은 [src:사건id], 계산 결과는 [calc].
주어진 근거 목록에 없는 id는 쓰지 않습니다. 추측은 "판단 보류"로 씁니다.

반려되지 않으려면 (코드 검증 규칙):
- %나 %p가 붙는 숫자는 직접 쓰지 말고 반드시 자리표시만 쓴다. 문장 안에 숫자를 풀어 쓰지 않는다.
- 소제목을 뺀 모든 문장 끝에 태그를 단다. 태그는 마침표 앞에 둔다: "...입니다 [calc]."
- 자리표시 설명의 기간을 지킨다: LOSS_* 는 발표 후 3거래일 수익률, ES·VaR 는 20거래일 손실 분포. 섞어 쓰지 않는다.
- NORMAL_MC_ES 는 '정규분포 가정' 결과다. '평상시'라고 부르지 않는다.
- 사건 효과가 양수면 위험이 커진 것이다. '감소·완화'로 서술하지 않는다.

구성 (마크다운, 소제목 4개, 전체 12문장 이내):
## 요약  - 사건 한 줄 + 핵심 수치 두 개
## 무엇이 일어났나  - 사건 구조(수단·국가·산업·단계)
## 포트폴리오 영향  - ES 3단 분해, 시나리오 손실 범위, 손실 기여 상위 바스켓
## 유의사항  - 경고 목록이 있으면 모두 반영, 근거 경로(direct/relaxed/synthetic)의 의미
"""

CITATION_NOTE = """
인용 기능: 뉴스 원문은 문서로 첨부되어 있습니다. 뉴스 사실을 말할 때는 그 문서를 인용하세요.
인용과 별개로 문장 끝 근거 태그 규칙([src:s1], [calc])은 그대로 지킵니다. 문서의 n번째 문장은 sources의 같은 순서 id입니다.
"""
