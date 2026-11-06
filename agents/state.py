"""에이전트끼리 주고받는 데이터 모양(스키마)과 헤드쿼터가 들고 다니는 State.

원칙: 자유 텍스트는 최종 브리핑뿐. 나머지는 전부 이 스키마로 검증된 값만 오간다.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------- 어휘 (사건 라이브러리와 같은 값만 허용)
INSTRUMENTS = ["tariff", "tariff_pause", "tech_export_control", "entity_list",
               "resource_export_control", "currency", "financial_sanction",
               "chokepoint_disruption", "industrial_subsidy", "investment_restriction",
               "domestic_regulation", "election_shift"]
# 국내 정책 사건: 발동국이 자국 산업을 겨냥한다(발동국 = 대상국 허용). 2026-11 중간선거 후 AI 데이터센터 규제 우려 같은 경우.
DOMESTIC_INSTRUMENTS = {"domestic_regulation", "election_shift"}
STAGES = ["announce", "escalate", "enforce", "retaliate", "ease"]
DIRECTIONS = ["escalate", "ease"]
FACTORS = ["risk_off", "dollar", "cny", "oil", "ust10y", "credit", "tech_war"]
COUNTRIES = ["US", "CN", "EU", "JP", "TW", "KR", "IN", "RU", "VN", "MX", "CA", "GB", "SA", "IR", "GLOBAL"]
SECTORS = ["semiconductor", "telecom", "defense", "autos", "industrials", "energy", "banks",
           "agriculture", "battery", "critical_minerals", "pharma", "shipping",
           "tech_platform", "ai_infra", "utilities"]

# 수단별 기본 영향 채널 (확대 기준). 완화 사건은 부호를 뒤집는다.
# Event Analyst가 채널을 못 정하거나 어휘 밖 값을 내면 이 표로 채운다.
CHANNEL_DEFAULTS: dict[str, dict[str, int]] = {
    "tariff": {"risk_off": 1, "tech_war": -1, "cny": 1},
    "tariff_pause": {"risk_off": 1, "tech_war": -1, "credit": -1},     # ease로 쓰이므로 뒤집혀 적용됨
    "tech_export_control": {"tech_war": -1, "risk_off": 1, "cny": 1},
    "entity_list": {"tech_war": -1, "risk_off": 1, "cny": 1},
    "resource_export_control": {"tech_war": -1, "risk_off": 1, "cny": 1},
    "currency": {"cny": 1, "risk_off": 1, "tech_war": -1},
    "financial_sanction": {"oil": 1, "risk_off": 1, "credit": -1},
    "chokepoint_disruption": {"risk_off": 1, "oil": 1, "tech_war": -1, "credit": -1},   # 해협·운하 봉쇄, 공급 차단
    "industrial_subsidy": {"tech_war": 1, "cny": 1},              # 보조금·세액공제·현지화 (자국 산업 지원 + 상대국 배제)
    "investment_restriction": {"tech_war": -1, "risk_off": 1, "cny": 1},   # 해외투자·인수 제한
    "domestic_regulation": {"tech_war": -1, "risk_off": 1},       # 자국 산업 규제·반독점·인프라 건설 제한
    "election_shift": {"risk_off": 1},                            # 선거·정권 교체로 정책 기대가 바뀜 (방향은 사건마다)
}


def default_channels(instrument: str, direction: str) -> dict[str, int]:
    base = CHANNEL_DEFAULTS.get(instrument, {"risk_off": 1})
    sign = -1 if direction == "ease" else 1
    return {f: d * sign for f, d in base.items()}


# ---------------------------------------------------------------- 스키마
class Sentence(BaseModel):
    id: str                 # s1, s2, ...
    text: str


class Event(BaseModel):
    """Event Analyst 출력. 모든 판단 필드는 근거 문장 id를 함께 가진다."""
    id: str = "new_event"
    date: str = ""
    title: str = ""
    instrument: Literal[tuple(INSTRUMENTS)]                  # type: ignore[valid-type]
    direction: Literal["escalate", "ease"] = "escalate"
    initiator: str
    target: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    stage: Literal[tuple(STAGES)] = "announce"               # type: ignore[valid-type]
    severity: int = Field(3, ge=1, le=5)                                  # 등급 (요소 점수에서 코드가 계산)
    severity_components: dict[str, int] = Field(default_factory=dict)     # 6요소 1~5
    severity_points: float = 50.0                                         # 0~100 연속 점수
    severity_samples: int = 1                                             # 강도 채점 횟수 (자기일관성)
    severity_spread: dict[str, int] = Field(default_factory=dict)         # 요소별 최대-최소 (채점 흩어짐)
    channels: dict[str, int] = Field(default_factory=dict)
    evidence: dict[str, list[str]] = Field(default_factory=dict)   # 필드명 -> 근거 문장 id
    concurrent_shocks: list[str] = Field(default_factory=list)     # 같은 시기 다른 큰 충격
    sovereign_asset_risk: bool = False                            # 국부펀드 자산이 직접 표적인가 (금융제재 등)
    uncertain_fields: list[str] = Field(default_factory=list)     # 근거가 없거나 코드가 보정한 필드
    notes: list[str] = Field(default_factory=list)                # 코드 가드레일이 바꾼 내용 기록
    injection_flags: list[dict] = Field(default_factory=list)     # 지시문 의심 문장 [{id, text, match, by}]

    @field_validator("channels")
    @classmethod
    def _channels(cls, v: dict[str, int]) -> dict[str, int]:
        bad = [k for k in v if k not in FACTORS]
        if bad:
            raise ValueError(f"알 수 없는 팩터: {bad}")
        if any(d not in (-1, 1) for d in v.values()):
            raise ValueError("채널 방향은 +1 또는 -1")
        return v

    def as_library_dict(self) -> dict:
        d = self.model_dump()
        d["clean"] = True
        return d


class AnalogPick(BaseModel):
    id: str
    score: float
    parts: dict[str, float]
    reason: str


class Issue(BaseModel):
    kind: str          # number_mismatch / unknown_source / untagged / placeholder / direction / evidence
    detail: str
    route: Literal["reporter", "event_analyst", "shock_mapper"]


# ---------------------------------------------------------------- 헤드쿼터 State
class ShockLensState(TypedDict, total=False):
    run_id: str
    seed: int
    data_version: str
    raw_input: str                     # 사건 해석에 쓰는 텍스트 (URL·헤드라인이면 가져온 기사 본문, 길면 핵심 문장)
    original_input: str                # 사용자가 넣은 그대로
    news_meta: dict                    # 입력 종류(url/headline/article)·가져온 URL·도구 호출 수·언어
    attention: dict                    # 사건 직전 보도량 백분위 {query, percentile, ratio, note, error}
    triage: dict                       # 중요도 1차 분류 {relevant, importance, reason, mode, model}
    sentences: list[dict]              # [{"id": "s1", "text": ...}]
    event: dict                        # Event.model_dump()
    event_complete: bool
    event_preset: bool                 # 라이브러리 사건으로 바로 시작(뉴스 해석 생략)
    event_errors: list[str]
    event_mode: str                    # llm / rule / preset
    injection_flags: list[dict]        # 뉴스 안 지시문 의심 문장 (근거·출처에서 제외)
    needs_user: str                    # 사용자에게 가정을 물어야 할 때 질문 문장
    analogs: list[dict]                # AnalogPick
    analog_tier: str                   # direct / relaxed / synthetic / user
    analog_table: list[dict]           # 상위 후보 점수표
    analog_excluded: list[str]         # clean: false 라 뺀 사건
    analog_notes: list[str]            # 방향 불일치·인접 등급 보충 등 Analog Finder 경고
    size_class: str                    # 충격 크기 등급 small / medium / large
    shock_method: str                  # weighted(n) / event_level(n) / synthetic
    exposure: dict[str, str]
    exposure_reasons: dict[str, str]
    factor_shock: dict                 # {factor: {mild, central, severe}}
    idio_shock: dict
    beta_regime: str
    regime_info: dict                  # 국면 판단 근거 (mode, VIX 최근값·백분위·기준)
    shock_warnings: list[str]
    zeroed_wrong_sign: list
    risk_result: dict
    calc_log_id: str
    draft_brief: str
    reporter_mode: str                 # llm / llm+citations / template
    citation_stats: dict | None        # 인용 기능: 인용된 구간 수, 자동 보강한 태그 수, 태그-인용 불일치
    reporter_error: str | None         # LLM 작성 실패 시 이유 (템플릿으로 대체됨)
    facts: dict[str, str]              # 플레이스홀더 -> 코드가 채운 숫자 문자열
    sources: dict[str, str]            # [src:id] -> 설명
    verification: dict
    final_brief: str
    status: str                        # ok / partial / need_user / skipped
    retries: dict[str, int]
    whatif_overrides: dict
    regime_override: str | None        # 'crisis'/'normal' 국면 강제 (시나리오·What-if)
    scenario: dict | None              # 가상 시나리오 실행이면 {id, title}
    trace: Annotated[list[dict[str, Any]], operator.add]
