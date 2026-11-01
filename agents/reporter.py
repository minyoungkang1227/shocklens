"""Reporter: 계산 결과 -> 투자위원회용 브리핑 (LLM은 문장만, 숫자는 코드가 채움).

강화 포인트
- 숫자 플레이스홀더 방식: LLM은 {{ES_STRESSED}} 같은 자리표시만 쓰고, 코드가 calc 결과로 채운다.
  -> LLM이 숫자를 지어내거나 잘못 옮길 여지를 없앤다.
- 모든 사실 문장에 근거 태그: 뉴스 [src:s3], 과거 사건 [src:cn_ban_us_2024], 계산 [calc].
- 경고는 LLM에 맡기지 않고 코드가 '점검 목록'으로 항상 붙인다
  (동시발생 충격, 국부펀드 표적, synthetic 경로, 이상치, 유사 사건 1건, 근거 부족 필드).
- Verifier가 반려하면 지적 사항을 받아 다시 쓴다. LLM이 없으면 고정 템플릿으로 쓴다.
"""
from __future__ import annotations

import json
import re

from . import prompts
from .context import Context
from .llm import LLM

TIER_KO = {"direct": "직접 유사 사례", "relaxed": "완화된 유사 사례", "synthetic": "강도 기반 합성", "user": "사용자 가정"}
INST_KO = {"tariff": "관세", "tariff_pause": "관세 유예·인하", "tech_export_control": "기술 수출통제",
           "entity_list": "거래제한 명단", "resource_export_control": "자원 수출통제",
           "currency": "환율", "financial_sanction": "금융제재"}


COUNTRY_KO = {"US": "미국", "CN": "중국", "EU": "EU", "JP": "일본", "TW": "대만", "KR": "한국", "IN": "인도",
              "RU": "러시아", "VN": "베트남", "MX": "멕시코", "CA": "캐나다", "GB": "영국", "SA": "사우디",
              "IR": "이란", "GLOBAL": "다수 국가"}
SECTOR_KO = {"semiconductor": "반도체", "telecom": "통신", "defense": "방산", "autos": "자동차",
             "industrials": "산업재", "energy": "에너지", "banks": "은행", "agriculture": "농산물",
             "battery": "배터리", "critical_minerals": "핵심광물", "pharma": "제약", "shipping": "해운"}
STAGE_KO = {"announce": "발표", "escalate": "확대", "enforce": "시행", "retaliate": "보복", "ease": "완화"}


def josa(word: str, pair: str = "이가") -> str:
    """받침 유무로 조사 선택: josa('중국') -> '중국이', josa('EU') -> 'EU가'"""
    ch = word[-1]
    if "가" <= ch <= "힣":
        has = (ord(ch) - 0xAC00) % 28 != 0
    else:
        has = ch.upper() in "LMNR"            # 영문 약어: 엘·엠·엔·알로 끝나면 받침 있음
    return word + (pair[0] if has else pair[1])


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def spct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def pp(x: float) -> str:
    return f"{x * 100:+.2f}%p"


def build_facts(state: dict, ctx: Context) -> dict[str, str]:
    r = state["risk_result"]
    d = r["decomposition"]
    f = {
        "HORIZON": f"{r['horizon_days']}거래일",
        "ES_NORMAL": pct(r["baseline_normal"]["ES"]),
        "ES_REGIME": pct(r["baseline"]["ES"]),
        "ES_STRESSED": pct(r["stressed"]["ES"]),
        "VAR_STRESSED": pct(r["stressed"]["VaR"]),
        "REGIME_EFFECT": pp(d["regime_effect_ES"]),
        "EVENT_EFFECT": pp(d["event_effect_ES"]),
        "TOTAL_EFFECT": pp(d["total_change_ES"]),
        "NORMAL_MC_ES": pct(r["normal_mc_ES"]),
        "ES_UNDER": f"{abs(r['es_underestimation_normal']) * 100:.2f}%p",
        "LOSS_MILD": spct(r["scenario_loss"]["mild"]),
        "LOSS_CENTRAL": spct(r["scenario_loss"]["central"]),
        "LOSS_SEVERE": spct(r["scenario_loss"]["severe"]),
        "REGIME": "위기 국면" if r["regime"] == "crisis" else "평상시 국면",
        "TIER": TIER_KO.get(state.get("analog_tier", ""), state.get("analog_tier", "")),
        "CALC_ID": state.get("calc_log_id", ""),
    }
    for i, (b, c) in enumerate(r["tail_top3"], 1):
        f[f"TOP{i}"] = ctx.labels.get(b, b)
        f[f"TOP{i}_SHARE"] = f"{c * 100:.0f}%"
    return f


def build_sources(state: dict, ctx: Context) -> dict[str, str]:
    src = {s["id"]: s["text"] for s in state.get("sentences", [])}
    for a in state.get("analogs", []):
        src[a["id"]] = a["reason"]
    if state["event"].get("concurrent_shocks"):     # 점검 목록에 인용되는 동시발생 충격 문구 (숫자 대조용)
        src["concurrent"] = " / ".join(state["event"]["concurrent_shocks"])
    return src


def warnings_list(state: dict) -> list[str]:
    ev = state["event"]
    w = []
    for c in ev.get("concurrent_shocks", []):
        w.append(f"같은 시기 다른 충격({c})이 있어 실제 시장 반응에는 사건 외 요인이 섞일 수 있습니다.")
    if ev.get("sovereign_asset_risk"):
        w.append("국부펀드·중앙은행 해외자산이 직접 제재 표적입니다. 시장 손실과 별개로 자산 동결·회수 위험을 점검해야 합니다.")
    w += state.get("shock_warnings", [])
    unc = [f for f in ev.get("uncertain_fields", []) if f in ("instrument", "initiator", "target", "channels")]
    if unc:
        w.append(f"뉴스에서 직접 근거를 찾지 못해 추정한 항목: {', '.join(unc)} (판단 보류 성격).")
    if state.get("analog_excluded"):
        w.append(f"반응 창에 다른 큰 충격이 겹친 과거 사건은 비교에서 뺐습니다: {', '.join(state['analog_excluded'])}.")
    return w


def template_draft(state: dict, ctx: Context) -> str:
    ev = state["event"]
    s_inst = ",".join(ev.get("evidence", {}).get("instrument", [])[:1]) or ""
    tag_ev = f"[src:{s_inst}]" if s_inst else ""
    analogs = state.get("analogs", [])
    a_tags = " ".join(f"[src:{a['id']}]" for a in analogs)
    who = COUNTRY_KO.get(ev["initiator"], ev["initiator"])
    title = ev.get("title") if 0 < len(ev.get("title", "")) <= 45 else f"{who}의 {INST_KO[ev['instrument']]} 조치"
    tg = ", ".join(COUNTRY_KO.get(c, c) for c in ev.get("target", [])) or "복수 국가"
    lines = [
        "## 요약",
        f"{title} 사건으로 포트폴리오 {{{{HORIZON}}}} ES97.5는 {{{{ES_STRESSED}}}}로 추정되며, "
        f"평상시 대비 {{{{TOTAL_EFFECT}}}} 변화입니다 [calc].",
        "",
        "## 무엇이 일어났나",
        f"{josa(who)} {josa(tg, '을를')} 대상으로 {INST_KO[ev['instrument']]} 조치를 "
        f"취했습니다({STAGE_KO.get(ev['stage'], ev['stage'])} 단계, 강도 {ev['severity']}/5) {tag_ev}.",
    ]
    if ev.get("sectors"):
        s_sec = ",".join(ev.get("evidence", {}).get("sectors", [])[:1])
        secs = ", ".join(SECTOR_KO.get(x, x) for x in ev["sectors"])
        lines.append(f"주요 대상 산업은 {secs}입니다 {f'[src:{s_sec}]' if s_sec else ''}.")
    if analogs:
        lines.append(f"충격 크기는 구조가 비슷한 과거 사건의 발표 후 3거래일 반응으로 잡았습니다 {a_tags}.")
    lines += [
        "",
        "## 포트폴리오 영향",
        "ES97.5는 평상시 {{ES_NORMAL}}, {{REGIME}} {{ES_REGIME}}, 사건 충격 반영 후 {{ES_STRESSED}}입니다 [calc].",
        "국면 효과는 {{REGIME_EFFECT}}, 사건 자체 효과는 {{EVENT_EFFECT}}입니다 [calc].",
        "발표 직후 3거래일 기준 시나리오 수익률은 약 {{LOSS_MILD}} ~ {{LOSS_SEVERE}}(중앙 {{LOSS_CENTRAL}})입니다 [calc].",
        "꼬리 구간 손실 기여는 {{TOP1}} {{TOP1_SHARE}}, {{TOP2}} {{TOP2_SHARE}}, {{TOP3}} {{TOP3_SHARE}} 순입니다 [calc].",
        ("정규분포를 가정하면 ES를 {{NORMAL_MC_ES}}로 계산해 {{ES_UNDER}} 과소평가합니다 [calc]."
         if state["risk_result"]["es_underestimation_normal"] > 0 else
         "이번 표본에서는 정규분포 가정 ES({{NORMAL_MC_ES}})가 실제 분포 기반 ES보다 작지 않았습니다 [calc]."),
        "",
        "## 유의사항",
        "충격 근거 경로는 {{TIER}}입니다 [calc].",
    ]
    return "\n".join(lines)


def append_checklist(text: str, state: dict) -> str:
    w = warnings_list(state)
    block = ["", "### 점검 목록 (자동)"] + ([f"- {x.rstrip('.')} [calc]." for x in w] or ["- 특이사항 없음 [calc]."])
    block.append("- 계산 기록: {{CALC_ID}} [calc].")
    return text.rstrip() + "\n" + "\n".join(block)


def fill(text: str, facts: dict[str, str]) -> str:
    return re.sub(r"\{\{([A-Z0-9_]+)\}\}", lambda m: facts.get(m.group(1), m.group(0)), text)


def write(state: dict, ctx: Context, llm: LLM, feedback: list[str] | None = None) -> dict:
    facts = build_facts(state, ctx)
    sources = build_sources(state, ctx)
    draft = None
    if llm.enabled:
        payload = {
            "event": {k: state["event"].get(k) for k in
                      ("title", "instrument", "direction", "initiator", "target", "sectors", "stage", "severity",
                       "channels", "evidence")},
            "placeholders": {k: f"(값: {v})" for k, v in facts.items()},
            "sources": sources,
            "analog_tier": state.get("analog_tier"),
            "warnings": warnings_list(state),
        }
        user = ("아래 자료로 브리핑을 쓰세요. 숫자는 placeholders의 키를 {{KEY}} 형태로만 쓰고, 값은 해석용 참고입니다.\n"
                + json.dumps(payload, ensure_ascii=False, indent=1))
        if feedback:
            user += "\n\n[검증에서 지적된 문제 - 반드시 고칠 것]\n- " + "\n- ".join(feedback)
        try:
            draft = llm.complete_text(prompts.REPORTER_SYSTEM, user, max_tokens=1500)
        except Exception:
            draft = None
    if not draft:
        draft = template_draft(state, ctx)
    draft = append_checklist(draft, state)
    return {"draft_brief": draft, "facts": facts, "sources": sources, "final_brief": fill(draft, facts)}
