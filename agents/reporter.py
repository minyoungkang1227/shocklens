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

TIER_KO = {"direct": "직접 유사 사례", "relaxed": "완화된 유사 사례", "pool": "같은 수단 계열 사건 풀", "synthetic": "강도 기반 합성", "user": "사용자 가정"}
INST_KO = {"tariff": "관세", "tariff_pause": "관세 유예·인하", "tech_export_control": "기술 수출통제",
           "entity_list": "거래제한 명단", "resource_export_control": "자원 수출통제",
           "currency": "환율", "financial_sanction": "금융제재", "chokepoint_disruption": "요충지·공급 차단",
           "industrial_subsidy": "산업 보조금·현지화", "investment_restriction": "투자 제한"}


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


BLOC_KEY = {"미국 블록": "US", "중국 블록": "CN", "비동맹·연결국": "OTHER"}


def structure_lines(state: dict) -> list[str]:
    """구조 점검(블록·집중도·국부펀드 고유 위험) 문장. 숫자는 build_facts와 같은 문자열을 쓴다."""
    st = (state.get("risk_result") or {}).get("structure") or {}
    if not st:
        return []
    ko = {"US": "미국", "CN": "중국", "EU": "EU", "JP": "일본", "TW": "대만", "KR": "한국", "IN": "인도", "RU": "러시아",
          "IR": "이란", "GB": "영국", "CA": "캐나다", "SA": "사우디", "VN": "베트남", "MX": "멕시코"}
    bl = st.get("bloc_shares") or {}
    tb = st.get("tail_by_bloc") or {}
    out = ["구조 점검(분절화): 블록별 비중 " + ", ".join(f"{k} {pct(v)}" for k, v in bl.items())
           + (", 꼬리손실 블록 기여 " + ", ".join(f"{k} {pct(v)}" for k, v in tb.items()) if tb else "")]
    out.append(f"구조 점검(집중도): 국가 분산은 유효 {st.get('effective_countries')}개국 수준입니다 (관할국 근사 기준).")
    adv = st.get("adverse_countries") or []
    if adv:
        sov = (f"국부펀드 고유 위험: 사건 당사국 중 상대 블록 관할({', '.join(ko.get(c, c) for c in adv)}) 자산 "
               f"{pct(st.get('adverse_share', 0))}는 대응조치·자본통제·동결에 직접 노출될 수 있습니다")
    else:
        sov = "국부펀드 고유 위험: 사건 당사국 중 상대 블록 관할 자산은 없습니다"
    if st.get("sovereign_flag"):
        sov += ". 중앙은행·국부펀드 자산이 직접 표적인 유형이라 동결 선례 위험을 별도 점검해야 합니다"
    out.append(sov)
    if st.get("flashpoint_share", 0) > 0:
        out.append(f"분쟁 당사 지역(대만) 관할 자산은 {pct(st['flashpoint_share'])}입니다.")
    return out


def build_facts(state: dict, ctx: Context) -> dict[str, str]:
    r = state["risk_result"]
    d = r["decomposition"]
    f = {
        "HORIZON": f"{r['horizon_days']}거래일",
        "ES_NORMAL": pct(r["baseline_normal"]["ES"]),
        "ES_REGIME": pct(r["baseline"]["ES"]),
        "ES_STRESSED": pct(r["stressed"]["ES"]),
        "ES_STRESSED_SEVERE": pct(r["stressed_severe"]["ES"]),
        "ES_STRESSED_MILD": pct(r["stressed_mild"]["ES"]),
        "EVENT_EFFECT_SEVERE": pp(d["event_effect_ES_severe"]),
        "VAR_STRESSED": pct(r["stressed"]["VaR"]),
        "VAR_STRESSED_SEVERE": pct(r["stressed_severe"]["VaR"]),
        "REGIME_EFFECT": pp(d["regime_effect_ES"]),
        "EVENT_EFFECT": pp(d["event_effect_ES"]),
        "TOTAL_EFFECT": pp(d["total_change_ES"]),
        "NORMAL_MC_ES": pct(r["normal_mc_ES"]),
        "ES_UNDER": f"{abs(r['es_underestimation_normal']) * 100:.2f}%p",
        "LOSS_MILD": spct(r["scenario_loss"]["mild"]),
        "LOSS_CENTRAL": spct(r["scenario_loss"]["central"]),
        "LOSS_SEVERE": spct(r["scenario_loss"]["severe"]),
        **({"LOSS_BAND_LO": spct(r["scenario_band"]["interval"]["lo"]),
            "LOSS_BAND_HI": spct(r["scenario_band"]["interval"]["hi"]),
            "LOSS_BAND_HALF": pct(r["scenario_band"]["half_width"])} if r.get("scenario_band") else {}),
        "REGIME": "위기 국면" if r["regime"] == "crisis" else "평상시 국면",
        "REGIME_REASON": (r.get("regime_info") or {}).get("reason", ""),
        "TIER": TIER_KO.get(state.get("analog_tier", ""), state.get("analog_tier", "")),
        "CALC_ID": state.get("calc_log_id", ""),
    }
    en = r.get("ensemble") or {}
    if en:
        f["ENS_ES"] = pct(en["ES"])
        f["ENS_ES_SEVERE"] = pct(en["ES_severe"])
        f["ENS_LO"], f["ENS_HI"] = pct(en["range"][0]), pct(en["range"][1])
        f["ENS_MAX"] = pct(en["conservative"])
        f["ENS_MAX_SEVERE"] = pct(en["conservative_severe"])
        for m, v in en["models"].items():
            f[f"ENS_M_{m.upper()}"] = pct(v["ES"])
    st = r.get("structure") or {}
    for k, v in (st.get("bloc_shares") or {}).items():
        f[f"BLOC_{BLOC_KEY.get(k, k)}"] = pct(v)
    for k, v in (st.get("tail_by_bloc") or {}).items():
        f[f"TAILBLOC_{BLOC_KEY.get(k, k)}"] = pct(v)
    if st:
        f["SOV_ADVERSE"] = pct(st.get("adverse_share", 0))
        f["FLASHPOINT"] = pct(st.get("flashpoint_share", 0))
        f["EFF_COUNTRIES"] = f"{st.get('effective_countries')}개국"
    for i, (b, c) in enumerate(r["tail_top3"], 1):
        f[f"TOP{i}"] = ctx.labels.get(b, b)
        f[f"TOP{i}_SHARE"] = f"{c * 100:.0f}%"
    return f


# 자리표시마다 '무엇을 재는 숫자인지'를 LLM에 알려준다 (기간·가정을 잘못 붙이는 오류 방지)
PLACEHOLDER_DESC = {
    "HORIZON": "손실 분포의 기간 (20거래일)",
    "ES_NORMAL": "① 평상시 표본의 20거래일 ES97.5",
    "ES_REGIME": "② 현재 국면 표본의 20거래일 ES97.5 (사건 충격 전)",
    "ES_STRESSED": "③ 국면 + 사건 중앙 충격을 반영한 20거래일 ES97.5 (기준 모형: 블록 부트스트랩, 분해용)",
    "ENS_ES": "대표값: 모형 3개(블록 부트스트랩·FHS-극단값-t코퓰라·국면 전환) 경로를 섞은 앙상블 20거래일 ES97.5 (사건 중앙 충격)",
    "ENS_ES_SEVERE": "앙상블 20거래일 ES97.5 (사건 최악 충격)",
    "ENS_LO": "모형 위험: 모형별 ES 중 최소", "ENS_HI": "모형 위험: 모형별 ES 중 최대",
    "ENS_MAX": "보수적 기준: 모형별 ES 최대 (보수적 보고용)", "ENS_MAX_SEVERE": "보수적 기준 + 최악 충격",
    "ENS_M_BOOTSTRAP": "블록 부트스트랩 모형 ES", "ENS_M_FHS_EVT_T": "FHS·극단값·t-코퓰라 모형 ES",
    "ENS_M_REGIME_SWITCH": "국면 전환 모형 ES",
    "ES_STRESSED_SEVERE": "국면 + 사건 최악(severe) 충격을 반영한 20거래일 ES97.5",
    "ES_STRESSED_MILD": "국면 + 사건 약한(mild) 충격을 반영한 20거래일 ES97.5",
    "VAR_STRESSED": "③의 VaR95", "VAR_STRESSED_SEVERE": "최악 충격 반영 VaR95",
    "REGIME_EFFECT": "국면 효과 = ② - ① (ES 차이)", "EVENT_EFFECT": "사건 효과 = ③ - ② (ES 차이)",
    "EVENT_EFFECT_SEVERE": "최악 충격 기준 사건 효과", "TOTAL_EFFECT": "합계 = ③ - ①",
    "NORMAL_MC_ES": "같은 충격을 '정규분포 가정' 몬테카를로로 계산한 ES (평상시 ES가 아님)",
    "ES_UNDER": "정규분포 가정이 ES를 과소평가한 폭 (③ - 정규분포 ES)",
    "LOSS_MILD": "발표 후 '3거래일' 시나리오 포트폴리오 수익률 (mild). 20거래일 손실이 아님",
    "LOSS_CENTRAL": "발표 후 '3거래일' 시나리오 포트폴리오 수익률 (central). 20거래일 손실이 아님",
    "LOSS_SEVERE": "발표 후 '3거래일' 시나리오 포트폴리오 수익률 (severe). 20거래일 손실이 아님",
    "LOSS_BAND_LO": "3거래일 예측구간 하단 = 시나리오 범위에 평소 시장 변동 띠를 더한 값", "LOSS_BAND_HI": "3거래일 예측구간 상단",
    "LOSS_BAND_HALF": "평소 시장 변동 띠 반폭 (사건 무관일 |3일 수익률| 75% 분위)",
    "REGIME": "현재 국면 이름", "REGIME_REASON": "국면 판단 근거 문장",
    "TIER": "충격 근거 경로", "CALC_ID": "계산 기록 번호",
    "TOP1": "꼬리 손실 기여 1위 바스켓", "TOP1_SHARE": "그 기여 비중",
    "TOP2": "기여 2위", "TOP2_SHARE": "기여 비중", "TOP3": "기여 3위", "TOP3_SHARE": "기여 비중",
}


def build_sources(state: dict, ctx: Context) -> dict[str, str]:
    bad = {f["id"] for f in state.get("injection_flags") or state["event"].get("injection_flags") or []}
    src = {s["id"]: s["text"] for s in state.get("sentences", []) if s["id"] not in bad}   # 지시문 의심 문장은 출처 아님
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
    w += state.get("analog_notes", [])
    w += state.get("shock_warnings", [])
    unc = [f for f in ev.get("uncertain_fields", []) if f in ("instrument", "initiator", "target", "channels")]
    if unc:
        w.append(f"뉴스에서 직접 근거를 찾지 못해 추정한 항목: {', '.join(unc)} (판단 보류 성격).")
    if ev.get("instrument") in ("industrial_subsidy", "investment_restriction"):
        w.append("산업정책·투자 제한 법령은 효과가 몇 달에 걸쳐 나타나 단기 ES에 다 담기지 않습니다. "
                 "과거 같은 유형 법령 이후의 장기 업종 초과수익은 '장기 효과(법령)' 분석에서 확인하십시오.")
    att = state.get("attention") or {}
    if att.get("note"):
        w.append(att["note"])
    if state.get("scenario"):
        w.insert(0, f"가상 시나리오입니다({state['scenario'].get('title', '')}): 사건 구조와 강도는 가정값이며 실제로 일어난 사건이 아닙니다.")
    nm = state.get("news_meta") or {}
    if nm.get("note"):
        w.append(f"{nm['note'].rstrip('.')}: {', '.join(nm.get('urls', [])[:2])}.")
    if nm.get("error"):
        w.append(f"뉴스 입력 도구 문제({nm['error']})로 입력 텍스트만으로 분석했습니다.")
    flags = state.get("injection_flags") or ev.get("injection_flags") or []
    if flags:
        w.append(f"뉴스 안에 AI에게 내리는 지시로 보이는 문장({', '.join(f['id'] for f in flags)})이 있어 분석 근거와 출처에서 뺐습니다.")
    spread = ev.get("severity_spread") or {}
    if spread and max(spread.values()) >= 2:
        wide = ", ".join(f"{k}" for k, v in spread.items() if v >= 2)
        w.append(f"강도를 {ev.get('severity_samples', 1)}번 채점한 결과 일부 요소({wide})가 흩어져 중앙값을 썼습니다. 강도 해석에 여유를 두십시오.")
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
    raw_t = str(ev.get("title", "")).strip().rstrip(".").strip()
    ok_title = 0 < len(raw_t) <= 45 and not raw_t.endswith(("다", "요"))      # 문장형 제목은 쓰지 않음
    title = raw_t if ok_title else f"{who}의 {INST_KO[ev['instrument']]} 조치"
    tg = ", ".join(COUNTRY_KO.get(c, c) for c in ev.get("target", [])) or "복수 국가"
    lines = [
        "## 요약",
        (f"{title} 사건으로 포트폴리오 {{{{HORIZON}}}} ES97.5는 모형 3개 앙상블 기준 {{{{ENS_ES}}}}"
         f"(모형 범위 {{{{ENS_LO}}}}~{{{{ENS_HI}}}}, 보수적 기준 {{{{ENS_MAX}}}})로 추정되며, "
         f"기준 모형상 평상시 대비 {{{{TOTAL_EFFECT}}}} 변화입니다 [calc]."
         if state["risk_result"].get("ensemble") else
         f"{title} 사건으로 포트폴리오 {{{{HORIZON}}}} ES97.5는 {{{{ES_STRESSED}}}}로 추정되며, "
         f"평상시 대비 {{{{TOTAL_EFFECT}}}} 변화입니다 [calc]."),
        "",
        "## 무엇이 일어났나",
        f"{josa(who)} {josa(tg, '을를')} 대상으로 {INST_KO[ev['instrument']]} 조치를 "
        f"취했습니다({STAGE_KO.get(ev['stage'], ev['stage'])} 단계, 강도 {ev['severity']}/5, 점수 {ev.get('severity_points', 0):.0f}/100) {tag_ev}.",
    ]
    if ev.get("sectors"):
        s_sec = ",".join(ev.get("evidence", {}).get("sectors", [])[:1])
        secs = ", ".join(SECTOR_KO.get(x, x) for x in ev["sectors"])
        lines.append(f"주요 대상 산업은 {secs}입니다 {f'[src:{s_sec}]' if s_sec else ''}.")
    if analogs:
        cls = {"small": "소", "medium": "중", "large": "대"}.get(state.get("size_class", ""), "")
        how = ("유사도 가중 분포로" if str(state.get("shock_method", "")).startswith("weighted")
               else "사건 단위로")
        lines.append(f"충격 크기는 사건 구조와 기사 단어가 비슷하고 충격 등급({cls}) 기준으로 고른 과거 사건 {len(analogs)}건의 "
                     f"발표 후 3거래일 반응을 {how} 반영해 잡았습니다 {a_tags}.")
    lines += [
        "",
        "## 포트폴리오 영향",
    ]
    en = state["risk_result"].get("ensemble") or {}
    if en:
        names = {"bootstrap": "블록 부트스트랩", "fhs_evt_t": "FHS·극단값·t-코퓰라", "regime_switch": "국면 전환"}
        parts = ", ".join(f"{names[m]} {{{{ENS_M_{m.upper()}}}}}" for m in en.get("models", {}))
        lines += [f"모형별 ES97.5는 {parts}이고, 최악 충격에서는 앙상블 {{{{ENS_ES_SEVERE}}}}(보수적 기준 {{{{ENS_MAX_SEVERE}}}})입니다 [calc].",
                  "아래 평상시·국면·사건 분해는 기준 모형(블록 부트스트랩) 결과입니다 [calc]."]
    lines += [
        "ES97.5는 평상시 {{ES_NORMAL}}, {{REGIME}} {{ES_REGIME}}, 사건 충격 반영 후 {{ES_STRESSED}}(최악 시나리오 {{ES_STRESSED_SEVERE}})입니다 [calc].",
        "국면 효과는 {{REGIME_EFFECT}}, 사건 자체 효과는 {{EVENT_EFFECT}}(최악 {{EVENT_EFFECT_SEVERE}})입니다 [calc].",
        "국면 판단 근거: {{REGIME_REASON}} [calc].",
        "발표 직후 3거래일 기준 시나리오 수익률은 약 {{LOSS_MILD}} ~ {{LOSS_SEVERE}}(중앙 {{LOSS_CENTRAL}})입니다 [calc].",
        *(["평소 시장 변동을 더한 3거래일 예측구간은 {{LOSS_BAND_LO}} ~ {{LOSS_BAND_HI}}입니다 [calc]."]
          if state["risk_result"].get("scenario_band") else []),
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
    w = warnings_list(state) + structure_lines(state)
    block = ["", "### 점검 목록 (자동)"] + ([f"- {x.rstrip('.')} [calc]." for x in w] or ["- 특이사항 없음 [calc]."])
    block.append("- 계산 기록: {{CALC_ID}} [calc].")
    return text.rstrip() + "\n" + "\n".join(block)


TAG_ANY = re.compile(r"\s*\[(?:src:[^\]]+|calc)\]")


def display_brief(text: str, sources: dict[str, str] | None = None) -> tuple[str, list[tuple[str, str]]]:
    """화면용: 문장 끝 근거 태그([src:..], [calc])를 지운 깔끔한 본문과, 본문이 인용한 근거 목록을 돌려준다.
    검증은 태그가 붙은 원문(final_brief)으로 이미 끝났고, 근거는 목록으로 따로 보여준다."""
    used = []
    for m in re.finditer(r"\[src:([^\]]+)\]", text):
        for sid in m.group(1).split(","):
            sid = sid.strip()
            if sid and sid not in [u[0] for u in used]:
                used.append((sid, (sources or {}).get(sid, "")))
    clean = TAG_ANY.sub("", text)
    clean = re.sub(r"[ \t]+([.,)])", r"\1", clean)          # 태그를 지우고 남은 '다 .' 같은 공백 정리
    clean = re.sub(r"\(\s*\)", "", clean)
    return clean, used


def fill(text: str, facts: dict[str, str]) -> str:
    return re.sub(r"\{\{([A-Z0-9_]+)\}\}", lambda m: facts.get(m.group(1), m.group(0)), text)


SRC_TAG = re.compile(r"\[src:([^\]]+)\]")


def apply_citations(blocks: list[dict], chunk_ids: list[str]) -> tuple[str, dict]:
    """인용 기능 응답 -> 본문. 인용된 뉴스 문장 id를 그 문장 끝 [src:..] 태그로 옮긴다.
    모델이 직접 단 태그와 API 인용이 다르면 불일치로 기록한다(태그는 지우지 않고 Verifier가 판단)."""
    raw = ""
    for b in blocks:
        raw += b["text"]
        ids = [chunk_ids[i] for i in b.get("cites", []) if 0 <= i < len(chunk_ids)]
        if ids:
            raw += "⟦" + ",".join(ids) + "⟧"
    stats = {"cited_spans": sum(1 for b in blocks if b.get("cites")), "added_tags": 0, "mismatch": []}
    out_lines = []
    for line in raw.splitlines():
        pieces = re.split(r"(?<=[.。])\s+(?=\S)", line)
        fixed = []
        for p in pieces:
            cited = []
            for m in re.findall(r"⟦([^⟧]+)⟧", p):
                cited += [x for x in m.split(",") if x not in cited]
            p = re.sub(r"⟦[^⟧]+⟧", "", p)
            have = {x.strip() for t in SRC_TAG.findall(p) for x in t.split(",")}
            news_have = {x for x in have if x in chunk_ids}
            if cited and not have:
                tag = f" [src:{','.join(cited)}]"
                p = (p.rstrip()[:-1] + tag + ".") if p.rstrip().endswith((".", "。")) else p.rstrip() + tag
                stats["added_tags"] += 1
            elif cited and news_have and not news_have & set(cited):
                stats["mismatch"].append(f"태그 {sorted(news_have)} vs 인용 {cited}: {p[:40]}")
            fixed.append(p)
        out_lines.append(" ".join(fixed))
    return "\n".join(out_lines), stats


ANY_TAG = re.compile(r"\[(?:src:[^\]]+|calc)\]")


def _add_tag(p: str, tag: str) -> str:
    s = p.rstrip()
    return (s[:-1] + f" {tag}" + s[-1]) if s.endswith((".", "。")) else s + f" {tag}"


def carry_calc_tags(draft: str) -> tuple[str, int]:
    """LLM이 문단 끝 문장에만 [calc]를 달고 앞 문장을 빼먹는 경우를 코드로 보정한다.
    - 자리표시자({{KEY}})가 든 문장: 값은 코드가 채우므로 [calc]를 단다.
    - 숫자 없는 문장: 같은 줄 바로 다음 문장이 [calc]만 달고 있으면 이어지는 설명으로 보고 [calc]를 물려받는다.
    뉴스 출처 태그([src:..])는 물려주지 않는다(출처를 지어내지 않기 위해)."""
    added = 0
    out = []
    for line in draft.splitlines():
        st = line.strip()
        if not st or st.startswith("#"):
            out.append(line)
            continue
        if st.startswith("- "):
            if "{{" in st and not ANY_TAG.search(st):
                line, added = _add_tag(line, "[calc]"), added + 1
            out.append(line)
            continue
        pieces = re.split(r"(?<=[.。])\s+(?=\S)", line)
        nxt_tags: list[str] = []
        for i in range(len(pieces) - 1, -1, -1):
            p = pieces[i]
            tags = ANY_TAG.findall(p)
            if not tags:
                if "{{" in p:
                    p, added = _add_tag(p, "[calc]"), added + 1
                elif nxt_tags == ["[calc]"] and len(p.strip()) > 15:
                    p, added = _add_tag(p, "[calc]"), added + 1
                pieces[i] = p
                tags = ANY_TAG.findall(p)
            nxt_tags = tags
        out.append(" ".join(pieces))
    return "\n".join(out), added


def write(state: dict, ctx: Context, llm: LLM, feedback: list[str] | None = None) -> dict:
    facts = build_facts(state, ctx)
    sources = build_sources(state, ctx)
    draft, err, cite_stats = None, None, None
    if llm.enabled:
        payload = {
            "event": {k: state["event"].get(k) for k in
                      ("title", "instrument", "direction", "initiator", "target", "sectors", "stage", "severity",
                       "channels", "evidence")},
            "placeholders": {k: f"{PLACEHOLDER_DESC.get(k, '')} (값: {v})" for k, v in facts.items()},
            "sources": sources,
            "analog_tier": state.get("analog_tier"),
            "warnings": warnings_list(state),
        }
        user = ("아래 자료로 브리핑을 쓰세요. 숫자는 placeholders의 키를 {{KEY}} 형태로만 쓰고, 값은 해석용 참고입니다.\n"
                + json.dumps(payload, ensure_ascii=False, indent=1))
        if feedback:
            user += "\n\n[검증에서 지적된 문제 - 반드시 고칠 것]\n- " + "\n- ".join(feedback)
        use_cite = (ctx.thresholds.get("reporter") or {}).get("citations", False) and llm.provider == "anthropic"
        bad = {f["id"] for f in state.get("injection_flags") or []}
        news = [x for x in state.get("sentences", []) if x["id"] not in bad]
        rcfg = ctx.thresholds.get("reporter") or {}
        max_tok = int(rcfg.get("max_tokens", 8000))
        try:
            if use_cite and news:
                blocks = llm.complete_cited(prompts.REPORTER_SYSTEM + prompts.CITATION_NOTE, user,
                                            [x["text"] for x in news], max_tokens=max_tok)
                if blocks:
                    draft, cite_stats = apply_citations(blocks, [x["id"] for x in news])
                meta = getattr(llm, "last_meta", {}) or {}
                # 인용 응답은 인용 원문(cited_text)도 출력 토큰에 들어가 길이 제한에 먼저 걸릴 수 있다.
                # 잘렸거나 본문이 너무 짧으면 인용 없이 한 번 더 쓴다 (출처 태그는 모델이 직접 단다).
                if meta.get("stop_reason") == "max_tokens" or not draft or len(draft) < 400:
                    cite_stats = {**(cite_stats or {}), "fallback": f"citations 응답 {meta.get('stop_reason')} "
                                                                     f"{len(draft or '')}자 -> 일반 텍스트로 재작성"}
                    draft = llm.complete_text(prompts.REPORTER_SYSTEM, user, max_tokens=max_tok)
            else:
                draft = llm.complete_text(prompts.REPORTER_SYSTEM, user, max_tokens=max_tok)
            if not draft:
                err = f"LLM 응답에 본문이 없음 {getattr(llm, 'last_meta', {})}"
        except Exception as e:
            draft, err = None, f"{type(e).__name__}: {e}"
    mode = "llm" if draft else "template"
    if not draft:
        draft = template_draft(state, ctx)
    draft, n_auto = carry_calc_tags(draft)
    if cite_stats is not None:
        cite_stats["auto_calc_tags"] = n_auto
    draft = append_checklist(draft, state)
    return {"draft_brief": draft, "facts": facts, "sources": sources, "final_brief": fill(draft, facts),
            "reporter_mode": mode + ("+citations" if cite_stats and mode == "llm" else ""),
            "citation_stats": cite_stats,
            "reporter_error": err if llm.enabled and mode == "template" else None}
