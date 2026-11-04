"""후속 질문 대화: 브리핑을 본 사용자가 "왜 반도체 손실이 제일 커?"처럼 물으면 답한다.

원칙은 브리핑과 같다. 숫자는 계산 장부(calc_log)에서만 나온다.
- LLM은 아래 조회 도구를 호출해 장부를 읽고, 도구가 돌려준 문자열 숫자만 옮겨 쓴다.
- 답변의 모든 %·%p 숫자를 도구 결과와 대조하고, 장부에 없는 숫자가 있으면 한 번 다시 쓰게 한다.
  그래도 틀리면 규칙 기반 답변으로 바꾸고 그 사실을 표시한다.
- LLM이 없으면 질문 키워드로 같은 도구를 골라 규칙 기반 답변을 만든다.
"""
from __future__ import annotations

import re

from engine import calclog

from .llm import LLM

NUM = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*%p?")


def _p(x: float) -> str:
    return f"{x * 100:.2f}%"


def _sp(x: float) -> str:
    return f"{x * 100:+.2f}%"


def _pp(x: float) -> str:
    return f"{x * 100:+.2f}%p"


class Ledger:
    """한 번의 실행(state)과 그 계산 장부를 감싸 조회 도구를 제공한다."""

    def __init__(self, state: dict, ctx=None):
        self.state = state
        self.labels = getattr(ctx, "labels", {}) or {}
        w = getattr(ctx, "weights", None)
        self.weights = (w / w.sum()).to_dict() if w is not None else {}
        cid = state.get("calc_log_id")
        try:
            self.log = calclog.load(cid) if cid else {}
        except FileNotFoundError:
            self.log = {}
        self.out = self.log.get("outputs") or {}
        self.calc_id = cid or "(장부 없음)"

    def name(self, k: str) -> str:
        return self.labels.get(k, k)

    # ---------------------------------------------------------- 도구
    def risk_summary(self) -> dict:
        r = self.state["risk_result"]
        d = r["decomposition"]
        return {
            "calc_log_id": self.calc_id,
            "기간": f"{r.get('horizon_days', 20)}거래일 손실 분포", "국면": r.get("regime"),
            "ES97.5": {"① 평상시": _p(r["baseline_normal"]["ES"]), "② 시장 국면": _p(r["baseline"]["ES"]),
                       "③ 국면+사건(중앙)": _p(r["stressed"]["ES"]), "약한 충격": _p(r["stressed_mild"]["ES"]),
                       "최악 충격": _p(r["stressed_severe"]["ES"])},
            "VaR95": {"② 시장 국면": _p(r["baseline"]["VaR"]), "③ 국면+사건(중앙)": _p(r["stressed"]["VaR"])},
            "분해": {"국면 효과": _pp(d["regime_effect_ES"]), "사건 효과": _pp(d["event_effect_ES"]),
                     "사건 효과(최악)": _pp(d["event_effect_ES_severe"]), "전체 변화": _pp(d["total_change_ES"])},
            "정규분포 가정 ES": _p(r["normal_mc_ES"]),
            **({"앙상블(대표값)": {"ES97.5": _p(r["ensemble"]["ES"]), "최악 충격 ES": _p(r["ensemble"]["ES_severe"]),
                               "모형 범위": f"{_p(r['ensemble']['range'][0])} ~ {_p(r['ensemble']['range'][1])}",
                               "보수적 기준(모형 최대)": _p(r["ensemble"]["conservative"]),
                               "모형별 ES": {k: _p(v["ES"]) for k, v in r["ensemble"]["models"].items()},
                               "가중치": r["ensemble"]["weights"],
                               "설명": "분해(①②③)는 기준 모형(블록 부트스트랩) 결과"}} if r.get("ensemble") else {}),
            "정규분포 가정의 과소추정": _pp(r["es_underestimation_normal"]),
            "발표 후 3거래일 시나리오 수익률": {k: _sp(v) for k, v in r["scenario_loss"].items()},
            **({"3거래일 예측구간(평소 변동 포함)": {"하단": _sp(r["scenario_band"]["interval"]["lo"]),
                                               "상단": _sp(r["scenario_band"]["interval"]["hi"])}}
               if r.get("scenario_band") else {}),
        }

    def basket_breakdown(self, basket: str | None = None) -> dict:
        tc = self.out.get("tail_contribution") or {}
        bs = self.out.get("basket_shock") or {}
        expo = self.state.get("exposure") or {}
        why = self.state.get("exposure_reasons") or {}
        rows = []
        for b in sorted(tc, key=lambda k: -tc[k]):
            if basket and basket not in (b, self.name(b)):
                continue
            rows.append({
                "바스켓": self.name(b), "코드": b,
                "비중": _p(self.weights[b]) if b in self.weights else "모름",
                "꼬리손실 기여 비중": _p(tc[b]),
                "3거래일 충격(중앙)": _sp(bs.get("central", {}).get(b, 0.0)),
                "3거래일 충격(최악)": _sp(bs.get("severe", {}).get(b, 0.0)),
                "직접 노출": expo.get(b, "판단 없음"), "노출 근거": why.get(b, ""),
            })
        return {"calc_log_id": self.calc_id, "설명": "꼬리손실 기여 비중 = ES 구간 경로에서 그 바스켓 손실이 차지한 몫 (합 100%)",
                "바스켓": rows or f"'{basket}' 바스켓을 찾지 못함"}

    def factor_shock(self) -> dict:
        fs = self.state.get("factor_shock") or {}
        ev = self.state.get("event") or {}
        rows = {}
        for f, v in fs.items():
            rows[self.name(f)] = {k: (f"{v[k]:+.2f}pt" if f == "risk_off" else _sp(v[k])) for k in ("mild", "central", "severe") if k in v}
        return {"calc_log_id": self.calc_id, "채널(방향)": ev.get("channels", {}), "요인 충격(발표 후 3거래일)": rows,
                "산출 방식": self.state.get("shock_method"), "베타 국면": self.state.get("beta_regime")}

    def event_and_analogs(self) -> dict:
        ev = self.state.get("event") or {}
        keys = ("title", "instrument", "direction", "initiator", "target", "sectors", "stage", "severity",
                "severity_components", "uncertain_fields")
        return {"사건": {k: ev.get(k) for k in keys},
                "강도 점수": f"{ev.get('severity_points', 0):.0f}점",
                "유사 사건": [{"id": a["id"], "가중치": f"{a.get('weight', 0):.2f}", "이유": a.get("reason", "")}
                           for a in self.state.get("analogs", [])],
                "근거 등급": self.state.get("analog_tier")}

    def structure(self) -> dict:
        st = (self.state.get("risk_result") or {}).get("structure") or {}
        if not st:
            return {"설명": "구조 점검 결과 없음"}
        return {"calc_log_id": self.calc_id,
                "블록별 비중": {k: _p(v) for k, v in st.get("bloc_shares", {}).items()},
                "꼬리손실 블록 기여": {k: _p(v) for k, v in st.get("tail_by_bloc", {}).items()},
                "국가별 비중(관할국 근사)": {k: _p(v) for k, v in list(st.get("country_shares", {}).items())[:8]},
                "유효 국가 수": st.get("effective_countries"),
                "상대 블록 관할 자산(대응조치·자본통제 노출)": _p(st.get("adverse_share", 0)),
                "분쟁 지역(대만) 관할 자산": _p(st.get("flashpoint_share", 0)),
                "중앙은행·국부펀드 자산 표적 여부": st.get("sovereign_flag")}

    def call(self, name: str, args: dict) -> dict:
        if name == "get_risk_summary":
            return self.risk_summary()
        if name == "get_basket_breakdown":
            return self.basket_breakdown(args.get("basket"))
        if name == "get_factor_shock":
            return self.factor_shock()
        if name == "get_event_and_analogs":
            return self.event_and_analogs()
        if name == "get_structure":
            return self.structure()
        raise ValueError(f"없는 도구: {name}")


TOOLS = [
    {"name": "get_risk_summary", "description": "ES97.5·VaR95 3단계 분해, 정규분포 가정 비교, 3거래일 시나리오 수익률을 계산 장부에서 읽는다.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_basket_breakdown", "description": "바스켓별 비중, 꼬리손실 기여 비중, 3거래일 충격, 직접 노출 판단과 근거를 읽는다. basket을 비우면 전체(기여 큰 순).",
     "input_schema": {"type": "object", "properties": {"basket": {"type": "string", "description": "바스켓 코드 또는 한글 이름"}}}},
    {"name": "get_factor_shock", "description": "요인(위험회피·달러·위안화·유가·금리·신용·반도체)별 충격 범위와 채널 방향을 읽는다.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_event_and_analogs", "description": "사건 구조·강도 요소와 근거로 쓴 과거 유사 사건 목록을 읽는다.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_structure", "description": "구조 점검: 블록별 비중(분절화), 국가 집중도, 상대 블록 관할 자산(국부펀드 고유 위험), 분쟁 지역 노출.",
     "input_schema": {"type": "object", "properties": {}}},
]

SYSTEM = """당신은 국부펀드 리스크 브리핑의 후속 질문에 답하는 분석가입니다.
반드시 도구로 계산 장부를 조회한 뒤 답합니다. 숫자는 도구가 돌려준 문자열을 그대로 옮기고, 새로 계산하거나 반올림하지 않습니다.
기간을 구분합니다: '3거래일 충격·시나리오 수익률'과 '20거래일 ES·VaR'를 섞지 않습니다. 정규분포 가정 ES를 '평상시'라 부르지 않습니다.
장부로 답할 수 없는 질문(미래 가격 예측, 매매 권고 등)은 '판단 보류'라고 답합니다.
답은 한국어 3~5문장, 마지막 줄에 [calc:장부번호] 를 붙입니다."""


def _allowed(outputs: list) -> set[float]:
    vals = set()
    for o in outputs:
        for m in NUM.finditer(str(o)):
            vals.add(round(abs(float(m.group(1))), 2))
    return vals


def check_numbers(answer: str, tool_outputs: list) -> list[str]:
    ok = _allowed(tool_outputs)
    return [m.group(0) for m in NUM.finditer(answer)
            if not any(abs(round(abs(float(m.group(1))), 2) - a) <= 0.011 for a in ok)]


# ---------------------------------------------------------- 규칙 기반 (LLM 없을 때·검증 실패 시)
def rule_answer(q: str, led: Ledger) -> tuple[str, list[dict]]:
    log = []

    def use(name, args=None):
        out = led.call(name, args or {})
        log.append({"name": name, "input": args or {}, "output": out})
        return out

    if re.search(r"블록|분절|집중|동결|국부펀드|관할|대만 노출", q):
        s = use("get_structure")
        if "블록별 비중" not in s:
            return f"{s['설명']} (판단 보류)\n[calc:{led.calc_id}]", log
        bl = ", ".join(f"{k} {v}" for k, v in s["블록별 비중"].items())
        return (f"블록별 비중은 {bl}이고, 국가 분산은 유효 {s['유효 국가 수']}개국 수준입니다. "
                f"상대 블록 관할 자산은 {s['상대 블록 관할 자산(대응조치·자본통제 노출)']}입니다.\n[calc:{led.calc_id}]"), log
    named = next((b for b in led.out.get("tail_contribution", {}) if led.name(b) in q or b in q), None)
    if named or re.search(r"왜|기여|제일|가장|많이|바스켓|자산", q):
        bd = use("get_basket_breakdown", {"basket": named} if named else {})
        rows = bd["바스켓"] if isinstance(bd["바스켓"], list) else []
        if not rows:
            return f"{bd['바스켓']} (판단 보류)\n[calc:{led.calc_id}]", log
        r = rows[0]
        txt = (f"{r['바스켓']}은(는) 포트폴리오 비중 {r['비중']}이지만 20거래일 꼬리손실의 {r['꼬리손실 기여 비중']}를 차지합니다. "
               f"이번 사건 시나리오에서 발표 후 3거래일 충격은 중앙 {r['3거래일 충격(중앙)']}, 최악 {r['3거래일 충격(최악)']}입니다. "
               f"직접 노출 판단은 '{r['직접 노출']}'이며 근거는 {r['노출 근거'] or '요인 베타'}입니다.")
        if not named and len(rows) > 1:
            txt += f" 기여 2위는 {rows[1]['바스켓']}({rows[1]['꼬리손실 기여 비중']})입니다."
        return txt + f"\n[calc:{led.calc_id}]", log
    if re.search(r"유사|과거|비슷|사례|근거", q):
        ea = use("get_event_and_analogs")
        an = ", ".join(f"{a['id']}({a['가중치']})" for a in ea["유사 사건"]) or "없음(합성 시나리오)"
        return f"충격 크기는 과거 유사 사건 {an}의 실제 시장 반응에서 가져왔습니다. 근거 등급은 {ea['근거 등급']}입니다.\n[calc:{led.calc_id}]", log
    if re.search(r"요인|팩터|채널|유가|달러|위안|금리|신용|VIX", q, re.I):
        fs = use("get_factor_shock")
        parts = ", ".join(f"{k} 중앙 {v.get('central')}" for k, v in fs["요인 충격(발표 후 3거래일)"].items())
        return f"발표 후 3거래일 요인 충격은 {parts}입니다. 산출 방식은 {fs['산출 방식']}입니다.\n[calc:{led.calc_id}]", log
    rs = use("get_risk_summary")
    e = rs["ES97.5"]
    en = rs.get("앙상블(대표값)")
    if en:
        return (f"20거래일 ES97.5 대표값은 모형 3개 앙상블 기준 {en['ES97.5']}(모형 범위 {en['모형 범위']}, "
                f"보수적 기준 {en['보수적 기준(모형 최대)']})입니다. 기준 모형(블록 부트스트랩) 분해로는 평상시 {e['① 평상시']}, "
                f"국면 반영 {e['② 시장 국면']}, 사건 반영 {e['③ 국면+사건(중앙)']}이고 사건 효과는 {rs['분해']['사건 효과']}입니다."
                f"\n[calc:{led.calc_id}]"), log
    return (f"20거래일 ES97.5는 평상시 {e['① 평상시']}, 시장 국면 반영 {e['② 시장 국면']}, 사건 반영 {e['③ 국면+사건(중앙)']}"
            f"(최악 {e['최악 충격']})입니다. 국면 효과 {rs['분해']['국면 효과']}, 사건 효과 {rs['분해']['사건 효과']}입니다."
            f"\n[calc:{led.calc_id}]"), log


def answer(question: str, state: dict, ctx=None, llm: LLM | None = None,
           history: list[dict] | None = None) -> dict:
    """반환: {answer, mode(llm/rule/rule_fallback), tool_calls, issues}"""
    led = Ledger(state, ctx)
    if not state.get("risk_result"):
        return {"answer": "아직 계산 결과가 없어 답할 수 없습니다 (판단 보류).", "mode": "rule", "tool_calls": [], "issues": []}
    if llm is not None and llm.enabled and getattr(llm, "provider", "") == "anthropic":
        msgs = list(history or []) + [{"role": "user", "content": question}]
        try:
            text, log = llm.chat_tools(SYSTEM, msgs, TOOLS, led.call)
            bad = check_numbers(text, [c["output"] for c in log])
            if bad or not log:
                fix = (f"직전 답변에 장부에 없는 숫자 {bad} 가 있습니다." if bad else "도구를 호출하지 않았습니다.")
                msgs2 = msgs + [{"role": "assistant", "content": text or "(빈 답변)"},
                                {"role": "user", "content": fix + " 도구를 조회해 장부 숫자만으로 다시 답하세요."}]
                text, log2 = llm.chat_tools(SYSTEM, msgs2, TOOLS, led.call)
                log += log2
                bad = check_numbers(text, [c["output"] for c in log])
            if text and log and not bad:
                if "[calc:" not in text:
                    text += f"\n[calc:{led.calc_id}]"
                return {"answer": text, "mode": "llm", "tool_calls": log, "issues": []}
            issues = [f"장부에 없는 숫자 {bad}"] if bad else ["도구 조회 없이 답변"]
        except Exception as e:
            issues = [f"LLM 호출 실패: {e}"]
        txt, log = rule_answer(question, led)
        return {"answer": txt, "mode": "rule_fallback", "tool_calls": log, "issues": issues}
    txt, log = rule_answer(question, led)
    return {"answer": txt, "mode": "rule", "tool_calls": log, "issues": []}
