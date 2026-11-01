"""메인 헤드쿼터: LangGraph로 에이전트 7개를 잇고, 판단 규칙(조건부 엣지)으로 흐름을 정한다.

흐름
  START -> event_analyst --(게이트)--> [analog_finder ∥ exposure_analyst] -> shock_mapper -> risk_engine
        -> reporter -> verifier --(게이트)--> finalize -> END
게이트 규칙 (전부 코드, LLM 아님)
  · 사건 필수 필드 비면 -> event_analyst 재실행(오류 피드백 포함), 재시도 초과 -> 판단 보류 종료
  · 영향 채널을 정할 수 없으면 -> 사용자에게 가정을 묻고 종료(need_user)
  · 사건 확정되면 analog_finder와 exposure_analyst를 동시에 실행, 둘 다 끝나면 shock_mapper로 합류
  · exposure_analyst가 실패해도 '직접 노출 판단 없음'으로 계속 진행
  · verifier 반려 -> 지적된 담당(reporter / event_analyst / shock_mapper)만 다시, 재시도 2회 초과 -> 판단 보류 종료
What-if: 이전 결과의 사건·유사 사례는 그대로 두고 shock_mapper부터만 다시 돈다.
"""
from __future__ import annotations

import re
import time
import uuid

from langgraph.graph import END, START, StateGraph

from . import analog_finder, event_analyst, exposure_analyst, reporter, risk_agent, shock_mapper, verifier
from .context import Context
from .llm import LLM
from .state import Event, ShockLensState


def _t(node: str, summary: str, t0: float) -> list[dict]:
    return [{"node": node, "summary": summary, "ms": int((time.time() - t0) * 1000)}]


def _retries(state: dict, node: str) -> dict:
    r = dict(state.get("retries") or {})
    r[node] = r.get(node, -1) + 1          # 첫 실행 = 0, 재실행마다 +1
    return r


def build_graph(ctx: Context, llm: LLM, entry: str = "event_analyst", verify_with_llm: bool = False):
    max_retries = ctx.thresholds["hq"]["max_retries"]

    # ------------------------------------------------------------ 노드
    def n_event(state: ShockLensState) -> dict:
        t0 = time.time()
        if state.get("event_preset"):
            return {"event_complete": True, "event_mode": "preset", "retries": _retries(state, "event_analyst"),
                    "trace": _t("event_analyst", "라이브러리 사건 사용(뉴스 해석 생략)", t0)}
        fb = state.get("event_errors") or []
        ver = state.get("verification") or {}
        fb += [i["detail"] for i in ver.get("issues", []) if i["route"] == "event_analyst"]
        out = event_analyst.analyze(state["raw_input"], llm, fb or None)
        ev = out["event"]
        summ = (f"{ev.get('instrument')} {ev.get('initiator')}→{ev.get('target')} 강도 {ev.get('severity')} "
                f"[{out['mode']}]" if ev else f"구조화 실패: {out['event_errors']}")
        return {"sentences": out["sentences"], "event": ev, "event_complete": out["event_complete"],
                "event_errors": out["event_errors"], "event_mode": out["mode"],
                "retries": _retries(state, "event_analyst"), "trace": _t("event_analyst", summ, t0)}

    def n_analog(state: ShockLensState) -> dict:
        t0 = time.time()
        out = analog_finder.find(state["event"], ctx)
        ids = [a["id"] for a in out["analogs"]]
        return {**out, "trace": _t("analog_finder", f"{out['analog_tier']} {ids}", t0)}

    def n_exposure(state: ShockLensState) -> dict:
        t0 = time.time()
        try:
            out = exposure_analyst.analyze(state["event"], ctx)
            high = [b for b, l in out["exposure"].items() if l == "high"]
            return {**out, "trace": _t("exposure_analyst", f"high {high}", t0)}
        except Exception as e:                          # 실패해도 전체는 계속
            return {"exposure": {}, "exposure_reasons": {"_error": f"직접 노출 판단 없음 ({e})"},
                    "trace": _t("exposure_analyst", f"실패 -> 직접 노출 판단 없음: {e}", t0)}

    def n_shock(state: ShockLensState) -> dict:
        t0 = time.time()
        ids = [a["id"] for a in state.get("analogs", [])]
        out = shock_mapper.map_shock(state["event"], ids, state.get("analog_tier", "synthetic"),
                                     state.get("exposure", {}), ctx, state.get("whatif_overrides"))
        c = {f: round(v["central"], 4) for f, v in out["factor_shock"].items() if v["central"]}
        return {**out, "retries": _retries(state, "shock_mapper"),
                "trace": _t("shock_mapper", f"{out['beta_regime']} central {c}", t0)}

    def n_risk(state: ShockLensState) -> dict:
        t0 = time.time()
        out = risk_agent.run(state, ctx)
        r = out["risk_result"]
        return {**out, "data_version": ctx.data_version,
                "trace": _t("risk_engine", f"ES97.5 {r['stressed']['ES']:.2%} (사건 효과 "
                            f"{r['decomposition']['event_effect_ES']:+.2%}p) {out['calc_log_id']}", t0)}

    def n_reporter(state: ShockLensState) -> dict:
        t0 = time.time()
        ver = state.get("verification") or {}
        fb = [i["detail"] for i in ver.get("issues", []) if i["route"] == "reporter"] or None
        out = reporter.write(state, ctx, llm, fb)
        return {**out, "retries": _retries(state, "reporter"),
                "trace": _t("reporter", f"{len(out['final_brief'])}자 {'(재작성)' if fb else ''}", t0)}

    def n_verifier(state: ShockLensState) -> dict:
        t0 = time.time()
        v = verifier.check(state["final_brief"], state["facts"], state["sources"], state["risk_result"],
                           state["event"], llm, verify_with_llm)
        return {"verification": v,
                "trace": _t("verifier", "통과" if v["passed"] else f"반려 {len(v['issues'])}건 -> {v['route']}", t0)}

    def n_finalize(state: ShockLensState) -> dict:
        return {"status": "ok", "trace": _t("hq", "완료", time.time())}

    def n_partial(state: ShockLensState) -> dict:
        reasons = state.get("event_errors") or []
        reasons += [i["detail"] for i in (state.get("verification") or {}).get("issues", [])]
        head = "## 판단 보류\n아래 문제를 해결하지 못해 일부 결과만 제공합니다.\n" + \
               "\n".join(f"- {r}" for r in reasons[:8])
        body = state.get("final_brief", "")
        return {"status": "partial", "final_brief": head + ("\n\n" + body if body else ""),
                "trace": _t("hq", "판단 보류 종료", time.time())}

    def n_need_user(state: ShockLensState) -> dict:
        q = "이 사건이 어떤 경로(위험회피·유가·위안화·반도체 등)로 시장에 영향을 줄지 가정을 알려주세요."
        return {"status": "need_user", "needs_user": q, "final_brief": f"## 추가 정보 필요\n{q}",
                "trace": _t("hq", "사용자 가정 요청", time.time())}

    # ------------------------------------------------------------ 게이트
    def gate_event(state: ShockLensState):
        if state.get("event_complete"):
            if not state["event"].get("channels"):
                return "need_user"
            return ["analog_finder", "exposure_analyst"]          # 병렬 분기
        if state["retries"].get("event_analyst", 0) < max_retries:
            return "event_analyst"
        return "partial_end"

    def gate_verify(state: ShockLensState):
        v = state["verification"]
        if v["passed"]:
            return "finalize"
        route = v["route"]
        if (state.get("retries") or {}).get(route, 0) < max_retries:
            return route
        return "partial_end"

    g = StateGraph(ShockLensState)
    for name, fn in [("event_analyst", n_event), ("analog_finder", n_analog), ("exposure_analyst", n_exposure),
                     ("shock_mapper", n_shock), ("risk_engine", n_risk), ("reporter", n_reporter),
                     ("verifier", n_verifier), ("finalize", n_finalize), ("partial_end", n_partial),
                     ("need_user", n_need_user)]:
        g.add_node(name, fn)
    g.add_edge(START, entry)
    g.add_conditional_edges("event_analyst", gate_event,
                            ["event_analyst", "analog_finder", "exposure_analyst", "need_user", "partial_end"])
    g.add_edge(["analog_finder", "exposure_analyst"], "shock_mapper")    # 둘 다 끝나야 합류
    g.add_edge("shock_mapper", "risk_engine")
    g.add_edge("risk_engine", "reporter")
    g.add_edge("reporter", "verifier")
    g.add_conditional_edges("verifier", gate_verify,
                            ["finalize", "reporter", "event_analyst", "shock_mapper", "partial_end"])
    for n in ("finalize", "partial_end", "need_user"):
        g.add_edge(n, END)
    return g.compile()


# ---------------------------------------------------------------- 실행 함수
def run_news(text: str, ctx: Context, llm: LLM | None = None, seed: int | None = None) -> dict:
    llm = llm or LLM()
    graph = build_graph(ctx, llm)
    init: ShockLensState = {"run_id": uuid.uuid4().hex[:8], "raw_input": text, "retries": {}, "trace": [],
                            "seed": seed or ctx.thresholds["risk"]["seed"]}
    return graph.invoke(init, {"recursion_limit": 40})


def run_library_event(event_id: str, ctx: Context, llm: LLM | None = None) -> dict:
    """라이브러리 사건을 '새 뉴스'로 가정해 실행 (뉴스 해석 생략, 자기 자신은 유사 사례에서 제외)."""
    llm = llm or LLM()
    lib = ctx.event_by_id(event_id)
    if lib is None:
        raise KeyError(event_id)
    keep = set(Event.model_fields)
    ev = Event(**{k: v for k, v in lib.items() if k in keep})
    ev.evidence = {f: ["s1"] for f in ("instrument", "initiator", "target", "sectors", "stage", "severity")}
    ev.concurrent_shocks = list(lib.get("confounders", []))
    text = f"{lib['title']}. {' '.join(str(lib.get('description', '')).split())}"
    graph = build_graph(ctx, llm)
    init: ShockLensState = {"run_id": uuid.uuid4().hex[:8], "raw_input": text,
                            "sentences": [{"id": "s1", "text": text}], "event": ev.model_dump(),
                            "event_preset": True, "retries": {}, "trace": [],
                            "seed": ctx.thresholds["risk"]["seed"]}
    return graph.invoke(init, {"recursion_limit": 40})


FACTOR_WORDS = {
    "oil": r"유가|원유|oil|WTI", "risk_off": r"VIX|변동성|공포|위험회피", "dollar": r"달러|dollar",
    "cny": r"위안|CNY|yuan", "ust10y": r"금리|국채|10년물|yield", "credit": r"신용|하이일드|credit",
    "tech_war": r"반도체|기술전쟁|tech",
}


def parse_whatif(text: str, llm: LLM | None = None) -> dict:
    """'유가 2배', 'VIX 1.5x', '반도체 충격 3배' -> {"oil": {"mult": 2.0}, ...}"""
    if llm is not None and llm.enabled:
        try:
            res = llm.complete_json(
                "사용자의 가정 문장을 팩터 덮어쓰기 JSON으로만 바꿉니다. 팩터: oil, risk_off, dollar, cny, ust10y, credit, "
                "tech_war. 형식 {\"oil\": {\"mult\": 2}} 또는 {\"oil\": {\"abs\": 0.1}}. 숫자 해석 외 판단 금지.", text)
            if res and all(k in FACTOR_WORDS and isinstance(v, dict) for k, v in res.items()):
                return res
        except Exception:
            pass
    out = {}
    for f, pat in FACTOR_WORDS.items():
        m = re.search(rf"(?:{pat})[^0-9,]{{0,12}}(\d+(?:\.\d+)?)\s*(?:배|x|X)", text, re.I)
        if m:
            out[f] = {"mult": float(m.group(1))}
            continue
        m = re.search(rf"(?:{pat})[^0-9,+-]{{0,12}}([+-]\d+(?:\.\d+)?)\s*(%p|%|pt)?", text, re.I)
        if m:                                           # '유가 +10%' -> 절댓값 0.10, 'VIX +8pt' -> 8
            v = float(m.group(1))
            unit = (m.group(2) or "").lower()
            out[f] = {"abs": v / 100 if unit == "%" else v}
    if not out:
        raise ValueError(f"가정을 해석하지 못했습니다: {text} (예: '유가 2배')")
    return out


def run_whatif(prev: dict, text: str, ctx: Context, llm: LLM | None = None) -> dict:
    llm = llm or LLM()
    overrides = parse_whatif(text, llm)
    graph = build_graph(ctx, llm, entry="shock_mapper")
    keep = {k: prev[k] for k in ("run_id", "raw_input", "sentences", "event", "event_complete", "analogs",
                                 "analog_tier", "analog_table", "analog_excluded", "exposure", "exposure_reasons",
                                 "seed") if k in prev}
    init = {**keep, "whatif_overrides": overrides, "retries": {}, "trace": []}
    return graph.invoke(init, {"recursion_limit": 40})
