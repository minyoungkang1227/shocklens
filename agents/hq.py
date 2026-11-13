"""메인 헤드쿼터: LangGraph로 에이전트 7개를 잇고, 판단 규칙(조건부 엣지)으로 흐름을 정한다.

흐름 (뉴스 입력)
  START -> news_input(URL·헤드라인이면 기사 검색·본문 추출) -> triage(작은 모델로 중요도 1차 분류)
        --(게이트: 중요도 미달이면 skipped 종료, force면 통과)--> event_analyst ...
흐름 (공통)
  event_analyst --(게이트)--> [analog_finder ∥ exposure_analyst] -> shock_mapper -> risk_engine
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

from . import (analog_finder, event_analyst, event_checker, exposure_analyst, news_input, reporter, risk_agent, shock_mapper, triage,
               verifier)
from .context import Context
from .llm import LLM
from .state import Event, ShockLensState


def _t(node: str, summary: str, t0: float) -> list[dict]:
    return [{"node": node, "summary": summary, "ms": int((time.time() - t0) * 1000)}]


def _retries(state: dict, node: str) -> dict:
    r = dict(state.get("retries") or {})
    r[node] = r.get(node, -1) + 1          # 첫 실행 = 0, 재실행마다 +1
    return r


def _derive(llm, model):
    """작은 모델용 LLM. 키가 없거나 모델을 바꿀 수 없는 LLM(테스트용 가짜)이면 None -> 규칙 경로."""
    if llm is None or not getattr(llm, "enabled", False) or not hasattr(llm, "derive"):
        return None
    return llm.derive(model) if model else llm.derive(small=True)


def _attention(ev: dict, acfg: dict) -> dict:
    """사건 직전 보도량 백분위. 네트워크 실패면 {error}만 남기고 진행."""
    from engine import attention
    from engine import severity as sev
    from tools import news
    q = attention.query_for(ev)
    tl = news.timeline_volume(q)
    m = attention.measure(tl["points"])
    out = {"query": q, "error": tl["error"], **(m or {})}
    note = attention.interpret(m, acfg.get("high_percentile", 80))
    if note:
        out["note"] = note
    if m and acfg.get("adjust_surprise") and m["percentile"] >= acfg.get("high_percentile", 80):
        comps = dict(ev.get("severity_components") or {})
        if comps.get("surprise", 1) > 1:
            comps["surprise"] -= 1
            pts = sev.score(comps)
            out["adjusted"] = {**ev, "severity_components": comps, "severity_points": pts, "severity": sev.level(pts),
                               "notes": list(ev.get("notes", [])) + ["주의도 높음 -> 의외성 1 낮춤"]}
    return out


def build_graph(ctx: Context, llm: LLM, entry: str = "event_analyst", verify_with_llm: bool | None = None,
                force: bool = False):
    max_retries = ctx.thresholds["hq"]["max_retries"]
    lcfg = ctx.thresholds.get("llm") or {}
    tcfg = ctx.thresholds.get("triage") or {}
    rerank_on = bool(lcfg.get("analog_rerank", False))
    llm_mode = lcfg.get("verifier_llm_check", "off")
    if verify_with_llm is None:
        verify_with_llm = llm_mode != "off"
    elif verify_with_llm and llm_mode == "off":
        llm_mode = "reject"
    verifier_llm = _derive(llm, lcfg.get("verifier_model"))      # 작성자(Reporter)와 다른 모델로 검증
    triage_llm = _derive(llm, lcfg.get("triage_model"))
    ccfg = ctx.thresholds.get("event_check") or {}
    checker_llm = _derive(llm, lcfg.get("checker_model")) if ccfg.get("use_llm", True) else None   # 해석과 다른 모델로 검수

    # ------------------------------------------------------------ 노드
    def n_news_input(state: ShockLensState) -> dict:
        t0 = time.time()
        out = news_input.gather(state["raw_input"], llm, ctx.thresholds.get("news"))
        m = out["news_meta"]
        summ = f"{m['kind']} 입력, 도구 {m['tool_calls']}회, {m['chars_in']}→{m['chars_out']}자 ({m['lang']})" + \
               (f" / {m['error']}" if m.get("error") else "")
        return {"raw_input": out["text"], "original_input": state["raw_input"], "news_meta": m,
                "trace": _t("news_input", summ, t0)}

    def n_triage(state: ShockLensState) -> dict:
        t0 = time.time()
        if force or not tcfg.get("enabled", True):
            t = {"relevant": True, "importance": 0, "reason": "강제 실행" if force else "분류 꺼짐", "mode": "skip", "model": None}
        else:
            t = triage.classify(state["raw_input"], triage_llm)
        return {"triage": t, "trace": _t("triage", f"관련 {t['relevant']} 중요도 {t['importance']} [{t['mode']}] {t['reason']}", t0)}

    def n_skipped(state: ShockLensState) -> dict:
        t = state["triage"]
        brief = ("## 분석 생략 (중요도 1차 분류)\n"
                 f"- 판정: {'지경학 정책 사건' if t['relevant'] else '지경학 정책 사건 아님'}, 중요도 {t['importance']}/5 "
                 f"(기준 {tcfg.get('min_importance', 2)} 이상)\n- 이유: {t['reason']}\n"
                 "- 그래도 분석하려면 강제 실행을 선택하세요.")
        return {"status": "skipped", "final_brief": brief, "trace": _t("hq", "중요도 미달로 분석 생략", time.time())}

    def n_event(state: ShockLensState) -> dict:
        t0 = time.time()
        if state.get("event_preset"):
            return {"event_complete": True, "event_mode": "preset", "retries": _retries(state, "event_analyst"),
                    "trace": _t("event_analyst", "라이브러리 사건 사용(뉴스 해석 생략)", t0)}
        fb = state.get("event_errors") or []
        ver = state.get("verification") or {}
        fb += [i["detail"] for i in ver.get("issues", []) if i["route"] == "event_analyst"]
        out = event_analyst.analyze(state["raw_input"], llm, fb or None, ctx.thresholds.get("severity"))
        ev = out["event"]
        if out.get("not_policy") and out["mode"] == "rule":     # 규칙 경로는 '못 찾음'이지 '아님'이 아니다 -> 판단 보류
            return {"sentences": out["sentences"], "event": {}, "event_complete": False, "event_mode": "rule",
                    "event_errors": ["규칙 경로(LLM 없음)가 정책 수단을 찾지 못함 — 정책 사건일 수도 있으니 LLM으로 다시 확인"],
                    "retries": {**_retries(state, "event_analyst"), "event_analyst": max_retries},
                    "trace": _t("event_analyst", "규칙 경로: 정책 수단 단어 없음 -> 판단 보류", t0)}
        if out.get("not_policy"):                               # (2026-11-11) 정책 사건 아님 -> 바로 종료
            return {"sentences": out["sentences"], "event": {}, "event_complete": False, "not_policy": True,
                    "not_policy_reason": out.get("not_policy_reason", ""), "event_mode": out["mode"],
                    "event_errors": [], "retries": _retries(state, "event_analyst"),
                    "trace": _t("event_analyst", f"정책 사건 아님 [{out['mode']}] {out.get('not_policy_reason', '')}", t0)}
        # (2026-11-11) 근거 검사를 Reporter 전에: LLM 해석의 필수 필드에 뉴스 근거가 없으면 여기서 재해석
        if ev and out["event_complete"] and out["mode"] == "llm":
            no_ev = [f for f in ("instrument", "initiator", "target") if f not in (ev.get("evidence") or {})]
            if no_ev:
                out["event_complete"] = False
                out["event_errors"] = [f"필수 필드에 뉴스 근거 문장 없음: {no_ev}. 근거 문장이 없으면 instrument='none'으로 내세요."]
        # 재해석이 직전과 같으면 더 돌려도 같은 답 -> 재시도 횟수를 소진시켜 바로 판단 보류
        prev = state.get("event") or {}
        key = lambda e: (e.get("instrument"), e.get("initiator"), tuple(e.get("target") or []),
                         tuple((e.get("evidence") or {}).get("instrument") or []))
        repeat = bool(prev and ev and "event_analyst" in (state.get("retries") or {}) and key(prev) == key(ev)
                      and not out["event_complete"])
        if repeat:
            out["event_errors"] = list(out["event_errors"]) + ["재해석 결과가 직전과 같아 재시도를 중단"]
        summ = (f"{ev.get('instrument')} {ev.get('initiator')}→{ev.get('target')} 강도 {ev.get('severity')}({ev.get('severity_points', 0):.0f}) "
                f"[{out['mode']}]" + (f" 지시문 의심 {len(out['injection_flags'])}건" if out.get("injection_flags") else "")
                + (f" / LLM 실패: {out['llm_error']}" if out.get("llm_error") else "")
                if ev else f"구조화 실패: {out['event_errors']}")
        att = {}
        acfg = ctx.thresholds.get("attention") or {}
        if ev and acfg.get("enabled") and out["event_complete"]:
            att = _attention(ev, acfg)
            if att.get("adjusted"):
                ev = att.pop("adjusted")
        retries = _retries(state, "event_analyst")
        if repeat:
            retries = {**retries, "event_analyst": max_retries}
        return {"attention": att, "sentences": out["sentences"], "event": ev, "event_complete": out["event_complete"],
                "event_errors": out["event_errors"], "event_mode": out["mode"],
                "injection_flags": out.get("injection_flags", []),
                "retries": retries, "trace": _t("event_analyst", summ + (" / 직전과 같은 해석 -> 중단" if repeat else ""), t0)}

    def n_analog(state: ShockLensState) -> dict:
        t0 = time.time()
        out = analog_finder.find(state["event"], ctx, state.get("raw_input"), llm=llm if rerank_on else None)
        ids = [f"{a['id']}({a['weight']:.2f})" for a in out["analogs"]]
        return {**out, "trace": _t("analog_finder", f"{out['analog_tier']} 등급 {out['size_class']} {ids}", t0)}

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
        w = {a["id"]: a.get("weight", 1.0) for a in state.get("analogs", [])}
        asof = state["event"].get("date") or None if state.get("event_preset") else None
        out = shock_mapper.map_shock(state["event"], ids, state.get("analog_tier", "synthetic"),
                                     state.get("exposure", {}), ctx, state.get("whatif_overrides"), w, asof,
                                     state.get("regime_override"))
        c = {f: round(v["central"], 4) for f, v in out["factor_shock"].items() if v["central"]}
        return {**out, "retries": _retries(state, "shock_mapper"),
                "trace": _t("shock_mapper", f"{out['beta_regime']}({out['regime_info'].get('reason', '')}) "
                            f"{out['shock_method']} central {c}", t0)}

    def n_risk(state: ShockLensState) -> dict:
        t0 = time.time()
        out = risk_agent.run(state, ctx)
        r = out["risk_result"]
        return {**out, "data_version": ctx.data_version,
                "trace": _t("risk_engine", f"ES97.5 {r['stressed']['ES']:.2%}~{r['stressed_severe']['ES']:.2%} (사건 효과 "
                            f"{r['decomposition']['event_effect_ES']:+.2%}p) {out['calc_log_id']}", t0)}

    def n_reporter(state: ShockLensState) -> dict:
        t0 = time.time()
        ver = state.get("verification") or {}
        fb = [i["detail"] for i in ver.get("issues", []) if i["route"] == "reporter"] or None
        out = reporter.write(state, ctx, llm, fb)
        return {**out, "retries": _retries(state, "reporter"),
                "trace": _t("reporter", f"{out['reporter_mode']} {len(out['final_brief'])}자 {'(재작성)' if fb else ''}"
                            + (f" / LLM 실패: {out['reporter_error']}" if out.get("reporter_error") else ""), t0)}

    def n_verifier(state: ShockLensState) -> dict:
        t0 = time.time()
        v = verifier.check(state["final_brief"], state["facts"], state["sources"], state["risk_result"],
                           state["event"], verifier_llm, verify_with_llm, llm_mode,
                           require_structure=state.get("reporter_mode", "").startswith("llm"))
        extra = f" (사실 검사 {v['checker_model']}, 경고 {len(v.get('warnings', []))})" if v.get("checker_model") else ""
        return {"verification": v,
                "trace": _t("verifier", ("통과" if v["passed"] else f"반려 {len(v['issues'])}건 -> {v['route']}") + extra, t0)}

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

    def n_check(state: ShockLensState) -> dict:
        """사건 검수관 (2026-11-13): Event Analyst와 다른 모델·코드 규칙으로 해석을 한 번 더 거른다."""
        t0 = time.time()
        if state.get("event_preset") or not ccfg.get("enabled", True):
            c = {"verdict": "pass", "reason": "라이브러리·가정 사건이라 검수 생략" if state.get("event_preset") else "검수 꺼짐",
                 "mode": "skip"}
            return {"event_check": c, "trace": _t("event_checker", c["reason"], t0)}
        flagged = [f["id"] for f in state.get("injection_flags") or []]
        c = event_checker.check(state["event"], state.get("sentences") or [], checker_llm, ccfg, flagged)
        upd: dict = {"event_check": c}
        if c["verdict"] == "not_policy":
            upd.update(not_policy=True, not_policy_reason=c["reason"])
        elif c["verdict"] == "off_topic":
            upd["event_errors"] = [f"사건 검수관 반려: {c['reason']}. 기사 제목·첫 문단의 중심 사건을 근거로 다시 해석하고, "
                                   f"중심 사건이 정책 조치가 아니면 instrument='none'으로 내세요."]
        summ = f"{c['verdict']} [{c['mode']}] {c['reason']}" + (f" / LLM 실패: {c['llm_error']}" if c.get("llm_error") else "")
        upd["trace"] = _t("event_checker", summ, t0)
        return upd

    def n_not_policy(state: ShockLensState) -> dict:
        r = state.get("not_policy_reason", "")
        brief = ("## 정책 사건 아님\n"
                 f"- Event Analyst 판단: 뉴스의 중심 사건이 정부·의회·선거가 만든 정책 조치가 아닙니다. {r}\n"
                 "- ShockLens는 정책 사건의 포트폴리오 꼬리 위험만 계산합니다. 시장 이벤트 자체의 영향은 계산하지 않습니다.")
        return {"status": "not_policy", "final_brief": brief, "trace": _t("hq", "정책 사건 아님으로 종료", time.time())}

    def n_need_user(state: ShockLensState) -> dict:
        q = "이 사건이 어떤 경로(위험회피·유가·위안화·반도체 등)로 시장에 영향을 줄지 가정을 알려주세요."
        return {"status": "need_user", "needs_user": q, "final_brief": f"## 추가 정보 필요\n{q}",
                "trace": _t("hq", "사용자 가정 요청", time.time())}

    # ------------------------------------------------------------ 게이트
    def gate_triage(state: ShockLensState):
        t = state["triage"]
        if force or t["mode"] == "skip" or triage.passes(t, tcfg.get("min_importance", 2)):
            return "event_analyst"
        return "skipped"

    def gate_event(state: ShockLensState):
        if state.get("not_policy"):
            return "not_policy"
        if state.get("event_complete"):
            if not state["event"].get("channels"):
                return "need_user"
            return "event_checker"
        if state["retries"].get("event_analyst", 0) < max_retries:
            return "event_analyst"
        return "partial_end"

    def gate_check(state: ShockLensState):
        v = (state.get("event_check") or {}).get("verdict", "pass")
        if v == "pass":
            return ["analog_finder", "exposure_analyst"]          # 병렬 분기
        if v == "not_policy":
            return "not_policy"
        if state["retries"].get("event_analyst", 0) < max_retries and state.get("event_mode") == "llm":
            return "event_analyst"                                 # 지적을 전달해 재해석 (규칙 경로는 다시 돌려도 같다)
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
                     ("need_user", n_need_user), ("news_input", n_news_input), ("triage", n_triage),
                     ("skipped", n_skipped), ("not_policy", n_not_policy), ("event_checker", n_check)]:
        g.add_node(name, fn)
    g.add_edge(START, entry)
    g.add_edge("news_input", "triage")
    g.add_conditional_edges("triage", gate_triage, ["event_analyst", "skipped"])
    g.add_conditional_edges("event_analyst", gate_event,
                            ["event_analyst", "event_checker", "need_user", "partial_end", "not_policy"])
    g.add_conditional_edges("event_checker", gate_check,
                            ["analog_finder", "exposure_analyst", "not_policy", "event_analyst", "partial_end"])
    g.add_edge(["analog_finder", "exposure_analyst"], "shock_mapper")    # 둘 다 끝나야 합류
    g.add_edge("shock_mapper", "risk_engine")
    g.add_edge("risk_engine", "reporter")
    g.add_edge("reporter", "verifier")
    g.add_conditional_edges("verifier", gate_verify,
                            ["finalize", "reporter", "event_analyst", "shock_mapper", "partial_end"])
    for n in ("finalize", "partial_end", "need_user", "skipped", "not_policy"):
        g.add_edge(n, END)
    return g.compile()


# ---------------------------------------------------------------- 실행 함수
def run_news(text: str, ctx: Context, llm: LLM | None = None, seed: int | None = None, force: bool = False) -> dict:
    """text: 뉴스 본문·헤드라인·URL 무엇이든. force=True면 중요도 분류와 상관없이 전체 분석."""
    llm = llm or LLM()
    graph = build_graph(ctx, llm, entry="news_input", force=force)
    init: ShockLensState = {"run_id": uuid.uuid4().hex[:8], "raw_input": text, "retries": {}, "trace": [],
                            "seed": seed or ctx.thresholds["risk"]["seed"]}
    return graph.invoke(init, {"recursion_limit": 40})


def scenario_event(sc: dict, ctx: Context) -> Event:
    """시나리오 dict -> Event (강도 요소 -> 점수·등급 계산)."""
    from engine import severity as sev
    keep = set(Event.model_fields)
    d = {k: v for k, v in sc.items() if k in keep}
    d.setdefault("direction", "escalate")
    d.setdefault("stage", "escalate")
    d["id"] = sc.get("id", "scenario")
    d["date"] = ""                                   # 날짜 없음 -> 모든 과거 사건을 유사 사례 후보로
    comps = {k: sev.clamp(v) for k, v in (sc.get("severity_components") or {}).items()}
    w = (ctx.thresholds.get("severity") or {}).get("component_weights")
    if comps:
        d["severity_components"] = comps
        d["severity_points"] = sev.score(comps, w)
        d["severity"] = sev.level(d["severity_points"])
    ev = Event(**d)
    ev.evidence = {f: ["s1"] for f in ("instrument", "initiator", "target", "sectors", "stage", "severity")}
    ev.notes.append("가상 시나리오: 사건 구조와 강도는 가정값")
    return ev


def scenario_from_text(text: str, ctx: Context, llm: LLM | None = None) -> dict:
    """사용자가 글로 쓴 가정("중국이 대만을 봉쇄하고 미국이 대중 금융제재로 맞서면?")을 시나리오 dict로 바꾼다.
    Event Analyst(LLM, 없으면 규칙)로 구조화하고, 결과는 화면에서 사람이 고친 뒤 run_scenario로 실행한다."""
    llm = llm or LLM()
    out = event_analyst.analyze(text, llm, None, ctx.thresholds.get("severity"))
    if not out["event_complete"]:
        raise ValueError("가정을 사건 구조로 바꾸지 못했습니다: " + "; ".join(out["event_errors"])[:200]
                         + " (누가·누구에게·어떤 조치를 하는지 적어 주세요)")
    ev = out["event"]
    from .reporter import COUNTRY_KO, INST_KO
    title = str(ev.get("title") or "").strip()
    if out["mode"] != "llm" or not title or len(title) > 30 or title.endswith(("?", "다", "면")):
        who = COUNTRY_KO.get(ev["initiator"], ev["initiator"])          # 문장 일부를 자르지 않고 구조로 제목을 만든다
        title = f"{who}의 {INST_KO.get(ev['instrument'], ev['instrument'])}"
    keys = ("instrument", "direction", "initiator", "target", "sectors", "stage", "severity_components",
            "channels", "sovereign_asset_risk")
    sc = {"id": "sc_custom", "title": title if title.startswith("(가정)") else f"(가정) {title}",
          "description": " ".join(text.split()), **{k: ev[k] for k in keys if k in ev}}
    sc["_mode"] = out["mode"]
    sc["_uncertain"] = ev.get("uncertain_fields", [])
    return sc


def run_scenario(scenario: str | dict, ctx: Context, llm: LLM | None = None, overrides: dict | None = None,
                 regime: str | None = None) -> dict:
    """가상 시나리오 실행. scenario = config/scenarios.yaml의 id 또는 같은 형식의 dict (사용자 정의).
    regime: 'crisis'/'normal' 강제 (없으면 시나리오의 regime, 그것도 없으면 현재 VIX로 판단)."""
    from pipeline import config as pconfig
    llm = llm or LLM()
    if isinstance(scenario, str):
        sc = next((s for s in pconfig.scenarios() if s["id"] == scenario), None)
        if sc is None:
            raise KeyError(f"시나리오 없음: {scenario}")
    else:
        sc = {k: v for k, v in scenario.items() if not k.startswith("_")}
        if sc.get("channels"):
            sc["channels"] = {f: int(v) for f, v in sc["channels"].items() if int(v) in (-1, 1)}
        else:
            from .state import default_channels
            sc["channels"] = default_channels(sc["instrument"], sc.get("direction", "escalate"))
    ev = scenario_event(sc, ctx)
    text = f"{sc.get('title', '')}. {' '.join(str(sc.get('description', '')).split())}"
    graph = build_graph(ctx, llm)
    init: ShockLensState = {"run_id": uuid.uuid4().hex[:8], "raw_input": text,
                            "sentences": [{"id": "s1", "text": text}], "event": ev.model_dump(),
                            "event_preset": True, "retries": {}, "trace": [],
                            "scenario": {"id": sc.get("id"), "title": sc.get("title")},
                            "regime_override": regime or sc.get("regime"),
                            "whatif_overrides": overrides or sc.get("overrides"),
                            "seed": ctx.thresholds["risk"]["seed"]}
    return graph.invoke(init, {"recursion_limit": 40})


def run_reverse(ctx: Context, loss: float | None = None, target_es: float | None = None,
                factors: list[str] | None = None, regime: str | None = None, horizon: int = 3,
                execute: bool = False, llm: LLM | None = None) -> dict:
    """역스트레스테스트. loss(3거래일 요인 손실, 0.05=5%) 또는 target_es(20거래일 ES 목표)를 준다.
    execute=True면 역산한 요인 충격을 가상 시나리오로 전체 파이프라인에 넣어 브리핑까지 만든다."""
    import numpy as np

    from engine import regime as regime_mod
    from engine import reverse, risk
    if ctx.frets is None:
        raise ValueError("요인 일별 데이터(factor_returns.parquet)가 없습니다. python -m pipeline.run_all 을 먼저 실행하세요.")
    th = ctx.thresholds
    vix = ctx.levels[ctx.vix_col]
    if regime not in ("crisis", "normal"):
        regime = regime_mod.decide({"severity": 3}, vix, th.get("regime", {"mode": "market"}),
                                   th["severity"]["crisis_regime_min"], None)["regime"]
    mask = vix >= vix.quantile(th["betas"]["crisis_vix_quantile"])
    beta = ctx.beta_crisis if regime == "crisis" else ctx.beta_normal
    es_base = None
    if target_es is not None:
        r = th["risk"]
        rng = np.random.default_rng(r["seed"])
        R = ctx.rets.to_numpy()
        days = (np.flatnonzero(mask.reindex(ctx.rets.index).fillna(False).to_numpy())
                if regime == "crisis" else np.arange(len(R)))
        paths = risk.block_bootstrap(R, days, r["horizon_days"], r["block_len"], r["n_paths"], rng)
        w = (ctx.weights.reindex(ctx.rets.columns) / ctx.weights.sum()).to_numpy()
        es_base = risk.var_es(risk.portfolio_return(paths, w), r["var_level"], r["es_level"])["ES"]
        loss = target_es - es_base
        if loss <= 0:
            raise ValueError(f"지금 ES({es_base:.2%})가 이미 목표({target_es:.2%}) 이상입니다.")
    if loss is None or loss <= 0:
        raise ValueError("목표 손실(loss) 또는 목표 ES(target_es)를 양수로 주세요.")
    out = reverse.run(beta, ctx.weights, ctx.frets, loss, horizon, factors,
                      mask if regime == "crisis" else None, ctx.f_event, ctx.events)
    out.update({"regime": regime, "es_base": es_base, "target_es": target_es,
                "labels": {f: ctx.labels.get(f, f) for f in out["factors"]}})
    res = None
    if execute:
        near = (out.get("nearest_events") or [{}])[0].get("id")
        base_ev = ctx.event_by_id(near) if near else None
        shock = {f: v for f, v in out["shock"].items() if abs(v) > 1e-9}
        sc = {"id": "sc_reverse", "title": f"(역산) {loss:.1%} 손실을 내는 가장 그럴듯한 요인 충격",
              "description": "역스트레스테스트로 구한 요인 충격을 그대로 넣은 가상 시나리오"
                             + (f" (가장 닮은 과거 사건: {near})" if near else ""),
              "instrument": (base_ev or {}).get("instrument", "tariff"),
              "initiator": (base_ev or {}).get("initiator", "US"), "target": (base_ev or {}).get("target", ["CN"]),
              "sectors": (base_ev or {}).get("sectors", []),
              "severity_components": {k: 3 for k in ("scope", "immediacy", "substitut", "escalation", "parties", "surprise")},
              "channels": {f: (1 if v > 0 else -1) for f, v in shock.items()}}
        res = run_scenario(sc, ctx, llm, overrides={f: {"abs": v} for f, v in shock.items()}, regime=regime)
        res["reverse"] = out
    return {"reverse": out, "run": res}


def run_library_event(event_id: str, ctx: Context, llm: LLM | None = None) -> dict:
    """라이브러리 사건을 '새 뉴스'로 가정해 실행 (뉴스 해석 생략, 자기 자신은 유사 사례에서 제외)."""
    llm = llm or LLM()
    lib = ctx.event_by_id(event_id)
    if lib is None:
        raise KeyError(event_id)
    keep = set(Event.model_fields)
    ev = Event(**{k: v for k, v in lib.items() if k in keep})
    ev.evidence = {f: ["s1"] for f in ("instrument", "initiator", "target", "sectors", "stage", "severity")}
    ev.severity_components = dict(lib.get("severity_components") or {})
    ev.severity_points = float(lib.get("severity_points", ev.severity_points))
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


WHATIF_SCHEMA = {"type": "object", "properties": {
    f: {"type": "object", "properties": {"mult": {"type": "number", "exclusiveMinimum": 0, "maximum": 10},
                                         "abs": {"type": "number", "minimum": -100, "maximum": 100}},
        "additionalProperties": False} for f in FACTOR_WORDS},
    "additionalProperties": False}
WHATIF_SCHEMA["properties"]["_regime"] = {"type": "string", "enum": ["crisis", "normal"],
                                          "description": "'위기 국면이라면' -> crisis, '평상시라면' -> normal"}
WHATIF_SYSTEM = """사용자의 가정 문장을 팩터 충격 덮어쓰기로 바꿉니다. 판단·추론 없이 문장에 있는 숫자만 옮깁니다.
팩터: oil=유가, risk_off=VIX·변동성·공포, dollar=달러, cny=위안화, ust10y=미 10년 금리, credit=신용·하이일드, tech_war=반도체·기술전쟁
- 'n배', 'nx' -> {"mult": n}
- 'VIX +8pt' -> {"risk_off": {"abs": 8}} (VIX는 포인트 그대로)
- '유가 +10%' -> {"oil": {"abs": 0.10}} (나머지 팩터의 %는 소수로)
- '금리 +50bp' -> {"ust10y": {"abs": 0.5}} (금리는 %p 단위: 50bp = 0.5)
- '위기 국면이라면' -> {"_regime": "crisis"}, '평상시 국면이라면' -> {"_regime": "normal"}
언급되지 않은 팩터는 넣지 않습니다. emit 도구로 제출합니다."""


def parse_whatif(text: str, llm: LLM | None = None) -> dict:
    """'유가 2배', 'VIX 1.5x', '반도체 충격 3배, 유가 +10%' -> {"oil": {"mult": 2.0}, ...}
    LLM이 있으면 tool use(스키마 강제)로, 없거나 실패하면 정규식 규칙으로 해석한다."""
    if llm is not None and llm.enabled:
        try:
            res = llm.complete_json(WHATIF_SYSTEM, text, max_tokens=300, schema=WHATIF_SCHEMA, name="emit")
            reg = (res or {}).get("_regime")
            res = {k: {kk: float(vv) for kk, vv in v.items() if kk in ("mult", "abs")}
                   for k, v in (res or {}).items() if k in FACTOR_WORDS and isinstance(v, dict)}
            res = {k: v for k, v in res.items() if v}
            if reg in ("crisis", "normal"):
                res["_regime"] = reg
            if res:
                return res
        except Exception:
            pass
    out = {}
    if re.search(r"위기 ?(국면|상황|장세)|위기라면|crisis", text, re.I):
        out["_regime"] = "crisis"
    elif re.search(r"평상시 ?(국면)?(라면|이라면)|평시라면|normal regime", text, re.I):
        out["_regime"] = "normal"
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
    keep = {k: prev[k] for k in ("run_id", "raw_input", "sentences", "event", "event_complete", "event_preset", "analogs",
                                 "analog_tier", "analog_table", "analog_excluded", "analog_notes", "size_class",
                                 "exposure", "exposure_reasons", "seed", "scenario", "regime_override",
                                 "news_meta", "injection_flags") if k in prev}
    init = {**keep, "whatif_overrides": overrides, "retries": {}, "trace": []}
    return graph.invoke(init, {"recursion_limit": 40})
