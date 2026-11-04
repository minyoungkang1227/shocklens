"""ShockLens 데모 화면 (Streamlit)

로컬 실행:  streamlit run app/streamlit_app.py
웹 배포:    Streamlit Community Cloud에서 이 저장소를 연결하고 Main file을 app/streamlit_app.py로 지정
LLM 키:     로컬은 .env, 웹은 Streamlit Secrets에 LLM_API_KEY = "..." 로 넣는다 (없으면 규칙 기반으로 동작)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import config  # noqa: E402

st.set_page_config(page_title="ShockLens", layout="wide")

# 웹 배포 시 Secrets의 키를 환경변수로 넘긴다
try:
    for k in ("LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL"):
        if k in st.secrets and not os.environ.get(k):
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass

from agents.context import load_context  # noqa: E402
from agents.hq import (run_library_event, run_news, run_reverse, run_scenario, run_whatif,  # noqa: E402
                       scenario_from_text)
from agents.llm import LLM  # noqa: E402

EXAMPLE_NEWS = (
    "중국 상무부는 9일 희토류 관련 기술과 해외에서 생산된 희토류 함유 제품까지 수출 허가 대상으로 확대한다고 발표했다.\n"
    "이번 조치는 미국의 반도체 수출통제에 대한 맞대응 성격이 강하다는 분석이 나온다.\n"
    "업계는 미국과 유럽의 반도체·방산·자동차 공급망에 타격이 불가피하다고 본다."
)
NODE_KO = {"event_analyst": "Event Analyst", "analog_finder": "Analog Finder", "exposure_analyst": "Exposure Analyst",
           "shock_mapper": "Shock Mapper", "risk_engine": "Risk Engine", "reporter": "Reporter",
           "verifier": "Verifier", "hq": "헤드쿼터", "news_input": "뉴스 입력 도구", "triage": "중요도 1차 분류"}


def model_ready() -> bool:
    return (config.MODEL_DIR / "beta_normal.csv").exists() and (config.PROCESSED_DIR / "returns.parquet").exists()


@st.cache_resource(show_spinner=False)
def get_context(_version: int):
    """데이터 준비 순서: 이미 있음 -> 저장소의 사본(data/snapshot) -> 직접 수집(pipeline.run_all)."""
    from pipeline import snapshot
    source = "로컬 데이터"
    if not model_ready():
        if snapshot.exists():
            snapshot.restore()
            source = f"저장소 사본 (기준일 {snapshot.meta().get('data_version')})"
        else:
            from pipeline import run_all
            run_all.main()
            source = "방금 수집"
    ctx = load_context()
    ctx._data_source = source
    return ctx


# ------------------------------------------------------------------ 사이드바
llm = LLM()
with st.sidebar:
    st.header("ShockLens")
    st.caption("지경학 뉴스 1건 → 국부펀드형 포트폴리오의 꼬리위험(ES97.5) 변화 브리핑")
    st.write("LLM:", f"`{llm.provider} {llm.model}`" if llm.enabled else "없음 (규칙 기반으로 동작)")
    if st.button("시장 데이터 새로 받기", help="ETF·FRED 데이터를 다시 내려받고 모델을 다시 만듭니다 (1~2분)"):
        from pipeline import run_all
        with st.spinner("데이터 수집과 모델 재계산 중..."):
            run_all.main()
        st.session_state["ctx_version"] = st.session_state.get("ctx_version", 0) + 1
        st.success("완료")

if not model_ready():
    from pipeline import snapshot as _snap
    if not _snap.exists():
        st.info("처음 실행이라 시장 데이터를 내려받고 모델을 만듭니다. 1~2분 걸립니다.")
try:
    with st.spinner("데이터 준비 중..."):
        ctx = get_context(st.session_state.get("ctx_version", 0))
except Exception as e:
    st.error(f"데이터 준비에 실패했습니다: {e}")
    st.stop()

with st.sidebar:
    st.write("데이터 기준일:", ctx.data_version, f"({getattr(ctx, '_data_source', '')})")
    st.write("사건 라이브러리:", f"{len(ctx.events)}건 (clean {sum(1 for e in ctx.events if e.get('clean'))}건)")
    st.caption("숫자는 전부 코드가 계산하고, LLM은 뉴스 해석과 문장 작성만 맡습니다. "
               "브리핑의 모든 숫자는 Verifier가 계산 기록과 대조합니다.")

# ------------------------------------------------------------------ 입력
st.title("ShockLens · 지경학 충격 스트레스테스트")
mode = st.radio("입력 방식", ["과거 사건으로 시연", "뉴스 붙여넣기", "가상 시나리오", "역스트레스테스트"], horizontal=True)
if mode == "역스트레스테스트":
    st.caption("목표 손실을 정하면, 그만큼 잃게 만드는 '가장 그럴듯한(가장 덜 드문)' 요인 충격을 거꾸로 계산합니다.")
    r1, r2, r3 = st.columns(3)
    kind = r1.radio("목표", ["3거래일 요인 손실", "20거래일 ES 목표"])
    val = r2.number_input("목표 (%)", 0.5, 50.0, 5.0 if kind.startswith("3") else 12.0, 0.5) / 100
    reg = r3.selectbox("국면", ["현재 VIX로 판단", "위기 국면", "평상시 국면"])
    fac_all = list(ctx.beta_normal.columns)
    facs = st.multiselect("움직일 수 있는 요인 (비우면 전체)", fac_all, default=fac_all,
                          format_func=lambda f: ctx.labels.get(f, f))
    regime_arg = {"위기 국면": "crisis", "평상시 국면": "normal"}.get(reg)
    if st.button("역산하기", type="primary"):
        try:
            with st.spinner("역산 중..."):
                out = run_reverse(ctx, val if kind.startswith("3") else None, None if kind.startswith("3") else val,
                                  facs or None, regime_arg, 3, False, llm)
                st.session_state["reverse"] = out["reverse"]
        except ValueError as e:
            st.error(str(e))
    rv = st.session_state.get("reverse")
    if rv:
        from agents.reverse import fmt_prob, fmt_shock
        st.subheader("가장 그럴듯한 요인 충격")
        st.dataframe(pd.DataFrame([{"요인": rv["labels"][f], "충격 (3거래일)": fmt_shock(f, rv["shock"][f]),
                                    "손실 기여": f"{-rv['contribution'][f]:.2%}"} for f in rv["factors"]]),
                     hide_index=True, width="stretch")
        h = rv["historical"]
        st.caption(f"목표 요인 손실 {rv['loss']:.2%}"
                   + (f" (지금 ES {rv['es_base']:.2%} → 목표 {rv['target_es']:.2%})" if rv.get("target_es") else "")
                   + f" · 국면 {rv['regime']} · 마할라노비스 거리 {rv['distance']:.2f}σ · 정규분포 가정 확률 {fmt_prob(rv['normal_prob'])}"
                   + (f" · 과거 실제 빈도 {fmt_prob(h['freq'])} (과거 최악 {h['worst']:.2%}, {h['worst_date']})" if h.get("freq") is not None else ""))
        if rv.get("nearest_events"):
            st.caption("가장 닮은 과거 사건: " + ", ".join(f"{e['id']}({e['similarity']})" for e in rv["nearest_events"]))
        if st.button("이 충격으로 전체 실행 (브리핑·구조 점검·후속 질문)"):
            with st.spinner("역산 충격으로 계산 중..."):
                out = run_reverse(ctx, rv["loss"], None, rv["factors"], rv["regime"], rv["horizon_days"], True, llm)
                st.session_state["res"] = out["run"]
                st.session_state.pop("whatif", None)
                st.session_state.pop("chat", None)
elif mode == "가상 시나리오":
    from agents.reporter import INST_KO, SECTOR_KO
    from agents.state import COUNTRIES, INSTRUMENTS, SECTORS
    how = st.radio("시나리오 고르는 방식", ["목록에서 고르기", "직접 작성"], horizontal=True)
    sc = None
    if how == "목록에서 고르기":
        scs = config.scenarios()
        pick = st.selectbox("시나리오 (실제 사건 아님)", [s["title"] for s in scs])
        sc = dict(next(s for s in scs if s["title"] == pick))
        st.caption(" ".join(str(sc.get("description", "")).split()))
    else:
        txt = st.text_area("가정을 문장으로 적어 주세요 (누가·누구에게·어떤 조치를)",
                           "중국이 대만 해협을 봉쇄하고, 미국과 일본이 중국 주요 은행에 금융제재로 맞선다면?", height=90)
        if st.button("가정 해석하기"):
            try:
                with st.spinner("가정을 사건 구조로 바꾸는 중..."):
                    st.session_state["custom_sc"] = scenario_from_text(txt, ctx, llm)
            except ValueError as e:
                st.error(str(e))
        base = st.session_state.get("custom_sc")
        if base:
            st.caption(f"해석 방식: {base.get('_mode')} · 아래 값을 확인하고 고친 뒤 실행하세요"
                       + (f" · 근거 부족: {', '.join(base['_uncertain'])}" if base.get("_uncertain") else ""))
            sc = dict(base)
            a1, a2 = st.columns(2)
            sc["title"] = a1.text_input("시나리오 이름", base["title"])
            sc["instrument"] = a2.selectbox("수단", INSTRUMENTS, index=INSTRUMENTS.index(base["instrument"]),
                                            format_func=lambda x: INST_KO.get(x, x))
            b1, b2, b3 = st.columns(3)
            sc["direction"] = b1.selectbox("방향", ["escalate", "ease"], index=0 if base.get("direction") != "ease" else 1,
                                           format_func=lambda x: "확대" if x == "escalate" else "완화")
            sc["initiator"] = b2.selectbox("발동국", COUNTRIES, index=COUNTRIES.index(base["initiator"]))
            sc["target"] = b3.multiselect("대상국", COUNTRIES, default=[c for c in base.get("target", []) if c in COUNTRIES])
            sc["sectors"] = st.multiselect("산업", SECTORS, default=[x for x in base.get("sectors", []) if x in SECTORS],
                                           format_func=lambda x: SECTOR_KO.get(x, x))
            sc["sovereign_asset_risk"] = st.checkbox("중앙은행·국부펀드 자산이 직접 표적", bool(base.get("sovereign_asset_risk")))
            if sc["instrument"] != base["instrument"] or sc.get("direction") != base.get("direction"):
                sc["channels"] = {}                        # 수단·방향을 바꾸면 영향 채널은 기본값으로 다시
            st.caption(f"영향 채널: {sc.get('channels') or '수단별 기본값'}")
    if sc is None:
        st.info("가정을 적고 '가정 해석하기'를 누르면 사건 구조가 채워집니다. 여러 조치가 섞이면 가장 큰 조치 하나로 잡히니, 나머지는 실행 후 What-if로 더하세요.")
    else:
        c1, c2 = st.columns(2)
        reg_ko = c1.selectbox("국면 가정", ["시나리오 기본값", "위기 국면", "평상시 국면", "현재 VIX로 판단"])
        with c2.expander("강도 6요소 조정"):
            comps = dict(sc.get("severity_components") or {})
            for k, lab in [("scope", "범위"), ("immediacy", "즉시성"), ("substitut", "대체 어려움"),
                           ("escalation", "확전"), ("parties", "당사국 규모"), ("surprise", "의외성")]:
                comps[k] = st.slider(lab, 1, 5, int(comps.get(k, 3)), key=f"sc_{sc['id']}_{k}_{how}")
            sc["severity_components"] = comps
        if reg_ko != "시나리오 기본값":
            sc["regime"] = {"위기 국면": "crisis", "평상시 국면": "normal", "현재 VIX로 판단": None}[reg_ko]
        if st.button("시나리오 실행", type="primary"):
            with st.spinner("가상 시나리오 계산 중..."):
                st.session_state["res"] = run_scenario(sc, ctx, llm)
                st.session_state.pop("whatif", None)
                st.session_state.pop("chat", None)
elif mode == "과거 사건으로 시연":
    labels = {f"{e['date']} · {e['title']}" + ("" if e.get("clean") else " (clean: false)"): e["id"]
              for e in sorted(ctx.events, key=lambda e: e["date"], reverse=True)}
    pick = st.selectbox("사건", list(labels))
    st.caption("선택한 사건을 새 뉴스로 가정하고, 그보다 앞선 다른 사건들만 유사 사례로 씁니다.")
    run = st.button("스트레스테스트 실행", type="primary")
    if run:
        with st.spinner("에이전트 실행 중..."):
            st.session_state["res"] = run_library_event(labels[pick], ctx, llm)
            st.session_state.pop("whatif", None)
            st.session_state.pop("chat", None)
else:
    text = st.text_area("뉴스 본문 · 헤드라인 · 기사 URL (한·영·중)", EXAMPLE_NEWS, height=150,
                        help="URL이면 본문을 가져오고, 헤드라인 한 줄이면 원문 기사를 검색해 붙입니다.")
    up = st.file_uploader("또는 보고서·기사 PDF/TXT 올리기", type=["pdf", "txt"])
    force = st.checkbox("중요도 1차 분류와 상관없이 강제 실행", value=False)
    run = st.button("스트레스테스트 실행", type="primary")
    if run:
        if up is not None:
            from tools.docs import read_pdf
            data = up.getvalue()
            text = read_pdf(data) if up.name.lower().endswith(".pdf") else data.decode("utf-8", "ignore")
        if text.strip():
            with st.spinner("에이전트 실행 중..."):
                st.session_state["res"] = run_news(text, ctx, llm, force=force)
                st.session_state.pop("whatif", None)
                st.session_state.pop("chat", None)

res = st.session_state.get("res")
if not res:
    st.stop()


# ------------------------------------------------------------------ 결과
def show_result(res: dict, title: str) -> None:
    status = res.get("status")
    if status == "ok":
        st.success(f"{title} · 검증 통과 · 계산 기록 {res.get('calc_log_id')}")
    elif status == "partial":
        st.warning(f"{title} · 판단 보류 (일부 결과만 제공)")
    elif status == "skipped":
        st.info(res.get("final_brief", "중요도 미달로 분석을 생략했습니다"))
        return
    else:
        st.info(res.get("needs_user", "추가 정보가 필요합니다"))
        return
    r = res.get("risk_result")
    if r:
        d = r["decomposition"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("① 평상시 ES97.5", f"{r['baseline_normal']['ES']:.2%}")
        c2.metric("② 국면 반영", f"{r['baseline']['ES']:.2%}", f"{d['regime_effect_ES'] * 100:+.2f}%p",
                  delta_color="inverse")
        c3.metric("③ 사건 충격 반영 (중앙 ~ 최악)", f"{r['stressed']['ES']:.2%} ~ {r['stressed_severe']['ES']:.2%}",
                  f"{d['event_effect_ES'] * 100:+.2f}%p ~ {d['event_effect_ES_severe'] * 100:+.2f}%p", delta_color="inverse")
        band = r.get("scenario_band") or {}
        rng_txt = (f"예측구간 {band['interval']['lo']:+.2%} ~ {band['interval']['hi']:+.2%}" if band.get("interval")
                   else f"범위 {r['scenario_loss']['mild']:+.2%} ~ {r['scenario_loss']['severe']:+.2%}")
        c4.metric("3거래일 시나리오 (중앙)", f"{r['scenario_loss']['central']:+.2%}", rng_txt, delta_color="off")
        en = r.get("ensemble") or {}
        if en:
            e1, e2, e3 = st.columns(3)
            e1.metric("앙상블 ES97.5 (대표값)", f"{en['ES']:.2%}", f"최악 충격 {en['ES_severe']:.2%}", delta_color="off")
            e2.metric("모형 위험 (최소 ~ 최대)", f"{en['range'][0]:.2%} ~ {en['range'][1]:.2%}")
            e3.metric("보수적 기준 (모형 최대)", f"{en['conservative']:.2%}", f"최악 충격 {en['conservative_severe']:.2%}",
                      delta_color="off")
            st.caption("모형별 ES: " + " · ".join(f"{k} {v['ES']:.2%}" for k, v in en["models"].items())
                       + " (동일 가중, 경로 섞기). 아래 ①②③ 분해는 기준 모형(블록 부트스트랩) 결과입니다.")
        ri = r.get("regime_info") or {}
        if ri.get("reason"):
            st.caption(f"국면 판단({'위기' if r['regime'] == 'crisis' else '평상시'}): {ri['reason']}")
    left, right = st.columns([3, 2])
    with left:
        from agents.reporter import display_brief
        raw = res.get("final_brief", "")
        show_tags = st.toggle("근거 태그 보기", False, key=f"tags_{title}")
        clean, used = display_brief(raw, res.get("sources"))
        brief = (raw if show_tags else clean).replace("\n### ", "\n##### ").replace("## ", "#### ")
        st.markdown(brief.replace("~", "\\~"))
        if used or res.get("calc_log_id"):
            with st.expander("근거 출처"):
                for sid, desc in used:
                    st.markdown(f"- **{sid}** {desc[:160]}")
                if res.get("calc_log_id"):
                    st.markdown(f"- **계산 결과** 계산 기록 `{res['calc_log_id']}` (모든 수치의 출처)")
    with right:
        ev = res.get("event", {})
        with st.expander("사건 카드 (Event Analyst)", expanded=True):
            st.write(f"**수단** {ev.get('instrument')} · **방향** {ev.get('direction')} · **단계** {ev.get('stage')} · "
                     f"**강도** {ev.get('severity')}/5 (점수 {ev.get('severity_points', 0):.0f}/100)")
            if ev.get("severity_components"):
                from engine.severity import describe
                st.caption("강도 요소: " + describe(ev["severity_components"]))
            st.write(f"**발동국** {ev.get('initiator')} → **대상국** {', '.join(ev.get('target', []))}")
            st.write(f"**산업** {', '.join(ev.get('sectors', [])) or '-'}")
            st.write(f"**영향 채널** {ev.get('channels')}")
            if ev.get("severity_samples", 1) > 1:
                sp = ev.get("severity_spread") or {}
                st.caption(f"강도 {ev['severity_samples']}회 채점 중앙값 · 요소별 흩어짐 최대 {max(sp.values(), default=0)}"
                           + (" (흩어짐 큼 → 강도 불확실)" if max(sp.values(), default=0) >= 2 else ""))
            if ev.get("uncertain_fields"):
                st.caption(f"근거 부족·보정 항목: {', '.join(ev['uncertain_fields'])}")
            for f in res.get("injection_flags") or ev.get("injection_flags") or []:
                st.caption(f"🛡 지시문 의심 문장 [{f['id']}] 근거에서 제외: {f['text'][:40]}…")
            st.caption(f"해석 방식: {res.get('event_mode')}")
            nm, tr = res.get("news_meta") or {}, res.get("triage") or {}
            if nm:
                st.caption(f"입력: {nm.get('kind')} · 언어 {nm.get('lang')} · 도구 {nm.get('tool_calls')}회"
                           + (f" · 출처 {', '.join(nm.get('urls', [])[:2])}" if nm.get("urls") else ""))
            if tr:
                st.caption(f"중요도 1차 분류: {tr.get('importance')}/5 [{tr.get('mode')} {tr.get('model') or ''}] {tr.get('reason')}")
        with st.expander(f"유사 과거 사건 (Analog Finder · {res.get('analog_tier')})", expanded=True):
            if res.get("analogs"):
                st.dataframe(pd.DataFrame([{"사건": a["id"], "등급": {"small": "소", "medium": "중", "large": "대"}.get(a.get("size_class"), ""),
                                            "구조": a.get("struct"), "문장": a.get("text"), "강도점수": a.get("severity_points"),
                                            "강도근접": a.get("closeness"), "가중치": a.get("weight"),
                                            "근거": a["reason"]} for a in res["analogs"]]),
                             hide_index=True, width="stretch")
                st.caption(f"충격 산정: {res.get('shock_method')} · 이번 사건 등급: {res.get('size_class')}")
            for n in res.get("analog_notes", []):
                st.caption(f"⚠ {n}")
            if not res.get("analogs"):
                st.write("유사 사건 없음 → 강도 기반으로 충격을 합성했습니다.")
        stc = (res.get("risk_result") or {}).get("structure") or {}
        if stc:
            with st.expander("구조 점검 (분절화·집중도·국부펀드 고유 위험)"):
                st.dataframe(pd.DataFrame({"포트폴리오 비중": stc.get("bloc_shares", {}),
                                           "꼬리손실 기여": stc.get("tail_by_bloc", {})}).style.format("{:.1%}"),
                             width="stretch")
                st.caption(f"국가 분산: 유효 {stc.get('effective_countries')}개국 · 상대 블록 관할 자산 "
                           f"{stc.get('adverse_share', 0):.1%} · 대만 관할 {stc.get('flashpoint_share', 0):.1%} (관할국 근사)")
        att = res.get("attention") or {}
        if att:
            with st.expander("주의도 (사건 직전 보도량)"):
                if att.get("percentile") is not None:
                    st.write(f"직전 7일 보도량 백분위 {att['percentile']:.0f} · 평소 대비 {att.get('ratio')}배 · 검색어 `{att['query']}`")
                    if att.get("note"):
                        st.caption(att["note"])
                else:
                    st.caption(f"측정 생략: {att.get('error')}")
        if ev.get("instrument") in ("industrial_subsidy", "investment_restriction"):
            from agents.law_effect import reference_events, summary_rows
            with st.expander("장기 효과 (법령·산업정책, 참고)"):
                rows = sum((summary_rows(e, ctx) for e in reference_events(ev["instrument"], ctx)
                            if e["id"] != ev.get("id")), [])
                if rows:
                    st.dataframe(pd.DataFrame(rows).style.format({"excess_60d": "{:+.2%}", "pct_60d": "{:.0f}"}),
                                 width="stretch")
                    st.caption("과거 같은 유형 법령 이후 60거래일 미국 주식 대비 초과수익 (pct: 무작위 창 대비 백분위). 인과가 아닌 사후 기록.")
                else:
                    st.caption("비교할 과거 같은 유형 법령이 없습니다.")
        with st.expander("직접 노출 (Exposure Analyst)"):
            for b, why in (res.get("exposure_reasons") or {}).items():
                st.write(f"- {why}")
        with st.expander("팩터 충격 (Shock Mapper)"):
            fs = pd.DataFrame.from_dict(res.get("factor_shock", {}), orient="index")
            if not fs.empty:
                fs.index = [ctx.labels.get(i, i) for i in fs.index]
                st.dataframe(fs.style.format("{:+.4f}"), width="stretch")
            for w in res.get("shock_warnings", []):
                st.caption(f"⚠ {w}")
        with st.expander("에이전트 실행 기록 (헤드쿼터)"):
            st.dataframe(pd.DataFrame([{"단계": NODE_KO.get(t["node"], t["node"]), "ms": t["ms"],
                                        "요약": t["summary"]} for t in res.get("trace", [])]),
                         hide_index=True, width="stretch")
        v = res.get("verification") or {}
        with st.expander(f"검증 (Verifier) · {'통과' if v.get('passed') else '반려 ' + str(len(v.get('issues', []))) + '건'}"):
            for i in v.get("issues", []):
                st.write(f"- [{i['kind']}] {i['detail']} → {i['route']}")
            if v.get("passed"):
                st.write("모든 숫자가 계산 기록과 일치하고, 모든 문장에 근거 태그가 있습니다.")
            for w in v.get("warnings", []):
                st.caption(f"⚠ {w}")
            if v.get("checker_model"):
                st.caption(f"사실 검사 모델: {v['checker_model']} (작성 모델과 분리)")
            cs = res.get("citation_stats")
            if cs:
                st.caption(f"인용 기능: 인용 구간 {cs['cited_spans']}개, 자동 보강 태그 {cs['added_tags']}개, 태그-인용 불일치 {len(cs['mismatch'])}건")


show_result(res, "스트레스테스트")

# ------------------------------------------------------------------ What-if
if res.get("status") == "ok":
    st.divider()
    st.subheader("What-if")
    c1, c2 = st.columns([4, 1])
    wtext = c1.text_input("가정 (예: 반도체 충격 3배, 유가 +10%, VIX +8pt)", "반도체 충격 3배")
    if c2.button("다시 계산", width="stretch"):
        try:
            with st.spinner("Shock Mapper부터 다시 계산 중..."):
                st.session_state["whatif"] = (wtext, run_whatif(res, wtext, ctx, llm))
        except ValueError as e:
            st.error(str(e))
    if st.session_state.get("whatif"):
        wtext, w = st.session_state["whatif"]
        e0, e1 = res["risk_result"]["stressed"]["ES"], w["risk_result"]["stressed"]["ES"]
        st.metric(f"ES97.5 ({wtext})", f"{e1:.2%}", f"{(e1 - e0) * 100:+.2f}%p vs 기본 시나리오", delta_color="inverse")
        with st.expander("What-if 브리핑 전체"):
            show_result(w, f"What-if: {wtext}")

# ------------------------------------------------------------------ 후속 질문
if res.get("risk_result"):
    from agents.followup import answer as follow_answer

    st.divider()
    st.subheader("후속 질문")
    st.caption("브리핑에 대해 물어보세요. 답의 숫자는 계산 기록(calc_log)을 조회해 가져오고, 기록에 없는 숫자는 걸러냅니다.")
    chat = st.session_state.setdefault("chat", [])
    for m in chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("tools"):
                st.caption("조회한 기록: " + ", ".join(m["tools"]) + f" · {m.get('mode')}")
    q = st.chat_input("예: 왜 반도체 손실이 제일 커? / 유사 사건은 뭐였어? / 국면 효과가 뭐야?")
    if q:
        hist = [{"role": m["role"], "content": m["content"]} for m in chat[-6:]]
        with st.spinner("계산 기록 조회 중..."):
            a = follow_answer(q, res, ctx, llm, hist)
        chat.append({"role": "user", "content": q})
        chat.append({"role": "assistant", "content": a["answer"], "mode": a["mode"],
                     "tools": [c["name"] for c in a["tool_calls"]]})
        st.rerun()
