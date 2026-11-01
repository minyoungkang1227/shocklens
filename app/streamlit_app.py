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
from agents.hq import run_library_event, run_news, run_whatif  # noqa: E402
from agents.llm import LLM  # noqa: E402

EXAMPLE_NEWS = (
    "중국 상무부는 9일 희토류 관련 기술과 해외에서 생산된 희토류 함유 제품까지 수출 허가 대상으로 확대한다고 발표했다.\n"
    "이번 조치는 미국의 반도체 수출통제에 대한 맞대응 성격이 강하다는 분석이 나온다.\n"
    "업계는 미국과 유럽의 반도체·방산·자동차 공급망에 타격이 불가피하다고 본다."
)
NODE_KO = {"event_analyst": "Event Analyst", "analog_finder": "Analog Finder", "exposure_analyst": "Exposure Analyst",
           "shock_mapper": "Shock Mapper", "risk_engine": "Risk Engine", "reporter": "Reporter",
           "verifier": "Verifier", "hq": "헤드쿼터"}


def model_ready() -> bool:
    return (config.MODEL_DIR / "beta_normal.csv").exists() and (config.PROCESSED_DIR / "returns.parquet").exists()


@st.cache_resource(show_spinner=False)
def get_context(_version: int):
    if not model_ready():
        from pipeline import run_all
        run_all.main()
    return load_context()


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
    st.info("처음 실행이라 시장 데이터를 내려받고 모델을 만듭니다. 1~2분 걸립니다.")
try:
    with st.spinner("데이터 준비 중..."):
        ctx = get_context(st.session_state.get("ctx_version", 0))
except Exception as e:
    st.error(f"데이터 준비에 실패했습니다: {e}")
    st.stop()

with st.sidebar:
    st.write("데이터 기준일:", ctx.data_version)
    st.write("사건 라이브러리:", f"{len(ctx.events)}건 (clean {sum(1 for e in ctx.events if e.get('clean'))}건)")
    st.caption("숫자는 전부 코드가 계산하고, LLM은 뉴스 해석과 문장 작성만 맡습니다. "
               "브리핑의 모든 숫자는 Verifier가 계산 기록과 대조합니다.")

# ------------------------------------------------------------------ 입력
st.title("ShockLens · 지경학 충격 스트레스테스트")
mode = st.radio("입력 방식", ["과거 사건으로 시연", "뉴스 붙여넣기"], horizontal=True)
if mode == "과거 사건으로 시연":
    labels = {f"{e['date']} · {e['title']}" + ("" if e.get("clean") else " (clean: false)"): e["id"]
              for e in sorted(ctx.events, key=lambda e: e["date"], reverse=True)}
    pick = st.selectbox("사건", list(labels))
    st.caption("선택한 사건을 새 뉴스로 가정하고, 그보다 앞선 다른 사건들만 유사 사례로 씁니다.")
    run = st.button("스트레스테스트 실행", type="primary")
    if run:
        with st.spinner("에이전트 실행 중..."):
            st.session_state["res"] = run_library_event(labels[pick], ctx, llm)
            st.session_state.pop("whatif", None)
else:
    text = st.text_area("뉴스 본문", EXAMPLE_NEWS, height=150)
    run = st.button("스트레스테스트 실행", type="primary")
    if run and text.strip():
        with st.spinner("에이전트 실행 중..."):
            st.session_state["res"] = run_news(text, ctx, llm)
            st.session_state.pop("whatif", None)

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
        c3.metric("③ 사건 충격 반영", f"{r['stressed']['ES']:.2%}", f"{d['event_effect_ES'] * 100:+.2f}%p",
                  delta_color="inverse")
        c4.metric("3거래일 시나리오 (중앙)", f"{r['scenario_loss']['central']:+.2%}",
                  f"범위 {r['scenario_loss']['mild']:+.2%} ~ {r['scenario_loss']['severe']:+.2%}", delta_color="off")
    left, right = st.columns([3, 2])
    with left:
        brief = res.get("final_brief", "").replace("\n### ", "\n##### ").replace("## ", "#### ")
        st.markdown(brief)
    with right:
        ev = res.get("event", {})
        with st.expander("사건 카드 (Event Analyst)", expanded=True):
            st.write(f"**수단** {ev.get('instrument')} · **방향** {ev.get('direction')} · **단계** {ev.get('stage')} · "
                     f"**강도** {ev.get('severity')}/5")
            st.write(f"**발동국** {ev.get('initiator')} → **대상국** {', '.join(ev.get('target', []))}")
            st.write(f"**산업** {', '.join(ev.get('sectors', [])) or '-'}")
            st.write(f"**영향 채널** {ev.get('channels')}")
            if ev.get("uncertain_fields"):
                st.caption(f"근거 부족·보정 항목: {', '.join(ev['uncertain_fields'])}")
            st.caption(f"해석 방식: {res.get('event_mode')}")
        with st.expander(f"유사 과거 사건 (Analog Finder · {res.get('analog_tier')})", expanded=True):
            if res.get("analogs"):
                st.dataframe(pd.DataFrame([{"사건": a["id"], "점수": a["score"], "근거": a["reason"]}
                                           for a in res["analogs"]]), hide_index=True, width="stretch")
            else:
                st.write("유사 사건 없음 → 강도 기반으로 충격을 합성했습니다.")
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
