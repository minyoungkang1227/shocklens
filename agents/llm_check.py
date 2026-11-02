"""LLM 실제 호출 점검: python -m agents.llm_check [--full]

.env에 LLM_API_KEY를 넣은 뒤 실행한다. 단계별로 무엇이 되고 안 되는지 보여준다.
  1) 설정 확인: 공급자·모델·키 존재 여부 (키 값은 앞 4자리만 표시)
  2) 연결 확인: 아주 짧은 JSON 응답 요청
  3) Event Analyst: 샘플 뉴스 3건을 LLM으로 구조화 -> 가드레일이 보정한 내용, 남은 오류, 소요시간
     (확대 / 완화 / '유예 종료 후 재인상'처럼 헷갈리는 기사까지)
  4) --full: 데이터(data/model)가 있으면 뉴스 1건으로 전체 에이전트를 LLM 모드로 실행하고 Verifier 결과 확인
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from .event_analyst import analyze
from .llm import LLM

SAMPLES = [
    ("확대", "중국 상무부는 9일 희토류 관련 기술과 해외에서 생산된 희토류 함유 제품까지 수출 허가 대상으로 확대한다고 발표했다. "
             "업계는 미국과 유럽의 반도체·방산 공급망에 타격이 불가피하다고 본다."),
    ("완화", "미국과 중국은 12일 제네바에서 상호 관세를 115%p 인하하고 90일간 추가 협상을 하기로 합의했다고 공동성명을 발표했다."),
    ("헷갈림", "미중 관세 휴전 유예가 종료되면서 미국은 중국산 제품에 대한 관세를 다시 인상한다고 발표했다. "
               "중국은 즉각 보복 관세로 맞대응하겠다고 밝혔다."),
]
EXPECT = {"확대": ("resource_export_control", "escalate"), "완화": ("tariff_pause", "ease"), "헷갈림": ("tariff", "escalate")}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="전체 에이전트를 LLM 모드로 1회 실행")
    a = ap.parse_args()

    llm = LLM()
    key = os.environ.get("LLM_API_KEY", "")
    print("1) 설정")
    print(f"   공급자: {llm.provider} / 모델: {llm.model or '-'} / 키: {(key[:4] + '...') if key else '없음'}")
    if not llm.enabled:
        print("   -> LLM이 꺼져 있습니다. .env 파일에 LLM_API_KEY=... 를 넣고 다시 실행하세요.")
        print("      (anthropic 외에는 LLM_PROVIDER=openai 와 `pip install openai` 필요)")
        return

    print("\n2) 연결")
    t0 = time.time()
    try:
        r = llm.complete_json("JSON 객체 하나만 출력합니다.", '{"ok": true} 를 그대로 출력하세요.', max_tokens=50)
        print(f"   응답: {r}  ({time.time() - t0:.1f}초)")
    except Exception as e:
        print(f"   실패: {type(e).__name__}: {e}")
        print("   -> 키가 맞는지, 모델 이름(LLM_MODEL)이 계정에서 쓸 수 있는 것인지 확인하세요.")
        return

    print("\n3) Event Analyst (샘플 뉴스 3건)")
    ok = 0
    for label, text in SAMPLES:
        t0 = time.time()
        out = analyze(text, llm)
        ev = out["event"]
        dt = time.time() - t0
        exp = EXPECT[label]
        got = (ev.get("instrument"), ev.get("direction"))
        hit = got == exp
        ok += hit
        print(f"\n   [{label}] {'O' if hit else 'X'}  해석 방식: {out['mode']}  ({dt:.1f}초)")
        print(f"     수단 {ev.get('instrument')} / 방향 {ev.get('direction')} / {ev.get('initiator')}→{ev.get('target')} "
              f"/ 산업 {ev.get('sectors')} / 강도 {ev.get('severity')} / 채널 {ev.get('channels')}")
        print(f"     기대: 수단 {exp[0]} / 방향 {exp[1]}")
        if ev.get("notes"):
            print(f"     가드레일 보정: {ev['notes']}")
        if ev.get("uncertain_fields"):
            print(f"     근거 부족 항목: {ev['uncertain_fields']}")
        if out["event_errors"]:
            print(f"     남은 오류: {out['event_errors']}")
    print(f"\n   기대와 일치: {ok}/{len(SAMPLES)}   LLM 호출 수: {llm.calls}")

    if a.full:
        print("\n4) 전체 실행 (LLM 모드)")
        try:
            from .context import load_context
            from .hq import run_news
            ctx = load_context()
        except Exception as e:
            print(f"   데이터가 없습니다 ({e}). 먼저 python -m pipeline.run_all")
            return
        t0 = time.time()
        res = run_news(SAMPLES[0][1], ctx, llm)
        v = res.get("verification") or {}
        print(f"   상태 {res.get('status')} / 검증 {'통과' if v.get('passed') else v.get('issues')} / "
              f"{time.time() - t0:.1f}초 / LLM 호출 {llm.calls}회")
        reps = [t for t in res.get("trace", []) if t["node"] == "reporter"]
        print(f"   Reporter 작성 횟수: {len(reps)} (1이면 첫 작성에 검증 통과)")
        for t in res.get("trace", []):
            if t["node"] == "verifier" and "반려" in t["summary"]:
                print(f"     - 검증 반려: {t['summary']}")
        print(f"   Reporter 방식: {res.get('reporter_mode')}"
              + (f"  <- LLM 작성 실패로 템플릿 사용: {res.get('reporter_error')}" if res.get("reporter_error") else ""))
        if res.get("reporter_mode") == "llm":
            print("   (LLM이 쓴 문장 + 코드가 채운 숫자. 템플릿 문장과 달라야 정상)")
        print("\n" + res.get("final_brief", ""))


if __name__ == "__main__":
    main()
