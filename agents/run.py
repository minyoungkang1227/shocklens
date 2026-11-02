"""에이전트 전체 실행 (선행: python -m pipeline.run_all)

  python -m agents.run --event cn_ree_oct_2025              # 라이브러리 사건으로 실행 (키 없이 가능)
  python -m agents.run --file news.txt                      # 뉴스 텍스트 파일
  python -m agents.run --text "중국 상무부는 ... 발표했다."
  python -m agents.run --text "https://기사주소"                  # URL이면 본문을 가져온다
  python -m agents.run --text "China expands rare earth export controls"   # 헤드라인이면 원문 기사를 검색
  python -m agents.run --file report.pdf                     # PDF·TXT (긴 문서는 핵심 문장만)
  python -m agents.run --text "..." --force                  # 중요도 1차 분류와 상관없이 실행
  python -m agents.run --event cn_ree_oct_2025 --whatif "유가 2배"
  python -m agents.run --event cn_ree_oct_2025 --ask "왜 반도체 손실이 제일 커?"   # 후속 질문 (여러 번 가능)
  python -m agents.run --list-scenarios                                            # 가상 시나리오 목록
  python -m agents.run --scenario sc_taiwan_blockade --whatif "평상시 국면이라면"     # 가상 시나리오
  python -m agents.run --event cn_ree_oct_2025 --chat                             # 후속 질문 대화 모드
옵션: --trace 단계별 기록 출력, --json 결과 전체를 run_결과.json으로 저장

LLM 키(.env의 LLM_API_KEY)가 없으면 Event Analyst·Reporter는 규칙 기반으로 동작한다.
"""
from __future__ import annotations

import argparse
import json
import sys

from .context import load_context
from .hq import run_library_event, run_news, run_scenario, run_whatif
from .llm import LLM


def show(res: dict, trace: bool) -> None:
    print("\n" + "=" * 70)
    print(res.get("final_brief", "(브리핑 없음)"))
    print("=" * 70)
    v = res.get("verification") or {}
    print(f"상태: {res.get('status')} | 검증: {'통과' if v.get('passed') else v.get('issues')} "
          f"| calc_log_id: {res.get('calc_log_id')} | 사건 해석: {res.get('event_mode')} | 보고서: {res.get('reporter_mode')}")
    if res.get("triage"):
        t = res["triage"]
        print(f"중요도 1차 분류: {t['importance']}/5 [{t['mode']}] {t['reason']}")
    for w in v.get("warnings", []):
        print(f"검증 경고: {w}")
    if trace:
        print("\n[단계별 기록]")
        for t in res.get("trace", []):
            print(f"  {t['node']:17s} {t['ms']:>6d}ms  {t['summary']}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")       # 윈도우 cmd 한글 깨짐 방지
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--event")
    g.add_argument("--file")
    g.add_argument("--text")
    g.add_argument("--scenario", help="가상 시나리오 id (config/scenarios.yaml)")
    g.add_argument("--list-scenarios", action="store_true")
    ap.add_argument("--whatif")
    ap.add_argument("--regime", choices=["crisis", "normal"], help="국면 강제 (시나리오·과거 사건)")
    ap.add_argument("--ask", action="append", help="브리핑 후 후속 질문 (여러 번 지정 가능)")
    ap.add_argument("--chat", action="store_true", help="브리핑 후 후속 질문 대화 (빈 줄 입력 시 종료)")
    ap.add_argument("--force", action="store_true", help="중요도 1차 분류 무시")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.list_scenarios:
        from pipeline import config
        for s in config.scenarios():
            print(f"  {s['id']:28s} {s['title']}")
        return
    ctx = load_context()
    llm = LLM()
    print(f"[LLM] {llm.provider} {llm.model if llm.enabled else '(규칙 기반 대체 경로)'}")
    if a.scenario:
        res = run_scenario(a.scenario, ctx, llm, regime=a.regime)
    elif a.event:
        res = run_library_event(a.event, ctx, llm)
    else:
        if a.text:
            text = a.text
        elif a.file.lower().endswith(".pdf"):
            from tools.docs import read_pdf
            text = read_pdf(a.file)
        else:
            text = open(a.file, encoding="utf-8-sig").read()
        res = run_news(text, ctx, llm, force=a.force)
    show(res, a.trace)
    if a.whatif and res.get("status") == "ok":
        print(f"\n### What-if: {a.whatif}")
        res2 = run_whatif(res, a.whatif, ctx, llm)
        show(res2, a.trace)
        d0, d1 = res["risk_result"]["stressed"]["ES"], res2["risk_result"]["stressed"]["ES"]
        print(f"\nES97.5 변화: {d0:.2%} -> {d1:.2%} ({(d1 - d0) * 100:+.2f}%p)")
    u = getattr(llm, "usage", None)
    if u and any(u.values()):
        print(f"[토큰] 입력 {u['input']} / 출력 {u['output']} / 캐시 읽기 {u['cache_read']} / 캐시 쓰기 {u['cache_write']}")
    if (a.ask or a.chat) and res.get("risk_result"):
        from .followup import answer
        hist: list[dict] = []

        def ask(q: str) -> None:
            r = answer(q, res, ctx, llm, hist[-6:])
            print(f"\nQ. {q}\nA. {r['answer']}\n   (조회: {[c['name'] for c in r['tool_calls']]} / {r['mode']})")
            hist.extend([{"role": "user", "content": q}, {"role": "assistant", "content": r["answer"]}])

        for q in a.ask or []:
            ask(q)
        while a.chat:
            q = input("\n질문 (빈 줄=종료)> ").strip()
            if not q:
                break
            ask(q)
    if a.json:
        with open("run_결과.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2, default=str)
        print("run_결과.json 저장")


if __name__ == "__main__":
    main()
