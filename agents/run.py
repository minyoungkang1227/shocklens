"""에이전트 전체 실행 (선행: python -m pipeline.run_all)

  python -m agents.run --event cn_ree_oct_2025              # 라이브러리 사건으로 실행 (키 없이 가능)
  python -m agents.run --file news.txt                      # 뉴스 텍스트 파일
  python -m agents.run --text "중국 상무부는 ... 발표했다."
  python -m agents.run --event cn_ree_oct_2025 --whatif "유가 2배"
옵션: --trace 단계별 기록 출력, --json 결과 전체를 run_결과.json으로 저장

LLM 키(.env의 LLM_API_KEY)가 없으면 Event Analyst·Reporter는 규칙 기반으로 동작한다.
"""
from __future__ import annotations

import argparse
import json
import sys

from .context import load_context
from .hq import run_library_event, run_news, run_whatif
from .llm import LLM


def show(res: dict, trace: bool) -> None:
    print("\n" + "=" * 70)
    print(res.get("final_brief", "(브리핑 없음)"))
    print("=" * 70)
    v = res.get("verification") or {}
    print(f"상태: {res.get('status')} | 검증: {'통과' if v.get('passed') else v.get('issues')} "
          f"| calc_log_id: {res.get('calc_log_id')} | 사건 해석: {res.get('event_mode')}")
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
    ap.add_argument("--whatif")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    ctx = load_context()
    llm = LLM()
    print(f"[LLM] {llm.provider} {llm.model if llm.enabled else '(규칙 기반 대체 경로)'}")
    if a.event:
        res = run_library_event(a.event, ctx, llm)
    else:
        text = a.text if a.text else open(a.file, encoding="utf-8-sig").read()
        res = run_news(text, ctx, llm)
    show(res, a.trace)
    if a.whatif and res.get("status") == "ok":
        print(f"\n### What-if: {a.whatif}")
        res2 = run_whatif(res, a.whatif, ctx, llm)
        show(res2, a.trace)
        d0, d1 = res["risk_result"]["stressed"]["ES"], res2["risk_result"]["stressed"]["ES"]
        print(f"\nES97.5 변화: {d0:.2%} -> {d1:.2%} ({(d1 - d0) * 100:+.2f}%p)")
    if a.json:
        with open("run_결과.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2, default=str)
        print("run_결과.json 저장")


if __name__ == "__main__":
    main()
