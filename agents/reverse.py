"""역스트레스테스트 CLI.

  python -m agents.reverse --loss 0.05                 # 3거래일에 요인 때문에 5% 잃는 가장 그럴듯한 충격
  python -m agents.reverse --es 0.12                   # 20거래일 ES를 12%로 만드는 충격
  python -m agents.reverse --loss 0.05 --factors tech_war,cny,risk_off   # 지경학 채널만 움직인다고 가정
  python -m agents.reverse --loss 0.05 --regime crisis --run             # 위기 국면, 그 충격으로 전체 실행
"""
from __future__ import annotations

import argparse
import sys

from .context import load_context
from .hq import run_reverse
from .llm import LLM


def fmt_prob(p: float | None) -> str:
    if p is None:
        return "모름"
    if p == 0:
        return "과거에 한 번도 없음"
    return f"{p:.2%}" if p >= 1e-4 else f"{p:.1e}"


def fmt_shock(f: str, v: float) -> str:
    if f == "risk_off":
        return f"{v:+.2f}pt"
    if f == "ust10y":
        return f"{v:+.2f}%p"
    return f"{v:+.2%}"


def show(rv: dict) -> None:
    print(f"\n목표: {rv['horizon_days']}거래일 요인 손실 {rv['loss']:.2%}"
          + (f" (지금 ES {rv['es_base']:.2%} → 목표 ES {rv['target_es']:.2%})" if rv.get("target_es") else "")
          + f" · 국면 {rv['regime']}")
    print(f"{'요인':24s} {'충격':>10s} {'손실 기여':>10s} {'노출(b)':>10s}")
    for f in rv["factors"]:
        print(f"{rv['labels'][f]:24s} {fmt_shock(f, rv['shock'][f]):>10s} {-rv['contribution'][f]:>10.2%} "
              f"{rv['exposure'][f]:>10.4f}")
    h = rv["historical"]
    print(f"\n그럴듯함: 마할라노비스 거리 {rv['distance']:.2f}σ · 정규분포 가정 확률 {fmt_prob(rv['normal_prob'])}"
          + (f" · 과거 실제 빈도 {fmt_prob(h['freq'])} ({h['n_windows']}개 창, 최악 {h['worst']:.2%} {h['worst_date']})"
             if h.get("freq") is not None else ""))
    for e in rv.get("nearest_events", []):
        print(f"  닮은 과거 사건: {e['id']} (유사도 {e['similarity']}) {e['title'][:40]}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--loss", type=float)
    g.add_argument("--es", type=float)
    ap.add_argument("--factors")
    ap.add_argument("--regime", choices=["crisis", "normal"])
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--run", action="store_true", help="역산한 충격으로 전체 파이프라인 실행")
    a = ap.parse_args()
    ctx = load_context()
    out = run_reverse(ctx, a.loss, a.es, a.factors.split(",") if a.factors else None, a.regime, a.horizon,
                      a.run, LLM() if a.run else None)
    show(out["reverse"])
    if out["run"]:
        from .reporter import display_brief
        print("\n" + display_brief(out["run"]["final_brief"])[0])


if __name__ == "__main__":
    main()
