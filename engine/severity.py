"""강도(severity) 세분화: 하나의 1~5 숫자 대신 6개 요소 점수 -> 0~100 연속 점수.

요소 (각 1~5)
  scope        범위: 얼마나 넓게 겨냥했나 (기업 몇 곳 1 ~ 국가 전체 교역 5)
  immediacy    즉시성: 검토·예고 1 ~ 즉시 시행 5
  substitut    대체 가능성: 대체 쉬움 1 ~ 한 나라가 공급 대부분을 쥠 5  (높을수록 충격 큼)
  escalation   확전 단계: 첫 조치 1 ~ 보복 연쇄 5
  parties      당사국 규모: 소국 간 1 ~ 미중·주요국 간 5
  surprise     예상 밖 정도: 이미 보도된 조치 1 ~ 완전한 기습 5   (시장은 알려진 조치에 거의 반응하지 않는다)

점수 = 100 x sum(w_i x (s_i - 1) / 4),  w는 thresholds.yaml의 severity.component_weights (합 1)
등급(1~5)은 점수를 5칸으로 나눠 호환성을 유지한다. 가중치는 사건이 쌓이면 실제 반응 크기에 회귀해 다시 정한다.
"""
from __future__ import annotations

COMPONENTS = ["scope", "immediacy", "substitut", "escalation", "parties", "surprise"]
DEFAULT_WEIGHTS = {"scope": 0.20, "immediacy": 0.15, "substitut": 0.15, "escalation": 0.15,
                   "parties": 0.15, "surprise": 0.20}
COMPONENT_KO = {"scope": "범위", "immediacy": "즉시성", "substitut": "대체 가능성", "escalation": "확전 단계",
                "parties": "당사국 규모", "surprise": "예상 밖 정도"}


def clamp(v, lo=1, hi=5) -> int:
    try:
        return int(min(max(round(float(v)), lo), hi))
    except (TypeError, ValueError):
        return 3


def score(components: dict[str, int] | None, weights: dict[str, float] | None = None,
          fallback_level: int | None = None) -> float:
    """요소 점수 -> 0~100. 요소가 없으면 fallback_level(1~5)을 같은 척도로 환산."""
    w = weights or DEFAULT_WEIGHTS
    if not components:
        lvl = clamp(fallback_level if fallback_level is not None else 3)
        return 100.0 * (lvl - 1) / 4
    total = sum(w.values())
    return 100.0 * sum(w[k] * (clamp(components.get(k, 3)) - 1) / 4 for k in w) / total


def level(points: float) -> int:
    """0~100 -> 1~5 등급 (0~20 -> 1, 20~40 -> 2, ... 80~100 -> 5)."""
    return int(min(max(points // 20 + 1, 1), 5))


def closeness(p_a: float, p_b: float, scale: float = 20.0) -> float:
    """강도 점수가 가까울수록 1에 가까운 가중치. scale 점 차이에서 0.5."""
    d = abs(p_a - p_b)
    return 1.0 / (1.0 + (d / scale) ** 2)


def describe(components: dict[str, int] | None) -> str:
    if not components:
        return "요소 점수 없음"
    return ", ".join(f"{COMPONENT_KO[k]} {clamp(components.get(k, 3))}" for k in COMPONENTS)
