"""'정책 사건 아님' 회귀 사례 (data/eval/not_policy_cases.yaml) — 규칙 경로에서 숫자가 나오지 않는지 본다 (네트워크 없음).

규칙 경로 기대값은 모두 hold(판단 보류)다. 브리핑까지 가서 ES를 내면(status ok) 시장 이벤트를 정책 사건으로 계산한 것이다.
LLM 경로 기대값(not_policy/hold)은 키가 있어야 해서 여기서 보지 않는다.
"""
from pathlib import Path

import pytest
import yaml

from agents.hq import run_news

from .test_agents import NoLLM, ctx  # noqa: F401  (ctx는 fixture)

CASES = yaml.safe_load((Path(__file__).parents[1] / "data/eval/not_policy_cases.yaml").read_text(encoding="utf-8"))["cases"]


def test_cases_file_is_well_formed():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids)) and len(ids) >= 11
    for c in CASES:
        assert c["text"].strip() and c["expect"]["rule"] == "hold" and c["expect"]["llm"] in ("not_policy", "hold")


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_rule_path_does_not_compute_market_events(case, ctx):
    st = run_news(case["text"], ctx, NoLLM(), force=True)
    assert st["status"] in ("partial", "skipped", "not_policy", "need_user"), f"{case['id']}: 규칙 경로가 시장 이벤트를 정책 사건으로 계산함 ({st.get('event')})"
    assert not st.get("calc_log_id")
