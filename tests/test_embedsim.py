"""engine/embedsim.py와 Analog Finder 연결 검사. 실제 임베딩 모델 대신 결정론적 가짜 인코더를 쓴다
(네트워크 없음). 여기서 통과해도 실제 모델이 유사 사건을 더 잘 찾는다는 뜻은 아니다 — 그 판정은 analog_eval."""
import re

import numpy as np
import pytest

from agents import analog_finder
from engine import embedsim
from engine.textsim import CONCEPTS
from tests.test_agents import REE_NEWS, ctx  # noqa: F401  (pytest 픽스처 재사용)


def _target(ctx):  # noqa: F811
    """라이브러리의 최근 수출통제 사건을 '새 사건'으로 쓴다(과거 사건만 후보가 된다)."""
    ec = [e for e in ctx.events if e.get("instrument") in ("tech_export_control", "resource_export_control")]
    return {**sorted(ec, key=lambda e: e["date"])[-1], "date": "2026-11-01"}

# 가짜 인코더: 개념 사전 + 사전에 없는 동의어 몇 개를 같은 축으로 본다. '뜻이 같으면 단어가 달라도 가깝다'를 흉내.
EXTRA = {"c_export_control": r"반출 ?제한|판매 ?금지", "c_ree": r"희귀 ?금속"}
AXES = list(CONCEPTS)


def fake_encoder(texts):
    out = np.zeros((len(texts), len(AXES)))
    for i, t in enumerate(texts):
        for j, name in enumerate(AXES):
            pat = CONCEPTS[name][2] + ("|" + EXTRA[name] if name in EXTRA else "")
            out[i, j] = len(re.findall(pat, t, re.I))
    out[:, 0] += 1e-3          # 0 벡터 방지
    return out


fake_encoder.model_name = "fake-concepts"


def test_blend_weights_and_bounds():
    lex, emb = {"a": 0.2, "b": 0.8}, {"a": 1.0, "c": 0.5}
    assert embedsim.blend(lex, emb, 0.0) == lex                       # 끄면 단어 유사도 그대로
    one = embedsim.blend(lex, emb, 1.0)
    assert one == {"a": 1.0, "b": 0.0, "c": 0.5}
    half = embedsim.blend(lex, emb, 0.5)
    assert half["a"] == pytest.approx(0.6) and half["b"] == pytest.approx(0.4) and half["c"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        embedsim.blend(lex, emb, 1.5)


def test_index_cosine_is_clipped_and_self_similar():
    evs = [{"id": "x", "title": "중국 희토류 수출통제", "description": ""},
           {"id": "y", "title": "미국 관세 인상", "description": ""}]
    idx = embedsim.EmbedIndex(evs, fake_encoder)
    s = idx.query("중국 희토류 수출통제")
    assert s["x"] == pytest.approx(1.0) and 0.0 <= s["y"] < s["x"]
    assert all(0.0 <= v <= 1.0 for v in s.values())


def test_weight_zero_leaves_finder_unchanged(ctx):  # noqa: F811
    ev = _target(ctx)
    base = analog_finder.find(ev, ctx, REE_NEWS)
    ctx._embed_encoder = fake_encoder
    ctx.thresholds["analog"]["embed_weight"] = 0.0
    again = analog_finder.find(ev, ctx, REE_NEWS)
    assert [p["id"] for p in base["analogs"]] == [p["id"] for p in again["analogs"]]
    assert [p["weight"] for p in base["analogs"]] == [p["weight"] for p in again["analogs"]]
    assert not any("임베딩" in n for n in again["analog_notes"])


def test_embedding_layer_is_used_and_noted(ctx):  # noqa: F811
    ev = _target(ctx)
    text = "중국 정부가 핵심 소재 반출 제한 조치를 발표했다."      # 사전 단어 '수출통제'가 없다
    ctx._embed_encoder = fake_encoder
    ctx.thresholds["analog"]["embed_weight"] = 0.5
    res = analog_finder.find(ev, ctx, text)
    assert any("임베딩(fake-concepts)" in n for n in res["analog_notes"])
    lex_only = analog_finder._text_index(ctx).query(text)[0]
    emb = analog_finder._embed_index(ctx, ctx.thresholds["analog"]["embed_model"])[0].query(text)
    # 사전 밖 표현이라 단어 유사도로는 못 잡는 수출통제 사건을 임베딩 층은 잡는다(가짜 인코더 기준)
    ec = [e["id"] for e in ctx.events if e.get("instrument") in ("tech_export_control", "resource_export_control")]
    assert ec and max(emb[i] for i in ec) > max(lex_only[i] for i in ec)


def test_unavailable_backend_falls_back_with_note(ctx, monkeypatch):  # noqa: F811
    def boom(model):
        raise embedsim.EmbeddingUnavailable("테스트: 모델 없음")
    monkeypatch.setattr(embedsim, "sentence_transformers_encoder", boom)
    ev = _target(ctx)
    base = analog_finder.find(ev, ctx, REE_NEWS)
    ctx.thresholds["analog"]["embed_weight"] = 0.3
    res = analog_finder.find(ev, ctx, REE_NEWS)
    assert any("쓸 수 없어 단어 유사도만" in n for n in res["analog_notes"])
    assert [p["id"] for p in res["analogs"]] == [p["id"] for p in base["analogs"]]   # 숫자는 단어 유사도 그대로


def test_eval_restores_setting(ctx):  # noqa: F811
    from agents import analog_eval
    ctx._embed_encoder = fake_encoder
    ctx.thresholds["analog"]["embed_weight"] = 0.0
    df = analog_eval.run(ctx, None, {}, backtest=False, embed=0.4)
    assert set(df.method) == {"code", "embed"}
    assert ctx.thresholds["analog"]["embed_weight"] == 0.0
