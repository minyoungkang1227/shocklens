"""학습형 1차 분류기(agents/triage_model.py) 테스트. 실제 임베딩 모델 대신 결정론적 가짜 인코더, 가짜 기사로 본다(네트워크 없음).
여기서 통과해도 실제 기사에서 규칙보다 낫다는 뜻은 아니다 — 그 판정은 실제 데이터로 돌린 `triage_model eval`."""
import json
import re

import numpy as np
import pytest
import yaml

from agents import triage, triage_model as tm

# 가짜 인코더: '정책다움' 축과 '시황다움' 축. 규칙 사전에 없는 표현(반출 제한, 판매 금지)도 정책 축으로 본다 = 임베딩의 장점 흉내
POL = r"수출 ?통제|관세|제재|반출 ?제한|판매 ?금지|export control|tariff|sanction"
MKT = r"실적|주가|급등|급락|CEO|금리|earnings|shares"


def fake_encoder(texts):
    return np.array([[len(re.findall(POL, t, re.I)), len(re.findall(MKT, t, re.I)), 1.0] for t in texts], float)


fake_encoder.model_name = "fake-2axis"

POLICY_T = ["미국, 중국에 반도체 수출통제 확대", "중국, 핵심 소재 반출 제한 발표", "EU, 대중 관세 부과",
            "미 재무부, 러시아 은행 제재", "US expands export controls on chips", "일본, 장비 판매 금지 조치"]
MARKET_T = ["엔비디아 실적 발표 후 주가 급등", "코스피 외국인 매도에 급락", "CEO 교체설에 주가 급락",
            "금리 인상 우려에 기술주 약세", "Nvidia shares jump after earnings", "삼성전자 실적 개선 기대"]


def make_rows(months=("2025-01", "2025-02", "2025-03", "2025-04", "2025-05")):
    rows, i = [], 0
    for m in months:
        for t in POLICY_T:
            rows.append({"id": f"a{i}", "seen": f"{m}-10", "title": t, "text": t, "keep": True,
                         "triage": {"mode": "llm", "relevant": True, "importance": 3}}); i += 1
        for t in MARKET_T:
            rows.append({"id": f"a{i}", "seen": f"{m}-12", "title": t, "text": t, "keep": False,
                         "triage": {"mode": "llm", "relevant": False, "importance": 1}}); i += 1
    rows.append({"id": "r0", "seen": "2025-01-01", "title": "x", "text": "규칙 판정 기사", "keep": True,
                 "triage": {"mode": "rule"}})
    return rows


def test_load_rows_drops_rule_labels(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in make_rows()), encoding="utf-8")
    assert all(r["triage"]["mode"] == "llm" for r in tm.load_rows(p))
    assert len(tm.load_rows(p, llm_only=False)) == len(tm.load_rows(p)) + 1


def test_split_by_month_no_month_on_both_sides():
    rows = [r for r in make_rows() if r["triage"]["mode"] == "llm"]
    tr, te = tm.split_by_month(rows, 0.2)
    mtr, mte = {tm.month(r) for r in tr}, {tm.month(r) for r in te}
    assert not (mtr & mte) and max(mtr) < min(mte)                # 과거로 학습, 최근으로 시험
    assert len(te) >= 0.2 * len(rows)
    with pytest.raises(ValueError):
        tm.split_by_month(make_rows(months=("2025-01",))[:-1])


def test_train_save_load_and_classify(tmp_path):
    rows = [r for r in make_rows() if r["triage"]["mode"] == "llm"]
    meta = tm.train(rows, encoder=fake_encoder, out=tmp_path / "m.npz")
    assert meta["model"] == "fake-2axis" and 0.2 <= meta["threshold"] <= 0.8 and meta["n_test"] > 0
    m = tm.LearnedTriage(tmp_path / "m.npz", encoder=fake_encoder)
    new = "한국, 핵심 소재 반출 제한 시행"                       # 규칙 사전에 없는 표현
    assert not triage.rule_triage(new)["relevant"]                 # 규칙은 놓친다
    hit = m.classify(new)
    assert hit["relevant"] and hit["importance"] == 3              # 학습 분류기는 잡는다(가짜 인코더 기준)
    assert not m.classify("엔비디아 실적 발표, 주가 급등")["relevant"]


def test_classify_order_llm_then_model_then_rule(tmp_path, monkeypatch):
    rows = [r for r in make_rows() if r["triage"]["mode"] == "llm"]
    path = tmp_path / "m.npz"
    tm.train(rows, encoder=fake_encoder, out=path)
    tm._CACHE.clear()
    monkeypatch.setattr(tm, "_encoder", lambda model_name=None, encoder=None: fake_encoder)
    cfg = {"enabled": True, "path": str(path)}
    out = triage.classify("중국, 핵심 소재 반출 제한 발표", None, learned_cfg=cfg)
    assert out["mode"] == "model" and out["relevant"] and "학습 분류기" in out["reason"]

    tm._CACHE.clear()
    off = triage.classify("중국, 핵심 소재 반출 제한 발표", None, learned_cfg={"enabled": False})
    assert off["mode"] == "rule"                                   # 기본(꺼짐)은 지금과 같다

    tm._CACHE.clear()
    miss = triage.classify("미국 관세 부과", None, learned_cfg={"enabled": True, "path": str(tmp_path / "없음.npz")})
    assert miss["mode"] == "rule" and "학습 분류기 사용 불가" in miss["reason"]   # 조용히 바꾸지 않고 이유를 남긴다
    tm._CACHE.clear()


def test_evaluate_tables_gold_traps_and_lora(tmp_path):
    rows = [r for r in make_rows() if r["triage"]["mode"] == "llm"]
    tm.train(rows, encoder=fake_encoder, out=tmp_path / "m.npz")
    m = tm.LearnedTriage(tmp_path / "m.npz", encoder=fake_encoder)
    _, te = tm.split_by_month(rows)
    gold = {r["id"]: int(r["keep"]) for r in te[:6]}
    gold[te[0]["id"]] = 1 - gold[te[0]["id"]]                       # 사람이 Haiku 판정 하나를 뒤집음
    traps = [{"id": "np_x", "text": "CEO 교체설에 주가 급락"}, {"id": "np_y", "text": "금리 인상 확률 90%"}]
    lora = {r["id"]: float(r["keep"]) for r in te} | {"np_x": 0.1, "np_y": 0.2}
    res = tm.evaluate(rows, model=m, lora=lora, gold=gold, traps=traps)
    assert set(res["vs_haiku"]) == {"규칙", "기준선(임베딩+회귀)", "LoRA"}
    assert res["vs_haiku"]["LoRA"]["acc"] == 1.0
    assert res["vs_gold"]["Haiku"]["acc"] < 1.0                     # Haiku의 실수가 사람 라벨에서 보인다
    assert res["traps"]["기준선(임베딩+회귀)"]["통과(오답)"] == 0 and res["traps"]["LoRA"]["통과(오답)"] == 0
    md = tm.report_md(res)
    assert "얼마나 흉내 냈나" in md and "실제로 얼마나 맞나" in md and "함정 시험지" in md


def test_export_gold_balanced_and_never_overwrites(tmp_path):
    rows = [r for r in make_rows() if r["triage"]["mode"] == "llm"]
    p = tmp_path / "gold.yaml"
    n = tm.export_gold(rows, n=10, path=p)
    items = yaml.safe_load(p.read_text(encoding="utf-8"))["items"]
    assert n == 10 and sum(i["haiku_keep"] for i in items) == 5 and all(i["gold"] is None for i in items)
    assert tm.load_gold(p) == {}                                    # 빈 칸은 라벨로 치지 않는다
    with pytest.raises(SystemExit):
        tm.export_gold(rows, n=10, path=p)


def test_real_trap_file_is_readable():
    traps = tm.load_traps()
    assert len(traps) >= 11 and all(c["id"].startswith("np_") for c in traps)
