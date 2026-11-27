"""실전 성적표 테스트: 예측 기록 -> 반응 창이 끝나면 실제 수익률로 채점 (네트워크 없음)."""
import datetime as dt

import numpy as np

from agents import track_record as tr
from agents.hq import run_news
from pipeline import config

from .test_agents import REE_NEWS, NoLLM, ctx  # noqa: F401  (ctx는 fixture)

ET = tr.ET


def test_window_start_and_pure():
    assert tr.window_start(dt.date(2026, 11, 7)) == dt.date(2026, 11, 9)          # 토요일 뉴스 -> 월요일
    assert tr.window_start(dt.date(2026, 11, 5)) == dt.date(2026, 11, 5)
    start = dt.date(2026, 11, 5)
    assert tr.is_pure(start, dt.datetime(2026, 11, 4, 20, 0, tzinfo=ET))
    assert tr.is_pure(start, dt.datetime(2026, 11, 5, 9, 0, tzinfo=ET))
    assert not tr.is_pure(start, dt.datetime(2026, 11, 5, 17, 40, tzinfo=ET))      # 아침 7:40 KST = 전날 장 마감 후


def test_record_score_roundtrip(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    st = run_news(REE_NEWS, ctx, NoLLM())
    now = dt.datetime(2026, 10, 20, 8, 0, tzinfo=ET)                                # 장 열리기 전 기록 -> 순수 예측
    rec = tr.make_record(st, "test", dt.date(2026, 10, 20), now)
    assert rec and rec["pure"] and rec["window_start"] == "2026-10-20"
    assert rec["mild"] is not None and rec["band_lo"] <= rec["band_hi"]
    assert set(rec["basket_central"]) == set(ctx.rets.columns)
    assert tr.append(rec) and not tr.append(rec)                                   # 같은 예측은 한 번만

    late = tr.make_record({**st, "calc_log_id": "calc_late"}, "test", dt.date(2026, 10, 29), now)
    tr.append(late)                                                                # 10/29~ 3거래일 중 하루만 데이터에 있음
    df = tr.score_all(ctx)
    assert len(df) == 1 and df.iloc[0]["id"] == rec["id"]                          # 창이 다 안 들어온 예측은 대기
    r = df.iloc[0]
    w = (ctx.weights / ctx.weights.sum()).reindex(ctx.rets.columns).to_numpy()
    win = ctx.rets.loc["2026-10-20":].iloc[:3].sum().to_numpy()
    assert np.isclose(r["actual"], np.expm1(win) @ w)
    assert r["in_band"] == (r["band_lo"] <= r["actual"] <= r["band_hi"])
    assert (tmp_path / "forecasts" / "scored.csv").exists()
    md = tr.report_md(df, 1)
    assert "순수 예측" in md and "대기 1건" in md


def test_skipped_runs_are_not_recorded():
    assert tr.make_record({"status": "skipped", "risk_result": {}}, "x") is None
    assert tr.make_record({"status": "partial", "risk_result": {"scenario_loss": {}}}, "x") is None


def test_excluded_records_stay_in_file_but_leave_the_score(ctx, tmp_path, monkeypatch):
    """오분류 기록은 forecasts.jsonl에서 지우지 않고 excluded.yaml로 집계에서만 뺀다(2026-11-27)."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    st = run_news(REE_NEWS, ctx, NoLLM())
    now = dt.datetime(2026, 10, 20, 8, 0, tzinfo=ET)
    rec = tr.make_record(st, "test", dt.date(2026, 10, 20), now)
    bad = tr.make_record({**st, "calc_log_id": "calc_bad"}, "test", dt.date(2026, 10, 20), now)
    tr.append(rec), tr.append(bad)
    assert len(tr.score_all(ctx)) == 2
    (tmp_path / "forecasts" / "excluded.yaml").write_text(
        "excluded:\n  - id: calc_bad\n    reason: 오분류\n", encoding="utf-8")
    df = tr.score_all(ctx)
    assert list(df["id"]) == [rec["id"]] and len(tr.load()) == 2                   # 기록은 남고 채점에서만 빠진다
    assert tr.pending_count(df) == 0                                               # 제외한 건은 '대기'로도 세지 않는다
    md = tr.report_md(df, 0)
    assert "집계 제외 1건" in md and "calc_bad (오분류)" in md


def test_repo_excluded_file_marks_nov10_misclassification():
    ex = tr.load_excluded(config.DATA_DIR / "forecasts")
    assert "calc_20261111_113846_96260f" in ex and "오분류" in ex["calc_20261111_113846_96260f"]
