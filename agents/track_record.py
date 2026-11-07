"""실전 성적표 (Track Record): 실시간 예측을 기록해 두고, 3거래일이 지나면 실제 수익률로 자동 채점한다.

    python -m agents.track_record                       # 채점 가능한 예측을 채점하고 성적표 출력
    python -m agents.track_record --add "뉴스 본문" --news-date 2026-11-05   # 뉴스 하나를 분석해 예측으로 기록
    python -m agents.track_record --list                # 기록된 예측 목록

왜: 백테스트는 과거 사건을 하나씩 빼고 맞히는 방식이라 표본이 작고(clean 9건), 사건을 이미 알고 고른 라이브러리라는
한계가 있다. 실제로 매일 낸 예측을 그대로 채점하면 누구도 손댈 수 없는 성적이 쌓인다.

기록 (data/forecasts/forecasts.jsonl, 한 줄 = 예측 1건, 지우거나 고치지 않고 추가만)
  예측 시점, 데이터 기준일, 사건 구조, 3거래일 시나리오(mild/central/severe), 예측구간(lo/hi), 바스켓별 중앙 충격,
  반응 창 시작일(window_start)과 '순수 예측' 여부.
  - window_start: 뉴스 날짜(미국 동부 시간 기준) 이후 첫 거래일.
  - pure: 예측을 기록한 시점에 반응 창 첫날 장이 아직 열리지 않았으면 True.
    아침 데일리 워치는 전날 미국 장이 이미 반응한 뉴스를 다루는 경우가 많다 -> pure=False ('사후 해석')로 따로 집계한다.

채점 (data/forecasts/scored.csv): 가격 데이터에 반응 창 3거래일이 모두 들어오면 채점한다. 지표는 backtest와 같다.
  방향 적중, 범위 적중([mild, severe]), 예측구간 적중, 구간 폭, severe 커버(실제가 최악 시나리오보다 덜 나빴나), 절대오차.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from engine import risk
from pipeline import config

ET = ZoneInfo("America/New_York")
OPEN_ET = dt.time(9, 30)


def forecast_dir() -> Path:
    return config.DATA_DIR / "forecasts"


def _f(x) -> float | None:
    try:
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
    except (TypeError, ValueError):
        return None


def window_start(news_date: dt.date) -> dt.date:
    """뉴스 날짜(미국 동부 기준) 이후 첫 영업일(주말 제외. 미국 공휴일은 채점 때 실제 거래일로 다시 맞춘다)."""
    return pd.Timestamp(np.busday_offset(news_date, 0, roll="forward")).date()


def is_pure(start: dt.date, now_et: dt.datetime) -> bool:
    """기록 시점에 반응 창 첫날 장이 아직 안 열렸으면 순수 예측."""
    return now_et.date() < start or (now_et.date() == start and now_et.time() < OPEN_ET)


def make_record(state: dict, source: str, news_date: dt.date | None = None, now: dt.datetime | None = None,
                title: str | None = None) -> dict | None:
    """브리핑 State -> 예측 기록. 사건 구조화나 계산이 안 된 실행(판단 보류·skipped)은 기록하지 않는다."""
    rr = state.get("risk_result") or {}
    sl = rr.get("scenario_loss") or {}
    if not sl or state.get("status") not in ("ok", "partial") or not state.get("calc_log_id"):
        return None
    now = (now or dt.datetime.now(dt.timezone.utc)).astimezone(ET)
    ev = state.get("event") or {}
    if news_date is None:
        try:
            news_date = dt.date.fromisoformat(str(ev.get("date") or ""))
        except ValueError:
            news_date = now.date()
    start = window_start(news_date)
    band = (rr.get("scenario_band") or {}).get("interval") or {}
    bshock = {}
    try:
        from engine import calclog                      # 계산 기록에 바스켓 충격이 있으면 같이 남긴다
        try:
            log = calclog.load(state["calc_log_id"])
        except FileNotFoundError:
            log = calclog.load(state["calc_log_id"], calclog.LOG_DIR)
        bshock = (log.get("outputs", {}).get("basket_shock") or {}).get("central", {})
    except Exception:
        pass
    return {
        "id": state["calc_log_id"], "source": source, "recorded_at": now.isoformat(timespec="minutes"),
        "data_version": state.get("data_version"), "news_date": news_date.isoformat(),
        "window_start": start.isoformat(), "pure": is_pure(start, now),
        "title": title or ev.get("title") or "", "instrument": ev.get("instrument"), "direction": ev.get("direction"),
        "initiator": ev.get("initiator"), "target": ev.get("target"), "sectors": ev.get("sectors"),
        "severity": ev.get("severity"), "event_mode": state.get("event_mode"), "tier": state.get("analog_tier"),
        "regime": rr.get("regime"),
        "mild": _f(sl.get("mild")), "central": _f(sl.get("central")), "severe": _f(sl.get("severe")),
        "band_lo": _f(band.get("lo")), "band_hi": _f(band.get("hi")),
        "basket_central": {k: _f(v) for k, v in bshock.items()},
    }


def append(rec: dict | None, path: Path | None = None) -> bool:
    """같은 calc_log_id는 한 번만 기록한다. 반환: 새로 기록했는지."""
    if not rec:
        return False
    path = path or forecast_dir() / "forecasts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(r["id"] == rec["id"] for r in load(path)):
        return False
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return True


def load(path: Path | None = None) -> list[dict]:
    path = path or forecast_dir() / "forecasts.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def realized(rec: dict, rets: pd.DataFrame, weights: pd.Series, days: int = 3) -> dict | None:
    """반응 창 실제 수익률. 창이 데이터에 다 안 들어왔으면 None."""
    i0 = rets.index.searchsorted(pd.Timestamp(rec["window_start"]))
    if i0 + days > len(rets):
        return None
    win = rets.iloc[i0:i0 + days]
    lr = win.sum()
    w = (weights / weights.sum()).reindex(rets.columns).fillna(0).to_numpy()
    port = float(risk.portfolio_return(lr.to_numpy()[None, :], w)[0])
    return {"actual": port, "window": f"{win.index[0].date()}~{win.index[-1].date()}",
            "basket_actual": {k: float(np.expm1(v)) for k, v in lr.items()}}


def score_one(rec: dict, real: dict) -> dict:
    mild, central, severe, act = rec["mild"], rec["central"], rec["severe"], real["actual"]
    lo, hi = min(mild, severe), max(mild, severe)
    blo = rec.get("band_lo") if rec.get("band_lo") is not None else lo
    bhi = rec.get("band_hi") if rec.get("band_hi") is not None else hi
    ease = rec.get("direction") == "ease"
    # 바스켓 단위: 예측 중앙 충격이 가장 큰 바스켓이 실제로도 가장 많이 빠졌나
    bc, ba = rec.get("basket_central") or {}, real["basket_actual"]
    common = [k for k in bc if k in ba and bc[k] is not None]
    worst_pred = min(common, key=lambda k: bc[k]) if common else None
    worst_act = min(ba, key=ba.get) if ba else None
    return {
        "id": rec["id"], "source": rec["source"], "title": rec["title"], "pure": rec["pure"],
        "news_date": rec["news_date"], "window": real["window"], "instrument": rec.get("instrument"),
        "tier": rec.get("tier"), "regime": rec.get("regime"), "data_version": rec.get("data_version"),
        "mild": mild, "central": central, "severe": severe, "band_lo": blo, "band_hi": bhi, "actual": act,
        "dir_hit": bool(np.sign(central) == np.sign(act)) if central else False,
        "in_range": bool(lo <= act <= hi),
        "in_band": bool(blo <= act <= bhi),
        "band_width": bhi - blo,
        "severe_cover": bool(act <= severe) if ease else bool(act >= severe),
        "abs_err": abs(central - act),
        "worst_pred": worst_pred, "worst_actual": worst_act,
        "worst_hit": worst_pred is not None and worst_pred == worst_act,
    }


def score_all(ctx, path: Path | None = None) -> pd.DataFrame:
    """기록된 예측 중 채점 가능한 것을 모두 채점해 scored.csv로 저장. 아직 창이 안 끝난 예측은 건너뛴다."""
    days = ctx.thresholds["event_window"]["react_days"]
    rows = []
    for rec in load(path):
        if rec.get("central") is None:
            continue
        real = realized(rec, ctx.rets, ctx.weights, days)
        if real is not None:
            rows.append(score_one(rec, real))
    df = pd.DataFrame(rows)
    out = (path.parent if path else forecast_dir()) / "scored.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.assign(구분=np.where(df["pure"], "순수 예측", "사후 해석"))
    parts = [df.assign(구분="전체"), df]
    d = pd.concat(parts)
    g = d.groupby("구분")
    return pd.DataFrame({
        "건수": g.size(), "방향 적중": g["dir_hit"].mean(), "범위 적중": g["in_range"].mean(),
        "예측구간 적중": g["in_band"].mean(), "평균 구간 폭": g["band_width"].mean(),
        "severe 커버": g["severe_cover"].mean(), "평균 절대오차": g["abs_err"].mean(),
        "최대 손실 바스켓 적중": g["worst_hit"].mean(),
    })


def report_md(df: pd.DataFrame, n_pending: int) -> str:
    L = [f"채점 {len(df)}건 · 반응 창 대기 {n_pending}건"]
    if df.empty:
        L.append("아직 채점할 수 있는 예측이 없습니다 (반응 창 3거래일이 가격 데이터에 들어오면 채점).")
        return "\n".join(L)
    s = summary(df)
    pct = lambda x: f"{x:.0%}"
    L += ["", "| 구분 | 건수 | 방향 | 범위 | 예측구간 | 구간 폭 | severe 커버 | 절대오차 | 최대손실 바스켓 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, r in s.iterrows():
        L.append(f"| {k} | {int(r['건수'])} | {pct(r['방향 적중'])} | {pct(r['범위 적중'])} | {pct(r['예측구간 적중'])} | "
                 f"{r['평균 구간 폭'] * 100:.2f}%p | {pct(r['severe 커버'])} | {r['평균 절대오차'] * 100:.2f}%p | "
                 f"{pct(r['최대 손실 바스켓 적중'])} |")
    L += ["", "최근 채점:", "", "| 뉴스일 | 사건 | 구분 | 중앙 | 최악 | 예측구간 | 실제 | 구간 적중 |", "|---|---|---|---|---|---|---|---|"]
    for _, r in df.sort_values("news_date", ascending=False).head(8).iterrows():
        L.append(f"| {r['news_date']} | {str(r['title'])[:28]} | {'순수' if r['pure'] else '사후'} | {r['central']:+.2%} | "
                 f"{r['severe']:+.2%} | {r['band_lo']:+.2%} ~ {r['band_hi']:+.2%} | {r['actual']:+.2%} | "
                 f"{'O' if r['in_band'] else 'X'} |")
    L += ["", "'사후 해석'은 기록 시점에 반응 창 첫날 장이 이미 열린 예측입니다(아침 브리핑이 전날 미국 장 뉴스를 다룰 때). "
          "모델 실력은 '순수 예측' 행으로 판단하세요."]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description="ShockLens 실전 성적표")
    ap.add_argument("--add", help="뉴스 텍스트·URL을 분석해 예측으로 기록")
    ap.add_argument("--news-date", help="뉴스 날짜 YYYY-MM-DD (미국 동부 기준, 기본: 사건 날짜 또는 오늘)")
    ap.add_argument("--force", action="store_true", help="중요도 1차 분류 무시 (--add와 함께)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    from .context import load_context
    ctx = load_context()
    if a.add:
        from . import hq
        from .llm import LLM
        st = hq.run_news(a.add, ctx, LLM(), force=a.force)
        nd = dt.date.fromisoformat(a.news_date) if a.news_date else None
        rec = make_record(st, "manual", nd)
        print("기록함" if append(rec) else f"기록 안 함 (상태 {st.get('status')}, 이미 있거나 계산 결과 없음)")
        if rec:
            print(f"  {rec['id']} 창 시작 {rec['window_start']} {'순수 예측' if rec['pure'] else '사후 해석'} "
                  f"중앙 {rec['central']:+.2%} 최악 {rec['severe']:+.2%} 구간 {rec['band_lo']:+.2%}~{rec['band_hi']:+.2%}")
    if a.list:
        for r in load():
            print(f"{r['news_date']} {r['id']} {'순수' if r['pure'] else '사후'} {r['source']:12s} {r['title'][:40]}")
    df = score_all(ctx)
    print(report_md(df, len(load()) - len(df)))
    print(f"\n데이터 기준일 {ctx.data_version} · 저장 {forecast_dir() / 'scored.csv'}")


if __name__ == "__main__":
    main()
