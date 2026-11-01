"""4단계: 사건 라이브러리 구간 계산.

사건마다 발표일 이후 첫 거래일(t0)을 잡고,
- 사전 구간: t0 이전 pre_days 거래일
- 반응 구간: t0부터 post_days 거래일
의 팩터 누적 변동, 바스켓 누적 수익률, 베타로 설명되지 않는 잔차 수익률을 계산한다.
잔차 = 실제 바스켓 수익률 - (평상시 베타 x 팩터 변동). Exposure Analyst의 고유충격 크기에 쓴다.
결과: data/model/event_factor_post.csv, event_basket_post.csv, event_resid_post.csv, event_factor_pre.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from . import config

LIBRARY_FILE = config.DATA_DIR / "library" / "events.yaml"


def load_events(path: Path = LIBRARY_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["events"]


def clean_ids(events: list[dict]) -> set[str]:
    """반응 창에 무관한 큰 충격이 겹치지 않은 사건 id. clean 표시가 없으면 깨끗하지 않은 것으로 본다."""
    return {e["id"] for e in events if e.get("clean") is True}


def select_analogs(available: list[str], events: list[dict], target_id: str, clean_only: bool = True) -> list[str]:
    """유사 사례 후보: 대상 사건 자신은 빼고, clean_only면 깨끗한 사건만 남긴다."""
    ok = clean_ids(events) if clean_only else {e["id"] for e in events}
    return [i for i in available if i != target_id and i in ok]


def first_trading_index(index: pd.DatetimeIndex, date: str) -> int:
    pos = index.searchsorted(pd.Timestamp(date))
    if pos >= len(index):
        raise ValueError(f"{date} 이후 거래일 데이터가 없습니다")
    return int(pos)


def event_windows(events, rets, frets, beta_normal, pre_days, post_days):
    f_post, f_pre, b_post, resid = {}, {}, {}, {}
    for ev in events:
        i0 = first_trading_index(rets.index, ev["date"])
        if i0 - pre_days < 0 or i0 + post_days > len(rets):
            print(f"[library] 건너뜀 (데이터 구간 밖): {ev['id']}")
            continue
        post = slice(i0, i0 + post_days)
        pre = slice(i0 - pre_days, i0)
        fp = frets.iloc[post].sum()
        bp = rets.iloc[post].sum()
        f_post[ev["id"]] = fp
        f_pre[ev["id"]] = frets.iloc[pre].sum()
        b_post[ev["id"]] = bp
        resid[ev["id"]] = bp - beta_normal.reindex(index=bp.index, columns=fp.index).to_numpy() @ fp.to_numpy()
    to_df = lambda d: pd.DataFrame(d).T  # 행 = 사건, 열 = 팩터 또는 바스켓
    return to_df(f_post), to_df(f_pre), to_df(b_post), to_df(resid)


def run() -> None:
    th = config.thresholds()["event_window"]
    rets = pd.read_parquet(config.PROCESSED_DIR / "returns.parquet")
    frets = pd.read_parquet(config.PROCESSED_DIR / "factor_returns.parquet")
    beta = pd.read_csv(config.MODEL_DIR / "beta_normal.csv", index_col=0)
    events = load_events()
    # 20거래일 누적 (참고용)
    f_post, f_pre, b_post, resid = event_windows(events, rets, frets, beta, th["pre_days"], th["post_days"])
    f_post.to_csv(config.MODEL_DIR / "event_factor_post.csv")
    f_pre.to_csv(config.MODEL_DIR / "event_factor_pre.csv")
    b_post.to_csv(config.MODEL_DIR / "event_basket_post.csv")
    resid.to_csv(config.MODEL_DIR / "event_resid_post.csv")
    # 발표 직후 반응 창 (충격 크기 산정의 기본값)
    f_re, _, b_re, resid_re = event_windows(events, rets, frets, beta, th["pre_days"], th["react_days"])
    f_re.to_csv(config.MODEL_DIR / "event_factor_react.csv")
    b_re.to_csv(config.MODEL_DIR / "event_basket_react.csv")
    resid_re.to_csv(config.MODEL_DIR / "event_resid_react.csv")
    print(f"[library] 사건 {len(f_post)}건 구간 계산 완료")
    print(f"[library] 발표 후 {th['react_days']}거래일 반응 (충격 산정 기본값):")
    print(f_re.round(4).to_string())
    print(f"[library] 발표 후 {th['post_days']}거래일 누적 (참고):")
    print(f_post.round(4).to_string())


if __name__ == "__main__":
    run()
