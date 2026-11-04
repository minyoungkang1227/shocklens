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


def load_events(path: Path = LIBRARY_FILE, severity_cfg: dict | None = None) -> list[dict]:
    """사건 목록. severity_components가 있으면 0~100 점수(severity_points)와 등급(severity)을 계산해 채운다."""
    from engine import severity as sev
    with open(path, encoding="utf-8") as f:
        events = yaml.safe_load(f)["events"]
    cfg = severity_cfg if severity_cfg is not None else config.thresholds().get("severity", {})
    weights = cfg.get("component_weights")
    use = cfg.get("use_components", True)
    blind = {}
    if cfg.get("labels", "human") == "blind":
        # agents.relabel_severity 가 만든 블라인드 채점(발표 당일 정보만으로 LLM이 채점)으로 바꿔 쓴다
        bf = path.parent / "severity_blind.yaml"
        if bf.exists():
            with open(bf, encoding="utf-8") as f:
                blind = (yaml.safe_load(f) or {}).get("events", {})
    for e in events:
        if e["id"] in blind:
            e["severity_components_human"] = e.get("severity_components")
            e["severity_components"] = blind[e["id"]]["severity_components"]
        comp = e.get("severity_components")
        if use and comp:
            e["severity_points"] = sev.score(comp, weights)
            e["severity_label"] = e.get("severity")               # 사람이 적은 원래 값 보관
            e["severity"] = sev.level(e["severity_points"])
        else:
            e["severity_points"] = sev.score(None, weights, e.get("severity", 3))
    return events


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


def event_windows(events, rets, frets, beta_normal, pre_days, post_days, frets_long=None):
    """사건별 반응 구간.

    팩터 변동은 frets_long(팩터 장기 데이터)이 있으면 그것으로 잰다 -> 2012년 이전 사건도 충격 크기를 잴 수 있다.
    바스켓 수익률·잔차는 바스켓 데이터(rets) 구간 안의 사건만 계산한다(그 밖은 행이 없다).
    """
    fsrc = frets_long if frets_long is not None else frets
    f_post, f_pre, b_post, resid = {}, {}, {}, {}
    for ev in events:
        start = str(ev.get("react_from") or ev["date"])     # 장 마감 후 발표면 react_from(다음 거래일)부터
        try:
            j0 = first_trading_index(fsrc.index, start)
        except ValueError:
            print(f"[library] 건너뜀 (데이터 구간 밖): {ev['id']}")
            continue
        if j0 - pre_days < 0 or j0 + post_days > len(fsrc) or fsrc.index[j0] < fsrc.index[0]:
            print(f"[library] 건너뜀 (데이터 구간 밖): {ev['id']}")
            continue
        fp = fsrc.iloc[j0:j0 + post_days].sum(min_count=1)        # 전부 비어 있으면 NaN 유지
        f_post[ev["id"]] = fp
        f_pre[ev["id"]] = fsrc.iloc[j0 - pre_days:j0].sum(min_count=1)
        if pd.Timestamp(start) < rets.index[0]:
            continue                                               # 바스켓 데이터 이전 사건: 팩터 반응만
        i0 = first_trading_index(rets.index, start)
        if i0 + post_days > len(rets):
            continue
        bp = rets.iloc[i0:i0 + post_days].sum()
        fp_common = frets.iloc[i0:i0 + post_days].sum()
        b_post[ev["id"]] = bp
        resid[ev["id"]] = bp - beta_normal.reindex(index=bp.index, columns=fp_common.index).to_numpy() @ fp_common.to_numpy()
    to_df = lambda d: pd.DataFrame(d).T  # 행 = 사건, 열 = 팩터 또는 바스켓
    return to_df(f_post), to_df(f_pre), to_df(b_post), to_df(resid)


def run() -> None:
    th = config.thresholds()["event_window"]
    rets = pd.read_parquet(config.PROCESSED_DIR / "returns.parquet")
    frets = pd.read_parquet(config.PROCESSED_DIR / "factor_returns.parquet")
    beta = pd.read_csv(config.MODEL_DIR / "beta_normal.csv", index_col=0)
    events = load_events()
    long_path = config.PROCESSED_DIR / "factor_returns_long.parquet"
    flong = pd.read_parquet(long_path) if long_path.exists() else None
    # 20거래일 누적 (참고용)
    f_post, f_pre, b_post, resid = event_windows(events, rets, frets, beta, th["pre_days"], th["post_days"], flong)
    f_post.to_csv(config.MODEL_DIR / "event_factor_post.csv")
    f_pre.to_csv(config.MODEL_DIR / "event_factor_pre.csv")
    b_post.to_csv(config.MODEL_DIR / "event_basket_post.csv")
    resid.to_csv(config.MODEL_DIR / "event_resid_post.csv")
    # 발표 직후 반응 창 (충격 크기 산정의 기본값)
    f_re, _, b_re, resid_re = event_windows(events, rets, frets, beta, th["pre_days"], th["react_days"], flong)
    f_re.to_csv(config.MODEL_DIR / "event_factor_react.csv")
    b_re.to_csv(config.MODEL_DIR / "event_basket_react.csv")
    resid_re.to_csv(config.MODEL_DIR / "event_resid_react.csv")
    print(f"[library] 사건 {len(f_post)}건 구간 계산 완료 (팩터 반응 {len(f_re)}건, 그중 바스켓 반응까지 있는 사건 {len(b_re)}건)")
    print(f"[library] 발표 후 {th['react_days']}거래일 반응 (충격 산정 기본값):")
    print(f_re.round(4).to_string())
    print(f"[library] 발표 후 {th['post_days']}거래일 누적 (참고):")
    print(f_post.round(4).to_string())


if __name__ == "__main__":
    run()
