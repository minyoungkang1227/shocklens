"""에이전트가 공유하는 읽기 전용 데이터 묶음 (설정, 사건 라이브러리, 베타·수익률 등 모델 산출물).

한 번 읽어서 모든 노드에 넘긴다. 테스트에서는 합성 데이터로 직접 만들어 넣는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import yaml

from pipeline import config, library


@dataclass
class Context:
    thresholds: dict
    weights: pd.Series
    exposure_map: dict
    events: list[dict]
    rets: pd.DataFrame
    levels: pd.DataFrame
    beta_normal: pd.DataFrame
    beta_crisis: pd.DataFrame
    f_event: pd.DataFrame            # 사건별 팩터 변동 (발표 후 3거래일)
    b_event: pd.DataFrame            # 사건별 바스켓 수익률
    resid_event: pd.DataFrame        # 사건별 잔차
    q_up: pd.DataFrame
    q_down: pd.DataFrame
    vix_col: str = "VIXCLS"
    labels: dict = field(default_factory=dict)   # 바스켓·팩터 한글 이름
    frets: pd.DataFrame | None = None            # 일별 요인 변동 (역스트레스테스트의 공분산 추정)
    stress: pd.DataFrame | None = None           # 보조 위기 지수 OFR_FSI·AI_GPR (pipeline.indicators, 없으면 None)

    @property
    def data_version(self) -> str:
        return str(self.rets.index.max().date())

    def event_by_id(self, eid: str) -> dict | None:
        return next((e for e in self.events if e["id"] == eid), None)


def load_context(window: str = "react") -> Context:
    m, p = config.MODEL_DIR, config.PROCESSED_DIR
    suffix = "_react" if window == "react" else ""
    post = "react" if window == "react" else "post"
    pf = config.portfolio()["baskets"]
    fac = config.factors()["factors"]
    with open(config.CONFIG_DIR / "exposure_map.yaml", encoding="utf-8") as f:
        exposure_map = yaml.safe_load(f)
    labels = {b: v.get("label", b) for b, v in pf.items()}
    labels.update({k: v.get("label", k) for k, v in fac.items()})
    return Context(
        thresholds=config.thresholds(),
        weights=pd.Series({b: v["weight"] for b, v in pf.items()}, dtype=float),
        exposure_map=exposure_map,
        events=library.load_events(),
        rets=pd.read_parquet(p / "returns.parquet"),
        levels=pd.read_parquet(p / "factor_levels.parquet"),
        beta_normal=pd.read_csv(m / "beta_normal.csv", index_col=0),
        beta_crisis=pd.read_csv(m / "beta_crisis.csv", index_col=0),
        f_event=pd.read_csv(m / f"event_factor_{post}.csv", index_col=0),
        b_event=pd.read_csv(m / f"event_basket_{post}.csv", index_col=0),
        resid_event=pd.read_csv(m / f"event_resid_{post}.csv", index_col=0),
        q_up=pd.read_csv(m / f"factor_quantiles{suffix}_up.csv", index_col=0),
        q_down=pd.read_csv(m / f"factor_quantiles{suffix}_down.csv", index_col=0),
        vix_col=fac["risk_off"]["series"],
        labels=labels,
        frets=pd.read_parquet(p / "factor_returns.parquet") if (p / "factor_returns.parquet").exists() else None,
        stress=_load_stress(),
    )


def _load_stress():
    try:
        from pipeline import indicators
        return indicators.load()
    except Exception:
        return None
