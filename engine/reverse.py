"""역스트레스테스트: "포트폴리오가 L만큼 잃으려면 요인이 어떻게 움직이는 게 가장 그럴듯한가"를 거꾸로 푼다. LLM 없음.

설정
  b  = 포트폴리오의 요인 노출 = sum_i w_i * beta_i   (바스켓 베타의 비중 가중합)
  Σ  = h거래일 요인 변동 공분산 (일별 공분산 x h, 위기 국면이면 위기 날짜로 추정)
  손실 근사: 포트폴리오 로그수익 ≈ b'x
문제
  min  x' Σ^-1 x   (마할라노비스 거리 = 얼마나 '드문' 움직임인가)
  s.t. b'x = -L
해 (닫힌 형태)
  x* = -L · Σb / (b'Σb),   거리 d = L / sqrt(b'Σb)
해석
  정규분포 가정이면 요인 때문에 L 이상 잃을 확률 = 1 - Φ(d)
  과거 실제 빈도 = 과거 모든 h일 창에서 요인으로 설명되는 손실(-b'x)이 L 이상이었던 비율
  가장 닮은 과거 사건 = x*와 과거 사건 요인 반응의 코사인 유사도 상위
'목표 ES'로 물으면 L = 목표 ES - 지금 ES (사건 충격은 손실 분포 전체를 L만큼 미는 것과 같다).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def portfolio_exposure(beta: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(beta.index).fillna(0.0)
    return (beta.mul(w / w.sum(), axis=0)).sum()


def solve(b: pd.Series, sigma: pd.DataFrame, loss: float, factors: list[str] | None = None) -> dict:
    f = [x for x in (factors or list(b.index)) if x in b.index and x in sigma.index]
    bs, S = b[f].to_numpy(), sigma.loc[f, f].to_numpy()
    v = S @ bs
    denom = float(bs @ v)
    if denom <= 0:
        raise ValueError("요인 노출이 0이라 역산할 수 없습니다")
    x = -loss * v / denom
    d = loss / np.sqrt(denom)
    contrib = bs * x
    return {"factors": f, "shock": dict(zip(f, x.tolist())), "contribution": dict(zip(f, contrib.tolist())),
            "distance": float(d), "normal_prob": float(1 - stats.norm.cdf(d)),
            "portfolio_sd": float(np.sqrt(denom))}


def historical_frequency(frets: pd.DataFrame, b: pd.Series, loss: float, h: int, factors: list[str]) -> dict:
    """과거 모든 h일 창에서 요인 설명 손실이 loss 이상이었던 비율과 최악값."""
    X = frets[factors].dropna().rolling(h).sum().dropna()
    pl = -(X.to_numpy() @ b[factors].to_numpy())
    if not len(pl):
        return {"freq": None, "worst": None, "n_windows": 0}
    return {"freq": float((pl >= loss).mean()), "worst": float(pl.max()), "n_windows": int(len(pl)),
            "worst_date": str(X.index[int(pl.argmax())].date())}


def nearest_events(shock: dict, f_event: pd.DataFrame, events: list[dict], top: int = 3) -> list[dict]:
    f = [k for k in shock if k in f_event.columns]
    x = np.array([shock[k] for k in f])
    F = f_event[f].fillna(0.0)
    scale = F.abs().max().replace(0, 1).to_numpy()           # 요인 단위가 달라 최대 반응으로 정규화
    xs, Fs = x / scale, F.to_numpy() / scale
    sims = Fs @ xs / (np.linalg.norm(Fs, axis=1) * np.linalg.norm(xs) + 1e-12)
    title = {e["id"]: e.get("title", "") for e in events}
    order = np.argsort(-sims)[:top]
    return [{"id": F.index[i], "similarity": round(float(sims[i]), 3), "title": title.get(F.index[i], "")}
            for i in order]


def run(beta: pd.DataFrame, weights: pd.Series, frets: pd.DataFrame, loss: float, h: int = 3,
        factors: list[str] | None = None, mask: pd.Series | None = None, f_event: pd.DataFrame | None = None,
        events: list[dict] | None = None) -> dict:
    b = portfolio_exposure(beta, weights)
    fr = frets if mask is None else frets[mask.reindex(frets.index).fillna(False)]
    sigma = fr.cov() * h
    out = solve(b, sigma, loss, factors)
    out["horizon_days"] = h
    out["loss"] = loss
    out["exposure"] = {k: float(b[k]) for k in out["factors"]}
    out["historical"] = historical_frequency(frets, b, loss, h, out["factors"])
    if f_event is not None:
        out["nearest_events"] = nearest_events(out["shock"], f_event, events or [])
    return out
