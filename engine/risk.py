"""리스크 계산 (Risk Engine, LLM 없음).

1) 결정론적 시나리오 손실: 바스켓 충격 = B x f + 고유충격 -> 포트폴리오 손실 (mild/central/severe)
2) 분포: 과거 수익률을 5일 덩어리로 다시 뽑는 블록 부트스트랩으로 20일 손실 분포를 만들고,
   그 위에 central 충격을 더해 '충격 후' 분포를 만든다. -> ES97.5, VaR95
3) 비교: 같은 평균·공분산의 정규분포 몬테카를로로 ES를 다시 계산해 과소평가 정도를 본다.
수익률은 로그수익률로 더하고, 포트폴리오 합산할 때만 단순수익률(exp-1)로 바꾼다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def basket_shock(beta: pd.DataFrame, factor_shock: pd.DataFrame, idio: pd.DataFrame | None = None) -> pd.DataFrame:
    """바스켓별 충격(로그) = B x f (+ 고유충격). 행 = 바스켓, 열 = 시나리오."""
    shock = beta.reindex(columns=factor_shock.index).to_numpy() @ factor_shock.to_numpy()
    out = pd.DataFrame(shock, index=beta.index, columns=factor_shock.columns)
    if idio is not None:
        out = out + idio.reindex(index=out.index, columns=out.columns).fillna(0.0)
    return out


def portfolio_return(log_rets: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(경로 x 바스켓) 로그수익률 -> 경로별 포트폴리오 단순수익률."""
    return np.expm1(log_rets) @ weights


def scenario_losses(bshock: pd.DataFrame, weights: pd.Series) -> pd.Series:
    w = weights.reindex(bshock.index).to_numpy()
    return pd.Series({s: float(portfolio_return(bshock[s].to_numpy()[None, :], w)[0]) for s in bshock.columns})


def block_bootstrap(rets: np.ndarray, eligible_starts: np.ndarray, horizon: int, block_len: int,
                    n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """(T x N) 일별 로그수익률에서 horizon일 누적 수익률 경로 n_paths개를 만든다.

    연속된 block_len일을 한 덩어리로 뽑아 이어 붙이므로, 위기 때 며칠 연속 떨어지는 흐름이 보존된다.
    eligible_starts: 덩어리 시작일로 쓸 수 있는 날짜 위치 (위기 국면이면 VIX 높은 날만).
    """
    T = rets.shape[0]
    starts_ok = eligible_starts[eligible_starts <= T - block_len]
    if len(starts_ok) == 0:
        raise ValueError("부트스트랩에 쓸 수 있는 시작일이 없습니다")
    n_blocks = int(np.ceil(horizon / block_len))
    starts = rng.choice(starts_ok, size=(n_paths, n_blocks))
    idx = starts[..., None] + np.arange(block_len)          # (경로, 덩어리, 일)
    idx = idx.reshape(n_paths, -1)[:, :horizon]              # 정확히 horizon일만
    return rets[idx].sum(axis=1)                             # (경로, 바스켓)


def var_es(port: np.ndarray, var_level: float, es_level: float) -> dict:
    """손실을 양수로 표현. VaR95 = 하위 5% 경계 손실, ES97.5 = 하위 2.5% 평균 손실."""
    var = -np.quantile(port, 1 - var_level)
    cut = np.quantile(port, 1 - es_level)
    es = -port[port <= cut].mean()
    return {"VaR": float(var), "ES": float(es)}


def tail_contributions(paths: np.ndarray, weights: np.ndarray, es_level: float) -> np.ndarray:
    """ES 구간(하위 2.5%) 경로들에서 바스켓별 손실이 차지하는 비중. 합은 1."""
    contrib = np.expm1(paths) * weights                      # (경로, 바스켓)
    port = contrib.sum(axis=1)
    tail = port <= np.quantile(port, 1 - es_level)
    return contrib[tail].mean(axis=0) / port[tail].mean()


def normal_mc(paths: np.ndarray, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """부트스트랩 경로와 같은 평균·공분산을 갖는 정규분포 경로."""
    mu = paths.mean(axis=0)
    cov = np.cov(paths, rowvar=False)
    return rng.multivariate_normal(mu, cov, size=n_paths, method="cholesky")


def run_stress(rets: pd.DataFrame, weights: pd.Series, beta: pd.DataFrame, factor_shock: pd.DataFrame,
               idio: pd.DataFrame | None, eligible_mask: pd.Series | None, horizon: int, block_len: int,
               n_paths: int, var_level: float, es_level: float, seed: int) -> dict:
    """Risk Engine 본체. 결과는 전부 숫자로 반환하고, 해석·서술은 하지 않는다."""
    rng = np.random.default_rng(seed)
    w = weights.reindex(rets.columns).to_numpy() / weights.sum()
    bshock = basket_shock(beta.reindex(rets.columns), factor_shock, idio)

    mask = np.ones(len(rets), bool) if eligible_mask is None else eligible_mask.reindex(rets.index).fillna(False).to_numpy()
    base = block_bootstrap(rets.to_numpy(), np.flatnonzero(mask), horizon, block_len, n_paths, rng)
    stressed = base + bshock["central"].to_numpy()

    base_m = var_es(portfolio_return(base, w), var_level, es_level)
    str_m = var_es(portfolio_return(stressed, w), var_level, es_level)
    normal_stressed = normal_mc(base, n_paths, rng) + bshock["central"].to_numpy()
    nrm_m = var_es(portfolio_return(normal_stressed, w), var_level, es_level)

    return {
        "scenario_loss": scenario_losses(bshock, pd.Series(w, index=rets.columns)).to_dict(),
        "basket_shock": bshock.to_dict(),
        "baseline": base_m,
        "stressed": str_m,
        "normal_mc_stressed": nrm_m,
        "es_underestimation_normal": str_m["ES"] - nrm_m["ES"],
        "tail_contribution": dict(zip(rets.columns, tail_contributions(stressed, w, es_level).round(6))),
        "settings": {"horizon": horizon, "block_len": block_len, "n_paths": n_paths,
                     "var_level": var_level, "es_level": es_level, "seed": seed,
                     "n_eligible_days": int(mask.sum())},
    }
