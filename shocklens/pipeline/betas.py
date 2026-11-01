"""3단계: 베타 행렬(바스켓 12 x 팩터 7) 추정.

방법: 바스켓별로 팩터 7개에 릿지 회귀.
- 팩터를 표준화한 뒤 릿지를 적용하고, 계수를 원래 단위로 되돌린다.
  (VIX와 하이일드 스프레드처럼 같이 움직이는 팩터 때문에 계수가 불안정해지는 것을 막기 위함)
- normal: 전 기간 / crisis: VIX 수준이 상위 20%인 날만
결과: data/model/beta_normal.csv, beta_crisis.csv, factor_cov_crisis.csv, beta_r2.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def ridge_betas(y: pd.DataFrame, X: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, pd.Series]:
    """y의 각 열을 X에 릿지 회귀. 반환: (베타 행렬[바스켓 x 팩터], 바스켓별 R^2)"""
    Xc = X - X.mean()
    sd = Xc.std(ddof=0).replace(0, 1.0)
    Z = (Xc / sd).to_numpy()
    Yc = (y - y.mean()).to_numpy()
    n, k = Z.shape
    # 표준화 계수: (Z'Z + n*alpha*I)^-1 Z'Y
    A = Z.T @ Z + n * alpha * np.eye(k)
    coef_std = np.linalg.solve(A, Z.T @ Yc)          # k x m
    coef = coef_std / sd.to_numpy()[:, None]         # 원래 단위로 환원
    resid = Yc - Z @ coef_std
    r2 = 1 - (resid ** 2).sum(axis=0) / (Yc ** 2).sum(axis=0)
    betas = pd.DataFrame(coef.T, index=y.columns, columns=X.columns)
    return betas, pd.Series(r2, index=y.columns, name="r2")


def crisis_mask(vix_level: pd.Series, quantile: float) -> pd.Series:
    return vix_level >= vix_level.quantile(quantile)


def build(rets: pd.DataFrame, frets: pd.DataFrame, vix_level: pd.Series, alpha: float, q: float):
    b_norm, r2_norm = ridge_betas(rets, frets, alpha)
    mask = crisis_mask(vix_level.reindex(rets.index), q)
    b_cris, r2_cris = ridge_betas(rets[mask], frets[mask], alpha)
    cov_cris = frets[mask].cov()
    r2 = pd.DataFrame({"normal": r2_norm, "crisis": r2_cris})
    return b_norm, b_cris, cov_cris, r2, int(mask.sum())


def run() -> None:
    th = config.thresholds()["betas"]
    rets = pd.read_parquet(config.PROCESSED_DIR / "returns.parquet")
    frets = pd.read_parquet(config.PROCESSED_DIR / "factor_returns.parquet")
    levels = pd.read_parquet(config.PROCESSED_DIR / "factor_levels.parquet")
    vix_series = config.factors()["factors"]["risk_off"]["series"]

    b_norm, b_cris, cov_cris, r2, n_crisis = build(
        rets, frets, levels[vix_series], th["ridge_alpha"], th["crisis_vix_quantile"]
    )
    b_norm.to_csv(config.MODEL_DIR / "beta_normal.csv")
    b_cris.to_csv(config.MODEL_DIR / "beta_crisis.csv")
    cov_cris.to_csv(config.MODEL_DIR / "factor_cov_crisis.csv")
    r2.to_csv(config.MODEL_DIR / "beta_r2.csv")
    print(f"[betas] 위기 구간 {n_crisis}일 / 전체 {len(rets)}일")
    print("[betas] 바스켓별 설명력(R^2):")
    print(r2.round(2).to_string())


if __name__ == "__main__":
    run()
