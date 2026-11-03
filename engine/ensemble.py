"""앙상블 Risk Engine: 손실 분포 모형 3개를 경로 섞기로 합친다. LLM 없음.

모형 (모두 20거래일 바스켓 로그수익 경로를 만든다)
  bootstrap     : 블록 부트스트랩 (기존). 국면 표본에서 5일 블록을 다시 뽑는다.
  fhs_evt_t     : 필터링 역사적 시뮬레이션 + 극단값 꼬리 + t-코퓰라
                  ① EWMA(λ=0.94) 변동성으로 일별 수익을 표준화 (지금 변동성 수준을 반영)
                  ② 표준화 잔차의 양쪽 꼬리(기본 5%)는 일반화 파레토분포(GPD, PWM 추정)로, 가운데는 경험분포로
                  ③ 바스켓 간 결합은 t-코퓰라(자유도 4, 켄달 타우로 상관 추정) -> 위기 때 같이 떨어지는 꼬리 의존성
                  ④ 시뮬레이션 중 EWMA 변동성을 매일 갱신 (큰 손실 뒤 변동성이 커지는 효과)
  regime_switch : VIX 기준 2국면(평상시/위기) 마르코프 전환. 매일 국면이 바뀔 수 있고,
                  각 날의 수익은 그 국면에 속한 과거 날짜에서 뽑는다 (국면 지속성·전환을 함께 반영)
합치기
  대표값 = 모형별 경로를 가중치대로 섞은 한 분포의 ES (숫자 평균보다 꼬리를 보수적으로 반영)
  모형 위험 = 모형별 ES의 최소~최대, 보수적 기준 = 모형별 ES 최대
정규분포 몬테카를로는 앙상블에 넣지 않고 기준선으로만 비교한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .risk import block_bootstrap, portfolio_return, var_es

MODELS = ("bootstrap", "fhs_evt_t", "regime_switch")
MODEL_KO = {"bootstrap": "블록 부트스트랩", "fhs_evt_t": "FHS·극단값·t-코퓰라", "regime_switch": "국면 전환"}


# ---------------------------------------------------------------- 공통
def weighted_var_es(port: np.ndarray, wts: np.ndarray, var_level: float, es_level: float) -> dict:
    """가중 경로의 VaR·ES (손실 양수). 경로 섞기 앙상블용."""
    order = np.argsort(port)
    p, w = port[order], wts[order] / wts.sum()
    cw = np.cumsum(w)

    def q(level):
        return p[min(np.searchsorted(cw, 1 - level), len(p) - 1)]

    var = -q(var_level)
    a = 1 - es_level
    k = np.searchsorted(cw, a)
    tail_w = cw[k - 1] if k > 0 else 0.0
    es_sum = (p[:k] * w[:k]).sum() + p[k] * (a - tail_w)       # 경계 경로는 남은 확률만큼만
    return {"VaR": float(var), "ES": float(-es_sum / a)}


# ---------------------------------------------------------------- 극단값 꼬리 (GPD, 확률가중적률)
def gpd_pwm(exceed: np.ndarray) -> tuple[float, float]:
    """초과분(양수)에 GPD를 PWM(Hosking & Wallis 1987)으로 맞춘다. (xi 형상, sigma 척도)"""
    y = np.sort(exceed)
    n = len(y)
    a0 = y.mean()
    p = (np.arange(1, n + 1) - 0.35) / n
    a1 = ((1 - p) * y).mean()
    den = a0 - 2 * a1
    if den <= 0:
        return 0.0, float(a0)
    xi = 2 - a0 / den
    sigma = 2 * a0 * a1 / den
    return float(np.clip(xi, -0.4, 0.6)), float(max(sigma, 1e-8))


def gpd_ppf(q: np.ndarray, xi: float, sigma: float) -> np.ndarray:
    q = np.clip(q, 0, 1 - 1e-12)
    if abs(xi) < 1e-6:
        return -sigma * np.log1p(-q)
    return sigma / xi * ((1 - q) ** (-xi) - 1)


class SemiParametricMarginal:
    """가운데는 경험분포, 양쪽 꼬리(tail 비율)는 GPD."""

    def __init__(self, z: np.ndarray, tail: float = 0.05):
        z = np.sort(z[np.isfinite(z)])
        self.z, self.tail = z, tail
        self.lo_u, self.hi_u = np.quantile(z, tail), np.quantile(z, 1 - tail)
        self.lo = gpd_pwm(self.lo_u - z[z < self.lo_u]) if (z < self.lo_u).sum() >= 20 else (0.0, z.std())
        self.hi = gpd_pwm(z[z > self.hi_u] - self.hi_u) if (z > self.hi_u).sum() >= 20 else (0.0, z.std())

    def ppf(self, u: np.ndarray) -> np.ndarray:
        out = np.quantile(self.z, np.clip(u, self.tail, 1 - self.tail))
        lo, hi = u < self.tail, u > 1 - self.tail
        out[lo] = self.lo_u - gpd_ppf(1 - u[lo] / self.tail, *self.lo)
        out[hi] = self.hi_u + gpd_ppf((u[hi] - (1 - self.tail)) / self.tail, *self.hi)
        return out


def ewma_vol(R: np.ndarray, lam: float) -> np.ndarray:
    """일별 EWMA 변동성 (T x N). 첫 값은 처음 60일 표준편차."""
    T, N = R.shape
    var = np.empty((T, N))
    var[0] = np.nanvar(R[:60], axis=0) + 1e-12
    for t in range(1, T):
        var[t] = lam * var[t - 1] + (1 - lam) * R[t - 1] ** 2
    return np.sqrt(var)


def t_copula_corr(Z: np.ndarray) -> np.ndarray:
    """켄달 타우 -> 타원 코퓰라 상관 sin(pi*tau/2), 양의 정부호로 보정."""
    N = Z.shape[1]
    C = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            tau = stats.kendalltau(Z[:, i], Z[:, j])[0]
            C[i, j] = C[j, i] = np.sin(np.pi * tau / 2)
    vals, vecs = np.linalg.eigh(C)
    C = vecs @ np.diag(np.clip(vals, 1e-6, None)) @ vecs.T
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def fhs_evt_t_paths(R: np.ndarray, horizon: int, n_paths: int, rng: np.random.Generator, lam: float = 0.94,
                    tail: float = 0.05, df: int = 4, sigma0_floor: np.ndarray | None = None,
                    copula_rows: np.ndarray | None = None) -> np.ndarray:
    """FHS + EVT 꼬리 + t-코퓰라 경로 (n_paths x N, horizon일 합계)."""
    vol = ewma_vol(R, lam)
    Z = R / vol
    marg = [SemiParametricMarginal(Z[:, k], tail) for k in range(R.shape[1])]
    zc = Z if copula_rows is None else Z[copula_rows]
    C = t_copula_corr(zc[-min(len(zc), 2500):])               # 최근 약 10년으로 결합 추정 (계산 시간)
    L = np.linalg.cholesky(C)
    N = R.shape[1]
    sig2 = (lam * vol[-1] ** 2 + (1 - lam) * R[-1] ** 2)      # 내일의 변동성 예측
    if sigma0_floor is not None:
        sig2 = np.maximum(sig2, sigma0_floor ** 2)
    sig2 = np.tile(sig2, (n_paths, 1))
    total = np.zeros((n_paths, N))
    for _ in range(horizon):
        g = rng.standard_normal((n_paths, N)) @ L.T
        w = rng.chisquare(df, (n_paths, 1)) / df
        u = stats.t.cdf(g / np.sqrt(w), df)
        z = np.column_stack([marg[k].ppf(u[:, k]) for k in range(N)])
        r = z * np.sqrt(sig2)
        total += r
        sig2 = lam * sig2 + (1 - lam) * r ** 2
    return total


# ---------------------------------------------------------------- 국면 전환
def regime_switch_paths(R: np.ndarray, state: np.ndarray, start_state: int, horizon: int, n_paths: int,
                        rng: np.random.Generator) -> np.ndarray:
    """state: 0=평상시, 1=위기 (일별). 전이확률은 과거 국면 순서에서 센다."""
    s0, s1 = state[:-1], state[1:]
    P = np.zeros((2, 2))
    for a in (0, 1):
        n = (s0 == a).sum()
        P[a, 1] = ((s0 == a) & (s1 == 1)).sum() / n if n else float(a)
        P[a, 0] = 1 - P[a, 1]
    days = [np.flatnonzero(state == 0), np.flatnonzero(state == 1)]
    if len(days[1]) == 0:
        days[1] = days[0]
    cur = np.full(n_paths, start_state)
    total = np.zeros((n_paths, R.shape[1]))
    for _ in range(horizon):
        for s in (0, 1):
            idx = np.flatnonzero(cur == s)
            if len(idx):
                total[idx] += R[rng.choice(days[s], len(idx))]
        flip = rng.random(n_paths) < np.where(cur == 0, P[0, 1], P[1, 0])
        cur = np.where(flip, 1 - cur, cur)
    return total


# ---------------------------------------------------------------- 실행
def run(rets: pd.DataFrame, weights: pd.Series, bshock: pd.DataFrame, base_bootstrap: np.ndarray,
        crisis_state: pd.Series, regime: str, cfg: dict, horizon: int, n_paths: int, var_level: float,
        es_level: float, seed: int) -> dict:
    """bshock: 바스켓 충격 (행 바스켓, 열 mild/central/severe). base_bootstrap: 기존 부트스트랩 경로(국면 표본)."""
    rng = np.random.default_rng(seed + 7)
    R = rets.to_numpy()
    w = weights.reindex(rets.columns).to_numpy() / weights.sum()
    state = crisis_state.reindex(rets.index).fillna(False).to_numpy().astype(int)
    mw = {m: float(v) for m, v in (cfg.get("weights") or {m: 1.0 for m in MODELS}).items() if m in MODELS and v > 0}
    paths = {}
    if "bootstrap" in mw:
        paths["bootstrap"] = base_bootstrap
    if "fhs_evt_t" in mw:
        floor = None
        if regime == "crisis" and state.sum() > 30:          # 위기 국면 가정이면 변동성을 위기 날짜의 중앙값 이상으로
            floor = np.median(ewma_vol(R, cfg.get("ewma_lambda", 0.94))[state == 1], axis=0)
        paths["fhs_evt_t"] = fhs_evt_t_paths(R, horizon, n_paths, rng, cfg.get("ewma_lambda", 0.94),
                                             cfg.get("evt_tail", 0.05), cfg.get("t_df", 4), floor)
    if "regime_switch" in mw:
        paths["regime_switch"] = regime_switch_paths(R, state, 1 if regime == "crisis" else 0, horizon, n_paths, rng)

    models, mix = {}, {}
    for scen in ("central", "severe"):
        port_all, wt_all = [], []
        for m, P in paths.items():
            port = portfolio_return(P + bshock[scen].to_numpy(), w)
            r = var_es(port, var_level, es_level)
            models.setdefault(m, {})["ES" if scen == "central" else "ES_severe"] = r["ES"]
            if scen == "central":
                models[m]["VaR"] = r["VaR"]
                models[m]["ES_base"] = var_es(portfolio_return(P, w), var_level, es_level)["ES"]
            port_all.append(port)
            wt_all.append(np.full(len(port), mw[m] / len(port)))
        mix[scen] = weighted_var_es(np.concatenate(port_all), np.concatenate(wt_all), var_level, es_level)
    es = [v["ES"] for v in models.values()]
    es_sev = [v["ES_severe"] for v in models.values()]
    tot = sum(mw.values())
    return {
        "weights": {m: round(v / tot, 4) for m, v in mw.items()},
        "models": models,
        "ES": mix["central"]["ES"], "VaR": mix["central"]["VaR"], "ES_severe": mix["severe"]["ES"],
        "range": [min(es), max(es)], "range_severe": [min(es_sev), max(es_sev)],
        "conservative": max(es), "conservative_severe": max(es_sev),
        "settings": {"n_paths_per_model": n_paths, "t_df": cfg.get("t_df", 4), "evt_tail": cfg.get("evt_tail", 0.05),
                     "ewma_lambda": cfg.get("ewma_lambda", 0.94), "mixing": "경로 섞기"},
    }
