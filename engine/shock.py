"""충격 벡터 계산 (Shock Mapper가 쓰는 순수 계산, LLM 없음).

충격은 항상 세 시나리오로 만든다: mild(약), central(중앙), severe(강).
표 모양: 행 = 팩터 7개, 열 = [mild, central, severe]
"""
from __future__ import annotations

import pandas as pd

SCENARIOS = ["mild", "central", "severe"]


def channel_filter(event_factor_post: pd.DataFrame, channels: dict[str, int] | None,
                   clip_wrong_sign: bool = True) -> tuple[pd.DataFrame, list[tuple[str, str, float]]]:
    """대상 사건의 영향 채널만 남긴다.

    - 채널에 없는 팩터는 0으로 둔다. 유사 사건 기간에 우연히 함께 움직인 달러·금리 등이
      이번 사건의 충격으로 섞여 들어가는 것을 막는다.
    - clip_wrong_sign이면 채널 팩터라도 기대 방향과 반대로 움직인 값은 0으로 둔다.
      (스트레스테스트에서 사건이 자기 경로로 포트폴리오를 '도와주는' 경우는 쓰지 않는다)
    돌려주는 값: (걸러진 표, [(사건 id, 팩터, 원래 값)] 0으로 바꾼 반대 부호 목록)
    """
    out = pd.DataFrame(0.0, index=event_factor_post.index, columns=event_factor_post.columns)
    flipped = []
    if not channels:
        return event_factor_post.copy(), flipped
    for f, d in channels.items():
        if f not in out.columns:
            continue
        col = event_factor_post[f]
        if clip_wrong_sign:
            wrong = col * d < 0
            flipped += [(i, f, float(col[i])) for i in col.index[wrong]]
            col = col.where(~wrong, 0.0)
        out[f] = col
    return out, flipped


def from_analogs(event_factor_post: pd.DataFrame, analog_ids: list[str],
                 beta: pd.DataFrame | None = None, weights: pd.Series | None = None) -> pd.DataFrame:
    """유사 사건들의 실제 팩터 변동으로 충격 범위를 잡는다.
    (영향 채널만 쓰려면 먼저 channel_filter를 거친 표를 넣는다.)

    beta와 weights를 주면 (권장): 사건 단위로 묶어서 고른다.
      각 사건의 팩터 변동 전체를 포트폴리오에 넣어 손실을 계산하고,
      severe = 손실이 가장 큰 사건의 팩터 변동, mild = 가장 작은 사건의 팩터 변동, central = 팩터별 중앙값.
      실제로 함께 일어난 팩터 조합만 쓰므로, 서로 다른 사건의 극단값을 섞는 문제가 없다.
    beta가 없으면: 팩터마다 따로 mild = 절댓값 최소, central = 중앙값, severe = 절댓값 최대.
    """
    sub = event_factor_post.loc[analog_ids]
    if beta is not None and weights is not None:
        w = weights.reindex(beta.index).fillna(0.0)
        w = w / w.sum()
        sub0 = sub.fillna(0.0)                                # 데이터가 없던 팩터(오래된 사건)는 0으로 본다
        impact = (beta.reindex(columns=sub0.columns).to_numpy() @ sub0.to_numpy().T).T @ w.to_numpy()
        impact = pd.Series(impact, index=sub0.index)          # 사건별 포트폴리오 로그수익률(근사)
        return pd.DataFrame({
            "mild": sub0.loc[impact.idxmax()],
            "central": sub.median().fillna(0.0),              # 중앙값은 데이터가 있는 사건들로만
            "severe": sub0.loc[impact.idxmin()],
        })[SCENARIOS]
    sub = sub.fillna(0.0)
    out = {}
    for f in sub.columns:
        col = sub[f]
        out[f] = {
            "mild": col.loc[col.abs().idxmin()],
            "central": col.median(),
            "severe": col.loc[col.abs().idxmax()],
        }
    return pd.DataFrame(out).T[SCENARIOS]


def weighted_quantile(values, weights, q: float) -> float:
    """가중 분위수. 값을 정렬하고 누적 가중치가 q를 넘는 지점을 선형 보간한다."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    o = np.argsort(v)
    v, w = v[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()          # 각 점의 가중 중앙 위치
    return float(np.interp(q, cw, v))


def from_analogs_weighted(event_factor_post: pd.DataFrame, weights: dict[str, float], channels: dict[str, int],
                          quantiles: tuple[float, float, float] = (0.25, 0.5, 0.9)) -> pd.DataFrame:
    """유사 사건이 여러 건일 때: 유사도 가중 분포로 충격을 잡는다.

    영향 채널 팩터마다 '충격 방향으로 본 크기'(방향 x 값)의 가중 분위수를 구해
    mild = 25%, central = 50%, severe = 90% 로 쓴다. 채널 밖 팩터는 0.
    (사건 단위 선택보다 팩터 간 동시성은 덜 보존하지만, 표본이 많을 때 범위를 안정적으로 잡는다)
    """
    ids = list(weights)
    w = [weights[i] for i in ids]
    out = pd.DataFrame(0.0, index=event_factor_post.columns, columns=SCENARIOS)
    for f, d in channels.items():
        if f not in out.index:
            continue
        col = event_factor_post.loc[ids, f] * d
        ok = col.notna().to_numpy()                       # 오래된 사건에서 데이터가 없던 팩터는 빼고 계산
        if not ok.any():
            continue
        mag, ww = col.to_numpy()[ok], [x for x, k in zip(w, ok) if k]
        for name, q in zip(SCENARIOS, quantiles):
            out.loc[f, name] = d * weighted_quantile(mag, ww, q)
    return out


def apply_severe_floor(shock: pd.DataFrame, channels: dict[str, int], severity: int, q_up: pd.DataFrame,
                       q_down: pd.DataFrame, quantile_map: dict[int, float], level_offset: int = -1
                       ) -> tuple[pd.DataFrame, list[str]]:
    """severe가 강도 기준 분위수보다 약하면 그 분위수까지 끌어올린다 (비교 사건이 적어 범위가 좁아지는 것 방지).

    기준 = 강도(severity + level_offset)에 해당하는 과거 3거래일 변동 분위수. 기본은 한 단계 아래 강도.
    돌려주는 값: (보정된 표, 끌어올린 팩터 목록)
    """
    out = shock.copy()
    lvl = min(max(severity + level_offset, 1), 5)
    q = quantile_map[lvl]
    raised = []
    for f, d in channels.items():
        if f not in out.index:
            continue
        table = q_up if d > 0 else q_down
        floor = float(table.loc[f, _closest(table.columns, q)])
        if d * out.loc[f, "severe"] < d * floor:
            out.loc[f, "severe"] = floor
            raised.append(f)
    return out, raised


def from_severity(channels: dict[str, int], severity: int, q_up: pd.DataFrame, q_down: pd.DataFrame,
                  quantile_map: dict[int, float], factors: list[str]) -> pd.DataFrame:
    """유사 사건이 없을 때(synthetic): 영향 채널과 강도만으로 충격을 만든다.

    채널에 있는 팩터만 움직이고, 방향(+1/-1)에 맞는 꼬리의 분위수를 쓴다.
    mild = 강도-1, central = 강도, severe = 강도+1 수준 (1~5로 제한).
    """
    def level(s: int) -> float:
        s = min(max(s, 1), 5)
        return quantile_map[s]

    out = pd.DataFrame(0.0, index=factors, columns=SCENARIOS)
    for f, direction in channels.items():
        table = q_up if direction > 0 else q_down
        for name, s in zip(SCENARIOS, (severity - 1, severity, severity + 1)):
            out.loc[f, name] = table.loc[f, _closest(table.columns, level(s))]
    return out


def _closest(cols, q: float):
    """분위수 표의 열 이름(문자열일 수도 있음) 중 q에 가장 가까운 것을 찾는다."""
    return min(cols, key=lambda c: abs(float(c) - q))


def apply_overrides(shock: pd.DataFrame, overrides: dict | None) -> pd.DataFrame:
    """What-if: {"oil": {"mult": 2}} 또는 {"oil": {"abs": 0.15}} 형태로 해당 팩터만 덮어쓴다."""
    if not overrides:
        return shock
    out = shock.copy()
    for f, rule in overrides.items():
        if f not in out.index:
            raise KeyError(f"알 수 없는 팩터: {f}")
        if "mult" in rule:
            out.loc[f] = out.loc[f] * rule["mult"]
        elif "abs" in rule:
            out.loc[f] = rule["abs"]
        else:
            raise ValueError(f"override 형식 오류: {rule}")
    return out


def idio_shock(exposure: dict[str, str], event_resid_post: pd.DataFrame, analog_ids: list[str],
               baskets: list[str]) -> pd.DataFrame:
    """직접 노출이 'high'인 바스켓에 베타로 설명되지 않는 고유충격을 더한다.

    잔차(실제 - B x f)에는 팩터로 설명되지 않은 '시장 전체 흐름'도 섞여 있다(팩터에 주식시장 전체 팩터가 없음).
    그래서 사건마다 12개 바스켓 잔차의 중앙값을 빼서, '다른 바스켓보다 더 맞은 정도'만 고유충격으로 쓴다.
    크기 = 유사 사건들에서 그 바스켓의 보정 잔차 (mild/central/severe = 절댓값 최소/중앙/최대).
    유사 사건이 없으면 0.
    """
    out = pd.DataFrame(0.0, index=baskets, columns=SCENARIOS)
    analog_ids = [i for i in analog_ids if i in event_resid_post.index]   # 바스켓 데이터 이전 사건은 잔차 없음
    if not analog_ids:
        return out
    sub = event_resid_post.loc[analog_ids].dropna()
    if sub.empty:
        return out
    sub = sub.sub(sub.median(axis=1), axis=0)          # 횡단면 보정
    for b, lvl in exposure.items():
        if lvl != "high" or b not in sub.columns:
            continue
        col = sub[b]
        out.loc[b] = [col.loc[col.abs().idxmin()], col.median(), col.loc[col.abs().idxmax()]]
    return out
