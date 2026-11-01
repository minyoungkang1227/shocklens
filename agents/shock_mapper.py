"""Shock Mapper: 사건 -> 팩터 충격(mild/central/severe) + 직접 노출 고유충격 (LLM 없음).

강화 포인트
- 경로별 산정: direct/relaxed = 유사 사건의 실제 3거래일 반응, synthetic = 강도 -> 분위수 표.
- 영향 채널 필터: 사건과 무관한 팩터는 0, 기대와 반대 부호는 0 (설정으로 끌 수 있음).
- 스스로 점검해 경고를 남긴다:
  · 충격이 과거 사건 최대치의 2배 초과 (이상치)
  · 유사 사건 1건뿐 -> 범위(mild~severe)가 한 점
  · synthetic 경로 -> 과거 실측 근거 없음
  · 반대 부호로 0 처리한 값이 있음
- What-if: overrides({"oil": {"mult": 2}})를 마지막에 덮어쓴다.
"""
from __future__ import annotations

import pandas as pd

from engine import shock

from .context import Context


def map_shock(event: dict, analog_ids: list[str], tier: str, exposure: dict, ctx: Context,
              overrides: dict | None = None) -> dict:
    th = ctx.thresholds
    F = ctx.f_event
    regime = "crisis" if event["severity"] >= th["severity"]["crisis_regime_min"] else "normal"
    beta = ctx.beta_crisis if regime == "crisis" else ctx.beta_normal
    warnings, flipped = [], []

    if analog_ids:
        F_ch, flipped = shock.channel_filter(F, event["channels"], th["analog"].get("clip_wrong_sign", True))
        flipped = [x for x in flipped if x[0] in analog_ids]
        f_shock = shock.from_analogs(F_ch, analog_ids, beta, ctx.weights)
        if len(analog_ids) == 1:
            warnings.append("유사 사건이 1건뿐이라 mild·central·severe가 같습니다(범위 추정 불가).")
        if flipped:
            warnings.append(f"유사 사건에서 기대와 반대로 움직인 값 {len(flipped)}개를 0으로 처리했습니다.")
    else:
        f_shock = shock.from_severity(event["channels"], event["severity"], ctx.q_up, ctx.q_down,
                                      th["severity"]["quantile_map"], list(F.columns))
        warnings.append("과거 유사 사건이 없어 강도 기반 분위수로 충격을 합성했습니다(synthetic).")

    if overrides:
        for f, rule in overrides.items():
            if "mult" in rule and f in f_shock.index and (f_shock.loc[f] == 0).all():
                warnings.append(f"{ctx.labels.get(f, f)}는 이 사건의 영향 채널이 아니어서 기준 충격이 0입니다. "
                                f"배수 가정은 효과가 없으니 절댓값으로 지정하세요.")
        f_shock = shock.apply_overrides(f_shock, overrides)
        warnings.append(f"사용자 가정 반영(What-if): {overrides}")

    # 이상치 점검: 과거 모든 사건의 3거래일 반응 절댓값 최대 대비
    hist_max = F.abs().max()
    mult = th["hq"]["shock_outlier_multiple"]
    for f in f_shock.index:
        c = abs(float(f_shock.loc[f, "central"]))
        if hist_max.get(f, 0) > 0 and c > mult * hist_max[f]:
            warnings.append(f"{ctx.labels.get(f, f)} 충격이 과거 사건 최대치의 {mult:g}배를 넘습니다(central {c:.4f}).")

    idio = shock.idio_shock(exposure, ctx.resid_event, analog_ids, list(ctx.rets.columns))
    return {
        "factor_shock": f_shock.to_dict(orient="index"),
        "idio_shock": idio.to_dict(orient="index"),
        "beta_regime": regime,
        "shock_warnings": warnings,
        "zeroed_wrong_sign": flipped,
    }


def to_frame(d: dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(d, orient="index")[shock.SCENARIOS]
