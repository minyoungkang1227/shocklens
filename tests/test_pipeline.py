"""네트워크 없이 합성 데이터로 파이프라인 계산 로직을 검증한다."""
import numpy as np
import pandas as pd
import pytest

from pipeline import betas, returns
from pipeline.fetch import parse_fred_csv

BASKETS = {
    "a": {"ticker": "AAA", "weight": 60},
    "b": {"ticker": "BBB", "weight": 40},
}
FACTORS = {
    "vol": {"source": "fred", "series": "VIXCLS", "kind": "level"},
    "fx": {"source": "fred", "series": "FX", "kind": "price"},
    "spread": {"source": "etf", "long": "AAA", "short": "BBB", "kind": "spread_ret"},
}


def test_parse_fred_csv_handles_missing_and_date_column():
    text = "observation_date,VIXCLS\n2024-01-02,13.2\n2024-01-03,.\n2024-01-04,14.0\n"
    s = parse_fred_csv(text, "VIXCLS")
    assert list(s.values) == [13.2, 14.0]
    assert s.index[0] == pd.Timestamp("2024-01-02")


def test_returns_kinds_and_alignment():
    idx = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"AAA": [100, 110, 121, 121], "BBB": [50, 50, 55, 55]}, index=idx)
    # FRED는 하루 빠진 날이 있다고 가정 -> 직전 값으로 채워져야 함
    # 시리즈마다 따로 받아 합치므로 VIX 칸만 비고 위안화 값은 남아 있다
    fred = pd.DataFrame({"VIXCLS": [20.0, 22.0, np.nan, 25.0], "FX": [7.0, 7.0, 7.7, 7.7]}, index=idx)
    r, f, lv = returns.build(prices, fred, BASKETS, FACTORS)

    assert r.loc[idx[1], "a"] == pytest.approx(np.log(1.1))
    assert f.loc[idx[1], "vol"] == pytest.approx(2.0)          # level -> 차분
    assert f.loc[idx[2], "vol"] == pytest.approx(0.0)          # 결측일은 직전값(22)으로 채움
    assert f.loc[idx[2], "fx"] == pytest.approx(np.log(1.1))   # price -> 로그수익률
    assert f.loc[idx[2], "spread"] == pytest.approx(np.log(1.1) - np.log(1.1))
    assert f.loc[idx[1], "spread"] == pytest.approx(np.log(1.1) - 0.0)


def test_ridge_recovers_known_betas():
    rng = np.random.default_rng(0)
    n = 5000
    X = pd.DataFrame(rng.normal(size=(n, 3)) * [1.0, 0.01, 0.5], columns=["f1", "f2", "f3"])
    true = np.array([[0.5, 30.0, -0.2], [-1.0, 0.0, 0.8]])
    Y = pd.DataFrame(X.to_numpy() @ true.T + rng.normal(scale=0.05, size=(n, 2)), columns=["a", "b"])
    B, r2 = betas.ridge_betas(Y, X, alpha=1e-6)
    np.testing.assert_allclose(B.to_numpy(), true, rtol=0.05, atol=0.05)
    assert (r2 > 0.9).all()


def test_ridge_shrinks_with_larger_alpha():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(500, 2)), columns=["f1", "f2"])
    Y = pd.DataFrame({"a": 2 * X["f1"] + rng.normal(scale=0.1, size=500)})
    small, _ = betas.ridge_betas(Y, X, alpha=1e-6)
    big, _ = betas.ridge_betas(Y, X, alpha=1.0)
    assert abs(big.loc["a", "f1"]) < abs(small.loc["a", "f1"])


def test_crisis_mask_picks_top_quantile():
    vix = pd.Series(np.arange(100, dtype=float))
    m = betas.crisis_mask(vix, 0.8)
    assert m.sum() == 20
    assert m.iloc[-1] and not m.iloc[0]
