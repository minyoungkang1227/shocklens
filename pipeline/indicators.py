"""위기·지정학 보조 지수 수집 (2026-11-13): OFR 금융스트레스지수, AI 지정학 리스크 지수(AI-GPR).

    python -m pipeline.indicators          # data/raw/stress_indicators.parquet 저장

- 두 지수 모두 CSV로 받는다. 열 이름이 공개돼 있지 않거나 바뀔 수 있어, 날짜 열은 자동으로 찾고 지수 열은
  config/indicators.yaml 의 value_patterns(정규식)로 고른다. 고른 열 이름은 출력으로 남긴다.
- 실패해도 예외를 밖으로 던지지 않는다(국면 판단은 VIX만으로 계속). 무엇이 실패했는지는 반환값·출력으로 남긴다.
- 외부 응답 크기 상한은 tools.news.get_capped 를 쓴다(보안 수정 c6b159a와 같은 기준).

저장 열: OFR_FSI, OFR_FSI_<구성요소>..., AI_GPR, AI_GPR_<하위 지수>...
"""
from __future__ import annotations

import io
import re

import pandas as pd
import yaml

from . import config

OUT = config.RAW_DIR / "stress_indicators.parquet"


def load_cfg() -> dict:
    with open(config.CONFIG_DIR / "indicators.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _date_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if re.search(r"date|day|time", str(c), re.I):
            return c
    c = df.columns[0]                                      # 이름이 없으면 첫 열이 날짜로 읽히는지 본다
    ok = pd.to_datetime(df[c], errors="coerce").notna().mean()
    return c if ok > 0.9 else None


def _pick(cols: list[str], patterns: list[str]) -> str | None:
    for p in patterns:
        for c in cols:
            if re.search(p, str(c).strip(), re.I):
                return c
    return None


def parse_csv(raw: bytes, prefix: str, value_patterns: list[str], component_patterns: list[str] | None = None,
              keep_all_numeric: bool = False) -> tuple[pd.DataFrame, dict]:
    """CSV 바이트 -> (날짜 인덱스 DataFrame, 메모). 메모에 고른 열 이름이 남는다."""
    df = pd.read_csv(io.BytesIO(raw))
    dc = _date_col(df)
    if dc is None:
        raise ValueError(f"날짜 열을 찾지 못함: {list(df.columns)[:8]}")
    df.index = pd.to_datetime(df[dc], errors="coerce")
    df = df[df.index.notna()].drop(columns=[dc]).sort_index()
    num = df.apply(pd.to_numeric, errors="coerce")
    num = num.loc[:, num.notna().mean() > 0.5]
    if num.empty:
        raise ValueError("숫자 열이 없음")
    main = _pick(list(num.columns), value_patterns) or num.columns[0]
    out = pd.DataFrame({prefix: num[main]})
    comps = []
    for c in num.columns:
        if c == main:
            continue
        if keep_all_numeric or (component_patterns and _pick([c], component_patterns)):
            name = f"{prefix}_{re.sub(r'[^A-Za-z0-9]+', '_', str(c)).strip('_').upper()}"
            out[name] = num[c]
            comps.append(str(c))
    out = out[~out.index.duplicated(keep="last")]
    return out, {"value_column": str(main), "components": comps, "rows": int(out[prefix].notna().sum()),
                 "first": str(out.index.min().date()), "last": str(out.index.max().date())}


def fetch_one(spec: dict, prefix: str, getter=None, keep_all_numeric: bool = False) -> tuple[pd.DataFrame | None, dict]:
    if getter is None:
        from tools.news import get_capped
        getter = lambda u: get_capped(u, timeout=30, max_bytes=50_000_000)       # noqa: E731
    errors = []
    for url in spec.get("urls", []):
        try:
            df, memo = parse_csv(getter(url), prefix, spec.get("value_patterns", []),
                                 spec.get("component_patterns"), keep_all_numeric)
            return df, {**memo, "url": url}
        except Exception as e:
            errors.append(f"{url}: {type(e).__name__}: {str(e)[:80]}")
    return None, {"error": "; ".join(errors) or "주소 없음"}


def run(getter=None, out_path=None) -> dict:
    cfg = load_cfg()
    frames, report = [], {}
    for key, prefix, keep in (("ofr_fsi", "OFR_FSI", True), ("ai_gpr", "AI_GPR", False)):
        if key not in cfg:
            continue
        df, memo = fetch_one(cfg[key], prefix, getter, keep_all_numeric=keep)   # OFR은 구성요소(분야·지역)를 모두 남긴다
        report[key] = memo
        if df is not None:
            frames.append(df)
        print(f"[indicators] {prefix}: " + (f"{memo['rows']}일 ({memo['first']}~{memo['last']}), 열 '{memo['value_column']}'"
                                           if df is not None else f"실패 -> {memo['error']}"))
    if frames:
        out = pd.concat(frames, axis=1).sort_index()
        path = out_path or OUT
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():                                   # 한쪽만 실패하면 이전에 받아 둔 열은 유지
            old = pd.read_parquet(path)
            for c in old.columns:
                if c not in out.columns:
                    out[c] = old[c]
        out.to_parquet(path)
        report["saved"] = str(path)
    return report


def load(path=None) -> pd.DataFrame | None:
    path = path or OUT
    return pd.read_parquet(path) if path.exists() else None


if __name__ == "__main__":
    run()
