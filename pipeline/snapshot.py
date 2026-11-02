"""데이터 사본: python -m pipeline.snapshot

웹 배포(Streamlit Cloud)에서는 시장 데이터 다운로드가 막히거나 느릴 수 있다.
그래서 내 PC에서 만든 결과물(data/processed, data/model)을 data/snapshot/에 복사해 GitHub에 같이 올리고,
앱은 데이터가 없을 때 이 사본을 먼저 쓴다. 크기는 1MB 안팎.

  python -m pipeline.snapshot            # 현재 결과물로 사본 만들기 (run_all 후 실행)
  python -m pipeline.snapshot --restore  # 사본을 data/processed, data/model 로 되돌리기
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime

import pandas as pd

from . import config

SNAP_DIR = config.DATA_DIR / "snapshot"
FILES = {
    "processed": ["returns.parquet", "factor_returns.parquet", "factor_levels.parquet", "factor_returns_long.parquet"],
    "model": ["beta_normal.csv", "beta_crisis.csv", "beta_r2.csv", "factor_cov_crisis.csv",
              "event_factor_post.csv", "event_factor_pre.csv", "event_basket_post.csv", "event_resid_post.csv",
              "event_factor_react.csv", "event_basket_react.csv", "event_resid_react.csv",
              "factor_quantiles_up.csv", "factor_quantiles_down.csv",
              "factor_quantiles_react_up.csv", "factor_quantiles_react_down.csv"],
}
SRC = {"processed": config.PROCESSED_DIR, "model": config.MODEL_DIR}


def exists() -> bool:
    return (SNAP_DIR / "meta.json").exists()


def meta() -> dict:
    return json.loads((SNAP_DIR / "meta.json").read_text(encoding="utf-8")) if exists() else {}


def make() -> dict:
    total, copied = 0, []
    for sub, names in FILES.items():
        (SNAP_DIR / sub).mkdir(parents=True, exist_ok=True)
        for n in names:
            src = SRC[sub] / n
            if src.exists():
                shutil.copy2(src, SNAP_DIR / sub / n)
                total += src.stat().st_size
                copied.append(f"{sub}/{n}")
    rets = pd.read_parquet(config.PROCESSED_DIR / "returns.parquet")
    m = {"created_at": datetime.now().isoformat(timespec="seconds"), "data_version": str(rets.index.max().date()),
         "files": copied, "bytes": total}
    (SNAP_DIR / "meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    return m


def restore() -> dict:
    if not exists():
        raise FileNotFoundError("data/snapshot 이 없습니다. 먼저 python -m pipeline.snapshot")
    config.ensure_dirs()
    for sub, names in FILES.items():
        for n in names:
            src = SNAP_DIR / sub / n
            if src.exists():
                shutil.copy2(src, SRC[sub] / n)
    return meta()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true")
    a = ap.parse_args()
    if a.restore:
        m = restore()
        print(f"[snapshot] 사본 복원 완료 (데이터 기준일 {m.get('data_version')})")
    else:
        m = make()
        print(f"[snapshot] 사본 생성: {len(m['files'])}개 파일, {m['bytes'] / 1e6:.2f}MB, 데이터 기준일 {m['data_version']}")
        print("[snapshot] data/snapshot 폴더를 GitHub에 함께 올리면 웹 데모가 다운로드 없이 바로 뜹니다.")


if __name__ == "__main__":
    main()
