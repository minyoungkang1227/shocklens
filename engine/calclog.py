"""계산 장부: Risk Engine이 계산할 때마다 입력·출력·시드를 JSON으로 남기고 번호(calc_log_id)를 붙인다.
Verifier는 브리핑의 숫자를 이 장부와만 대조한다.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

LOG_DIR = Path(__file__).resolve().parents[1] / "data" / "calc_logs"


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (pd.DataFrame, pd.Series)):
        return o.to_dict()
    raise TypeError(f"직렬화할 수 없는 타입: {type(o)}")


def save(inputs: dict, outputs: dict, data_version: str, log_dir: Path = LOG_DIR) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    body = {"inputs": inputs, "outputs": outputs, "data_version": data_version}
    text = json.dumps(body, ensure_ascii=False, default=_default, sort_keys=True)
    digest = hashlib.sha256(text.encode()).hexdigest()[:6]
    calc_id = f"calc_{datetime.now():%Y%m%d_%H%M%S}_{digest}"
    record = {"calc_log_id": calc_id, "created_at": datetime.now().isoformat(timespec="seconds"), **body}
    (log_dir / f"{calc_id}.json").write_text(
        json.dumps(record, ensure_ascii=False, default=_default, indent=2), encoding="utf-8")
    return calc_id


def load(calc_id: str, log_dir: Path = LOG_DIR) -> dict:
    return json.loads((log_dir / f"{calc_id}.json").read_text(encoding="utf-8"))


def numbers(obj) -> list[float]:
    """장부 안의 모든 숫자를 평평하게 꺼낸다 (Verifier의 숫자 대조용)."""
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))
    return out
