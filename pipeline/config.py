"""설정 파일과 데이터 경로를 한 곳에서 관리한다."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "model"


def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def portfolio() -> dict:
    return load_yaml("portfolio.yaml")


def factors() -> dict:
    return load_yaml("factors.yaml")


def thresholds() -> dict:
    return load_yaml("thresholds.yaml")


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
