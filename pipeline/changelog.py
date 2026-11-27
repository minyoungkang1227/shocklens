"""사건 라이브러리의 손 라벨 변경 기록: python -m pipeline.changelog add --event ID --by 이름 --reason "사유"

events.yaml의 강도 6요소·채널·clean 등은 사람이 손으로 매긴다(BCBS 239 원칙 3: 수작업은 사유를 남긴다).
라벨을 고치면 changes.yaml에 언제·누가·왜를 남겨야 하고, 테스트(tests/test_changelog.py)가 기록 없는 변경을 막는다.

  python -m pipeline.changelog check                       # 기록 없이 바뀐 사건 목록
  python -m pipeline.changelog add --event us_301_2018 --by 강민영 --reason "확전단계 1→2: 보복 예고 반영"
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import yaml

from . import config

EVENTS_FILE = config.DATA_DIR / "library" / "events.yaml"
CHANGES_FILE = config.DATA_DIR / "library" / "changes.yaml"
# 손으로 매기고 분석 결과를 바꾸는 필드. 설명문·출처 문구 수정은 여기에 넣지 않는다.
LABEL_FIELDS = ("date", "react_from", "instrument", "direction", "stage", "initiator", "target", "sectors",
                "severity_components", "channels", "clean", "confounder_type")


def fingerprint(event: dict) -> str:
    core = {k: event.get(k) for k in LABEL_FIELDS}
    return hashlib.sha256(json.dumps(core, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]


def load(path: Path = CHANGES_FILE) -> list[dict]:
    if not path.exists():
        return []
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("changes", [])


def unlogged(events_path: Path = EVENTS_FILE, changes_path: Path = CHANGES_FILE) -> list[str]:
    """현재 라벨이 마지막 기록과 다른(또는 기록이 아예 없는) 사건 id."""
    latest = {}
    for c in load(changes_path):
        latest[c["event"]] = c["fingerprint"]
    events = yaml.safe_load(events_path.read_text(encoding="utf-8"))["events"]
    return [e["id"] for e in events if latest.get(e["id"]) != fingerprint(e)]


def add(event_id: str, by: str, reason: str, events_path: Path = EVENTS_FILE, changes_path: Path = CHANGES_FILE) -> dict:
    if not reason.strip():
        raise ValueError("사유(reason)가 비어 있다")
    events = {e["id"]: e for e in yaml.safe_load(events_path.read_text(encoding="utf-8"))["events"]}
    if event_id not in events:
        raise ValueError(f"사건 없음: {event_id}")
    row = {"event": event_id, "date": datetime.date.today().isoformat(), "by": by, "reason": reason,
           "fingerprint": fingerprint(events[event_id])}
    with open(changes_path, "a", encoding="utf-8") as f:
        f.write("".join("  " + line + "\n" for line in yaml.safe_dump([row], allow_unicode=True, sort_keys=False).splitlines()))
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline.changelog")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    a_ = sub.add_parser("add")
    a_.add_argument("--event", required=True)
    a_.add_argument("--by", required=True)
    a_.add_argument("--reason", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        bad = unlogged()
        print("모든 라벨 변경에 기록이 있다" if not bad else "기록 없이 바뀐 사건: " + ", ".join(bad))
        return 1 if bad else 0
    print(add(a.event, a.by, a.reason))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
