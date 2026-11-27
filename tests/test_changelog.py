"""사건 라이브러리의 손 라벨은 기록 없이 바뀌면 안 된다(BCBS 239 원칙 3)."""
import shutil

import yaml

from pipeline import changelog as C


def test_every_event_label_is_logged():
    bad = C.unlogged()
    assert bad == [], f"기록 없이 바뀐 사건: {bad}. python -m pipeline.changelog add --event ID --by 이름 --reason 사유"


def test_change_without_reason_is_caught_then_logged(tmp_path):
    ev, ch = tmp_path / "events.yaml", tmp_path / "changes.yaml"
    shutil.copy(C.EVENTS_FILE, ev)
    shutil.copy(C.CHANGES_FILE, ch)
    data = yaml.safe_load(ev.read_text(encoding="utf-8"))
    target = data["events"][0]
    target["severity_components"]["surprise"] = 5 if target["severity_components"]["surprise"] != 5 else 1
    ev.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    assert C.unlogged(ev, ch) == [target["id"]]
    C.add(target["id"], "test", "예상밖 점수 조정", ev, ch)
    assert C.unlogged(ev, ch) == []
    assert C.load(ch)[-1]["reason"] == "예상밖 점수 조정"


def test_description_edit_does_not_need_log(tmp_path):
    ev = tmp_path / "events.yaml"
    data = yaml.safe_load(C.EVENTS_FILE.read_text(encoding="utf-8"))
    data["events"][0]["description"] = "오타 수정"
    ev.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    assert C.unlogged(ev, C.CHANGES_FILE) == []
