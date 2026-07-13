#!/usr/bin/env python3
"""Lightweight regression test for PlateVision person history logic.

The test extracts PersonHistoryManager from app.py, so Flask, Ultralytics and
model files are not imported or executed.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "platevision" / "src" / "app.py"


class _Config:
    def __init__(self):
        self.data = {
            "people": {
                "history_enabled": True,
                "person_recount_block_enabled": True,
                "person_recount_block_minutes": 15,
                "person_recount_identity_mode": "track_or_position",
                "person_recount_position_tolerance_percent": 12,
                "session_gap_minutes": 5,
                "present_timeout_minutes": 10,
                "image_history_enabled": False,
                "image_history_retention_days": 10,
                "image_history_auto_cleanup_enabled": False,
                "retention_days": 30,
            },
            "general": {"max_history_entries": 1000},
            "dashboard": {"default_range_days": 7},
        }

    def get(self, *args):
        if len(args) == 1:
            return self.data.get(args[0])
        return self.data.get(args[0], {}).get(args[1])


def _load_json(path, default):
    target = Path(path)
    return json.loads(target.read_text(encoding="utf-8")) if target.exists() else json.loads(json.dumps(default))


def _write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")


def _load_manager_class():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == "PersonHistoryManager")
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {
        "Path": Path,
        "datetime": datetime,
        "timedelta": timedelta,
        "threading": threading,
        "time": time,
        "uuid": uuid,
        "hashlib": hashlib,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "cv2": object(),
        "logger": logging.getLogger("people-history-test"),
        "config_manager": _Config(),
        "_load_json_with_backup": _load_json,
        "_atomic_write_json": _write_json,
    }
    exec(compile(ast.fix_missing_locations(module), str(APP_PATH), "exec"), namespace)
    return namespace["PersonHistoryManager"]


def main():
    manager_class = _load_manager_class()
    previous_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="platevision-people-test-") as temp_dir:
        os.chdir(temp_dir)
        try:
            manager = manager_class()
            base = datetime.now() - timedelta(minutes=20)
            events = [
                {"id": "e1", "timestamp": base.isoformat(), "source": "rtsp", "event_type": "appearance", "direction": "down", "counted": False, "confidence": 0.8, "track_id": 1, "bbox": [0, 0, 10, 20], "frame_width": 100, "frame_height": 100},
                {"id": "e2", "timestamp": (base + timedelta(seconds=30)).isoformat(), "source": "rtsp", "event_type": "line_crossing", "direction": "down", "counted": True, "confidence": 0.9, "track_id": 1, "bbox": [0, 0, 10, 20], "frame_width": 100, "frame_height": 100},
                {"id": "e3", "timestamp": (base + timedelta(minutes=10)).isoformat(), "source": "rtsp", "event_type": "line_crossing", "direction": "up", "counted": True, "confidence": 0.7, "track_id": 1, "bbox": [20, 0, 30, 20], "frame_width": 100, "frame_height": 100},
            ]
            for event in events:
                manager.add_event(event)

            page = manager.paginated_history({"days": 1, "counted": "all", "limit": 2})
            assert page["total"] == 3 and len(page["entries"]) == 2 and page["has_more"]

            sessions = manager.get_sessions({"days": 1, "counted": "all", "limit": 10})
            assert sessions["total"] == 2

            statistics = manager.get_statistics({"days": 1})["summary"]
            assert statistics["events"] == 3
            assert statistics["total_persons"] == 1
            assert statistics["repeat_blocked"] == 1

            updated = manager.update_event("e2", {"label": "Eingang", "note": "Geprüft", "review_status": "confirmed"})
            assert updated and updated["review_status"] == "confirmed"

            # Calendar archive groups only events that still have image metadata.
            manager.history[0]["images"] = {"crop": "crops/test/e3.jpg"}
            manager.history[1]["images"] = {"crop": "crops/test/e2.jpg"}
            image_days = manager.image_days({"days": 3650})
            assert image_days["total_events_with_images"] == 2
            assert image_days["total_days"] >= 1

            # Old image files are removed while newer statistics and image files stay.
            old_file = manager.IMAGE_ROOT / "crops" / "old" / "old.jpg"
            new_file = manager.IMAGE_ROOT / "crops" / "new" / "new.jpg"
            old_file.parent.mkdir(parents=True, exist_ok=True)
            new_file.parent.mkdir(parents=True, exist_ok=True)
            old_file.write_bytes(b"old")
            new_file.write_bytes(b"new")
            manager.history.append({"id": "old-image", "timestamp": (datetime.now()-timedelta(days=15)).isoformat(), "counted": False, "images": {"crop": "crops/old/old.jpg"}})
            manager.history.append({"id": "new-image", "timestamp": (datetime.now()-timedelta(days=2)).isoformat(), "counted": False, "images": {"crop": "crops/new/new.jpg"}})
            cleanup = manager.cleanup_images(10, delete_orphan_files=False, delete_records=False)
            assert cleanup["deleted_images"] == 1
            assert not old_file.exists() and new_file.exists()
            assert next(x for x in manager.history if x["id"] == "old-image")["images"] == {}

            deleted = manager.bulk_delete(["e1", "e3"])
            assert deleted["deleted"] == 2 and len(manager.history) == 3
        finally:
            os.chdir(previous_cwd)

    print("PlateVision person history tests: OK")


if __name__ == "__main__":
    main()
