from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from filelock import FileLock


class JsonStorageClient:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock_path = Path(f"{self.path}.lock")

    def ensure_exists(self, default: dict[str, Any] | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            return
        with FileLock(str(self.lock_path)):
            if not self.path.exists():
                self.path.write_text(
                    json.dumps(default or {}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    def load(self) -> dict[str, Any]:
        self.ensure_exists(default={})
        with FileLock(str(self.lock_path)):
            return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock_path)):
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
