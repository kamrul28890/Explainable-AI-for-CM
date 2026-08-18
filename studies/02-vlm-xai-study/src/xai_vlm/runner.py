"""Resumable stage execution: checkpointing, fingerprinting, guard rails.

Stage 2 runs for hours. A crash at hour eight without resumability costs a day --
but the worse failure is quieter: half a run at one configuration concatenated with
half at another produces a complete-looking CSV that no downstream metric can tell
is wrong. This module exists to make the first cheap and the second impossible.

Deliberately free of torch and transformers. Every behaviour here is exercised on
CPU against FakeBackend, because logic that can only be tested on the one machine
with a GPU is logic that does not get tested.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

# (image_id, rule_id) -- the natural key of a manifest row.
#
# Resume is keyed on this, never on row position. Images arrive in dataset
# streaming order rather than manifest order, so a positional cursor would desync
# the first time iteration changed, and would do so silently.
RowKey = tuple[str, str]


class CheckpointWriter:
    """Append-only CSV writer whose rows are on disk before the next call begins.

    Buffering would lose up to a bufferful on a crash, and rewriting the whole file
    periodically risks truncating it mid-write. Appending one row and flushing to
    the platform costs a few milliseconds against a ~3 s model call.
    """

    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self._fh = None
        self._writer = None

    def done_keys(self) -> set[RowKey]:
        """Keys already recorded, read from the file on disk."""
        if not self.path.exists():
            return set()
        with self.path.open(newline="", encoding="utf-8") as fh:
            return {
                (r["image_id"], r["rule_id"])
                for r in csv.DictReader(fh)
                if r.get("image_id")
            }

    def _open(self) -> None:
        if self._fh is not None:
            return
        is_new = not self.path.exists() or self.path.stat().st_size == 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames)
        if is_new:
            self._writer.writeheader()

    def append(self, row: dict) -> None:
        self._open()
        self._writer.writerow({k: row.get(k, "") for k in self.fieldnames})
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._writer = None
