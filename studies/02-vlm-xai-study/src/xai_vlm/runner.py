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
import hashlib
import json
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


def manifest_sha256(path: Path) -> str:
    """Content hash of the manifest, read in binary so line endings count."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class FingerprintMismatch(RuntimeError):
    """The run on disk was produced under a different configuration.

    Raised rather than warned about. A mismatch means the rows already written and
    the rows about to be written answer different questions, and once concatenated
    nothing downstream can separate them again.
    """


class RunState:
    """The sidecar recording what configuration produced a results file."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))

    def verify_or_create(self, fingerprint: dict, force: bool = False) -> None:
        existing = self.load()
        if existing is None:
            self._write(dict(fingerprint))
            return

        diffs = [
            f"  {k}: on disk {existing.get(k)!r} -> now {v!r}"
            for k, v in fingerprint.items()
            if existing.get(k) != v
        ]
        if diffs and not force:
            raise FingerprintMismatch(
                "this results file was produced under a different configuration:\n"
                + "\n".join(diffs)
                + "\n\nResuming would concatenate rows answering different questions.\n"
                "Start a fresh output file, or pass --force-resume if you are "
                "certain the difference does not affect the measurement."
            )
        if diffs:
            merged = dict(fingerprint)
            merged["forced"] = True
            self._write(merged)

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
