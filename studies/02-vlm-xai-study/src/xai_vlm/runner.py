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
import statistics
from pathlib import Path
from typing import Callable, Iterable, Optional

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


def backend_device(backend) -> str:
    """The device a backend computes on, for the run fingerprint.

    Greedy decoding reproduces on one machine and not across machines: the
    2026-11-09 parity run (identical weights, code, data and pinned libraries,
    RTX 3070 vs A40) agreed on 96.9% of verdicts and 42.7% of replies byte for
    byte. Rows from two devices must therefore never share a results file, and
    recording the device lets resume refuse to mix them.

    Read by attribute so this module stays torch-free.
    """
    return str(getattr(backend, "device_name", "unknown"))


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


# PRE-REGISTERED, fixed before any Stage 2 numbers were seen -- the same discipline
# the Stage 0 gate thresholds were set under. Not to be tuned to fit a run.
PARSE_RATE_FLOOR = 0.95

# CALIBRATED FROM MEASURED IN-RUN LATENCY, not from the Stage 0 budget.
#
# The original value was 12.0 s = 2 x LATENCY_BUDGET_S (6.0), and 6.0 was itself
# set against Stage 0's 2.9 s median. Every one of those numbers was measured on
# single images with nothing else running, and that context does not survive
# contact with a real run: the SAME images that answer in 3.1 s standalone take
# 10-11 s inside a run. Measured healthy medians are 9.5 s and 10.9 s.
#
# So a 12.0 s ceiling sat roughly 10% above the healthy baseline and would abort
# perfectly good runs on ordinary variation. 25.0 s is ~2.3x the measured baseline
# and still well under the 34.0 s degradation actually observed, so it keeps a
# real detector without firing on healthy work.
#
# The 3.5x standalone-vs-in-run gap is NOT understood. It is not VRAM (peak 6.97 GB
# against 8 GB, stable), not system RAM (18 GB free), not image size, not reply
# length. The leading unverified hypothesis is CPU contention between parquet
# decoding and Qwen's CPU-side image preprocessing, which is inside the timed call.
# Until that is settled this ceiling is empirical, not principled.
LATENCY_CEILING_S = 25.0

# Guards evaluate rows from the CURRENT PROCESS only, and not until this many exist.
# A resumed run is therefore never aborted by history it cannot change, and a short
# --limit smoke run never trips them.
GUARD_MIN_ROWS = 100


class GuardTripped(RuntimeError):
    """A pre-registered quality guard failed; the run stops."""


def _row_ok(out_row: dict) -> bool:
    """Stage 2's definition of a healthy row, and the default for run_stage."""
    return bool(out_row.get("verdict")) and not out_row.get("parse_error")


class Guards:
    """Running quality checks over the rows this process has produced."""

    def __init__(self, parse_rate_floor: float = PARSE_RATE_FLOOR,
                 latency_ceiling_s: float = LATENCY_CEILING_S,
                 min_rows: int = GUARD_MIN_ROWS):
        self.parse_rate_floor = parse_rate_floor
        self.latency_ceiling_s = latency_ceiling_s
        self.min_rows = min_rows
        self.n = 0
        self._ok = 0
        self._latencies: list[float] = []

    def record(self, parsed_ok: bool, latency_s: float) -> None:
        self.n += 1
        self._ok += bool(parsed_ok)
        self._latencies.append(latency_s)

    @property
    def parse_rate(self) -> float:
        return self._ok / self.n if self.n else float("nan")

    @property
    def median_latency_s(self) -> float:
        return statistics.median(self._latencies) if self._latencies else float("nan")

    def check(self) -> None:
        if self.n < self.min_rows:
            return
        if self.parse_rate < self.parse_rate_floor:
            raise GuardTripped(
                f"parse rate {self.parse_rate:.1%} is below the pre-registered "
                f"floor of {self.parse_rate_floor:.0%} over {self.n} rows. The "
                "prompt is degrading; every further row would be unusable. Fix the "
                "prompt, then start a fresh output file."
            )
        if self.median_latency_s > self.latency_ceiling_s:
            raise GuardTripped(
                f"median latency {self.median_latency_s:.1f}s exceeds "
                f"{self.latency_ceiling_s:.1f}s over {self.n} rows, against a "
                "measured healthy baseline of 9-11s.\n"
                "Check, in this order: another process using the GPU "
                "(nvidia-smi); free system RAM; then peak VRAM against the 8 GB "
                "card. Completed rows are on disk -- resume once it is clear.\n"
                "Do NOT lower MAX_PIXELS to work around this. It is a "
                "pre-registered setting whose value was measured, and in the one "
                "degradation observed so far VRAM was not the cause."
            )


def run_stage(
    rows: Iterable[dict],
    work_fn: Callable[[dict], dict],
    out_csv: Path,
    state_json: Path,
    fieldnames: list[str],
    fingerprint: dict,
    *,
    resume: bool = True,
    force_resume: bool = False,
    checkpoint_every: int = 100,
    progress_fn: Optional[Callable[[dict], None]] = None,
    guards: Optional[Guards] = None,
    ok_fn: Optional[Callable[[dict], bool]] = None,
    latency_fn: Optional[Callable[[dict], float]] = None,
) -> dict:
    """Run `work_fn` over `rows`, checkpointing every row and skipping done work.

    `work_fn` raising is not fatal: the row is recorded with its error and the run
    continues, because one unreadable image should not end an overnight run. A
    tripped guard *is* fatal, because it means every subsequent row would be
    unusable too.

    `ok_fn` and `latency_fn` say how THIS stage's health is read from a result
    row. The defaults are Stage 2's columns, which is where the guards came from,
    and every other stage writes something different:

      - latency defaults to `latency_ms`. Stage 4 spends its time in
        `latency_ms_black` + `latency_ms_tokens`; Stage 5 and 6 wrote no such
        column at all. A missing column reads as 0.0 s, so the ceiling can never
        be crossed and the stage reports a healthy `median 0.0s` forever.
      - success defaults to a non-empty `verdict`. Because `out_row` is
        `{**row, **result}`, a stage whose input rows come from an earlier
        stage's CSV *inherits that CSV's verdict column* and grades the previous
        stage's output instead of its own.

    Both failures are silent and both look like health. Guards run in-process
    over this run's rows only and never touch what is written, so supplying these
    cannot change a recorded number -- only whether a bad run is caught.
    """
    state = RunState(state_json)
    state.verify_or_create(fingerprint, force=force_resume)

    writer = CheckpointWriter(out_csv, fieldnames)
    done = writer.done_keys() if resume else set()
    guards = guards if guards is not None else Guards()

    summary = {"done": 0, "skipped": 0, "errors": 0, "guards": guards}

    try:
        for row in rows:
            key = (row["image_id"], row["rule_id"])
            if key in done:
                summary["skipped"] += 1
                continue

            try:
                result = work_fn(row)
                error = ""
            except (KeyboardInterrupt, SystemExit):
                raise
            except GuardTripped:
                raise
            except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
                result = {}
                error = f"{type(exc).__name__}: {exc}"
                summary["errors"] += 1

            out_row = {**row, **result, "error": error}
            writer.append(out_row)
            summary["done"] += 1

            guards.record(
                parsed_ok=_row_ok(out_row) if ok_fn is None else bool(ok_fn(out_row)),
                latency_s=(float(out_row.get("latency_ms") or 0.0) / 1000.0
                           if latency_fn is None else float(latency_fn(out_row))),
            )

            if summary["done"] % checkpoint_every == 0:
                guards.check()
                if progress_fn is not None:
                    progress_fn(summary)
    finally:
        writer.close()

    return summary
