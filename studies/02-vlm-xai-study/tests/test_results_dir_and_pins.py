"""Where results go, and that every weight the code loads is the pinned one."""
import os
import re
import subprocess
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))

from xai_vlm import config  # noqa: E402


def _results_dir_with(env_value):
    env = dict(os.environ)
    env.pop("XAI_RESULTS_DIR", None)
    if env_value is not None:
        env["XAI_RESULTS_DIR"] = env_value
    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(STUDY_ROOT / 'src')!r}); "
         "from xai_vlm import config; print(config.RESULTS_DIR)"],
        capture_output=True, text=True, env=env, check=True)
    return Path(out.stdout.strip())


def test_default_results_dir_is_unchanged():
    # The frozen local results live here; no flag must mean no change.
    assert _results_dir_with(None) == STUDY_ROOT / "results"


def test_results_dir_can_be_redirected_for_a_separate_machine(tmp_path):
    assert _results_dir_with(str(tmp_path)) == tmp_path


def test_no_stage_script_hardcodes_the_results_dir():
    # One hardcoded path and a Scholar run reads or writes the frozen local tree
    # for that stage alone -- a mixed run that still parses.
    offenders = [p.name for p in sorted((STUDY_ROOT / "scripts").glob("[0-9]*.py"))
                 if 'STUDY_ROOT / "results"' in p.read_text(encoding="utf-8")]
    assert offenders == []


def _lock_revision(tier):
    lock = STUDY_ROOT.parents[1] / "requirements" / "models.lock"
    for line in lock.read_text(encoding="utf-8").splitlines():
        parts = line.split("#")[0].split()
        if parts[:1] == [tier]:
            return parts[2]
    raise AssertionError(f"{tier} not in models.lock")


def test_embedder_defaults_to_the_locked_revision():
    # Stage 10 constructs MiniLMEmbedder() with no arguments. Unpinned, a fresh
    # machine downloads whatever `main` is that day and metric 7 silently moves.
    from xai_vlm.metrics.text_similarity import EMBED_REVISION, MiniLMEmbedder
    assert EMBED_REVISION == _lock_revision("minilm-l6-v2")
    assert MiniLMEmbedder()._revision == EMBED_REVISION
