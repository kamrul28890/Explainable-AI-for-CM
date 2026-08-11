#!/usr/bin/env bash
# Move pilot/ -> studies/01-florence2-pilot/ and repair everything the move breaks.
#
# WHY A SCRIPT: three things hold absolute paths that silently break on a move --
# the editable-install .pth file, the Jupyter kernelspec, and the VS Code settings.
# Moving the folder by hand leaves an environment that imports nothing and a kernel
# that points at a directory which no longer exists.
#
# PREREQUISITE: close VS Code (or at least shut down all Jupyter kernels). Windows
# refuses to move a directory while any executable inside it is running, and the
# venv's python.exe lives inside pilot/.
#
# Usage:  bash scripts/migrate_pilot_to_studies.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OLD="pilot"
NEW="studies/01-florence2-pilot"

echo "==> repository: $REPO_ROOT"

# ---------------------------------------------------------------- preflight
if [ ! -d "$OLD" ]; then
  if [ -d "$NEW" ]; then
    echo "==> already migrated ($NEW exists). Nothing to do."
    exit 0
  fi
  echo "ERROR: neither $OLD nor $NEW exists. Wrong directory?" >&2
  exit 1
fi

# Refuse to run while something inside pilot/ is executing -- the move would fail
# halfway and leave a partially-moved tree.
if command -v powershell.exe >/dev/null 2>&1; then
  BUSY=$(powershell.exe -NoProfile -Command \
    "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*${OLD}*' } | Measure-Object).Count" 2>/dev/null | tr -d '\r' || echo 0)
  if [ "${BUSY:-0}" != "0" ]; then
    echo "ERROR: $BUSY python process(es) are running from inside $OLD/." >&2
    echo "       Close VS Code and shut down all Jupyter kernels, then re-run." >&2
    exit 1
  fi
fi

# ---------------------------------------------------------------- move
echo "==> moving $OLD -> $NEW"
mkdir -p studies
mv "$OLD" "$NEW"

# ---------------------------------------------------------------- repair
VENV_PY="$NEW/.venv/Scripts/python.exe"
[ -x "$VENV_PY" ] || VENV_PY="$NEW/.venv/bin/python"

if [ -x "$VENV_PY" ]; then
  echo "==> repairing editable install (the .pth file holds an absolute path)"
  # 'python -m pip' rather than the pip.exe shim: shims bake in the old path and
  # break on a move, whereas 'python -m' resolves at runtime.
  "$VENV_PY" -m pip install -e "$NEW" --quiet --no-deps

  echo "==> re-registering the Jupyter kernel"
  "$VENV_PY" -m ipykernel install --user \
      --name xai-pilot --display-name "Python (xai-pilot)" >/dev/null

  # The study-2 package is installed editable against the same venv and holds its
  # own absolute path, so it needs the same repair.
  if [ -f "studies/02-vlm-xai-study/pyproject.toml" ]; then
    echo "==> repairing study-2 editable install"
    "$VENV_PY" -m pip install -e studies/02-vlm-xai-study --quiet --no-deps
  fi

  echo "==> verifying"
  "$VENV_PY" - <<'PY'
import sys
import xai_pilot
from xai_pilot.config import PILOT_ROOT, RESULTS_DIR
print(f"    interpreter : {sys.executable}")
print(f"    xai_pilot   : {xai_pilot.__file__}")
print(f"    PILOT_ROOT  : {PILOT_ROOT}")
assert PILOT_ROOT.exists(), "PILOT_ROOT does not resolve"
assert RESULTS_DIR.exists(), "RESULTS_DIR does not resolve"

try:
    import xai_vlm
    from xai_vlm.rules import prompt_for
    assert "ANSWER:" in prompt_for("rule_1")
    print(f"    xai_vlm     : {xai_vlm.__file__}")
except ImportError as exc:
    print(f"    xai_vlm     : NOT importable ({exc})")

print("    imports and paths OK")
PY
else
  echo "==> no venv found at $NEW/.venv -- skipping repair."
  echo "    Create one with the steps in docs/setup/SETUP.md."
fi

# ---------------------------------------------------------------- vscode
echo "==> updating .vscode/settings.json paths"
if [ -f .vscode/settings.json ]; then
  sed -i.bak "s#\${workspaceFolder}/pilot/#\${workspaceFolder}/${NEW}/#g; s#\${workspaceFolder}/pilot\"#\${workspaceFolder}/${NEW}\"#g" .vscode/settings.json
  rm -f .vscode/settings.json.bak
fi

cat <<EOF

==> migration complete.

    $OLD  ->  $NEW

    Nothing has been staged in git. Review with:
        git status
        git add -A pilot studies    # records it as a rename

    Then reopen VS Code and pick the "Python (xai-pilot)" kernel.
EOF
