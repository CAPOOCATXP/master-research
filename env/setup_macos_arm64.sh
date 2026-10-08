#!/usr/bin/env bash
# Reproducible environment setup for master-research on macOS / Apple Silicon.
#
# Records what it does. Fails loudly rather than silently substituting anything.
#
# Usage:
#   bash env/setup_macos_arm64.sh              # full setup
#   bash env/setup_macos_arm64.sh --repos-only # just re-clone the pinned repos
#   bash env/setup_macos_arm64.sh --check      # verify an existing setup
#
# See env/environment_lock.md for the recorded versions and the code-signing
# blocker that makes the interpreter bootstrap below necessary.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PATCHCORE_COMMIT="fcaa92f124fb1ad74a7acf56726decd4b27cbcad"
ANOMALYDINO_COMMIT="b9d1c2648e3a5247437d4d953d907a8f3d994457"
WR50_SHA256="95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8"

log() { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn ]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[fail ]\033[0m %s\n' "$*" >&2; exit 1; }

clone_repos() {
  mkdir -p third_party
  if [ ! -d third_party/patchcore-inspection ]; then
    log "cloning PatchCore"
    git clone --quiet https://github.com/amazon-science/patchcore-inspection.git \
      third_party/patchcore-inspection
  fi
  git -C third_party/patchcore-inspection checkout --quiet "$PATCHCORE_COMMIT"

  if [ ! -d third_party/AnomalyDINO ]; then
    log "cloning AnomalyDINO"
    git clone --quiet https://github.com/dammsi/AnomalyDINO.git third_party/AnomalyDINO
  fi
  git -C third_party/AnomalyDINO checkout --quiet "$ANOMALYDINO_COMMIT"

  local got_pc got_ad
  got_pc="$(git -C third_party/patchcore-inspection rev-parse HEAD)"
  got_ad="$(git -C third_party/AnomalyDINO rev-parse HEAD)"
  [ "$got_pc" = "$PATCHCORE_COMMIT" ] || die "PatchCore commit mismatch: $got_pc"
  [ "$got_ad" = "$ANOMALYDINO_COMMIT" ] || die "AnomalyDINO commit mismatch: $got_ad"
  log "repos pinned: patchcore=${got_pc:0:12} anomalydino=${got_ad:0:12}"
}

resolve_interpreter() {
  # Preferred: the DSH bundled runtime, re-signed ad-hoc in place if the hardened
  # runtime flag would block loading torch's ad-hoc-signed dylibs.
  local bundled="/Users/zhangege/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/bin"
  if [ -x "$bundled/python3.12" ]; then
    if [ ! -x "$bundled/python3.12-mps" ]; then
      log "creating ad-hoc signed interpreter copy (library-validation workaround)"
      cp "$bundled/python3.12" "$bundled/python3.12-mps"
      codesign --force --sign - "$bundled/python3.12-mps" >/dev/null 2>&1 \
        || warn "codesign failed; torch may fail to import"
    fi
    echo "$bundled/python3.12-mps"
    return
  fi
  # Fallback: any python3 on PATH. On macOS this may itself be hardened; if torch
  # then fails to import, see env/environment_lock.md.
  command -v python3 || die "no python3 found"
}

make_venv() {
  local py
  py="$(resolve_interpreter)"
  log "interpreter: $py"
  [ -d .venv ] || "$py" -m venv .venv
  .venv/bin/python -m pip install --quiet --upgrade pip
  log "installing torch + torchvision (MPS build)"
  .venv/bin/python -m pip install --quiet torch torchvision
  log "installing scientific stack"
  .venv/bin/python -m pip install --quiet numpy scipy scikit-learn Pillow matplotlib pandas pyyaml tqdm
  log "installing the package in editable mode"
  .venv/bin/python -m pip install --quiet -e .
  .venv/bin/python -m pip install --quiet pytest
  fix_pth_flags
}

fix_pth_flags() {
  # The bundled Python's site.addpackage() SKIPS .pth files that carry the
  # macOS UF_HIDDEN flag, silently. pip-written .pth files can end up with
  # st_flags=0x8040 here, which makes the editable install vanish with no error
  # (``ModuleNotFoundError: No module named 'master_research'`` while the
  # package is plainly installed). Clear the flag on every .pth in the venv.
  # See env/environment_lock.md §2b.
  local f flags
  while IFS= read -r -d '' f; do
    flags="$(ls -lO "$f" 2>/dev/null | awk '{print $5}')"
    case "$flags" in
      *hidden*)
        chflags nohidden "$f" 2>/dev/null && log "cleared UF_HIDDEN on $(basename "$f")" ;;
    esac
  done < <(find .venv -name "*.pth" -print0 2>/dev/null)
}

check() {
  log "verifying environment"
  .venv/bin/python - <<'PY'
import sys
import numpy as np
import torch, torchvision
import master_research
print(f"python        {sys.version.split()[0]}")
print(f"torch         {torch.__version__}")
print(f"torchvision   {torchvision.__version__}")
print(f"numpy         {np.__version__}")
print(f"mps built     {torch.backends.mps.is_built()}")
print(f"mps available {torch.backends.mps.is_available()}")
print(f"package       {master_research.__version__}")
PY
  local wr50="$HOME/.cache/torch/hub/checkpoints/wide_resnet50_2-95faca4d.pth"
  if [ -f "$wr50" ]; then
    local got
    got="$(shasum -a 256 "$wr50" | awk '{print $1}')"
    if [ "$got" = "$WR50_SHA256" ]; then
      log "wide_resnet50_2 weights verified"
    else
      die "wide_resnet50_2 sha256 mismatch: got $got"
    fi
  else
    warn "wide_resnet50_2 weights not cached yet; run scripts/00_env_check.py"
  fi
}

case "${1:-}" in
  --repos-only) clone_repos ;;
  --check) check ;;
  "") clone_repos; make_venv; check; log "done. Next: .venv/bin/python scripts/00_env_check.py" ;;
  *) die "unknown option: $1" ;;
esac
