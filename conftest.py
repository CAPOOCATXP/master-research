"""Pytest bootstrap: make the repository importable without relying on the
editable install.

Why this file exists
--------------------
Tests originally imported ``master_research`` through the editable install's
``__editable__*.pth`` file in ``.venv/lib/python3.12/site-packages``. That
silently stopped working, because the Python runtime used here (the DSH bundled
build) contains an extra guard in ``site.addpackage``:

    if ((getattr(st, 'st_flags', 0) & stat.UF_HIDDEN) or ...):
        _trace(f"Skipping hidden .pth file: {fullname!r}")
        return

On macOS the pip-written ``.pth`` files carried ``UF_HIDDEN`` (``st_flags``
``0x8040``), so they were **skipped without any error message** and
``import master_research`` began failing with ``ModuleNotFoundError`` while the
package was demonstrably installed. Clearing the flag with
``chflags nohidden`` fixes it, but a test suite should not depend on a
site-packages flag at all.

Adding ``src`` to ``sys.path`` here makes the tests independent of both the
install mode and that platform quirk. See ``env/environment_lock.md`` §2b.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
