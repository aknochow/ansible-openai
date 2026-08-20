# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
import shutil

# Deliberately bare, not `from tests.unit.conftest import ...` -- verified
# empirically that the qualified form actually BREAKS when pytest is
# invoked from within tests/unit itself (ModuleNotFoundError: No module
# named 'tests'), which is exactly the invocation pattern a qualified
# import would supposedly protect against. The bare form works correctly
# under both `pytest tests/unit/` from the repo root and `pytest
# test_conftest_namespace_shim.py` from within tests/unit -- pytest
# inserts each test file's own directory into sys.path during collection,
# independent of the invocation cwd. Matches both sibling collections'
# identical, working pattern.
from conftest import _create_namespace_shim


def test_namespace_shim_registers_working_cleanup(monkeypatch, tmp_path):
    registered = []
    monkeypatch.setattr(
        atexit, "register", lambda fn, *args, **kwargs: registered.append((fn, args, kwargs))
    )

    result = _create_namespace_shim("test_shim_", "llama", tmp_path)
    try:
        assert result.exists()
        assert len(registered) == 1
        fn, args, kwargs = registered[0]

        # Regression check: it's not enough that *something* was registered --
        # invoking exactly what was registered must actually remove the
        # directory. This is what catches a wrong path, wrong function, or a
        # missing atexit.register call entirely.
        fn(*args, **kwargs)
        assert not result.exists()
    finally:
        # monkeypatching atexit.register above means the real cleanup this
        # fix adds never actually gets registered for this test's own
        # directory -- without this, a failed assertion above would leak
        # exactly the kind of directory this whole fix exists to stop
        # leaking.
        shutil.rmtree(str(result), ignore_errors=True)
