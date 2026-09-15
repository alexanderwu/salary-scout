"""The JavaScript port is verified by replaying the exported fixtures in Node."""

import shutil
import subprocess

import pytest

from salary_scout.data import REPO_ROOT

WEB = REPO_ROOT / "web"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.skipif(not (WEB / "model" / "fixtures.json").exists(), reason="web/model not exported")
def test_javascript_matches_python_fixtures():
    proc = subprocess.run(
        ["node", str(WEB / "test" / "verify.mjs")], capture_output=True, text=True, timeout=300, check=False
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "checks passed" in proc.stdout
