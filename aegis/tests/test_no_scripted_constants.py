"""Regression test to guarantee no scripted constants or hard-coded benchmark values appear in aegis/."""

import os
from pathlib import Path

FORBIDDEN_PATTERNS = [
    "180," + "000",
    "18," + "500",
    "89" + ".4",
    "ISO" + " 10816",
    "ieee" + "xplore",
    "aegis" + "-engine-v1",
]


def test_no_scripted_constants_in_aegis():
    """Verify that none of the forbidden scripted values or strings appear anywhere in aegis/."""
    # Resolve aegis/ directory: parent of tests/
    aegis_dir = Path(__file__).resolve().parent.parent
    violations = []

    for root, dirs, files in os.walk(aegis_dir):
        # Exclude hidden directories and pycache
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        for file in files:
            if file.endswith((".pyc", ".pyo", ".pyd")):
                continue
            file_path = Path(root) / file
            # Skip the test file itself
            if file_path.resolve() == Path(__file__).resolve():
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                lines = content.splitlines()
                for line_no, line in enumerate(lines, start=1):
                    for pattern in FORBIDDEN_PATTERNS:
                        if pattern in line:
                            violations.append(
                                f"{file_path.relative_to(aegis_dir)}:{line_no} contains forbidden pattern '{pattern}' -> {line.strip()}"
                            )
            except Exception as e:
                violations.append(f"Could not read {file_path}: {e}")

    assert not violations, "Forbidden scripted strings found in aegis/:\n" + "\n".join(violations)
