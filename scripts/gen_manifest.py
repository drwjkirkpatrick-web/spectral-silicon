#!/usr/bin/env python3
"""Reproducible build manifest generator for the Spectral Silicon project.

Generates a JSON manifest containing:
  - Tool version information (yosys, openroad, iverilog, python, torch)
  - SHA-256 hashes of all RTL (.v) files
  - OpenLane configuration
  - SHA-256 hashes of all Python module files (spectral_silicon/)

The manifest enables reproducible builds: the same source tree and tool
versions should produce identical bitstreams, and the manifest can be
verified at any later time to detect tampering or drift.

Usage
-----
Generate a manifest::

    python scripts/gen_manifest.py --output build_manifest.json

Verify an existing manifest::

    python scripts/gen_manifest.py --verify build_manifest.json

The script uses only Python stdlib (hashlib, json, subprocess, sys,
platform, argparse) plus optionally torch (if available) for version
reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────────
# Path resolution
# ──────────────────────────────────────────────────────────────────────────

# Project root = parent of the scripts/ directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RTL_DIR = PROJECT_ROOT / "rtl"
PYTHON_PKG_DIR = PROJECT_ROOT / "spectral_silicon"
TESTS_DIR = PROJECT_ROOT / "tests"
OPENLANE_DIR = PROJECT_ROOT / "openlane"  # may not exist yet

MANIFEST_VERSION = 1


def _sha256_file(filepath: Path) -> str:
    """Compute the SHA-256 hash of a file.

    Parameters
    ----------
    filepath : Path
        Path to the file to hash.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest string.
    """
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in 64KB chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    """Compute the SHA-256 hash of raw bytes.

    Parameters
    ----------
    data : bytes
        Data to hash.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest string.
    """
    return hashlib.sha256(data).hexdigest()


def _run_version_cmd(cmd: List[str]) -> str:
    """Run a command and return its stdout, or 'not found' if unavailable.

    Parameters
    ----------
    cmd : list of str
        Command and arguments to execute.

    Returns
    -------
    str
        Trimmed stdout output, or an error string if the command fails.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout.strip() if result.returncode == 0 else (
            result.stderr.strip() or f"exit code {result.returncode}"
        )
        # Truncate to the first line only (some tools like iverilog print
        # long copyright notices on version commands)
        if output:
            first_line = output.splitlines()[0].strip()
            # Keep just the version-relevant first line for clean manifests
            return first_line
        return "unknown"
    except FileNotFoundError:
        return "not found"
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception as e:
        return f"error: {e}"


def _get_tool_versions() -> Dict[str, str]:
    """Collect version strings for all build tools.

    Returns
    -------
    dict
        Mapping of tool name to version string.  Tools that are not
        installed are recorded as ``"not found"``.
    """
    versions: Dict[str, str] = {}

    # yosys — `yosys -V`
    versions["yosys"] = _run_version_cmd(["yosys", "-V"])

    # openroad — `openroad -version`
    versions["openroad"] = _run_version_cmd(["openroad", "-version"])

    # iverilog — `iverilog -V`
    versions["iverilog"] = _run_version_cmd(["iverilog", "-V"])

    # Python version
    versions["python"] = platform.python_version()

    # torch version (optional — import guard)
    try:
        import torch
        versions["torch"] = torch.__version__
    except ImportError:
        versions["torch"] = "not installed"
    except Exception as e:
        versions["torch"] = f"error: {e}"

    return versions


def _hash_directory(
    directory: Path, patterns: List[str], exclude_patterns: Optional[List[str]] = None
) -> Dict[str, str]:
    """Hash all files matching *patterns* in *directory*.

    Parameters
    ----------
    directory : Path
        Directory to scan.
    patterns : list of str
        File glob patterns to match (e.g. ``["*.v"]``, ``["*.py"]``).
    exclude_patterns : list of str, optional
        Filename substrings to exclude (e.g. ``["__pycache__"]``).

    Returns
    -------
    dict
        Mapping of relative file path to SHA-256 hex digest.
    """
    exclude_patterns = exclude_patterns or []
    hashes: Dict[str, str] = {}

    if not directory.exists():
        return hashes

    for pattern in patterns:
        for filepath in sorted(directory.rglob(pattern)):
            # Skip excluded patterns
            rel = filepath.relative_to(PROJECT_ROOT)
            if any(ex in str(rel) for ex in exclude_patterns):
                continue
            # Skip __pycache__ directories
            if "__pycache__" in filepath.parts:
                continue
            hashes[str(rel)] = _sha256_file(filepath)

    return hashes


def _collect_openlane_config() -> Dict[str, Any]:
    """Collect OpenLane configuration from the project.

    Looks for a ``openlane/`` directory and collects any ``config.json``
    or ``*.json`` configuration files.

    Returns
    -------
    dict
        OpenLane configuration data, or an empty dict if not found.
    """
    config: Dict[str, Any] = {}

    if OPENLANE_DIR.exists():
        # Hash all config files
        for filepath in sorted(OPENLANE_DIR.rglob("*.json")):
            rel = filepath.relative_to(PROJECT_ROOT)
            config[str(rel)] = _sha256_file(filepath)

        # Also hash any .tcl or .cfg files
        for pattern in ["*.tcl", "*.cfg", "*.yml", "*.yaml"]:
            for filepath in sorted(OPENLANE_DIR.rglob(pattern)):
                rel = filepath.relative_to(PROJECT_ROOT)
                config[str(rel)] = _sha256_file(filepath)

    if not config:
        config["status"] = "openlane directory not found"

    return config


def generate_manifest() -> Dict[str, Any]:
    """Generate the complete build manifest.

    Returns
    -------
    dict
        A JSON-serializable manifest dictionary with:
        - ``manifest_version``: schema version (1)
        - ``generated_at``: ISO-8601 timestamp
        - ``generator``: script name
        - ``project_root``: absolute path to the project root
        - ``tools``: tool version strings
        - ``rtl_files``: SHA-256 hashes of RTL files
        - ``python_modules``: SHA-256 hashes of Python source files
        - ``openlane``: OpenLane configuration hashes
        - ``tests``: SHA-256 hashes of test files
    """
    manifest: Dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/gen_manifest.py",
        "project_root": str(PROJECT_ROOT),
        "host": {
            "machine": platform.machine(),
            "system": platform.system(),
            "python_implementation": platform.python_implementation(),
        },
        "tools": _get_tool_versions(),
        "rtl_files": _hash_directory(RTL_DIR, ["*.v"]),
        "python_modules": _hash_directory(
            PYTHON_PKG_DIR,
            ["*.py"],
            exclude_patterns=["__pycache__"],
        ),
        "tests": _hash_directory(
            TESTS_DIR,
            ["*.py"],
            exclude_patterns=["__pycache__"],
        ),
        "openlane": _collect_openlane_config(),
    }

    # Compute a top-level manifest hash for quick integrity check
    # (hash of all file hashes, sorted)
    all_hashes = []
    for section in ["rtl_files", "python_modules", "tests"]:
        for path, h in sorted(manifest[section].items()):
            all_hashes.append(f"{path}:{h}")
    manifest["manifest_hash"] = _sha256_bytes(
        "\n".join(all_hashes).encode("utf-8")
    )

    return manifest


def verify_manifest(manifest_path: str) -> int:
    """Verify an existing manifest against the current source tree.

    Parameters
    ----------
    manifest_path : str
        Path to a JSON manifest file.

    Returns
    -------
    int
        0 if all hashes match, 1 if any mismatch is found.
    """
    with open(manifest_path, "r") as f:
        stored = json.load(f)

    print(f"Verifying manifest: {manifest_path}")
    print(f"  Generated at: {stored.get('generated_at', 'unknown')}")
    print()

    mismatches: List[str] = []
    missing_files: List[str] = []

    # Verify RTL files
    for section_name in ["rtl_files", "python_modules", "tests"]:
        section = stored.get(section_name, {})
        print(f"Checking {section_name} ({len(section)} files)...")
        for rel_path, stored_hash in section.items():
            full_path = PROJECT_ROOT / rel_path
            if not full_path.exists():
                missing_files.append(rel_path)
                print(f"  MISSING: {rel_path}")
                continue
            current_hash = _sha256_file(full_path)
            if current_hash != stored_hash:
                mismatches.append(rel_path)
                print(f"  MISMATCH: {rel_path}")
                print(f"    stored:   {stored_hash}")
                print(f"    current: {current_hash}")

    # Check for new files not in the manifest
    for section_name, directory, patterns in [
        ("rtl_files", RTL_DIR, ["*.v"]),
        ("python_modules", PYTHON_PKG_DIR, ["*.py"]),
        ("tests", TESTS_DIR, ["*.py"]),
    ]:
        section = stored.get(section_name, {})
        for pattern in patterns:
            for filepath in sorted(directory.rglob(pattern)):
                if "__pycache__" in filepath.parts:
                    continue
                rel = str(filepath.relative_to(PROJECT_ROOT))
                if rel not in section:
                    print(f"  NEW FILE (not in manifest): {rel}")
                    mismatches.append(rel)

    # Verify tool versions (informational)
    print("\nTool versions:")
    current_tools = _get_tool_versions()
    for tool, version in current_tools.items():
        stored_version = stored.get("tools", {}).get(tool, "N/A")
        match = "OK" if version == stored_version else "CHANGED"
        print(f"  {tool}: {version} [{match}]")

    print()
    if mismatches:
        print(f"FAILED: {len(mismatches)} file(s) changed or missing.")
        for m in mismatches:
            print(f"  - {m}")
        return 1
    if missing_files:
        print(f"FAILED: {len(missing_files)} file(s) missing.")
        for m in missing_files:
            print(f"  - {m}")
        return 1

    print("OK: All file hashes match the manifest.")
    return 0


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate or verify a reproducible build manifest "
        "for the Spectral Silicon project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Generate a manifest
  python scripts/gen_manifest.py --output build_manifest.json

  # Verify an existing manifest
  python scripts/gen_manifest.py --verify build_manifest.json

  # Print manifest to stdout
  python scripts/gen_manifest.py
""",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for the manifest JSON (default: stdout).",
    )
    parser.add_argument(
        "--verify",
        "-v",
        type=str,
        default=None,
        metavar="MANIFEST",
        help="Verify an existing manifest file against the current tree.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level (default: 2).",
    )
    args = parser.parse_args()

    if args.verify:
        return verify_manifest(args.verify)

    manifest = generate_manifest()
    json_str = json.dumps(manifest, indent=args.indent, sort_keys=True)

    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)
            f.write("\n")
        print(f"Manifest written to {args.output}")
        print(f"  {len(manifest.get('rtl_files', {}))} RTL files hashed")
        print(f"  {len(manifest.get('python_modules', {}))} Python files hashed")
        print(f"  {len(manifest.get('tests', {}))} test files hashed")
        print(f"  Manifest hash: {manifest.get('manifest_hash', 'N/A')}")
    else:
        print(json_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())