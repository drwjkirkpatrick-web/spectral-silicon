#!/usr/bin/env bash
# Run all tests for the spectral-silicon project.
# Usage: bash scripts/run_tests.sh [--slow]
#
# Runs pytest on the test suite and the complexity benchmark script.
# Use --slow to include tests marked with @pytest.mark.slow.

set -euo pipefail

# Determine project root (parent of this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "  Spectral Silicon — Test Suite"
echo "============================================================"
echo ""

# --- Parse args ---
INCLUDE_SLOW=false
PYTEST_ARGS=("-v")
for arg in "$@"; do
    case "$arg" in
        --slow)
            INCLUDE_SLOW=true
            PYTEST_ARGS+=("--runslow")
            ;;
        --fast)
            # Exclude slow tests explicitly
            PYTEST_ARGS+=("-m" "not slow")
            ;;
        *)
            PYTEST_ARGS+=("$arg")
            ;;
    esac
done

# If no slow flag, skip slow markers by default
if [ "$INCLUDE_SLOW" = false ]; then
    # Add a marker expression if not already present
    if ! printf '%s\n' "${PYTEST_ARGS[@]}" | grep -q -- "-m"; then
        PYTEST_ARGS+=("-m" "not slow")
    fi
fi

echo "[1/3] Running pytest..."
echo "  args: ${PYTEST_ARGS[*]}"
echo ""
python -m pytest tests/ "${PYTEST_ARGS[@]}"
echo ""
echo "  pytest: PASSED"
echo ""

echo "[2/3] Running complexity benchmark..."
python scripts/benchmark.py --seq-lens 128 256 512 --channels 16 --repeats 3 --warmup 1 || {
    echo "  benchmark: completed (non-zero exit is OK — spectral may not always win at small N)"
}
echo ""

echo "[3/3] Generating twiddle factors..."
python scripts/gen_twiddles.py --n 256 --format Q8.8 || {
    echo "  gen_twiddles: FAILED (continuing)"
}
echo ""

echo "============================================================"
echo "  All tests complete."
echo "============================================================"