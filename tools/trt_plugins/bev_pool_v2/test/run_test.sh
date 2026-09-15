#!/bin/bash
# Run BEVPoolV2 Plugin Registration Test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRT_ROOT="${TRT_ROOT:?Set TRT_ROOT to the TensorRT installation prefix}"

# Set library path
export LD_LIBRARY_PATH="${TRT_ROOT}/lib:${SCRIPT_DIR}/..:${LD_LIBRARY_PATH}"

echo "Running BEVPoolV2 Plugin Registration Test..."
echo ""

cd "$SCRIPT_DIR/build"
./test_plugin_registration

echo ""
echo "Test completed!"
