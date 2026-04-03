#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT_DIR/eval"

BASE_URL="https://raw.githubusercontent.com/CuriousCaliBoi/CORAL/main/examples/kernel_builder"

if [ ! -f "$ROOT_DIR/kernel_builder.py" ] || grep -q "Task bootstrap file" "$ROOT_DIR/kernel_builder.py"; then
  curl -fsSL "$BASE_URL/seed/kernel_builder.py" -o "$ROOT_DIR/kernel_builder.py"
fi

if [ ! -f "$ROOT_DIR/eval/frozen_problem.py" ]; then
  curl -fsSL "$BASE_URL/eval/frozen_problem.py" -o "$ROOT_DIR/eval/frozen_problem.py"
fi

echo "prepare complete"
