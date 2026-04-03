#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -f "$ROOT_DIR/eval/frozen_problem.py" ]; then
  echo "missing eval/frozen_problem.py; run: bash prepare.sh"
  exit 1
fi

if [ ! -f "$ROOT_DIR/kernel_builder.py" ]; then
  echo "missing kernel_builder.py; run: bash prepare.sh"
  exit 1
fi

ROOT_DIR="$ROOT_DIR" python3 - <<'PY'
import json
import os
import sys
import traceback

ROOT = os.environ["ROOT_DIR"]
sys.path.insert(0, os.path.join(ROOT, "eval"))

from frozen_problem import Machine, build_mem_image, reference_kernel2, Tree, Input, N_CORES

BASELINE_CYCLES = 147_734
BEST_KNOWN_CYCLES = 1_363


def load_builder():
    path = os.path.join(ROOT, "kernel_builder.py")
    ns = {"__name__": "__main__"}
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    exec(source, ns)
    if "KernelBuilder" not in ns:
        raise RuntimeError("KernelBuilder class not found")
    return ns["KernelBuilder"]


def run_once(KernelBuilder):
    forest = Tree.generate(10)
    inp = Input.generate(forest, 256, 16)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), 16)

    machine = Machine(mem, kb.instrs, kb.debug_info(), n_cores=N_CORES)
    machine.enable_pause = False
    machine.enable_debug = False
    machine.run()

    for ref_mem in reference_kernel2(mem):
        pass

    inp_values_p = ref_mem[6]
    actual = machine.mem[inp_values_p : inp_values_p + len(inp.values)]
    expected = ref_mem[inp_values_p : inp_values_p + len(inp.values)]
    if actual != expected:
        return machine.cycle, False, "Incorrect output values"
    return machine.cycle, True, ""


def cycle_to_score(cycles: int) -> float:
    denom = BASELINE_CYCLES - BEST_KNOWN_CYCLES
    if denom <= 0:
        return 0.0
    raw = (BASELINE_CYCLES - cycles) / denom
    return max(0.0, min(1.0, raw))


try:
    KernelBuilder = load_builder()
    cycles, ok, err = run_once(KernelBuilder)
    if not ok:
        print("cycles:", cycles)
        print("score: 0.0")
        print("error:", err)
        raise SystemExit(0)
    score = cycle_to_score(cycles)
    print("cycles:", cycles)
    print(f"score: {score:.6f}")
except Exception as e:
    print("cycles: 999999")
    print("score: 0.0")
    print("error:", str(e))
    traceback.print_exc()
PY
