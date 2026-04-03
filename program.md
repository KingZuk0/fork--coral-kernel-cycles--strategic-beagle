Task: optimize `kernel_builder.py` for lower cycle count while preserving correctness.

Context:
- This is based on the CORAL kernel builder challenge.
- You are optimizing a VLIW SIMD kernel for tree traversal.
- Baseline cycle count is around 147734.
- Best-known cycle count is around 1363.

What to change:
- Edit only `kernel_builder.py`.
- Keep output exactly correct under the simulator checks.

How to run:
1) `bash prepare.sh`
2) `bash eval/eval.sh`

Scoring:
- Eval prints both `cycles:` and `score:`.
- Score is normalized to [0, 1]:
  - 0.0 at baseline (147734 cycles)
  - 1.0 at best-known (1363 cycles)

Notes:
- Correctness is mandatory. Incorrect output gets score 0.0.
- Lower cycles is better.
