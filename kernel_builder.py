"""
Baseline KernelBuilder for kernel optimization.

This is the naive scalar implementation that serves as the starting point
for optimization. It processes one element at a time without exploiting
VLIW parallelism or SIMD vectorization.

Key optimization opportunities:
1. VLIW: Pack multiple operations per cycle (up to 12 ALU, 6 VALU, 2 load, 2 store)
2. SIMD: Use vector operations (VLEN=8) to process 8 elements at once
3. Loop unrolling: Reduce loop overhead
4. Memory access patterns: Use vload/vstore for contiguous access
5. Instruction scheduling: Avoid dependencies, maximize ILP

Performance:
- Baseline: ~147,734 cycles
- Best known: ~1,363 cycles (108x speedup)

This file is self-contained and can be executed via exec() for grading.
"""

from dataclasses import dataclass

# Constants for the VLIW SIMD machine
SCRATCH_SIZE = 1536

# Hash function stages: (op1, val1, op2, op3, val3)
# Each stage computes: val = op2(op1(val, val1), op3(val, val3))
HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),
    ("^", 0xC761C23C, "^", ">>", 19),
    ("+", 0x165667B1, "+", "<<", 5),
    ("+", 0xD3A2646C, "^", "<<", 9),
    ("+", 0xFD7046C5, "+", "<<", 3),
    ("^", 0xB55A4F09, "^", ">>", 16),
]


@dataclass
class DebugInfo:
    """Debug information for the simulator."""

    scratch_map: dict[int, tuple[str, int]]


class KernelBuilder:
    """
    Kernel builder for the VLIW SIMD machine.

    This class builds a program (list of instructions) that implements
    the tree traversal algorithm. Each instruction is a dict mapping
    engine names to lists of operations.

    Engines and slot limits per cycle:
    - alu: 12 scalar operations
    - valu: 6 vector operations (each on VLEN=8 elements)
    - load: 2 memory reads
    - store: 2 memory writes
    - flow: 1 control flow operation
    """

    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        # pending slots collected during kernel construction; these will be
        # packed together with the main body to produce efficient VLIW
        # instructions. Using a pending list avoids emitting many single-slot
        # instructions via self.add() which prevented global packing.
        self.pending_slots = []

    def debug_info(self):
        """Return debug info for the simulator."""
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots, vliw=False):
        """
        Pack a list of (engine, operation) tuples into VLIW instructions.

        This greedy packer tries to fill available engine slots per cycle
        while preserving correctness: it forbids packing two operations
        into the same instruction when one reads a scratch location that
        another operation in the same instruction writes. It also forbids
        multiple writes to the same scratch location in one instruction.

        This significantly reduces cycle count by emitting fewer
        instructions (one instruction can contain multiple slots).
        """
        instrs = []
        # Slot limits per engine
        limits = {"alu": 12, "valu": 6, "load": 2, "store": 2, "flow": 1}

        def rw_for_slot(engine: str, slot: tuple):
            """Return (reads:set, writes:set) scratch indices for the slot."""
            reads = set()
            writes = set()
            # slot may be like ("+", dst, a, b) for alu, or for load engine the slot
            # itself encodes op name such as ("load", dst, addr) or ("const", dst, val)
            if engine == "alu":
                _, dst, a, b = slot
                reads.update([a, b])
                writes.add(dst)
            elif engine == "load":
                op = slot[0]
                if op == "load":
                    _, dst, addr = slot
                    reads.add(addr)
                    writes.add(dst)
                elif op == "load_offset":
                    _, dst, addr, offset = slot
                    reads.add(addr + offset)
                    writes.add(dst + offset)
                elif op == "vload":
                    _, dst, addr = slot
                    reads.add(addr)
                    writes.update(range(dst, dst + 8))
                elif op == "const":
                    _, dst, _ = slot
                    writes.add(dst)
                else:
                    # conservative default
                    try:
                        _, dst, addr = slot
                        reads.add(addr)
                        writes.add(dst)
                    except Exception:
                        pass
            elif engine == "store":
                op = slot[0]
                if op == "store":
                    _, addr, src = slot
                    reads.update([addr, src])
                elif op == "vstore":
                    _, addr, src = slot
                    reads.add(addr)
                    reads.update(range(src, src + 8))
                else:
                    try:
                        _, addr, src = slot
                        reads.update([addr, src])
                    except Exception:
                        pass
            elif engine == "flow":
                op = slot[0]
                if op == "select":
                    _, dst, cond, a, b = slot
                    reads.update([cond, a, b])
                    writes.add(dst)
                elif op == "vselect":
                    _, dst, cond, a, b = slot
                    reads.update(range(cond, cond + 8))
                    reads.update(range(a, a + 8))
                    reads.update(range(b, b + 8))
                    writes.update(range(dst, dst + 8))
                elif op == "add_imm":
                    _, dst, a, _ = slot
                    reads.add(a)
                    writes.add(dst)
                elif op == "trace_write":
                    _, val = slot
                    reads.add(val)
                elif op in ("cond_jump", "cond_jump_rel"):
                    _, cond, _ = slot
                    reads.add(cond)
                elif op == "jump_indirect":
                    _, addr = slot
                    reads.add(addr)
                elif op == "coreid":
                    _, dst = slot
                    writes.add(dst)
                else:
                    # ops like pause/halt/jump have no scratch effects
                    pass
            elif engine == "valu":
                # valu ops operate on vectors; they read/write contiguous ranges
                op = slot[0]
                if op == "vbroadcast":
                    _, dst, src = slot
                    reads.add(src)
                    writes.update(range(dst, dst + 8))
                elif op == "multiply_add":
                    _, dst, a, b, c = slot
                    reads.update(range(a, a + 8))
                    reads.update(range(b, b + 8))
                    reads.update(range(c, c + 8))
                    writes.update(range(dst, dst + 8))
                else:
                    # generic binary op: (op, dst, a, b)
                    try:
                        _, dst, a, b = slot
                        reads.update(range(a, a + 8))
                        reads.update(range(b, b + 8))
                        writes.update(range(dst, dst + 8))
                    except Exception:
                        pass
            else:
                # conservative: no info
                pass
            return reads, writes

        i = 0
        n = len(slots)
        while i < n:
            instr = {}
            used = {"alu": 0, "valu": 0, "load": 0, "store": 0, "flow": 0}
            curr_reads = set()
            curr_writes = set()
            # try to pack as many following slots as possible
            j = i
            while j < n:
                engine, slot = slots[j]
                if used.get(engine, 0) >= limits.get(engine, 0):
                    break
                reads, writes = rw_for_slot(engine, slot)
                # forbid if this slot reads something written earlier in this instr
                if reads & curr_writes:
                    break
                # forbid if this slot writes something already written in this instr
                if writes & curr_writes:
                    break
                # all good; add
                instr.setdefault(engine, []).append(slot)
                used[engine] = used.get(engine, 0) + 1
                curr_reads |= reads
                curr_writes |= writes
                j += 1
            # if we couldn't pack even a single slot (shouldn't happen), force one
            if i == j:
                engine, slot = slots[i]
                instr.setdefault(engine, []).append(slot)
                i += 1
            else:
                i = j
            instrs.append(instr)
        return instrs

    def add(self, engine, slot):
        """Add a single-slot instruction into pending slots for later packing.

        Storing slots in pending_slots instead of emitting immediate single-slot
        instructions allows the global packer (build) to consider constants,
        broadcasts and header loads together with the main body and pack them
        into full VLIW words. This reduces instruction count and improves
        utilization across engines.
        """
        self.pending_slots.append((engine, slot))

    def alloc_scratch(self, name=None, length=1):
        """Allocate scratch space (like registers)."""
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        """Load a constant into scratch space, reusing if already loaded."""
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        """
        Build instructions for the hash function.

        The hash function has 6 stages, each with:
        - First ALU op: tmp1 = val op1 const
        - Second ALU op: tmp2 = val op3 const
        - Third ALU op: val = tmp1 op2 tmp2
        """
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
        return slots

    def scratch_const_vector(self, val, name=None):
        """Create or return a vectorized broadcast of an immediate constant.

        This allocates an 8-word scratch and emits a vbroadcast valu slot that
        fills it with the scalar constant. The scalar constant itself is
        allocated via scratch_const and will be reused.
        """
        if not hasattr(self, "const_vec_map"):
            self.const_vec_map = {}
        if val not in self.const_vec_map:
            scalar = self.scratch_const(val)
            vec = self.alloc_scratch(name, length=8)
            # Emit a vbroadcast to fill vector scratch
            self.add("valu", ("vbroadcast", vec, scalar))
            self.const_vec_map[val] = vec
        return self.const_vec_map[val]

    def build_hash_vector(self, val_vec_addr, tmp1_vec, tmp2_vec):
        """Build vectorized hash stages operating on 8 lanes.

        Uses valu operations and vbroadcasted constant vectors.
        Hash stages with op + use multiply_add(val, ones, c) == val + c to vary valu slot mix for
        the packer (same semantics as binary +).
        """
        slots = []
        ones = self.scratch_const_vector(1)
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            c1 = self.scratch_const_vector(val1)
            c3 = self.scratch_const_vector(val3)
            if op1 == "+":
                slots.append(("valu", ("multiply_add", tmp1_vec, val_vec_addr, ones, c1)))
            else:
                slots.append(("valu", (op1, tmp1_vec, val_vec_addr, c1)))
            if op3 == "+":
                slots.append(("valu", ("multiply_add", tmp2_vec, val_vec_addr, ones, c3)))
            else:
                slots.append(("valu", (op3, tmp2_vec, val_vec_addr, c3)))
            slots.append(("valu", (op2, val_vec_addr, tmp1_vec, tmp2_vec)))
        return slots

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        """
        Build the main kernel with SIMD (VLEN=8) vectorization for the batch
        dimension. This version processes 8 elements at a time using valu
        operations and vload/vstore where possible. Remaining tail elements
        (batch_size % 8) are handled with the scalar fallback.
        """
        # Scalars used for occasional scalar ops
        tmp_addr = self.alloc_scratch("tmp_addr")
        tmp_addr2 = self.alloc_scratch("tmp_addr2")
        tmp_i0 = self.alloc_scratch("tmp_i0")
        tmp_v0 = self.alloc_scratch("tmp_v0")
        tmp_i1 = self.alloc_scratch("tmp_i1")
        tmp_v1 = self.alloc_scratch("tmp_v1")

        # Load initial values from memory header into reserved scratches.
        # mem[3] is forest_height; this kernel does not use it — skip load/slots.
        init_spec = [
            ("rounds", 0),
            ("n_nodes", 1),
            ("batch_size", 2),
            ("forest_values_p", 4),
            ("inp_indices_p", 5),
            ("inp_values_p", 6),
        ]
        for v, _ in init_spec:
            self.alloc_scratch(v, 1)
        for v, idx in init_spec:
            self.add("load", ("load", self.scratch[v], self.scratch_const(idx)))

        def alloc_v8(name):
            return self.alloc_scratch(name, length=8)

        idx_a = alloc_v8("idx_a")
        val_a = alloc_v8("val_a")
        node_addr_a = alloc_v8("node_addr_a")
        node_val_a = alloc_v8("node_val_a")
        hash_tmp1_a = alloc_v8("hash_tmp1_a")
        hash_tmp2_a = alloc_v8("hash_tmp2_a")
        mod_a = alloc_v8("mod_a")
        add1_a = alloc_v8("add1_a")
        mask_a = alloc_v8("mask_a")

        idx_b = alloc_v8("idx_b")
        val_b = alloc_v8("val_b")
        node_addr_b = alloc_v8("node_addr_b")
        node_val_b = alloc_v8("node_val_b")
        hash_tmp1_b = alloc_v8("hash_tmp1_b")
        hash_tmp2_b = alloc_v8("hash_tmp2_b")
        mod_b = alloc_v8("mod_b")
        add1_b = alloc_v8("add1_b")
        mask_b = alloc_v8("mask_b")

        # Broadcasted vector constants
        one_vec = self.scratch_const_vector(1, name="one_vec")
        two_vec = self.scratch_const_vector(2, name="two_vec")

        # Broadcast scalar header values into vector registers for comparisons/adds
        n_nodes_vec = self.alloc_scratch("n_nodes_vec", length=8)
        self.add("valu", ("vbroadcast", n_nodes_vec, self.scratch["n_nodes"]))
        forest_base_vec = self.alloc_scratch("forest_base_vec", length=8)
        self.add("valu", ("vbroadcast", forest_base_vec, self.scratch["forest_values_p"]))

        body = []

        # Vectorized main loop: process 8 elements per iteration
        vec_step = 8
        vec_iters = batch_size // vec_step
        tail = batch_size % vec_step

        # Pre-allocate scalar temporaries used by tail handling to avoid
        # repeated allocations inside the inner loop which bloated the
        # instruction stream and exhausted scratch space.
        tmp_idx_tail = self.alloc_scratch("tmp_idx_tail")
        tmp_val_tail = self.alloc_scratch("tmp_val_tail")
        tmp_node_val_tail = self.alloc_scratch("tmp_node_val_tail")
        h_tmp1 = self.alloc_scratch("h_tmp1")
        h_tmp2 = self.alloc_scratch("h_tmp2")
        tmp3_tail = self.alloc_scratch("tmp3_tail")
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        def emit_vec_round(idx_v, val_v, node_addr_v, node_val_v, h1, h2, mod_v, add1_v, mask_v):
            body.append(("valu", ("+", node_addr_v, idx_v, forest_base_vec)))
            for lane in range(8):
                body.append(("load", ("load_offset", node_val_v, node_addr_v, lane)))
            body.append(("valu", ("^", val_v, val_v, node_val_v)))
            body.extend(self.build_hash_vector(val_v, h1, h2))
            body.append(("valu", ("&", mod_v, val_v, one_vec)))
            body.append(("valu", ("multiply_add", add1_v, mod_v, one_vec, one_vec)))
            body.append(("valu", ("multiply_add", idx_v, idx_v, two_vec, add1_v)))
            body.append(("valu", ("<", mask_v, idx_v, n_nodes_vec)))
            body.append(("valu", ("*", idx_v, idx_v, mask_v)))

        vi = 0
        while vi < vec_iters:
            if vi + 1 < vec_iters:
                i0 = vi * vec_step
                i1 = (vi + 1) * vec_step
                i0c = self.scratch_const(i0)
                i1c = self.scratch_const(i1)
                body.append(("alu", ("+", tmp_i0, self.scratch["inp_indices_p"], i0c)))
                body.append(("alu", ("+", tmp_v0, self.scratch["inp_values_p"], i0c)))
                body.append(("alu", ("+", tmp_i1, self.scratch["inp_indices_p"], i1c)))
                body.append(("alu", ("+", tmp_v1, self.scratch["inp_values_p"], i1c)))
                body.append(("load", ("vload", idx_a, tmp_i0)))
                body.append(("load", ("vload", val_a, tmp_v0)))
                body.append(("load", ("vload", idx_b, tmp_i1)))
                body.append(("load", ("vload", val_b, tmp_v1)))
                for _r in range(rounds):
                    emit_vec_round(
                        idx_b, val_b, node_addr_b, node_val_b,
                        hash_tmp1_b, hash_tmp2_b, mod_b, add1_b, mask_b,
                    )
                    emit_vec_round(
                        idx_a, val_a, node_addr_a, node_val_a,
                        hash_tmp1_a, hash_tmp2_a, mod_a, add1_a, mask_a,
                    )
                body.append(("store", ("vstore", tmp_i0, idx_a)))
                body.append(("store", ("vstore", tmp_v0, val_a)))
                body.append(("store", ("vstore", tmp_i1, idx_b)))
                body.append(("store", ("vstore", tmp_v1, val_b)))
                vi += 2
            else:
                i = vi * vec_step
                ic = self.scratch_const(i)
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], ic)))
                body.append(("alu", ("+", tmp_addr2, self.scratch["inp_values_p"], ic)))
                body.append(("load", ("vload", idx_a, tmp_addr)))
                body.append(("load", ("vload", val_a, tmp_addr2)))
                for _r in range(rounds):
                    emit_vec_round(
                        idx_a, val_a, node_addr_a, node_val_a,
                        hash_tmp1_a, hash_tmp2_a, mod_a, add1_a, mask_a,
                    )
                body.append(("store", ("vstore", tmp_addr, idx_a)))
                body.append(("store", ("vstore", tmp_addr2, val_a)))
                vi += 1

        if tail:
            start = vec_iters * vec_step
            for i in range(start, start + tail):
                i_const = self.scratch_const(i)
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("alu", ("+", tmp_addr2, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("load", tmp_idx_tail, tmp_addr)))
                body.append(("load", ("load", tmp_val_tail, tmp_addr2)))

                for r in range(rounds):
                    body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx_tail)))
                    body.append(("load", ("load", tmp_node_val_tail, tmp_addr)))

                    body.append(("alu", ("^", tmp_val_tail, tmp_val_tail, tmp_node_val_tail)))
                    body.extend(self.build_hash(tmp_val_tail, h_tmp1, h_tmp2, r, i))

                    body.append(("alu", ("%", tmp_idx_tail, tmp_val_tail, two_const)))
                    body.append(("alu", ("+", tmp3_tail, tmp_idx_tail, one_const)))
                    body.append(("alu", ("*", tmp_idx_tail, tmp_idx_tail, two_const)))
                    body.append(("alu", ("+", tmp_idx_tail, tmp_idx_tail, tmp3_tail)))

                    body.append(("alu", ("<", tmp_idx_tail, tmp_idx_tail, self.scratch["n_nodes"])))
                    body.append(("alu", ("*", tmp_idx_tail, tmp_idx_tail, tmp_idx_tail)))

                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("store", tmp_addr, tmp_idx_tail)))

                body.append(("alu", ("+", tmp_addr2, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("store", tmp_addr2, tmp_val_tail)))

        # Pack into VLIW instructions
        # Include any pending slots (constants, broadcasts, header loads) so
        # they can be packed together with the main body. This reduces
        # instruction count and improves slot utilization.
        all_slots = self.pending_slots + body
        body_instrs = self.build(all_slots)
        self.instrs.extend(body_instrs)

        # Clear pending slots now that they've been emitted
        self.pending_slots = []

        # Final pause
        self.instrs.append({"flow": [("pause",)]})
