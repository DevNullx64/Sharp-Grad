# Compilation Plan

## Points
1. Forward API and execution contract
- status: pending
- notes: design signatures, single vs multi-output, activations exposure
2. Slot ordering and binding (stable mapping Value->buffers)
- status: pending
3. Subgraph partitioning (barriers/reductions)
- status: pending
4. Kernel generation (forward, reduction) with nested loops (no linear index)
- status: pending
5. Parallel scheduling (lock-free strategy)
- status: pending
6. Constant binding (single-time validation/cache)
- status: pending
7. Backward kernels and gradient accumulation strategy
- status: pending
8. Validation/tests
- status: pending
9. Developer ergonomics (descriptors, helpers)
- status: pending

## Evolution Log
- 2026-01-17: Plan initialized. Focus next on Point 1: Forward API.
