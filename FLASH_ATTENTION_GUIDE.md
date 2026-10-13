# Flash Attention from Scratch in CUDA — Build Guide

Goal: implement Flash Attention v1 (fp16, warp-tiling) from scratch, validated against a
reference. You write all the code. This is the map. Climb the rungs in order — do not
jump to fp16 warp-tiling before you have a *correct* fp32 version.

Target: real FlashAttention v1 design (Dao et al. 2022, Algorithm 1).

---

## The core idea (read once, then keep the skeleton below in your head)

Standard attention materializes an N×N score matrix:

    S = Q @ Kᵀ / sqrt(d)     # (N×d)@(d×N) → N×N
    P = softmax(S)           # row-wise
    O = P @ V                # (N×N)@(N×d) → N×d

The N×N matrix S is the enemy: N=4096 → 16M floats of memory traffic. Flash Attention
NEVER writes S to global memory. It streams K/V in tiles and keeps a running softmax
per query row using the **online softmax** recurrence.

### Online softmax (the only new math)

Per query row, keep three running values:
- `m`  = running max of scores seen so far          (scalar)
- `l`  = running sum of exp(score - m)              (scalar)
- `O`  = running weighted output accumulator        (vector length d)

When a new score block arrives with local max `m_new`:

    m'   = max(m, m_new)
    corr = exp(m - m')                # rescales the old running state
    l'   = corr*l + sum(exp(scores - m'))
    O'   = corr*O + exp(scores - m') @ V_block
    m    = m'

Normalize O by l ONCE at the very end. That's the whole trick.

Two bugs everyone hits — decide now and be consistent everywhere:
1. The `1/sqrt(d)` scale goes on scores BEFORE softmax, in every phase.
2. `S - max` subtraction is numerical stability only; softmax is shift-invariant. It IS
   the online running-max. If it feels foreign, re-derive it on paper.

### Kernel skeleton (the destination)

    one thread-block owns one tile of query rows (Br × d), for one (batch, head)
      load Q_tile → shared mem, keep resident
      init  m = -inf,  l = 0,  O_acc = 0        (per query row)
      for each K/V tile (Bc × d) along the sequence:
          load K_tile, V_tile → shared mem
          S = (Q_tile @ K_tileᵀ) * scale         ← matmul #1
          m_new = rowmax(S)
          m'    = max(m, m_new)
          P     = exp(S - m')
          corr  = exp(m - m')
          l     = corr*l + rowsum(P)
          O_acc = corr*O_acc + P @ V_tile         ← matmul #2
          m     = m'
      O_tile = O_acc / l                          ← normalize once, at the end
      write O_tile → global mem

---

## Phases

### Phase 0 — Reference oracle (½ day)
Plain attention in NumPy. Ground truth for every later check. Dump Q,K,V,O as raw
binaries so the CUDA side can load identical inputs.

```python
import numpy as np, os
os.makedirs("data", exist_ok=True)
def attention_ref(Q, K, V):
    d = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d)
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S); P = P / P.sum(axis=-1, keepdims=True)
    return P @ V
np.random.seed(0)
N, d = 128, 64
Q,K,V = (np.random.randn(N,d).astype(np.float32) for _ in range(3))
O = attention_ref(Q,K,V)
for n,a in [("Q",Q),("K",K),("V",V),("O",O)]: a.tofile(f"data/{n}.bin")
print("wrote", N, d)
```
Done when: it runs and writes 4 .bin files.

### Phase 1 — Naive CUDA attention (1 day)
Three separate kernels, materialize S in global memory:
  1. `S = QKᵀ * scale`   (reuse your tiled matmul)
  2. `softmax(S)`        (one row per block; max-reduce, exp, sum-reduce, divide)
  3. `O = P @ V`         (reuse tiled matmul)
Load Q,K,V from the .bin files, compare O to data/O.bin.
Goal: CORRECT numerics, not speed.
Done when: max abs error vs oracle < 1e-3.

### Phase 2 — Online softmax on CPU (½ day)
Implement the running m/l/O recurrence in single-threaded C++. Verify it matches Phase 1
bit-for-bit-ish. This de-risks the MATH from the PARALLELISM before they combine.
Done when: CPU online version matches Phase 1 output.

### Phase 3 — Flash Attention v1 kernel, fp32 (3–5 days) ← the main event
One fused kernel, the skeleton above. Start simple:
  - grid: one block per (query-tile × batch × head)
  - one warp (or one thread) per query row
  - Q_tile loaded to shared mem once; K/V tiles streamed through shared mem
  - keep m, l in registers; O_acc in registers or shared mem
  - float everywhere, no tensor cores yet
Done when: matches oracle < 1e-3, AND no N×N buffer is ever allocated.

### Phase 4 — Optimize + fp16 WMMA (the summit)
Only after Phase 3 is green:
  - tune Br, Bc tile sizes (watch shared-mem budget)
  - coalesce global loads (K/V should be contiguous per tile)
  - swap both matmuls to WMMA tensor cores (start from your wmma1.cu)
  - inputs fp16, accumulate softmax stats in fp32 (never accumulate softmax in fp16)
  - double-buffer K/V loads to overlap load with compute (uses your streams knowledge)
Done when: matches oracle, and you can benchmark it vs Phase 1 and vs PyTorch SDPA.

---

## Numerical gotchas (bookmark this)
- Accumulate m, l, and O in **fp32** even when Q/K/V are fp16. Softmax in fp16 = NaN city.
- `exp(m - m')` when m starts at -inf: guard so exp(-inf) = 0, not NaN. Init m = -1e30.
- Apply `1/sqrt(d)` exactly once. If off by that factor, softmax looks "too flat/sharp".
- Compare with tolerance (max abs err, or rel err), never `==`. fp is not associative.
- When you add batch/heads: O[b,h] must use K[b,h],V[b,h] — index bugs read wrong head.

## Validation harness (build once, reuse every phase)
A tiny C++/host function: load Q,K,V,O.bin → run your kernel → print max_abs_err and
mean_abs_err vs O.bin. Every phase reports the same two numbers. If a change makes the
error jump, you know exactly which change did it.

## Suggested layout
    reference/attn_ref.py       # Phase 0 oracle + test-case generator
    data/                       # Q.bin K.bin V.bin O.bin
    src/attention_naive.cu      # Phase 1
    src/flash_attn.cu           # Phase 3-4
    src/check.h                 # max/mean abs error helper (Phase 1+)
    Makefile
    README.md                   # your notes: derivations + benchmark table

## Reference
- FlashAttention paper, Algorithm 1 (the pseudocode maps 1:1 to Phase 3).
- Keep a running benchmark table in README: (N, d) → naive ms / flash ms / speedup.
```
