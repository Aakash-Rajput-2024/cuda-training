# CUDA Learning

Learning to write CUDA kernels on rented/free NVIDIA GPUs — since Apple Silicon has no NVIDIA GPU
and CUDA can't run locally on this Mac. The daily driver is **Google Colab's free T4**.

```
cuda-learning/
├── notebooks/   → Colab-ready notebooks (start with cuda_kernel_ladder.ipynb)
├── kernels/     → raw .cu files once you outgrow notebook cells (nvcc file.cu -o out)
└── benchmarks/  → timing logs (e.g. tiled matmul vs cuBLAS)
```

## Get onto a GPU (Google Colab, free)

Colab gives a free NVIDIA **T4 (~15 GB)** with the CUDA toolkit pre-installed. Pick one way to get
the notebook there:

| Route | Setup | Best when |
|-------|-------|-----------|
| **A. Manual upload** | Colab → `File → Upload notebook` → pick `cuda_kernel_ladder.ipynb` | You just want to start *today*. Zero setup. |
| **B. GitHub** *(recommended)* | Push this folder to a GitHub repo, then Colab → `File → Open notebook → GitHub` → paste repo URL. Save back with `File → Save a copy in GitHub`. | You want version control + a portfolio artifact. |
| **C. Google Drive** | Install *Google Drive for Desktop*, drop this folder in your synced Drive, then Colab → `File → Open notebook → Google Drive`. | You want auto-sync without git. |

**Then, every session:** `Runtime → Change runtime type → T4 GPU → Save`. Run the first cell —
`!nvidia-smi` should show a Tesla T4. That's a real NVIDIA GPU running your kernels.

Backups when Colab's free tier is busy: **Kaggle Notebooks** (~30 GPU-hrs/week) and
**[leetgpu.com](https://leetgpu.com)** (browser CUDA playground, no GPU needed — great for drills).

## The kernel ladder (in the starter notebook)

`vector add → tiled matmul → reduction → softmax → fused attention`

Kernels 1–2 are written and runnable; 3–5 are stubs with the problem stated. This exact sequence is
the GPU-systems interview / RA skillset. **Tie-in to your accelerator work:** Kernel 2 (tiled matmul
with shared memory) is the software echo of a systolic array — reimplement your NPU-project matmul
as a CUDA kernel and benchmark it against cuBLAS. That's a memorable SOP line.

## Week-by-week plan

- **Week 1 — Fundamentals.** Colab running. Vector add + tiled matmul from the ladder. Read PMPP ch.
  1–5. Understand: grid/block/thread, global vs shared memory, coalescing, `__syncthreads`.
- **Week 2 — Reduction & memory.** Kernel 3 through all three optimisation milestones. Learn warp
  divergence, bank conflicts, `__shfl_*` warp shuffles. PMPP ch. 6–8.
- **Week 3 — Softmax & fusion.** Kernel 4, then start Kernel 5. Numerical stability, two-pass
  reductions, keeping intermediates on-chip. Watch the relevant GPU MODE lectures.
- **Week 4 — Attention + benchmarking.** Finish the fused attention kernel. Benchmark tiled matmul
  vs `cublasSgemm`; log numbers in `benchmarks/`. Skim a FlashAttention explainer.
- **Next — Triton & llm.c.** Rewrite a kernel or two in OpenAI **Triton** (Python, runs on NVIDIA in
  Colab) — the modern ML-systems path. Then read Karpathy's **llm.c** (GPT-2 in raw C/CUDA).

## Resources

- **PMPP** — *Programming Massively Parallel Processors* (Kirk/Hwu). The standard CUDA book; work the
  exercises in Colab.
- **GPU MODE** (formerly CUDA MODE) — free YouTube lectures + Discord + exercise repo. Best community.
- **Triton** — OpenAI's Python kernel language; easier than raw CUDA, still real GPU code.
- **llm.c** — Karpathy's raw C/CUDA GPT-2 training, once the fundamentals click.

## Renting bigger GPUs later

When you outgrow free tiers: **RunPod / Vast.ai / Lambda** rent NVIDIA GPUs by the hour. Also check
whether IIT Patna's HPC/GPU cluster is open to you through your lab — free real hardware for big runs.
