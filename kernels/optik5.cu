#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

typedef unsigned int uint;

// ---- Kernel 5 tiling parameters (siboehm's canonical config) ----
#define BM 128 // rows of C computed per block tile
#define BN 128 // cols of C computed per block tile
#define BK 8   // depth of the K-slice staged in shared memory
#define TM 8   // rows of C computed per thread
#define TN 8   // cols of C computed per thread

#define WARMUP_ITERS 3
#define BENCH_ITERS 20

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Kernel 5: 2D block tiling + 2D register tiling.
// Each block computes a BM x BN tile of C. Each thread computes a TM x TN
// sub-tile of that block. Threads cooperatively stage a BM x BK slab of A and
// a BK x BN slab of B into shared memory, then accumulate outer products.
__global__ void sgemm_2d_blocktiling(int M, int N, int K, float alpha,
                                     const float *A, const float *B, float beta,
                                     float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // (BM*BN)/(TM*TN) = 256 threads cover the whole block tile.
  const uint numThreadsBlocktile = (BM * BN) / (TM * TN);
  assert(numThreadsBlocktile == blockDim.x);

  // Where this thread's TM x TN output tile sits inside the block tile.
  // BN/TN = 16 threads span a row, BM/TM = 16 threads span a column.
  const uint threadCol = threadIdx.x % (BN / TN); // 0..15
  const uint threadRow = threadIdx.x / (BN / TN); // 0..15

  __shared__ float As[BM * BK]; // 128 x 8
  __shared__ float Bs[BK * BN]; // 8 x 128

  // Move the block tile to the start of this block's row of A / col of B.
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // Cooperative-load indices for A -> As. Each thread loads one element per
  // step; strideA rows are covered per step, so BM/strideA = 4 steps total.
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  const uint strideA = numThreadsBlocktile / BK; // 32

  // Cooperative-load indices for B -> Bs. Consecutive threads hit consecutive
  // columns of B, keeping the global load coalesced.
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  const uint strideB = numThreadsBlocktile / BN; // 2

  // Register file storage: the 8x8 accumulators plus the slices we reuse.
  float threadResults[TM * TN] = {0.0f};
  float regM[TM] = {0.0f}; // one column slice of As
  float regN[TN] = {0.0f}; // one row slice of Bs

  // Outer loop over the K dimension, BK columns at a time.
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // ---- stage GMEM -> SMEM ----
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance the block tile one BK-slice along K
    A += BK;
    B += BK * N;

    // ---- compute: accumulate BK outer products into the 8x8 tile ----
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // pull this thread's slices out of SMEM once...
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      // ...then do TM*TN = 64 FMAs from just TM+TN = 16 register reads.
      for (uint resM = 0; resM < TM; ++resM) {
        for (uint resN = 0; resN < TN; ++resN) {
          threadResults[resM * TN + resN] += regM[resM] * regN[resN];
        }
      }
    }
    __syncthreads();
  }

  // ---- write the 8x8 tile back to C ----
  for (uint resM = 0; resM < TM; ++resM) {
    for (uint resN = 0; resN < TN; ++resN) {
      uint gRow = threadRow * TM + resM;
      uint gCol = threadCol * TN + resN;
      C[gRow * N + gCol] =
          alpha * threadResults[resM * TN + resN] + beta * C[gRow * N + gCol];
    }
  }
}

void sgemm_cpu(int M, int N, int K, float alpha, const float *A, const float *B,
               float beta, float *C) {
  for (int r = 0; r < M; r++) {
    for (int c = 0; c < N; c++) {
      float t = 0.0f;
      for (int i = 0; i < K; i++) {
        t += A[r * K + i] * B[i * N + c];
      }
      C[r * N + c] = alpha * t + beta * C[r * N + c];
    }
  }
}

bool check_answer(const float *ref, const float *test, int n,
                  float atol = 1e-3f, float rtol = 1e-3f) {
  int mismatches = 0;
  for (int i = 0; i < n; i++) {
    float diff = fabsf(ref[i] - test[i]);
    float denom = fmaxf(fabsf(ref[i]), 1e-8f);
    if (diff > atol && (diff / denom) > rtol) {
      if (mismatches < 5) {
        printf("  MISMATCH [%d]: ref=%.6f  got=%.6f  diff=%.6f\n", i, ref[i],
               test[i], diff);
      }
      mismatches++;
    }
  }
  if (mismatches > 0) {
    printf("  Total mismatches: %d / %d\n", mismatches, n);
  }
  return mismatches == 0;
}

int main() {
  const int M = 1024;
  const int N = 1024;
  const int K = 1024;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  printf("=======================================================\n");
  printf("  KERNEL 5 — SGEMM 2D BLOCKTILING\n");
  printf("  M=%d N=%d K=%d | BM=%d BN=%d BK=%d TM=%d TN=%d\n", M, N, K, BM, BN,
         BK, TM, TN);
  printf("=======================================================\n\n");

  size_t sizeA = M * K * sizeof(float);
  size_t sizeB = K * N * sizeof(float);
  size_t sizeC = M * N * sizeof(float);

  float *hA = (float *)malloc(sizeA);
  float *hB = (float *)malloc(sizeB);
  float *hC = (float *)malloc(sizeC);
  float *hCref = (float *)malloc(sizeC);
  float *hCgpu = (float *)malloc(sizeC);

  srand(42);
  for (int i = 0; i < M * K; i++)
    hA[i] = (float)(rand() % 100) / 100.0f;
  for (int i = 0; i < K * N; i++)
    hB[i] = (float)(rand() % 100) / 100.0f;
  for (int i = 0; i < M * N; i++)
    hC[i] = 0.0f;

  printf("Computing CPU reference SGEMM...\n");
  memcpy(hCref, hC, sizeC);
  sgemm_cpu(M, N, K, alpha, hA, hB, beta, hCref);
  printf("CPU reference done.\n\n");

  float *dA, *dB, *dC, *dC_init;
  CUDA_CHECK(cudaMalloc(&dA, sizeA));
  CUDA_CHECK(cudaMalloc(&dB, sizeB));
  CUDA_CHECK(cudaMalloc(&dC, sizeC));
  CUDA_CHECK(cudaMalloc(&dC_init, sizeC));

  CUDA_CHECK(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC_init, hC, sizeC, cudaMemcpyHostToDevice));

  // 256 threads per block, an 8x8 grid of block tiles for 1024x1024.
  dim3 block((BM * BN) / (TM * TN));
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

  printf("Grid (%d,%d) | Block (%d)\n", grid.x, grid.y, block.x);

  // correctness
  CUDA_CHECK(cudaMemcpy(dC, dC_init, sizeC, cudaMemcpyDeviceToDevice));
  sgemm_2d_blocktiling<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(hCgpu, dC, sizeC, cudaMemcpyDeviceToHost));
  bool correct = check_answer(hCref, hCgpu, M * N);
  printf("Verification : %s\n", correct ? "PASS ✓" : "FAIL ✗");

  // benchmark
  for (int i = 0; i < WARMUP_ITERS; i++) {
    CUDA_CHECK(cudaMemcpy(dC, dC_init, sizeC, cudaMemcpyDeviceToDevice));
    sgemm_2d_blocktiling<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < BENCH_ITERS; i++) {
    CUDA_CHECK(cudaMemcpy(dC, dC_init, sizeC, cudaMemcpyDeviceToDevice));
    sgemm_2d_blocktiling<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  float avg_ms = total_ms / BENCH_ITERS;

  double gflops = (2.0 * M * N * K) / (avg_ms * 1e6);
  printf("Avg Latency  : %.3f ms\n", avg_ms);
  printf("Throughput   : %.1f GFLOP/s\n", gflops);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(dC_init));
  free(hA);
  free(hB);
  free(hC);
  free(hCref);
  free(hCgpu);

  return 0;
}
