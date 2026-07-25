#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

typedef unsigned int uint;

#define BLOCKSIZE 32
#define WARMUP_ITERS 3
#define BENCH_ITERS 20

#define BM 64
#define BK 8
#define BN 64
#define TM 8

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void sgemm_naive_coalesced(int M, int N, int K, float alpha,
                                      const float *A, const float *B,
                                      float beta, float *C) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < M && col < N) {
    float t = 0.0f;
    for (int i = 0; i < K; i++) {
      t += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = alpha * t + beta * C[row * N + col];
  }
}

__global__ void sgemm_naive_anticoalesced(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  int col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < M && col < N) {
    float t = 0.0f;
    for (int i = 0; i < K; i++) {
      t += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = alpha * t + beta * C[row * N + col];
  }
}

__global__ void sgemm_shared_tiled(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  const int threadRow = threadIdx.y;
  const int threadCol = threadIdx.x;

  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  A += cRow * BLOCKSIZE * K;
  B += cCol * BLOCKSIZE;
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  float tmp = 0.0f;

  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    __syncthreads();

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    for (int i = 0; i < BLOCKSIZE; i++) {
      tmp += As[threadRow * BLOCKSIZE + i] * Bs[i * BLOCKSIZE + threadCol];
    }

    __syncthreads();
  }

  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}

__global__ void sgemm_1d_blocktiling(int M, int N, int K, float alpha,
                                     const float *A, const float *B, float beta,
                                     float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint threadCol = threadIdx.x % BN;
  const uint threadRow = threadIdx.x / BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;

  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;

  float threadResult[TM] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];

    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float Btmp = Bs[dotIdx * BN + threadCol];
      for (uint i = 0; i < TM; ++i) {
        threadResult[i] += As[(threadRow * TM + i) * BK + dotIdx] * Btmp;
      }
    }

    __syncthreads();
  }

  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    uint gr = threadRow * TM + resIdx;
    uint gc = threadCol;

    if ((cRow * BM + gr) < M && (cCol * BN + gc) < N) {
      C[gr * N + gc] = alpha * threadResult[resIdx] + beta * C[gr * N + gc];
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

typedef void (*KernelLauncher)(int M, int N, int K, float alpha,
                               const float *dA, const float *dB, float beta,
                               float *dC, dim3 grid, dim3 block);

void launch_coalesced(int M, int N, int K, float alpha, const float *dA,
                      const float *dB, float beta, float *dC, dim3 grid,
                      dim3 block) {
  sgemm_naive_coalesced<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
}

void launch_anticoalesced(int M, int N, int K, float alpha, const float *dA,
                          const float *dB, float beta, float *dC, dim3 grid,
                          dim3 block) {
  sgemm_naive_anticoalesced<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
}

void launch_shared(int M, int N, int K, float alpha, const float *dA,
                   const float *dB, float beta, float *dC, dim3 grid,
                   dim3 block) {
  sgemm_shared_tiled<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
}

void launch_1d_blocktiling(int M, int N, int K, float alpha, const float *dA,
                           const float *dB, float beta, float *dC, dim3 grid,
                           dim3 block) {
  sgemm_1d_blocktiling<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
}

float benchmark_kernel(KernelLauncher launcher, int M, int N, int K,
                       float alpha, const float *dA, const float *dB,
                       float beta, float *dC, float *dC_init, dim3 grid,
                       dim3 block) {
  for (int i = 0; i < WARMUP_ITERS; i++) {
    CUDA_CHECK(cudaMemcpy(dC, dC_init, M * N * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    launcher(M, N, K, alpha, dA, dB, beta, dC, grid, block);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < BENCH_ITERS; i++) {
    CUDA_CHECK(cudaMemcpy(dC, dC_init, M * N * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    launcher(M, N, K, alpha, dA, dB, beta, dC, grid, block);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return total_ms / BENCH_ITERS;
}

int main() {
  const int M = 1024;
  const int N = 1024;
  const int K = 1024;
  const float alpha = 1.0f;
  const float beta = 0.0f;

  printf("====================================================================="
         "==\n");
  printf("  SGEMM BENCHMARK — MATRIX MULTIPLICATION KERNEL OPTIMIZATION SUITE  "
         "  \n");
  printf("  Matrix Dimensions: M=%d, N=%d, K=%d\n", M, N, K);
  printf("====================================================================="
         "==\n\n");

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

  printf("Computing CPU reference SGEMM solution...\n");
  memcpy(hCref, hC, sizeC);
  sgemm_cpu(M, N, K, alpha, hA, hB, beta, hCref);
  printf("CPU reference computation complete.\n\n");

  float *dA, *dB, *dC, *dC_init;
  CUDA_CHECK(cudaMalloc(&dA, sizeA));
  CUDA_CHECK(cudaMalloc(&dB, sizeB));
  CUDA_CHECK(cudaMalloc(&dC, sizeC));
  CUDA_CHECK(cudaMalloc(&dC_init, sizeC));

  CUDA_CHECK(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dC_init, hC, sizeC, cudaMemcpyHostToDevice));

  dim3 blockCoal(BLOCKSIZE, BLOCKSIZE);
  dim3 gridCoal((N + BLOCKSIZE - 1) / BLOCKSIZE,
                (M + BLOCKSIZE - 1) / BLOCKSIZE);

  dim3 blockAnti(BLOCKSIZE, BLOCKSIZE);
  dim3 gridAnti((M + BLOCKSIZE - 1) / BLOCKSIZE,
                (N + BLOCKSIZE - 1) / BLOCKSIZE);

  dim3 block1D((BM / TM) * BN);
  dim3 grid1D((N + BN - 1) / BN, (M + BM - 1) / BM);

  struct KernelInfo {
    const char *name;
    KernelLauncher launcher;
    dim3 grid;
    dim3 block;
  };

  KernelInfo kernels[] = {
      {"Kernel 1: Naive (Coalesced — col=threadIdx.x)", launch_coalesced,
       gridCoal, blockCoal},
      {"Kernel 2: Naive (Anti-Coalesced — row=threadIdx.x)",
       launch_anticoalesced, gridAnti, blockAnti},
      {"Kernel 3: 2D Shared Memory Tiling (32x32)", launch_shared, gridCoal,
       blockCoal},
      {"Kernel 4: 1D Block Tiling + Register Tiling (64x64 Tile, TM=8)",
       launch_1d_blocktiling, grid1D, block1D},
  };

  int numKernels = sizeof(kernels) / sizeof(kernels[0]);

  for (int k = 0; k < numKernels; k++) {
    printf("-------------------------------------------------------------------"
           "----\n");
    printf("  Executing: %s\n", kernels[k].name);
    printf("  Grid Layout : (%d, %d, %d) | Block Layout: (%d, %d, %d)\n",
           kernels[k].grid.x, kernels[k].grid.y, kernels[k].grid.z,
           kernels[k].block.x, kernels[k].block.y, kernels[k].block.z);
    printf("-------------------------------------------------------------------"
           "----\n");

    CUDA_CHECK(cudaMemcpy(dC, dC_init, sizeC, cudaMemcpyDeviceToDevice));
    kernels[k].launcher(M, N, K, alpha, dA, dB, beta, dC, kernels[k].grid,
                        kernels[k].block);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("  LAUNCH ERROR: %s\n\n", cudaGetErrorString(err));
      continue;
    }

    CUDA_CHECK(cudaMemcpy(hCgpu, dC, sizeC, cudaMemcpyDeviceToHost));
    bool correct = check_answer(hCref, hCgpu, M * N);
    printf("  Verification : %s\n", correct ? "PASS ✓" : "FAIL ✗");

    float avg_ms =
        benchmark_kernel(kernels[k].launcher, M, N, K, alpha, dA, dB, beta, dC,
                         dC_init, kernels[k].grid, kernels[k].block);

    double gflops = (2.0 * M * N * K) / (avg_ms * 1e6);
    printf("  Avg Latency  : %.3f ms\n", avg_ms);
    printf("  Throughput   : %.1f GFLOP/s\n\n", gflops);
  }

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