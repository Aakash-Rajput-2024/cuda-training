
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define BLOCKSIZE 32
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

__global__ void sgemm_shared(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C) {
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
  sgemm_shared<<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
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

  printf("==========================================================\n");
  printf("  SGEMM Benchmark — M=%d  N=%d  K=%d  (BLOCKSIZE=%d)\n", M, N, K,
         BLOCKSIZE);
  printf("==========================================================\n\n");

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

  memcpy(hCref, hC, sizeC);
  sgemm_cpu(M, N, K, alpha, hA, hB, beta, hCref);

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

  dim3 blockShared(BLOCKSIZE, BLOCKSIZE);
  dim3 gridShared((N + BLOCKSIZE - 1) / BLOCKSIZE,
                  (M + BLOCKSIZE - 1) / BLOCKSIZE);

  struct KernelInfo {
    const char *name;
    KernelLauncher launcher;
    dim3 grid;
    dim3 block;
  };

  KernelInfo kernels[] = {
      {"Naive (COALESCED — col=threadIdx.x)", launch_coalesced, gridCoal,
       blockCoal},
      {"Naive (ANTI-COALESCED — row=threadIdx.x)", launch_anticoalesced,
       gridAnti, blockAnti},
      {"Shared-memory tiled", launch_shared, gridShared, blockShared},
  };
  int numKernels = sizeof(kernels) / sizeof(kernels[0]);

  for (int k = 0; k < numKernels; k++) {
    printf("----------------------------------------------------------\n");
    printf("  Kernel: %s\n", kernels[k].name);
    printf("----------------------------------------------------------\n");

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
    printf("  Correctness : %s\n", correct ? "PASS ✓" : "FAIL ✗");

    float avg_ms =
        benchmark_kernel(kernels[k].launcher, M, N, K, alpha, dA, dB, beta, dC,
                         dC_init, kernels[k].grid, kernels[k].block);

    double gflops = (2.0 * M * N * K) / (avg_ms * 1e6);
    printf("  Avg time    : %.3f ms\n", avg_ms);
    printf("  Throughput  : %.1f GFLOP/s\n\n", gflops);
  }

  printf("==========================================================\n");
  printf("  COALESCING COMPARISON SUMMARY\n");
  printf("==========================================================\n");
  printf("  If coalescing matters, the 'COALESCED' kernel should be\n");
  printf("  noticeably faster than the 'ANTI-COALESCED' kernel.\n");
  printf("  The shared-memory kernel avoids the issue via tiling.\n");
  printf("==========================================================\n");

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