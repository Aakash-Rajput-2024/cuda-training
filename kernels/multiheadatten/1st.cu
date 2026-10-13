#include <cuda_runtime.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void compute_scores(const float *Q, const float *K, float *score,
                               int N, int d_model, int dk, int h_idx) {
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;

  if (row < N && col < N) {
    float sum = 0.0f;

    int q_i = row * d_model + h_idx * dk;
    int k_i = col * d_model + h_idx * dk;

    for (int m = 0; m < dk; m++) {
      sum += Q[q_i + m] * K[k_i + m];
    }

    score[row * N + col] = sum / sqrtf((float)dk);
  }
}

__global__ void row_softmax(float *scores, int N) {
  int row = threadIdx.x + blockDim.x * blockIdx.x;

  if (row < N) {
    float max_val = scores[row * N];

    for (int i = 0; i < N; i++) {
      if (scores[row * N + i] > max_val)
        max_val = scores[row * N + i];
    }

    float sum = 0.0f;

    for (int col = 0; col < N; col++) {
      scores[col + row * N] = expf(scores[col + row * N] - max_val);
      sum += scores[col + row * N];
    }

    for (int col = 0; col < N; col++) {
      scores[col + row * N] = scores[col + row * N] / sum;
    }
  }
}

__global__ void compute_output(const float *scores, const float *V,
                               float *output, int N, int d_model, int dk,
                               int h_idx) {
  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int local_c = threadIdx.x + blockDim.x * blockIdx.x;

  if (row < N && local_c < dk) {
    float sum = 0;
    for (int k = 0; k < N; k++) {
      sum += scores[row * N + k] * V[k * d_model + h_idx * dk + local_c];
    }
    output[row * d_model + h_idx * dk + local_c] = sum;
  }
}

extern "C" void solve(const float *Q, const float *K, const float *V,
                      float *output, int N, int d_model, int h) {
  int dk = d_model / h;

  float *d_scores;
  cudaMalloc(&d_scores, (size_t)N * N * sizeof(float));

  dim3 block16(16, 16);
  dim3 grid16((N + 15) / 16, (N + 15) / 16);

  dim3 block_dk(16, 16);
  dim3 grid_dk((dk + 15) / 16, (N + 15) / 16);

  int threads_per_block_1d = 256;
  int blocks_1d = (N + threads_per_block_1d - 1) / threads_per_block_1d;

  for (int i = 0; i < h; ++i) {
    compute_scores<<<grid16, block16>>>(Q, K, d_scores, N, d_model, dk, i);
    row_softmax<<<blocks_1d, threads_per_block_1d>>>(d_scores, N);
    compute_output<<<grid_dk, block_dk>>>(d_scores, V, output, N, d_model, dk,
                                          i);
  }

  cudaDeviceSynchronize();
  cudaFree(d_scores);
}

// ---------------------------------------------------------------------------
// Host verification & benchmarking harness for Colab execution
// ---------------------------------------------------------------------------

bool verify_correctness(const float *h_Q, const float *h_K, const float *h_V,
                        const float *gpu_out, int N, int d_model, int h,
                        float tol) {
  int dk = d_model / h;
  float scale = 1.0f / sqrtf((float)dk);

  // 1. Sanity check: Ensure all GPU output elements are finite (no NaN, no Inf)
  for (size_t i = 0; i < (size_t)N * d_model; i++) {
    if (isnan(gpu_out[i]) || isinf(gpu_out[i])) {
      printf("  [FAIL] GPU output contains NaN/Inf at index %zu!\n", i);
      return false;
    }
  }

  // 2. Mathematical CPU verification:
  // For small N (<= 1024), check every row.
  // For large N (stress test), check representative sample rows to keep verification sub-second
  // and prevent host RAM overflow.
  int num_sample_rows = 5;
  int sample_rows[5];
  if (N <= 1024) {
    num_sample_rows = N;
  } else {
    sample_rows[0] = 0;
    sample_rows[1] = N / 4;
    sample_rows[2] = N / 2;
    sample_rows[3] = 3 * N / 4;
    sample_rows[4] = N - 1;
  }

  float *row_scores = (float *)malloc(N * sizeof(float));
  if (!row_scores) {
    fprintf(stderr, "Failed to allocate row_scores buffer\n");
    return false;
  }

  float max_diff = 0.0f;

  for (int s = 0; s < num_sample_rows; s++) {
    int r = (N <= 1024) ? s : sample_rows[s];

    for (int h_idx = 0; h_idx < h; h_idx++) {
      int q_offset = r * d_model + h_idx * dk;
      float max_val = -1e30f;

      for (int c = 0; c < N; c++) {
        float dot = 0.0f;
        int k_offset = c * d_model + h_idx * dk;
        for (int m = 0; m < dk; m++) {
          dot += h_Q[q_offset + m] * h_K[k_offset + m];
        }
        float sc = dot * scale;
        row_scores[c] = sc;
        if (sc > max_val)
          max_val = sc;
      }

      float sum_exp = 0.0f;
      for (int c = 0; c < N; c++) {
        row_scores[c] = expf(row_scores[c] - max_val);
        sum_exp += row_scores[c];
      }
      for (int c = 0; c < N; c++) {
        row_scores[c] /= sum_exp;
      }

      for (int c = 0; c < dk; c++) {
        float ref_val = 0.0f;
        for (int k = 0; k < N; k++) {
          ref_val += row_scores[k] * h_V[k * d_model + h_idx * dk + c];
        }
        int out_idx = r * d_model + h_idx * dk + c;
        float diff = fabsf(ref_val - gpu_out[out_idx]);
        if (diff > max_diff)
          max_diff = diff;

        if (diff > tol) {
          printf("  [FAIL] Mismatch at row %d, head %d, dim %d: ref=%.6f, "
                 "gpu=%.6f, diff=%.6f\n",
                 r, h_idx, c, ref_val, gpu_out[out_idx], diff);
          free(row_scores);
          return false;
        }
      }
    }
  }

  free(row_scores);
  printf("  [PASS] Mathematical verification successful (checked %d rows, "
         "max diff: %.6e, tol: %.6e)\n",
         num_sample_rows, max_diff, tol);
  return true;
}

int main(int argc, char **argv) {
  int N = 512;       // Sequence length
  int d_model = 512; // Hidden dimension
  int h = 8;         // Number of heads

  if (argc >= 2)
    N = atoi(argv[1]);
  if (argc >= 3)
    d_model = atoi(argv[2]);
  if (argc >= 4)
    h = atoi(argv[3]);

  int dk = d_model / h;
  size_t bytes_qkv = (size_t)N * d_model * sizeof(float);
  size_t bytes_scores = (size_t)N * N * sizeof(float);
  double total_gpu_gb = (4.0 * bytes_qkv + bytes_scores) / (1024.0 * 1024.0 * 1024.0);

  printf(
      "===================================================================\n");
  printf("  Multi-Head Attention Benchmark: 1st (Baseline Naive)\n");
  printf("  Config: N=%d, d_model=%d, heads=%d, dk=%d\n", N, d_model, h, dk);
  printf("  Total GPU Memory Footprint: %.2f GB\n", total_gpu_gb);
  printf(
      "===================================================================\n");

  float *h_Q = (float *)malloc(bytes_qkv);
  float *h_K = (float *)malloc(bytes_qkv);
  float *h_V = (float *)malloc(bytes_qkv);
  float *h_out = (float *)malloc(bytes_qkv);

  if (!h_Q || !h_K || !h_V || !h_out) {
    fprintf(stderr, "Host memory allocation failed\n");
    return 1;
  }

  // Deterministic initialization in [-0.5, 0.5]
  srand(42);
  for (size_t i = 0; i < (size_t)N * d_model; i++) {
    h_Q[i] = ((float)rand() / (float)RAND_MAX) - 0.5f;
    h_K[i] = ((float)rand() / (float)RAND_MAX) - 0.5f;
    h_V[i] = ((float)rand() / (float)RAND_MAX) - 0.5f;
  }

  float *d_Q, *d_K, *d_V, *d_out;
  cudaMalloc(&d_Q, bytes_qkv);
  cudaMalloc(&d_K, bytes_qkv);
  cudaMalloc(&d_V, bytes_qkv);
  cudaMalloc(&d_out, bytes_qkv);

  cudaMemcpy(d_Q, h_Q, bytes_qkv, cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, h_K, bytes_qkv, cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, h_V, bytes_qkv, cudaMemcpyHostToDevice);

  // 1. Correctness verification against CPU reference
  printf("\n[1/2] Verifying correctness against reference...\n");
  solve(d_Q, d_K, d_V, d_out, N, d_model, h);
  cudaDeviceSynchronize();
  cudaMemcpy(h_out, d_out, bytes_qkv, cudaMemcpyDeviceToHost);

  bool passed = verify_correctness(h_Q, h_K, h_V, h_out, N, d_model, h, 1e-3f);

  // 2. Benchmarking with CUDA events
  printf("\n[2/2] Benchmarking...\n");
  int warmup_iters = (N >= 16384) ? 2 : 5;
  int bench_iters = (N >= 16384) ? 5 : 20;

  for (int i = 0; i < warmup_iters; i++) {
    solve(d_Q, d_K, d_V, d_out, N, d_model, h);
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < bench_iters; i++) {
    solve(d_Q, d_K, d_V, d_out, N, d_model, h);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, start, stop);
  float avg_ms = total_ms / bench_iters;

  // FLOPs calculation: ~ 4 * N^2 * d_model
  double flops = 4.0 * (double)N * (double)N * (double)d_model;
  double gflops = flops / (avg_ms * 1e6);

  printf(
      "-------------------------------------------------------------------\n");
  printf("  Verification : %s\n", passed ? "PASS" : "FAIL");
  printf("  Avg Latency  : %.4f ms\n", avg_ms);
  printf("  Throughput   : %.2f GFLOP/s\n", gflops);
  printf(
      "-------------------------------------------------------------------\n");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_V);
  cudaFree(d_out);
  free(h_Q);
  free(h_K);
  free(h_V);
  free(h_out);

  return passed ? 0 : 1;
}