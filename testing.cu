// testing.cu — smoke test for the CUDA setup.
//
// Confirms the whole chain works end to end:
//   [0] the CUDA runtime sees an NVIDIA GPU (so nvcc + drivers are fine)
//   [1] vector add          — correct vs a CPU reference
//   [2] tiled matmul        — correct vs a CPU reference (double precision)
//
// Kernels are identical to notebooks/cuda_kernel_ladder.ipynb, so a green run
// here means those notebook cells are trustworthy too.
//
// Run it in Colab (Runtime -> T4 GPU), NOT on your Mac (no NVIDIA GPU locally):
//   !nvcc -O2 testing.cu -o testing && ./testing
// Optional, tuned for the T4 (compute capability 7.5):
//   !nvcc -O2 -arch=sm_75 testing.cu -o testing && ./testing
//
// Exit code is 0 only if every test passes (handy in scripts / CI).

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Abort with file:line if a CUDA runtime call fails.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(_e));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Catch both launch-time errors and errors surfaced during execution.
#define CUDA_CHECK_KERNEL()                                                     \
    do {                                                                        \
        CUDA_CHECK(cudaGetLastError());                                         \
        CUDA_CHECK(cudaDeviceSynchronize());                                    \
    } while (0)

#define TILE 16

// ---- kernels (same as the notebook) ---------------------------------------

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void matmulTiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int t = 0; t < N / TILE; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE; k++)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * N + col] = acc;
}

// ---- [0] device query -----------------------------------------------------

static int deviceInfo(void) {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        printf("  No CUDA device found. In Colab: Runtime -> Change runtime type -> T4 GPU.\n");
        return 0;
    }
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    printf("  GPU 0             : %s\n", p.name);
    printf("  Compute capability: %d.%d\n", p.major, p.minor);
    printf("  Global memory     : %.2f GB\n", (double)p.totalGlobalMem / 1e9);
    printf("  Streaming multiproc: %d\n", p.multiProcessorCount);
    printf("  Shared mem / block: %zu KB\n", p.sharedMemPerBlock / 1024);
    return count;
}

// ---- [1] vector add -------------------------------------------------------

static int testVecAdd(void) {
    const int N = 1 << 20;                    // ~1M elements
    size_t bytes = (size_t)N * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {             // deterministic, non-trivial input
        h_a[i] = (float)(i % 100) * 0.5f;
        h_b[i] = (float)(i % 7);
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    vecAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    double maxErr = 0.0;
    for (int i = 0; i < N; i++)
        maxErr = fmax(maxErr, fabs((double)h_c[i] - (double)(h_a[i] + h_b[i])));

    int pass = maxErr < 1e-5;
    printf("  max abs error = %.2e  ->  %s\n", maxErr, pass ? "[PASS]" : "[FAIL]");

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return pass;
}

// ---- [2] tiled matmul -----------------------------------------------------

static int testMatmul(void) {
    const int N = 512;                        // must be a multiple of TILE
    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    for (int i = 0; i < N * N; i++) {         // deterministic values in [0,1)
        h_A[i] = (float)((i * 3 + 1) % 13) / 13.0f;
        h_B[i] = (float)((i * 5 + 2) % 17) / 17.0f;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 threads(TILE, TILE);
    dim3 blocks(N / TILE, N / TILE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    matmulTiled<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK_KERNEL();

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // CPU reference in double, then compare relative error per element.
    double maxRel = 0.0;
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < N; c++) {
            double ref = 0.0;
            for (int k = 0; k < N; k++)
                ref += (double)h_A[r * N + k] * (double)h_B[k * N + c];
            double diff = fabs((double)h_C[r * N + c] - ref);
            maxRel = fmax(maxRel, diff / (fabs(ref) + 1e-6));
        }
    }

    double gflops = 2.0 * N * N * N / (ms / 1e3) / 1e9;
    int pass = maxRel < 1e-3;
    printf("  N=%d  %.3f ms  (%.1f GFLOP/s)\n", N, ms, gflops);
    printf("  max relative error = %.2e  ->  %s\n", maxRel, pass ? "[PASS]" : "[FAIL]");

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return pass;
}

// ---- driver ---------------------------------------------------------------

int main(void) {
    printf("=== CUDA setup smoke test ===\n\n");

    printf("[0] Device query\n");
    if (!deviceInfo()) return 1;

    int passed = 0, total = 0;

    printf("\n[1] Vector add\n");
    total++; passed += testVecAdd();

    printf("\n[2] Tiled matmul\n");
    total++; passed += testMatmul();

    printf("\n=== %d/%d tests passed ===\n", passed, total);
    return (passed == total) ? 0 : 1;
}
