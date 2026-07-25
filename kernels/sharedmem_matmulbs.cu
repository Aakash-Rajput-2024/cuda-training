#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define WARMUP 2
#define ITTER 10

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void sgemm_shared_dynamic(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C, int BS) {
    extern __shared__ float s[];
    float *As = s;
    float *Bs = s + BS * BS;

    int cRow = blockIdx.y;
    int cCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    A += cRow * BS * K;
    B += cCol * BS;
    C += cRow * BS * N + cCol * BS;

    float tmp = 0.0f;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BS) {
        As[threadRow * BS + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BS + threadCol] = B[threadRow * N + threadCol];
        __syncthreads();

        A += BS;
        B += BS * N;

        for (int i = 0; i < BS; i++) {
            tmp += As[threadRow * BS + i] * Bs[i * BS + threadCol];
        }
        __syncthreads();
    }

    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}

bool test_launch(int BS, int M, int N, int K, float alpha, const float *dA, const float *dB, float beta, float *dC, float *dC_init, float &out_ms) {
    if (BS * BS > 1024 || M % BS != 0 || N % BS != 0 || K % BS != 0) {
        return false;
    }

    dim3 block(BS, BS);
    dim3 grid(N / BS, M / BS);
    size_t shared_mem_bytes = 2 * BS * BS * sizeof(float);
    size_t bytes = M * N * sizeof(float);

    for (int i = 0; i < WARMUP; i++) {
        cudaMemcpy(dC, dC_init, bytes, cudaMemcpyDeviceToDevice);
        sgemm_shared_dynamic<<<grid, block, shared_mem_bytes>>>(M, N, K, alpha, dA, dB, beta, dC, BS);
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < ITTER; i++) {
        cudaMemcpy(dC, dC_init, bytes, cudaMemcpyDeviceToDevice);
        sgemm_shared_dynamic<<<grid, block, shared_mem_bytes>>>(M, N, K, alpha, dA, dB, beta, dC, BS);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&out_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    out_ms /= ITTER;
    return true;
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    const float alpha = 1.0f, beta = 0.0f;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *hA = (float *)malloc(sizeA);
    float *hB = (float *)malloc(sizeB);
    float *hC = (float *)malloc(sizeC);

    srand(42);
    for (int i = 0; i < M * K; i++) hA[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) hB[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < M * N; i++) hC[i] = 0.0f;

    float *dA, *dB, *dC, *dC_init;
    CUDA_CHECK(cudaMalloc(&dA, sizeA));
    CUDA_CHECK(cudaMalloc(&dB, sizeB));
    CUDA_CHECK(cudaMalloc(&dC, sizeC));
    CUDA_CHECK(cudaMalloc(&dC_init, sizeC));

    CUDA_CHECK(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC_init, hC, sizeC, cudaMemcpyHostToDevice));

    printf("Binary Searching for Highest Valid Block Size:\n");
    int low = 1, high = 64;
    int max_valid_bs = -1;
    float best_ms = 0.0f;

    while (low <= high) {
        int mid = low + (high - low) / 2;
        float ms = 0.0f;
        bool ok = test_launch(mid, M, N, K, alpha, dA, dB, beta, dC, dC_init, ms);
        if (ok) {
            double gflops = (2.0 * M * N * K) / (ms * 1e6);
            printf("BS = %2d -> VALID   | Time: %7.3f ms | Throughput: %7.1f GFLOP/s\n", mid, ms, gflops);
            max_valid_bs = mid;
            best_ms = ms;
            low = mid + 1;
        } else {
            printf("BS = %2d -> INVALID (Exceeds hardware limits / invalid config)\n", mid);
            high = mid - 1;
        }
    }

    printf("\nHighest Valid Block Size found: %dx%d (Time: %.3f ms)\n", max_valid_bs, max_valid_bs, best_ms);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dC_init));
    free(hA);
    free(hB);
    free(hC);

    return 0;
}
