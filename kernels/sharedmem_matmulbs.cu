#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define WARMUP 3
#define ITTER 20

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

template <int BS>
__global__ void sgemm_shared(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    __shared__ float As[BS * BS];
    __shared__ float Bs[BS * BS];

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

template <int BS>
float bench_bs(int M, int N, int K, float alpha, const float *dA, const float *dB, float beta, float *dC, const float *dC_init) {
    dim3 block(BS, BS);
    dim3 grid((N + BS - 1) / BS, (M + BS - 1) / BS);
    size_t bytes = M * N * sizeof(float);

    for (int i = 0; i < WARMUP; i++) {
        CUDA_CHECK(cudaMemcpy(dC, dC_init, bytes, cudaMemcpyDeviceToDevice));
        sgemm_shared<BS><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITTER; i++) {
        CUDA_CHECK(cudaMemcpy(dC, dC_init, bytes, cudaMemcpyDeviceToDevice));
        sgemm_shared<BS><<<grid, block>>>(M, N, K, alpha, dA, dB, beta, dC);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / ITTER;
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

    printf("BlockSize | Time (ms) | GFLOP/s\n");
    printf("--------------------------------\n");

    auto print_row = [](int bs, float ms, int M, int N, int K) {
        double gflops = (2.0 * M * N * K) / (ms * 1e6);
        printf("  %2dx%-2d  |  %7.3f  | %7.1f\n", bs, bs, ms, gflops);
    };

    print_row(4,  bench_bs<4>(M, N, K, alpha, dA, dB, beta, dC, dC_init), M, N, K);
    print_row(8,  bench_bs<8>(M, N, K, alpha, dA, dB, beta, dC, dC_init), M, N, K);
    print_row(16, bench_bs<16>(M, N, K, alpha, dA, dB, beta, dC, dC_init), M, N, K);
    print_row(32, bench_bs<32>(M, N, K, alpha, dA, dB, beta, dC, dC_init), M, N, K);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dC_init));
    free(hA);
    free(hB);
    free(hC);

    return 0;
}
