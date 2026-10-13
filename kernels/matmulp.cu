#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <math.h>

#define TW 32

__global__ void matmul(float * Nd, float * Md, float * out, int N, int K, int M) {
    
    __shared__ float Nds[TW][TW];
    __shared__ float Mds[TW][TW];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y; 

    int row = ty + by * blockDim.y;
    int col = tx + bx * blockDim.x;

    float p = 0.0f;

    for (int b = 0; b < K / TW; b++) {

        Nds[ty][tx] = Nd[row * K + (b * TW + tx)];
        Mds[ty][tx] = Md[(b * TW + ty) * M + col];

        __syncthreads();
    
        for (int k = 0; k < TW; ++k) {
            p += Nds[ty][k] * Mds[k][tx];
        }
        
        
        __syncthreads();
    }

    if (row < N && col < M) {
        out[row * M + col] = p;
    }
}
int main() {
    int N = 1024;
    int M = 1024;
    int K = 512;

    size_t size_N = N * K * sizeof(float);
    size_t size_M = K * M * sizeof(float);
    size_t size_out = N * M * sizeof(float);

    float * h_N = (float *)malloc(size_N);
    float * h_M = (float *)malloc(size_M);

    float * h_out = (float *)malloc(size_out);

    for (int i = 0; i < N * K; i++) h_N[i] = 0.123f;
    for (int i = 0; i < M * K; i++) h_M[i] = 0.223f;

    float *d_N, *d_M, *d_out;


    cudaMalloc((void **)&d_N, size_N); 
    cudaMalloc((void **)&d_M, size_M);
    cudaMalloc((void **)&d_out, size_out);

    
    cudaMemcpy(d_N, h_N, size_N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, size_M, cudaMemcpyHostToDevice);

    dim3 threads(TW, TW);
    dim3 numBlocks((M + TW - 1) / TW, (N + TW - 1) / TW);

 
    matmul<<<numBlocks, threads>>>(d_N, d_M, d_out, N, K, M);

    cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost);

    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_out);

    free(h_N);
    free(h_M);
    free(h_out);

    printf("Matrix Multiplication Completed Successfully (no thanks to you).\n");

    return 0;
}