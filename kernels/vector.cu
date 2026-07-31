#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

// =========================================================================
// DEVICE KERNEL: Vectorized SGEMM (Kernel 6)
// =========================================================================
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
sgemmVectorize(int M, int N, int K, float alpha,
               float *A, float *B, float beta, float *C) {

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // ---- which 8x8 output tile does THIS thread own? ----
  const uint threadCol = threadIdx.x % (BN / TN);   // 0..15
  const uint threadRow = threadIdx.x / (BN / TN);   // 0..15

  __shared__ float As[BK * BM];   // TRANSPOSED: As[k * BM + m]
  __shared__ float Bs[BK * BN];   //      normal: Bs[k * BN + n]

  // advance pointers to this block's tile
  A += cRow * BM * K;             // A: (BM x K) strip
  B += cCol * BN;                 // B: (K x BN) strip
  C += cRow * BM * N + cCol * BN;

  // ---- which float4 does THIS thread load? ----
  const uint innerRowA = threadIdx.x / (BK / 4);   // BK/4 = 2  -> 0..127
  const uint innerColA = threadIdx.x % (BK / 4);   //           -> 0..1
  
  const uint innerRowB = threadIdx.x / (BN / 4);   // BN/4 = 32 -> 0..7
  const uint innerColB = threadIdx.x % (BN / 4);   //           -> 0..31

  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {

    // ================= GMEM -> SMEM =================
    // A: one LDG.E.128, then transpose via 4 scalar STS
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    // B: one LDG.E.128 straight into one STS.128 (no transpose needed)
    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];

    __syncthreads();

    A += BK;        // slide right along K
    B += BK * N;    // slide down  along K

    // ================= SMEM -> regs -> FFMA =================
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      for (uint i = 0; i < TM; ++i)
        regM[i] = As[dotIdx * BM + threadRow * TM + i];   
      for (uint i = 0; i < TN; ++i)
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];   

      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
    }
    __syncthreads();
  }


  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
      float *cPtr = &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
      float4 tmp = reinterpret_cast<float4 *>(cPtr)[0];  
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN + 0] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      reinterpret_cast<float4 *>(cPtr)[0] = tmp;         
    }
  }
}


void verifyCPU(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}


int main() {

    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    
    const float alpha = 1.0f;
    const float beta = 0.0f;

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_C_ref(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;


    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));


    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);


    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;
    
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN)); // 128*128 / 64 = 256 threads per block

    sgemmVectorize<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();


    int num_runs = 20;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for(int i = 0; i < num_runs; i++) {
        sgemmVectorize<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_ms = milliseconds / num_runs;
    
   
    double tflops = (2.0 * M * N * K) / (avg_ms * 1e6);
    std::cout << "Kernel execution time (avg): " << avg_ms << " ms" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;

   
    std::cout << "\nValidating against CPU reference..." << std::endl;
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    verifyCPU(M, N, K, alpha, h_A.data(), h_B.data(), beta, h_C_ref.data());

    bool passed = true;
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(h_C[i] - h_C_ref[i]) > 1e-3) {
            std::cout << "Mismatch at index " << i << ": GPU=" << h_C[i] << ", CPU=" << h_C_ref[i] << std::endl;
            passed = false;
            break;
        }
    }

    if (passed) std::cout << "SUCCESS! GPU and CPU results match." << std::endl;
    else std::cout << "FAILED! Results do not match." << std::endl;

   
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}