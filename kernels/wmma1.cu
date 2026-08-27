#include <iostream> 
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WPB 4

#define TPB (WPB * 32)



__global__ void gemm_wmma_kernel (const half * A , const half * B , half * C , float alpha , float beta , int M , int N , int K ){

    
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;

    
    int tilesN = (N + 15) / 16;


    int globalWarpId  = (blockIdx.x * WPB) + warpId;

    int warpRow = globalWarpId / tilesN;
    int warpCol = globalWarpId % tilesN;

    int row = warpRow * 16;
    int col = warpCol * 16;

    if(row >= M || col >= N ){
        return ;
    }

    wmma::fragment <wmma:: matrix_a , 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment <wmma:: matrix_b , 16, 16, 16, half, wmma::row_major> b_frag;
    
    // FIXED: Added the missing acc_frag declaration before filling it
    wmma::fragment <wmma::accumulator, 16, 16, 16, float> acc_frag;

    wmma:: fill_fragment (acc_frag ,  0.0f);

    for(int i = 0 ; i< K ; i+=16){

        wmma :: load_matrix_sync (a_frag , A + row * K + i , K);
        wmma :: load_matrix_sync (b_frag , B + i*N + col , N);

        wmma :: mma_sync (acc_frag , a_frag , b_frag , acc_frag);

    }

    __shared__ float s_tile [WPB][16*16];
    wmma:: store_matrix_sync (s_tile [warpId] , acc_frag , 16 , wmma::mem_row_major);

    for(int i = 0 ; i<8 ; i++){
        int elemIdx = laneId + (i*32);
        int r = elemIdx / 16;
        int c = elemIdx % 16;

        if(row + r < M && col + c < N){

            int global_idx = (row + r) * N + (col + c);
            
            float ab_val = s_tile[warpId][elemIdx];
            float c_val  = __half2float(C[global_idx]);
            float final_result = (alpha * ab_val) + (beta* c_val);

            
            C[global_idx] = __float2half(final_result);
        }
    }
}

void solve (const half * A , const half * B , half * C , float alpha , float beta , int M , int N , int K){


    int tilesM = (M + 15) / 16;
    int tilesN = (N + 15) / 16;

    int totalTiles = tilesM * tilesN ;

    int numBlocks = (totalTiles + WPB - 1) / WPB;
    
    
    gemm_wmma_kernel <<<numBlocks , TPB >>> (A, B , C , alpha , beta , M , N , K);

    cudaDeviceSynchronize();
}

__global__ void init_matrix(half* mat, float val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        mat[idx] = __float2half(val);
    }
}

int main() {
    // Performance test size as requested
    int M = 1024;
    int N = 1024;
    int K = 1024;

    float alpha = 2.0f;
    float beta = 1.0f;

    size_t sizeA = M * K * sizeof(half);
    size_t sizeB = K * N * sizeof(half);
    size_t sizeC = M * N * sizeof(half);

    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

   
    int total_elements = 1024 * 1024;
    init_matrix<<<(total_elements + 255) / 256, 256>>>(d_A, 1.0f, M * K);
    init_matrix<<<(total_elements + 255) / 256, 256>>>(d_B, 2.0f, K * N);
    init_matrix<<<(total_elements + 255) / 256, 256>>>(d_C, 3.0f, M * N);
    cudaDeviceSynchronize();

   
    std::cout << "Launching solve() for 1024x1024..." << std::endl;
    solve(d_A, d_B, d_C, alpha, beta, M, N, K);


    half* h_C = (half*)malloc(sizeC);
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    std::cout << "\nTop-left 4x4 of the Result Matrix (C):\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << __half2float(h_C[i * N + j]) << " ";
        }
        std::cout << "\n";
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_C);

    return 0;
}