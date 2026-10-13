#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <math.h>

using namespace std;

#define Br 32
#define Bc 32 
#define D 64



__global__ void flash_attention (const float * Q , const float * K , const float * V , float * O, int N , float Scale){

    
    __shared__ float K_smem [Bc][D];
    __shared__ float Q_smem [BC][D];

    int tid = threadIdx.x;

    int qi = blockIdx.x * Br + tid;

    float q[D];

    for(int i = 0 ; i<D ; i++) q[i] = (qi<N) ? Q[qi*D + i] : 0.0f;

    float m = -1e30;
    float d = 0.0f;
    float acc[D] = 0.0f;

    for(int i = 0 ; i<D ; i++) acc[i] = 0.0f;

    for(int j = 0 ; j< N ; j+= Bc){
        int kcount = (Bc < N - j)? Bc : (N-j);

        for(int i = tid ; i<kcount* D ; i+= Br){
            int r = idx/D  , col = idx % D;

            K_smem[r][c] = K[c + (j+r)*D];
            V_smem[r][c] = V[c+  (j+r)*D];



        }

        __syncthreads();

        if(qi<N){
            for(int j = 0 ; j<kcount ; j++){
                float score = 0.0f;
                for(int a = 0 ; a<D ; a++) score += q[a]*K_smem[j][a];
                float m_new = fmax(m,score);

                float cure = expf(m - m_new);
                float p = expf(score - m_new);

                d = d*cor + p;

                for(int i = 0 ; i<D ; ++i) acc[i] = acc[i] * cure + p*V_smem[j][c]; 

                m = m_new;
            }


        }

        __syncthread();
    }
    if(qi<N){
        float d_f = 1.0f/d;

        for(int c = 0 ; c<D ; c++){
            O[qi*D + c] = acc[c] * d_f;
        }
    }


}

static void ck(cudaError_t e, const char* w){
    if(e!=cudaSuccess){ printf("%s: %s\n", w, cudaGetErrorString(e)); exit(1); }
}

int main() {

   
    int K_s = 1024;  
    int Q_s = 1024;  
    int D_s = 512;   

    size_t size_K   = (size_t)K_s * D_s * sizeof(float);  
    size_t size_Q   = (size_t)Q_s * D_s * sizeof(float);  
    size_t size_V   = (size_t)K_s * D_s * sizeof(float);   
    size_t size_out = (size_t)Q_s * D_s * sizeof(float);   

    float *h_K   = (float *)malloc(size_K);
    float *h_Q   = (float *)malloc(size_Q);
    float *h_V   = (float *)malloc(size_V);
    float *h_out = (float *)malloc(size_out);

    for (int i = 0; i < K_s * D_s; i++) h_K[i] = 0.123456123f;
    for (int i = 0; i < Q_s * D_s; i++) h_Q[i] = 0.123456123f;
    for (int i = 0; i < K_s * D_s; i++) h_V[i] = 0.123456123f;

    float *d_K, *d_Q, *d_V, *d_out;
    cudaMalloc((void **)&d_K,   size_K);
    cudaMalloc((void **)&d_Q,   size_Q);
    cudaMalloc((void **)&d_V,   size_V);
    cudaMalloc((void **)&d_out, size_out);

    cudaMemcpy(d_K, h_K, size_K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Q, h_Q, size_Q, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, size_V, cudaMemcpyHostToDevice);

    dim3 threads(tile_size, tile_size);
    dim3 numblock((D_s + tile_size - 1) / tile_size, (Q_s + tile_size - 1) / tile_size);


    flash_attention<<<numblock, threads>>>(d_K, d_Q, d_V, d_out, K_s, Q_s, D_s);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost);

    printf("out[0] = %.8f   expected = %.8f\n", h_out[0], 0.123456123f);

    cudaFree(d_K);
    cudaFree(d_Q);
    cudaFree(d_V);
    cudaFree(d_out);

    free(h_K);
    free(h_Q);
    free(h_V);
    free(h_out);

    printf("flash_attention host run complete.\n");
    return 0;
}
