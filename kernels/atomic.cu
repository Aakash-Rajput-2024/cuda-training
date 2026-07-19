#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_THREADS 1024
#define NUM_BLOCKS 1000

__global__ void inc(int * count){
    int old  = * count;
    ++old;
    *count = old;
}

__global__ void incA(int * count){
    atomicAdd(count, 1); 
}

int main (){
    int h_c = 0;
    int h_ca = 0;

    int *d_c, *d_ca;

    cudaMalloc((void**)&d_c, sizeof(int));
    cudaMalloc((void**)&d_ca, sizeof(int));

    cudaMemcpy(d_c, &h_c, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ca, &h_ca, sizeof(int), cudaMemcpyHostToDevice);

    inc<<<NUM_BLOCKS, NUM_THREADS>>>(d_c);
    incA<<<NUM_BLOCKS, NUM_THREADS>>>(d_ca);
    
    cudaDeviceSynchronize(); 

    cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_ca, d_ca, sizeof(int), cudaMemcpyDeviceToHost);

    printf("Naive: %d    Atomic: %d\n", h_c, h_ca);

    cudaFree(d_c);
    cudaFree(d_ca);

    return 0;
}