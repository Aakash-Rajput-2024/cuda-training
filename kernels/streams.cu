#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorADD(const float * A , const float * B , float * C , int n ){

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if( i < n){
        C[i] = A[i] + B[i];
    }
}

int main (){
    int n = 5000000;

    size_t size = n * sizeof(float);
    float * h_a , *h_b , *h_c;
    float * d_a , *d_b , *d_c;

    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c = (float *)malloc(size);

    for(int i = 0 ; i<n ; i++){
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    cudaMalloc((void **) &d_a , size);
    cudaMalloc((void **) &d_b , size);
    cudaMalloc((void **) &d_c , size);

    cudaStream_t s1, s2;

    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    cudaMemcpyAsync(d_a , h_a , cudaMemcpyHostToDevice , s1);
    cudaMemcpyAsync(d_b , h_b , cudaMemcpyHostToDevice , s2);

    cudaStreamSynchronize(s2);

    int thB = 256;
    int blockPerGrid = (n + thB - 1 )/thB;

    vectorADD <<<blockPerGrid,thB , 0 ,s1>>> (d_a , d_b ,d_c ,n);

    cudaMemcpyAsync(h_c , d_c , size , cudaMemcpyDeviceToHost ,s1);

    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);

    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaStreamDestroy(s1);
    cudaStreamDestory(s2);

    free(h_a);
    free(h_b);
    free(h_c);

  
}