#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h> // Added for fabs()

#define N 10000000  // Vector size = 10 million
#define BLOCK_SIZE 512

void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// FIXED: Changed last parameter to nz, and fixed the boundary check math
__global__ void vector_add_3d(float * a , float * b , float * c , int nx , int ny , int nz ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i < nx && j < ny && k < nz){
        // Flattening the 3D index to 1D
        int idx = i + (j * nx) + (k * nx * ny);
        c[idx] = a[idx] + b[idx];
    }
}


void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);


    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
 
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    // --- CPU BENCHMARK ---
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // --- 1D GPU BENCHMARK ---
    printf("Benchmarking 1D GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // --- 3D GPU BENCHMARK ---
    // We need our 3D dimensions to equal 10,000,000 total elements
    int nx = 100, ny = 100, nz = 1000; 
    
    // Create a 3D block of 8x8x8 = 512 threads
    dim3 threads3D(8, 8, 8); 
    
    // Calculate how many blocks we need in each dimension (using ceiling math)
    dim3 blocks3D((nx + threads3D.x - 1) / threads3D.x, 
                  (ny + threads3D.y - 1) / threads3D.y, 
                  (nz + threads3D.z - 1) / threads3D.z);

    printf("Benchmarking 3D GPU implementation...\n");
    double gpu_3d_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_add_3d<<<blocks3D, threads3D>>>(d_a, d_b, d_c, nx, ny, nz);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_3d_total_time += end_time - start_time;
    }
    double gpu_3d_avg_time = gpu_3d_total_time / 20.0;


    printf("\n--- RESULTS ---\n");
    printf("CPU average time:    %f ms\n", cpu_avg_time * 1000);
    printf("1D GPU average time: %f ms (Speedup: %fx)\n", gpu_avg_time * 1000, cpu_avg_time / gpu_avg_time);
    printf("3D GPU average time: %f ms (Speedup: %fx)\n", gpu_3d_avg_time * 1000, cpu_avg_time / gpu_3d_avg_time);

    // Verify results
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");
   
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}