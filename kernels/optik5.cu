#include <cassert>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

typedef unsigned int uint ;
using namespace std;


#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_2d(int M , int N , int K , float alpha , const float * A , const float * B , const float * C , float beta , float * C){

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint numThreadBlockTile = (BM * BN) / (TM * TN);
    assert (numThreadBlockTile == blockIdx.x);


    const uint threadCol = threadIdx.x % (BN/TN);
    const uint threadRow = threadIdx.x / (BN/TN);

    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];

    A += cRow * BM * K;
    B += cCol * BN;

    C += cRow * BM * N + cCol* BN;

    const uint innerRowA = threadIdx.x /BK;
    const uint innerColA = threadIdx.x % BK;

    const uint strideA = numThreadsBlocktile /BK;

    const uint innerRowB = threadIdx.y / BN;
    const uint innerColB = threadIdx.y / BN;

    const uint strideB = numThreadsBlocktile /BN;


    float threadResults [TM*TN] ={0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for(uitn bkIdx = 0 ; bkIdx < K ; bkIdx += BK){

      for(int load = 0 ; load <BM ; load += strideA){
        As[(innerRowA + load) * BK + innerColA] = A[ (innerRowB + load) * K + innerColA];
      }

      for(int load = 0 ; load <BM ; load += strideA){
        As[(innerRowA + load) * BK + innerColA] = A[ (innerRowB + load) * K + innerColA];
      }

      
    }
} 