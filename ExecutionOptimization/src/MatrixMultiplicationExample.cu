/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include "MatrixMultiplicationExample.h"

/**
 * This boolean value defined if debug information is printed within macros
 */
#define DEBUG true
/**
 * The number of items in the partial sum array
 */
#define MAT_SIZE 512

#define SEED 41887

#define TILE_WIDTH 16


/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
/**
 * This macro changes the active device to the device with the provided integer index.
 */
#define SET_DEVICE(value) {																\
	cudaDeviceProp devProp;																\
	cudaGetDeviceProperties(&devProp, value);											\
	if(DEBUG)printf("Changing the gpu to device id: %i name: %s\n",value,devProp.name);	\
	CUDA_CHECK_RETURN(cudaSetDevice(value));											\
																						\
}




__global__ void cu_matrixMultiplication(){

}

__global__ void improved_matrixMultiplication(int* A, int* B, float* AxB, int dim){

	unsigned int tile_width = TILE_WIDTH;

	__shared__ float A_cache[TILE_WIDTH][TILE_WIDTH];
	__shared__ float B_cache[TILE_WIDTH][TILE_WIDTH];

	//find the cell I'm working on
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = by*tile_width + ty;
	int col = bx*tile_width + tx;
	if(row < dim && col < dim){
	float finalValue = 0;

	//Data Cache Loop
	for(int i = 0; i < dim/tile_width; ++i){

			A_cache[ty][tx] =  (float)A[row*dim + (i*tile_width + tx)];
			B_cache[ty][tx] =  (float)B[(i*tile_width + ty)*dim + col];
			__syncthreads();

			for(int j =0; j < tile_width; ++j){
				finalValue += A_cache[ty][j]*B_cache[j][tx];
			}
	}
	//printf("%i\n", (blockIdx.y*blockDim.y+threadIdx.y));
	AxB[row*dim+col] = finalValue;
	}
	__syncthreads();
	return;
}






/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int matrixMultiplication(bool improved) {
	SET_DEVICE(0);
	CUDA_CHECK_RETURN(cudaDeviceReset()); //pre-clear the device

	int A[MAT_SIZE*MAT_SIZE], B[MAT_SIZE*MAT_SIZE], AxB[MAT_SIZE*MAT_SIZE];
	//populate both matrices with random values
	srand(SEED);
	for(int i = 0; i < MAT_SIZE; i++){
		for(int j=0; j<MAT_SIZE; j++){
			A[i*MAT_SIZE+j] = (int)(((float)rand()/RAND_MAX)*100);
			B[i*MAT_SIZE+j] = (int)(((float)rand()/RAND_MAX)*100);
		}
	}

	//Device pointers for A and B
	int *d_A, *d_B;
	//host and device pointers for the result
	float  *d_AxB;

	//allocate device side memory for d_A, d_B, d_AxB
	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_A, sizeof(int)*MAT_SIZE*MAT_SIZE));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_B, sizeof(int)*MAT_SIZE*MAT_SIZE));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_AxB, sizeof(float)*MAT_SIZE*MAT_SIZE));

	//Memcpy both device side A and B matrices
	CUDA_CHECK_RETURN(cudaMemcpy((void*) d_A, A, sizeof(int)*MAT_SIZE*MAT_SIZE, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy((void*) d_B, B, sizeof(int)*MAT_SIZE*MAT_SIZE, cudaMemcpyHostToDevice));

	//calc num threads and blocks
	int nThreads_perDim = 16;
	int nBlocks_perDim = MAT_SIZE / nThreads_perDim +1;

	if(!improved){
			printf("Running naive sum reduction\n");

		}
		else{
			printf("Running improved sum reduction\n");
			improved_matrixMultiplication<<<dim3(nBlocks_perDim,nBlocks_perDim,1),dim3(nThreads_perDim,nThreads_perDim,1),0,0>>>(d_A,d_B,d_AxB,MAT_SIZE);
		}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	CUDA_CHECK_RETURN(cudaMemcpy(AxB, d_AxB, sizeof(float)*MAT_SIZE*MAT_SIZE, cudaMemcpyDeviceToHost));

	return 0;
}
