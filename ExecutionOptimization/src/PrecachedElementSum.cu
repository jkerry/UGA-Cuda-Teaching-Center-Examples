/* *
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
#include "ElementSum.h"

/**
 * This boolean value defined if debug information is printed within macros
 */
#define DEBUG true
/**
 * The number of items in the partial sum array
 */
#define MAT_SIZE 512

#define SEED 41887


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


__global__ void ElementSum(int* matrix, int* result){
	//get this threads column index
	const unsigned int column =  threadIdx.x;
	//allocate shared memory for partial sums
	__shared__ int columnSums[MAT_SIZE];
	//loop through all elements in global memory and keep a running sum.  Finally leave that total in the shared memory space.
	int i;
	int sum = 0;
	for(i = 0; i < MAT_SIZE; i++){
		sum += matrix[(i*MAT_SIZE) +column]; //only difference in computation is here, the data access pattern
	}
	columnSums[column] = sum;
	__syncthreads();
	if(column == 0){
		int finalSum = 0;
		for(i=0; i < MAT_SIZE; i++){
			finalSum += columnSums[i];
		}
		*result = finalSum;
	}
	__syncthreads();
	return;
}

__global__ void ElementSum_precached(int* matrix, int* result){
	//get this threads column index
	const unsigned int column =  threadIdx.x;
	//allocate shared memory for partial sums
	__shared__ int columnSums[MAT_SIZE];
	//loop through all elements in global memory and keep a running sum.  Finally leave that total in the shared memory space.
	int i;
	int sum = 0;
	int nextValue = matrix[column];
	for(i = 1; i < MAT_SIZE; i++){
		sum += nextValue;
		nextValue = matrix[(i*MAT_SIZE) +column];
	}
	sum+= nextValue;
	columnSums[column] = sum;
	__syncthreads();
	if(column == 0){
		int finalSum = 0;
		for(i=0; i < MAT_SIZE; i++){
			finalSum += columnSums[i];
		}
		*result = finalSum;
	}
	__syncthreads();
	return;
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int precachedElementSum(bool improved) {
	SET_DEVICE(0);
	CUDA_CHECK_RETURN(cudaDeviceReset()); //pre-clear the device

	int A[MAT_SIZE*MAT_SIZE];
	//populate both matrices with random values
	srand(SEED);
	for(int i = 0; i < MAT_SIZE; i++){
		for(int j=0; j<MAT_SIZE; j++){
			A[i*MAT_SIZE+j] = (int)(((float)rand()/RAND_MAX)*100);
		}
	}

	//Device pointers for A and gpuResult
	int *d_A, *d_gpuResult;


	//allocate device side memory for d_A
	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_A, sizeof(int)*MAT_SIZE*MAT_SIZE));
	//allocate the array of size 1 for return value
	CUDA_CHECK_RETURN(cudaMalloc((void**) &d_gpuResult, sizeof(int)));
	//Memcpy device side A matrix
	CUDA_CHECK_RETURN(cudaMemcpy((void*) d_A, A, sizeof(int)*MAT_SIZE*MAT_SIZE, cudaMemcpyHostToDevice));


	if(!improved){
		printf("Calculating element sum without pre-caching.\n");
		ElementSum<<<dim3(1,1,1),dim3(MAT_SIZE,1,1),0,0>>>(d_A,d_gpuResult);
	}
	else{
		printf("Calculating element sum with pre-caching.\n");
		ElementSum_precached<<<dim3(1,1,1),dim3(MAT_SIZE,1,1),0,0>>>(d_A,d_gpuResult);
	}
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	//Allocate local memory for GPU result and serial result
	int GPU_Answer, Serial_Answer=0;
	CUDA_CHECK_RETURN(cudaMemcpy(&GPU_Answer, d_gpuResult, sizeof(int), cudaMemcpyDeviceToHost));

	for(int i =0; i < MAT_SIZE*MAT_SIZE; i++){
		Serial_Answer += A[i];
	}

	printf("GPU Answer:\t%i\nSerial Answer:\t%i\n",GPU_Answer,Serial_Answer);


	//Clean up
	cudaFree(d_A);
	cudaFree(d_gpuResult);
	CUDA_CHECK_RETURN(cudaDeviceReset()); //clear the device after all work is completed
	return 0;
}
