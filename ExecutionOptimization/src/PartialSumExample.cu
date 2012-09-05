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
#include "PartialSumExample.h"

/**
 * This boolean value defined if debug information is printed within macros
 */
#define DEBUG true
/**
 * The number of items in the partial sum array
 */
#define ARR_SIZE 512



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


__device__ int result;
__global__ void sumReduction(int* data, int length){
	const unsigned int tid = threadIdx.x;
	if(tid < length){
		//allocate shared memory within the block for the partial sums
		__shared__ float partialSum[ARR_SIZE];
		//half the number of active threads at any time, compute the current partial sum
		for( unsigned int stride = 1; stride < blockDim.x; stride *= 2){
			__syncthreads();
			if(tid%(2*stride==0)){
				partialSum[tid]+=partialSum[tid+stride];
			}
		}
		//return the last computed partial sum
		result = partialSum[0];
		return;
	}
	else{ 	//these threads are idle, thread id is outside the array bounds
		return;
	}
}

/**
 * Wrapper function to initialize host data elements so that the example appears simpler.
 */
void ps_setupData(int* arrData, int arrData_length){
	arrData =  (int*) malloc(ARR_SIZE*sizeof(int));
	srand(time(NULL));
	for(int i = 0; i < arrData_length; i++){
		arrData[i] = rand();
	}
}
/**
 * wrapper function to initialize device element pointers so that the example appears simpler
 */
void ps_setupDeviceData(int* arrData, int arrData_length, int* arrData_Local,int* psResult){
	//allocate arrData pointer on device
	CUDA_CHECK_RETURN(cudaMalloc((void**) &arrData, arrData_length*sizeof(int)));
	//copy local generated data to the device
	CUDA_CHECK_RETURN(cudaMemcpy((void*)arrData, arrData_Local, arrData_length*sizeof(int), cudaMemcpyHostToDevice));
	//allocate the array of size 1 for return value
	CUDA_CHECK_RETURN(cudaMalloc((void**) &psResult, 1*sizeof(int)));
	//CUDA_CHECK_RETURN();
}



/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int partialSum(void) {
	int *randomData, *randomData_Device, *psResult_Device;
	ps_setupData(randomData, ARR_SIZE);
	ps_setupDeviceData(randomData_Device, ARR_SIZE,randomData, psResult_Device);

	SET_DEVICE(0);
	CUDA_CHECK_RETURN(cudaDeviceReset()); //pre-clear the device
	//launch the kernel
	sumReduction<<<dim3(1,1,1),dim3(ARR_SIZE,1,1),0,0>>>(randomData_Device, ARR_SIZE);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
//	CUDA_CHECK_RETURN();
	CUDA_CHECK_RETURN(cudaDeviceReset()); //clear the device after all work is completed
	return 0;
}
