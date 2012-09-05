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


/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {

	partialSum();

	return 0;
}
