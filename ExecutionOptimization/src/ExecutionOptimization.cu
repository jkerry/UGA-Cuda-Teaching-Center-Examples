
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
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {

	partialSum(false);
	printf("Finished with naive sum reduction\n");
	partialSum(true);
	printf("Finished with improved sum reduction\n");

	return 0;
}
