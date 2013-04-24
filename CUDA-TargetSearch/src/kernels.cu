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
#include <math.h>
#include "kernel_wrapper.h"
#include <cuda.h>

#define THREADS_PER_BLOCK 256
#define DEBUG true
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

#define CUDA_PRINT_LAST_ERROR(value) {										\
																			\
	cudaError_t error = cudaGetLastError();									\
	printf("%s\t::\tlast error %s\n",value, cudaGetErrorString(error));		\
																			\
}

#define SET_DEVICE(value) {																\
	cudaDeviceProp devProp;																\
	cudaGetDeviceProperties(&devProp, value);											\
	if(DEBUG)printf("Changing the gpu to device id: %i name: %s\n",value,devProp.name);	\
	CUDA_CHECK_RETURN(cudaSetDevice(value));											\
																						\
}

//cudaMalloc((void **) &devTranscripts, nTranscripts * nLibs * sizeof(int));
//cudaMemcpy(devTranscripts, hostTranscripts,nTranscripts * nLibs * sizeof(int), cudaMemcpyHostToDevice);
__global__ void findAllTargets ( unsigned char* r, unsigned char* g, unsigned char* b, int nPixels, int* targetPositions){
	
	//find the pixel id
	const unsigned int pixel_id        = blockIdx.x * (blockDim.x) + threadIdx.x;
	if(pixel_id <= nPixels-3){
		//uchar3 potentialTarget_r;
		//uchar3 potentialTarget_g;
		//uchar3 potentialTarget_b;

		//potentialTarget_r.x = r[pixel_id];
		//potentialTarget_g.x = g[pixel_id];
		//potentialTarget_b.x = b[pixel_id];
	
		//potentialTarget_r.y = r[pixel_id+1];
        	//potentialTarget_g.y = g[pixel_id+1];
        	//potentialTarget_b.y = b[pixel_id+1];
	
		//potentialTarget_r.z = r[pixel_id+2];
                //potentialTarget_g.z = g[pixel_id+2];
                //potentialTarget_b.z = b[pixel_id+2];

		if(r[pixel_id]==(unsigned char)255 && g[pixel_id]==(unsigned char)255 && b[pixel_id]==(unsigned char)255 && r[pixel_id+2]==(unsigned char)255 && g[pixel_id+2]==(unsigned char)255 && b[pixel_id+2]==(unsigned char)255){
			targetPositions[pixel_id]=1;
		}
		else{
			targetPositions[pixel_id]=0;
		}
	}
	return;
}

//(dev_r1,dev_g1,dev_b1,dev_r2,dev_g2,dev_b2, dev_positions1, dev_positions2, &loc1, &loc2)
__global__ void isolateTarget(unsigned char* r1, unsigned char* g1, unsigned char* b1,unsigned char* r2, unsigned char* g2, unsigned char* b2,int ntarg1, int ntarg2 ,int* positions1, int* positions2,int* finalPositions){
	const unsigned int gpu1_id        = blockIdx.x * (blockDim.x) + threadIdx.x;
	const unsigned int gpu2_id        = blockIdx.y * (blockDim.y) + threadIdx.y;
	//if(blockIdx.x==0)printf("%i::%i\n",gpu1_id,gpu2_id);
	if(gpu1_id < ntarg1 && gpu2_id < ntarg2){
		printf("%i::%i\n",gpu1_id,gpu2_id);
		//find the positions i am concerned with
		int gpu1_position = positions1[gpu1_id];
		int gpu2_position = positions2[gpu2_id];

		//get data
		unsigned char R1 = r1[gpu1_position+1];
		unsigned char G1 = g1[gpu1_position+1];
		unsigned char B1 = b1[gpu1_position+1];
		
		unsigned char R2 = 0;//r2[gpu2_position+1];
        unsigned char G2 = 0;//g2[gpu2_position+1];
        unsigned char B2 = 0;//b2[gpu2_position+1];

	
		//check if the internal colors match
		if(R1==R2 && G1 == G2 && B1 == B2){
			finalPositions[0]=gpu1_position;
			finalPositions[1]=gpu2_position;
			printf("The match was found at indices:\t(%i , %i )", gpu1_position, gpu2_position );
		}	
	}
}
__global__ void testKernel(int nPos1, int nPos2, int* positions1, int* positions2){
	const unsigned int gpu1_id        = blockIdx.x * (blockDim.x) + threadIdx.x;
	const unsigned int gpu2_id        = blockIdx.y * (blockDim.y) + threadIdx.y;
	if(gpu1_id < nPos1 && gpu2_id < nPos2){
		printf("%i::%i\n",positions1[gpu1_id],positions2[gpu2_id]);
	}

}
int kernelLaunchpad (unsigned char* r1, unsigned char* g1, unsigned char* b1, unsigned char* r2, unsigned char* g2, unsigned char* b2, int nPixels1, int nPixels2){
	//TODO: Implement the kernel launcher and GPU-subfunctions.
	//instantiate two streams to use for each image	
	cudaStream_t stream1,stream2;
	cudaEvent_t event1,event2;
	//set device to dev0, the 560ti
	SET_DEVICE(0);
	cudaDeviceReset();
	cudaStreamCreate(&stream1);
	cudaEventCreate(&event1);
	//allocate and copy memory to the gpus
	unsigned char *dev_r1, *dev_g1, *dev_b1;
	int* dev_targetPositions1,*targetPositions1;
	cudaMalloc((void**) &dev_r1, nPixels1*sizeof(unsigned char));
	cudaMalloc((void**) &dev_g1, nPixels1*sizeof(unsigned char));
	cudaMalloc((void**) &dev_b1, nPixels1*sizeof(unsigned char));
	cudaMalloc((void**) &dev_targetPositions1, nPixels1*sizeof(int));
	cudaMallocHost((void**) &targetPositions1, nPixels1*sizeof(int));	

	//set device to dev1, the gtx490
	SET_DEVICE(1);
	cudaDeviceReset();
	cudaStreamCreate(&stream2);
	cudaEventCreate(&event2);
	unsigned char *dev_r2, *dev_g2, *dev_b2;
	int* dev_targetPositions2, *targetPositions2;
        cudaMalloc((void**) &dev_r2, nPixels2*sizeof(unsigned char));
        cudaMalloc((void**) &dev_g2, nPixels2*sizeof(unsigned char));
        cudaMalloc((void**) &dev_b2, nPixels2*sizeof(unsigned char));
	cudaMalloc((void**) &dev_targetPositions2, nPixels2*sizeof(int));
	cudaMallocHost((void**) &targetPositions2, nPixels2*sizeof(int));
	
	//set device to dev0, the 560ti
        SET_DEVICE(0);
	CUDA_CHECK_RETURN(cudaMemcpyAsync((void*)dev_r1,r1,nPixels1*sizeof(unsigned char),cudaMemcpyHostToDevice,stream1));
	CUDA_CHECK_RETURN(cudaMemcpyAsync((void*)dev_g1,g1,nPixels1*sizeof(unsigned char),cudaMemcpyHostToDevice,stream1));
	CUDA_CHECK_RETURN(cudaMemcpyAsync((void*)dev_b1,b1,nPixels1*sizeof(unsigned char),cudaMemcpyHostToDevice,stream1));
	
	//set device to dev1, the gtx490
        SET_DEVICE(1);
	CUDA_CHECK_RETURN(cudaMemcpyAsync((void*)dev_r2,r2,nPixels2*sizeof(unsigned char),cudaMemcpyHostToDevice,stream2));
        CUDA_CHECK_RETURN(cudaMemcpyAsync((void*)dev_g2,g2,nPixels2*sizeof(unsigned char),cudaMemcpyHostToDevice,stream2));
        CUDA_CHECK_RETURN(cudaMemcpyAsync((void*)dev_b2,b2,nPixels2*sizeof(unsigned char),cudaMemcpyHostToDevice,stream2));
	
	//launch target identification kernels
	//set device to dev0, the 560ti
        SET_DEVICE(0);
	findAllTargets<<<dim3((int)nPixels1/THREADS_PER_BLOCK+1,1,1),dim3(THREADS_PER_BLOCK,1,1),0,stream1>>>(dev_r1,dev_g1,dev_b1,nPixels1,dev_targetPositions1);
	
	//set device to dev0, the gtx490
        SET_DEVICE(1);
	findAllTargets<<<dim3((int)nPixels1/THREADS_PER_BLOCK+1,1,1),dim3(THREADS_PER_BLOCK,1,1),0,stream2>>>(dev_r2,dev_g2,dev_b2,nPixels2,dev_targetPositions2);
	printf("data loaded, kernels launched, waiting for return.\n");

	//set device to dev0, the 560ti
        SET_DEVICE(0);
	cudaMemcpy(targetPositions1,dev_targetPositions1,nPixels1*sizeof(int), cudaMemcpyDeviceToHost);
	
	//set device to dev0, the gtx490
        SET_DEVICE(1);
	cudaMemcpy(targetPositions2,dev_targetPositions2,nPixels2*sizeof(int), cudaMemcpyDeviceToHost);	
	
	printf("kernels returned.  targets identified:\n");

	/*
	*	Stage 2, process target hit information
	*/
	
	//count the found targets in image 1
	int counter = 0;
	int targetsFound1 = 0;
	for(int i = 0; i < nPixels1; i++){
		if(targetPositions1[i]==1) targetsFound1++;
	}
	int* positions1 = (int*)malloc(targetsFound1*sizeof(int));
	for(int i = 0; i < nPixels1; i++){
                if(targetPositions1[i]==1){
			positions1[counter]=i;
			counter++;
		}
        }
	printf("%i targets found in the first image.\n",targetsFound1);
	
	//count the found targets in image 2	
	int targetsFound2 = 0;
        for(int i = 0; i < nPixels2; i++){
                if(targetPositions2[i]==1) targetsFound2++;
        }
	counter=0;
	int* positions2 = (int*)malloc(targetsFound2*sizeof(int));
	for(int i = 0; i < nPixels2; i++){
                if(targetPositions2[i]==1){
			positions2[counter]=i;
                        counter++;
		}
        }
	printf("%i targets found in the second image.\n",targetsFound2);

	
	/*
	*	Isolate the matching target
	*/

	//first, add device side data structures for the target indices
	//set device to dev0, the 560ti
    SET_DEVICE(0);
	
	int* dev_positions1, *dev_positions2;
	CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_positions1, targetsFound1*sizeof(int)));
	CUDA_CHECK_RETURN(cudaMalloc((void**) &dev_positions2, targetsFound2*sizeof(int)));
	//(void*)dev_r2,r2,nPixels2*sizeof(unsigned char),cudaMemcpyHostToDevice
	CUDA_CHECK_RETURN(cudaMemcpy((void*)dev_positions1,positions1,targetsFound1*sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy((void*)dev_positions2,positions2,targetsFound2*sizeof(int), cudaMemcpyHostToDevice));
	printf("Positions of targets:\n\timg1\timg2\n");
	for(int i = 0; i < targetsFound1; i++){
		printf("%i\t%i\t%i\n",i,positions1[i],positions2[i]);
	}
	int* finalPositions, *dev_finalPositions;
	finalPositions = (int*)malloc(2*sizeof(int));
	cudaMalloc((void**) &dev_finalPositions, 2*sizeof(int));

	//testing
	int devCount;
	cudaGetDeviceCount(&devCount);
	printf("GPU count: %i\n", devCount);
	//END  TESTING

	//enable the 560ti access to the 490 memory
	int canAccess;
	cudaDeviceCanAccessPeer(&canAccess, 0,1);
	printf("can the 560 access the 480 memory?\t%i\n",canAccess);
	cudaDeviceCanAccessPeer(&canAccess, 1,0);
	printf("can the 480 access the 560 memory?\t%i\n",canAccess);

	CUDA_PRINT_LAST_ERROR("Pre-peer-access, 1-0");
		cudaDeviceEnablePeerAccess(1,0);
		CUDA_PRINT_LAST_ERROR("Post-peer-access, 1-0");
	//launch the kernel to compare all pairs of targets.
	int twoDThreadsPerBlock;
	twoDThreadsPerBlock = sqrt((double)THREADS_PER_BLOCK);
	printf("Isolate Target Kernel: blocks: (%i,%i,1), threads per block: (%i,%i,1)\n",(int)targetsFound1/twoDThreadsPerBlock+1,(int)targetsFound2/twoDThreadsPerBlock+1,twoDThreadsPerBlock,twoDThreadsPerBlock);

	CUDA_PRINT_LAST_ERROR("Pre-kernel");


	isolateTarget<<<dim3((int)(targetsFound1/twoDThreadsPerBlock)+1,(int)(targetsFound2/twoDThreadsPerBlock)+1,1),dim3(twoDThreadsPerBlock,twoDThreadsPerBlock,1),0,stream2>>>(
				dev_r1,
				dev_g1,
				dev_b1,

				dev_r2,
				dev_g2,
				dev_b2,

				targetsFound1,
				targetsFound2,
				dev_positions1,
				dev_positions2,
				dev_finalPositions
	);
	cudaDeviceSynchronize();
	CUDA_PRINT_LAST_ERROR("post-kernel");

	//testKernel<<<dim3((int)(targetsFound1/THREADS_PER_BLOCK)+1,(int)(targetsFound2/THREADS_PER_BLOCK)+1,1),dim3(THREADS_PER_BLOCK,THREADS_PER_BLOCK,1),1,stream1>>>(targetsFound1,targetsFound2,dev_positions1,dev_positions2);

	CUDA_CHECK_RETURN(cudaMemcpyAsync(finalPositions,dev_finalPositions,2*sizeof(int),cudaMemcpyDeviceToHost,stream2));
	cudaDeviceSynchronize();

	printf("finished, target was at index %i,%i\n",finalPositions[0],finalPositions[1]);	
	//free memory before exit
	free(positions1);
	free(positions2);
	cudaFree(dev_r1);
	cudaFree(dev_g1);
	cudaFree(dev_b1);
	cudaFree(dev_r2);
	cudaFree(dev_g2);
	cudaFree(dev_b2);

	return 1;
}
