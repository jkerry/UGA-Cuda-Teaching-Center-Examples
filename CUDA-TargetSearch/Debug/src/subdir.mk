################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/CUDA-TargetSearch.cu \
../src/kernels.cu 

CU_DEPS += \
./src/CUDA-TargetSearch.d \
./src/kernels.d 

OBJS += \
./src/CUDA-TargetSearch.o \
./src/kernels.o 


# Each subdirectory must supply rules for building sources it contributes
src/CUDA-TargetSearch.o: ../src/CUDA-TargetSearch.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -G -g -O0 -gencode arch=compute_20,code=sm_20 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -O0 -g -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/kernels.o: ../src/kernels.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -G -g -O0 -gencode arch=compute_20,code=sm_20 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -O0 -g -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20 -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


