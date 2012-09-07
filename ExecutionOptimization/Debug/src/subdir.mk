################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/ExecutionOptimization.cu \
../src/MatrixMultiplicationExample.cu \
../src/PartialSumExample.cu 

CU_DEPS += \
./src/ExecutionOptimization.d \
./src/MatrixMultiplicationExample.d \
./src/PartialSumExample.d 

OBJS += \
./src/ExecutionOptimization.o \
./src/MatrixMultiplicationExample.o \
./src/PartialSumExample.o 


# Each subdirectory must supply rules for building sources it contributes
src/ExecutionOptimization.o: ../src/ExecutionOptimization.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -G -g -O0 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -O0 -g -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21 -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/MatrixMultiplicationExample.o: ../src/MatrixMultiplicationExample.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -G -g -O0 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -O0 -g -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21 -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/PartialSumExample.o: ../src/PartialSumExample.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -G -g -O0 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -O0 -g -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21 -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


