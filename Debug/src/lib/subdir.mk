################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/lib/cmdLib.cu \
../src/lib/devlib.cu \
../src/lib/inputLib.cu \
../src/lib/opsLib.cu 

CU_DEPS += \
./src/lib/cmdLib.d \
./src/lib/devlib.d \
./src/lib/inputLib.d \
./src/lib/opsLib.d 

OBJS += \
./src/lib/cmdLib.o \
./src/lib/devlib.o \
./src/lib/inputLib.o \
./src/lib/opsLib.o 


# Each subdirectory must supply rules for building sources it contributes
src/lib/%.o: ../src/lib/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -G -g -O0 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -odir "src/lib" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -G -O0 -g -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


