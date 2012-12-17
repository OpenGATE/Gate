project(hGATE)

cmake_minimum_required(VERSION 2.8)

find_package(CUDA required)

set(CUDA_NVCC_FLAGS "-arch;compute_20;-use_fast_math")

cuda_add_executable(
		hgate
		mc_main_pet.cu
)
