#ifndef track_gpu_h
#define track_gpu_h 1

#include "fun_gpu.cu"

void GateGPU_VoxelSource_GeneratePrimaries(const GateGPUIO_Input * input, 
                                           GateGPUIO_Output & output);
	
#endif
