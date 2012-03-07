#ifndef track_gpu_h
#define track_gpu_h 1

#include "fun_gpu.cu"

void GateGPUGeneratePrimaries(const GateSourceGPUVoxellizedInput * input, 
                              GateSourceGPUVoxellizedOutputParticles * output);

	
#endif
