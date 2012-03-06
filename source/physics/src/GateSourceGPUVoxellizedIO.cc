/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSourceGPUVoxellizedIO.hh"
#include <iostream>

// To be used in GPU
GateSourceGPUVoxellizedOutputParticles * 
GateSourceGPUVoxellizedOutputParticles_new(unsigned long size) 
{
  std::cout << " GateSourceGPUVoxellizedOutputParticles_new" << std::endl;
  return new GateSourceGPUVoxellizedOutputParticles;
}

// To be used in CPU
void GateSourceGPUVoxellizedOutputParticles_delete(GateSourceGPUVoxellizedOutputParticles * output)
{
  std::cout << " GateSourceGPUVoxellizedOutputParticles_delete" << std::endl;

}

// 
void GateGPUGeneratePrimaries(const GateSourceGPUVoxellizedInput * input, 
                              GateSourceGPUVoxellizedOutputParticles * output) 
{
  std::cout << " GateGPUGeneratePrimaries " << std::endl;

}


