/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSourceGPUVoxellizedIO.hh"
#include "GateRandomEngine.hh"
#include <iostream>

GateSourceGPUVoxellizedInput* GateSourceGPUVoxellizedInput_new()
{
	GateSourceGPUVoxellizedInput* input = new GateSourceGPUVoxellizedInput;
	input->nb_events = 10000;
	input->E = 511*keV/MeV;
	input->seed = static_cast<unsigned int>(*GateRandomEngine::GetInstance()->GetRandomEngine());

	input->phantom_size_x = -1;
	input->phantom_size_y = -1;
	input->phantom_size_z = -1;
	input->phantom_spacing = -1*mm/mm;
	input->phantom_activity_data = NULL;
	input->phantom_material_data = NULL;

	return input;
}

void GateSourceGPUVoxellizedInput_delete(GateSourceGPUVoxellizedInput* input)
{
	if (input->phantom_activity_data) delete input->phantom_activity_data;
	if (input->phantom_material_data) delete input->phantom_material_data;
}

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


