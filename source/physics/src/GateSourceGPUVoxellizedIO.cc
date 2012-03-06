/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"
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

#ifndef GATE_USE_CUDA
void GateGPUGeneratePrimaries(const GateSourceGPUVoxellizedInput*, GateSourceGPUVoxellizedOutput&)
{
	std::cout << "DUMMY GPU CALL" << std::endl;
}
#endif



