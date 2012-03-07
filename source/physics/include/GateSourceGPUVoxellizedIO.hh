/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATESOURCEGPUVOXELLIZEDIO_H
#define GATESOURCEGPUVOXELLIZEDIO_H

#include <list>
#include <map>
#include <vector>

struct GateSourceGPUVoxellizedInput {
	long nb_events;
	double E; // MeV
	unsigned int seed;

	// Phantom and activity map (same size)
	// Coordinate system : 0,0,0 corner 
	int phantom_size_x;
	int phantom_size_y;
	int phantom_size_z;
    float phantom_spacing; // mm // to be change into spacingx spacingy spacingz FIXME
    std::vector<unsigned short int> phantom_material_data; // value = ID material CONSTANT (mc_cst_pet.cu l550)

	std::vector<float> activity_data; // no unit (relative) FIXME ? GATE PROCESS ????
	std::vector<unsigned int> activity_index; 
	// GPU (later) : ID gpu, nb thread/block
	// Physics (later) : 
};

typedef std::map<std::vector<int>,double> ActivityMap;

GateSourceGPUVoxellizedInput* GateSourceGPUVoxellizedInput_new();
void GateSourceGPUVoxellizedInput_delete(GateSourceGPUVoxellizedInput* input);
void GateSourceGPUVoxellizedInput_parse_activities(const ActivityMap& activities, GateSourceGPUVoxellizedInput* input);

struct GateSourceGPUVoxellizedOutputParticle {
	float E;  // MeV
	float dx; // direction (unary)
	float dy; // direction (unary)
	float dz; // direction (unary)
	float px; // mm position
	float py; // mm position
	float pz; // mm position
	float t;  // ns time
};

struct GateSourceGPUVoxellizedOutput { 
	typedef std::list<GateSourceGPUVoxellizedOutputParticle> ParticlesList;
	ParticlesList particles;
};


// Main function that lunch GPU calculation
void GateGPUGeneratePrimaries(const GateSourceGPUVoxellizedInput * input, 
		GateSourceGPUVoxellizedOutput & output);

#endif
