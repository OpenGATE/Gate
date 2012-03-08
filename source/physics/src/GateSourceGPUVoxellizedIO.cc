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
#include "GateMessageManager.hh"
#include <iostream>
#include <cassert>
using std::cout;
using std::endl;

GateSourceGPUVoxellizedInput* GateSourceGPUVoxellizedInput_new()
{
	GateSourceGPUVoxellizedInput* input = new GateSourceGPUVoxellizedInput;
	input->nb_events = 10000;
	input->E = 511*keV/MeV;
	input->seed = static_cast<unsigned int>(*GateRandomEngine::GetInstance()->GetRandomEngine());
    input->startTime = 0.0;

	input->phantom_size_x = -1;
	input->phantom_size_y = -1;
	input->phantom_size_z = -1;
	input->phantom_spacing = -1*mm/mm;

	return input;
}

void GateSourceGPUVoxellizedInput_delete(GateSourceGPUVoxellizedInput*)
{
}

struct ActivityMaterialTuple
{
	unsigned int index;
	float activity;
};

struct ActivityMaterialTupleStrictWeakOrdering
{
	bool operator()(const ActivityMaterialTuple& a, const ActivityMaterialTuple& b)
	{
		return a.activity < b.activity;
	}
};

typedef std::vector<ActivityMaterialTuple> ActivityMaterialTuplesVector;

void GateSourceGPUVoxellizedInput_parse_activities(const ActivityMap& activities, GateSourceGPUVoxellizedInput* input)
{
	assert(input);
	assert(input->activity_data.empty());
	assert(input->activity_index.empty());

	assert(input->phantom_size_x > 0);
	assert(input->phantom_size_y > 0);
	assert(input->phantom_size_z > 0);

	cout << "PARSING ACTIVITIES HERE " << activities.size() << endl;

	ActivityMaterialTuplesVector tuples;
	double total_activity = 0;
	{ // fill tuples structure
		for (ActivityMap::const_iterator iter = activities.begin(); iter != activities.end(); iter++)
		{
			const int ii = iter->first[0];
			const int jj = iter->first[1];
			const int kk = iter->first[2];
			assert(ii >= 0);
			assert(jj >= 0);
			assert(kk >= 0);
			assert(ii < input->phantom_size_x);
			assert(jj < input->phantom_size_y);
			assert(kk < input->phantom_size_z);

			const int index = ii + jj*input->phantom_size_x + kk*input->phantom_size_y*input->phantom_size_x;
			assert(index >= 0);
			assert(index < input->phantom_size_x*input->phantom_size_y*input->phantom_size_z);

			ActivityMaterialTuple tuple;
			tuple.index = index;
			tuple.activity = iter->second;

			tuples.push_back(tuple);
			total_activity += tuple.activity;
		}
	}

	{ // sort tuples by activities
		std::sort(tuples.begin(),tuples.end(),ActivityMaterialTupleStrictWeakOrdering());
	}

	{ // allocate and fill gpu input structure
		double cumulated_activity = 0;
		for (ActivityMaterialTuplesVector::const_iterator iter = tuples.begin(); iter != tuples.end(); iter++)
		{
			cumulated_activity += iter->activity;
			input->activity_data.push_back(cumulated_activity/total_activity);
			input->activity_index.push_back(iter->index);
		}
	}

	//{
	//	int kk = 0;
	//	for (ActivityMaterialTuplesVector::const_iterator iter = tuples.begin(); iter != tuples.end(); iter++)
	//	{
	//		cout << iter->index << " " << iter->activity << endl;
	//		if (kk>10) break;
	//		kk++;
	//	}
	//}

	//cout << total_activity << endl;

	//{
	//	for (int kk=0; kk<10; kk++)
	//	{
	//		cout << input->activity_index[kk] << " " << input->activity_data[kk] << endl;
	//	}
	//	cout << "....." << endl;
	//	for (int kk=tuples.size()-5; kk<tuples.size(); kk++)
	//	{
	//		cout << input->activity_index[kk] << " " << input->activity_data[kk] << endl;
	//	}
	//}
}

#ifndef GATE_USE_CUDA
void GateGPUGeneratePrimaries(const GateSourceGPUVoxellizedInput*, GateSourceGPUVoxellizedOutput&)
{
  GateError("Gate is compiled without CUDA enabled. You cannot use 'GPUvoxel' as source (GateSourceGPUVoxellized), use 'voxel' instead.");
}
#endif



