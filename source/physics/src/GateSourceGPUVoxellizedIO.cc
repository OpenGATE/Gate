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
#include <cassert>
using std::cout;
using std::endl;

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
	input->phantom_material_data = NULL;

	input->activity_size = -1;
	input->activity_data = NULL;
	input->activity_index = NULL;

	return input;
}

void GateSourceGPUVoxellizedInput_delete(GateSourceGPUVoxellizedInput* input)
{
	if (input->phantom_material_data) delete [] input->phantom_material_data;

	if (input->activity_data) delete [] input->activity_data;
	if (input->activity_index) delete [] input->activity_index;
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
		return a.activity > b.activity;
	}
};

typedef std::vector<ActivityMaterialTuple> ActivityMaterialTuplesVector;

void GateSourceGPUVoxellizedInput_parse_activities(const ActivityMap& activities, GateSourceGPUVoxellizedInput* input)
{
	assert(input);
	assert(input->activity_data == NULL);
	assert(input->activity_index == NULL);
	assert(input->activity_size < 0);

	cout << "PARSING ACTIVITIES HERE " << activities.size() << endl;

	ActivityMaterialTuplesVector tuples;
	double total_activity = 0;
	{ // fill tuples structure
		unsigned int current_index = 0;
		for (ActivityMap::const_iterator iter = activities.begin(); iter != activities.end(); iter++)
		{
			ActivityMaterialTuple tuple;
			tuple.index = current_index; // FIXME index should be linear
			tuple.activity = iter->second;

			tuples.push_back(tuple);
			total_activity += tuple.activity;

			current_index++;
		}
	}

	{ // sort tuples by activities
		std::sort(tuples.begin(),tuples.end(),ActivityMaterialTupleStrictWeakOrdering());
	}

	{ // allocate and fill gpu input structure
		input->activity_size = tuples.size();
		input->activity_data = new float[input->activity_size];
		input->activity_index = new unsigned int[input->activity_size];

		float* activity_iter = input->activity_data;
		unsigned int* index_iter = input->activity_index;
		double cumulated_activity = 0;
		for (ActivityMaterialTuplesVector::const_iterator iter = tuples.begin(); iter != tuples.end(); iter++)
		{
			cumulated_activity += iter->activity;
			*activity_iter = cumulated_activity/total_activity;
			*index_iter = iter->index;

			activity_iter++;
			index_iter++;
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
	cout << "DUMMY GPU CALL" << endl;
}
#endif



