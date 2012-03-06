/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATESOURCEGPUVOXELLIZEDIO_H
#define GATESOURCEGPUVOXELLIZEDIO_H


struct GateSourceGPUVoxellizedInput {
  long nb_events;
  double E; // MeV
  long seed;
  
  // Phantom and activity map (same size)
  // Coordinate system : 0,0,0 corner 
  int phantom_size_x;
  int phantom_size_y;
  int phantom_size_z;
  float phantom_spacing; // to be change into spacingx spacingy spacingz FIXME
  float * phantom_activity_data; // no unit (relative) FIXME ? GATE PROCESS ????
  unsigned short int * phantom_phantom_data; // value = ID material CONSTANT (mc_cst_pet.cu l550)
  // GPU (later) : ID gpu, nb thread/block
  // Physics (later) : 

};

struct GateSourceGPUVoxellizedOutputParticles{ // Allocated by GPU
  unsigned long size; // size of the array
  float* E; // MeV
  float* dx; // direction (unary)
  float* dy; // direction (unary)
  float* dz; // direction (unary)
  float* px; // mm position
  float* py; // mm position
  float* pz; // mm position
  float* t;  // ns time

  unsigned long int index; // current index

  // Internal GPU
  unsigned int* seed;
  unsigned char* interaction; // internal gpu
  unsigned char* live; // internal  gpu
  unsigned char* endsimu;  // internal  gpu
  unsigned char* ct_cpt; //internal gpu
  unsigned char* ct_pe;  //internal gpu
  unsigned char* ct_ray;  //internal gpu
  unsigned long* table_x_brent;  //internal gpu
}; //


// To be used in GPU
GateSourceGPUVoxellizedOutputParticles * GateSourceGPUVoxellizedOutputParticles_new(unsigned long size);

// To be used in CPU
void GateSourceGPUVoxellizedOutputParticles_delete(GateSourceGPUVoxellizedOutputParticles * output);

// 
void GateGPUGeneratePrimaries(const GateSourceGPUVoxellizedInput * input, 
                              GateSourceGPUVoxellizedOutputParticles * output);

#endif


