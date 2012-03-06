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
  std::list<GateSourceGPUVoxellizedOutputParticle> particles;
}; //

// for int i== 00
//  {
//    if (!sparse) {
//     GateSourceGPUVoxellizedOutputParticle a;
//     a.E = 12;
//     a .dx = 8;
//     list.push_back(a);
//    }
// }



// 
void GateGPUGeneratePrimaries(const GateSourceGPUVoxellizedInput * input, 
                              GateSourceGPUVoxellizedOutputParticles & output);

#endif


