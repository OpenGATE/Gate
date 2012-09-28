/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATETRACKINGGPUACTORIO_H
#define GATETRACKINGGPUACTORIO_H

#include "GateVImageVolume.hh"
#include <list>
#include <map>
#include <vector>

struct GateTrackingGPUActorParticle {
  float E;  // MeV
  float dx; // direction (unary)
  float dy; // direction (unary)
  float dz; // direction (unary)
  float px; // mm position
  float py; // mm position
  float pz; // mm position
  float t;  // ns time
  int   eventID; // event ID of the particle
  int   trackID; // track ID of the particle
  int   type; // gamma = 0 ; e- = 1
};

struct GateTrackingGPUActorInput {
  typedef std::vector<GateTrackingGPUActorParticle> ParticlesList;
  ParticlesList particles;
  
  // Phantom 
  // Coordinate system : 0,0,0 corner 
  int phantom_size_x; 
  int phantom_size_y;
  int phantom_size_z;
  float phantom_spacing_x; // mm 
  float phantom_spacing_y; // mm 
  float phantom_spacing_z; // mm 
  std::vector<unsigned short int> phantom_material_data; // value = ID material CONSTANT (mc_cst_pet.cu l550)

  int cudaDeviceID;
  long seed;

  // Number of elements per material
  int nb_materials;
  int nb_elements_total;
  unsigned short int * mat_nb_elements;

  unsigned short int * mat_index;
  unsigned short int * mat_mixture;
  float * mat_atom_num_dens;
  float * mat_nb_atoms_per_vol;
  float * mat_nb_electrons_per_vol;

  float * electron_cut_energy;
  float * electron_max_energy;
  float * electron_mean_excitation_energy;

  float * fX0;
  float * fX1;
  float * fD0;
  float * fC;
  float * fA;
  float * fM;
};


struct GateTrackingGPUActorOutput
 { 
  typedef std::vector<GateTrackingGPUActorParticle> ParticlesList;
  ParticlesList particles;
};


GateTrackingGPUActorInput* GateTrackingGPUActorInput_new();
void GateTrackingGPUActorInput_delete(GateTrackingGPUActorInput* input);
void GateTrackingGPUActorParticle_Print(const GateTrackingGPUActorParticle & p);
void GateTrackingGPUActorInput_Print_mat(GateTrackingGPUActorInput * input, int i);
void GateTrackingGPUActorInput_Init_Materials(GateTrackingGPUActorInput* input, 
                                              GateVImageVolume * v);


GateTrackingGPUActorOutput* GateTrackingGPUActorOutput_new();
void GateTrackingGPUActorOutput_delete(GateTrackingGPUActorOutput* output);

// Main function that lunch GPU calculation
void GateTrackingGPUActorTrack(const GateTrackingGPUActorInput * input, 
                               GateTrackingGPUActorOutput * output);

#endif
