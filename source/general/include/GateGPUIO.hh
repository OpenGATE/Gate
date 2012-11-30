/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATEGPUIO_H
#define GATEGPUIO_H

#include <vector>
#include "GateVImageVolume.hh"

//----------------------------------------------------------
struct GateGPUIO_Particle {
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
  int   initialID; // initial ID of the particle
};
//----------------------------------------------------------


//----------------------------------------------------------
struct GateGPUIO_Input {
  typedef std::vector<GateGPUIO_Particle> ParticlesList;
  ParticlesList particles;
  
  // Phantom 
  // Coordinate system : 0,0,0 corner 
  int phantom_size_x; 
  int phantom_size_y;
  int phantom_size_z;
  float phantom_spacing_x; // mm 
  float phantom_spacing_y; // mm 
  float phantom_spacing_z; // mm 
  
  // value = ID material CONSTANT (mc_cst_pet.cu l550)
  std::vector<unsigned short int> phantom_material_data; 

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

  // When the IO is used as a SourceGPUVoxellized : 
  std::vector<float> activity_data; 
  std::vector<unsigned int> activity_index; 
  // The ID of each particle will start from this number : 
  int firstInitialID; 
  double E; // MeV
  long nb_events;
};
//----------------------------------------------------------


//----------------------------------------------------------
struct GateGPUIO_Output
{ 
  typedef std::vector<GateGPUIO_Particle> ParticlesList;
  ParticlesList particles;
  double deltaTime;
};
//----------------------------------------------------------


//----------------------------------------------------------
typedef std::map<std::vector<int>,double> ActivityMap;
void GateGPUIO_Particle_Print(const GateGPUIO_Particle & p);
GateGPUIO_Input* GateGPUIO_Input_new();
void GateGPUIO_Input_delete(GateGPUIO_Input* input);
void GateGPUIO_Input_Print_mat(GateGPUIO_Input * input, int i);
void GateGPUIO_Input_Init_Materials(GateGPUIO_Input* input, 
                                    std::vector<G4Material*> & m, 
                                    G4String & name);
void GateGPUIO_Input_parse_activities(const ActivityMap& activities, 
                                      GateGPUIO_Input * input);
GateGPUIO_Output* GateGPUIO_Output_new();
void GateGPUIO_Output_delete(GateGPUIO_Output* output);
//----------------------------------------------------------


//----------------------------------------------------------
// Main function that lunch GPU calculation : Actor Tracking
void GateGPU_ActorTrack(const GateGPUIO_Input * input, 
                        GateGPUIO_Output * output);
// Main function that lunch GPU calculation : Voxelized Source 
void GateGPU_VoxelSource_GeneratePrimaries(const GateGPUIO_Input * input, 
                                           GateGPUIO_Output * output);
//----------------------------------------------------------

#endif
