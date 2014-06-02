/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATEGPUIO_H
#define GATEGPUIO_H

#include <vector>
#include "GateVImageVolume.hh"

#ifndef __CUDACC__
#ifndef FLOAT3
#define FLOAT3
    struct float3 {
        float x, y, z;
    };
#endif

#ifndef INT3
#define INT3
    struct int3 {
        int x, y, z;
    };
#endif
#endif

#ifndef DOSIMETRY
#define DOSIMETRY
struct Dosimetry {
    float *edep;
    float *edep2;
    
    unsigned int mem_data;
    float3 size_in_mm;
    int3 size_in_vox;
    float3 voxel_size;
    int nb_voxel_volume;
    int nb_voxel_slice;
    float3 position;
};
#endif

#ifndef MATERIALS
#define MATERIALS
// Structure for materials
struct Materials{
    unsigned int nb_materials;              // n
    unsigned int nb_elements_total;         // k
    
    unsigned short int *nb_elements;        // n
    unsigned short int *index;              // n
    unsigned short int *mixture;            // k
    float *atom_num_dens;                   // k
    float *nb_atoms_per_vol;                // n
    float *nb_electrons_per_vol;            // n
    float *electron_cut_energy;             // n
    float *electron_max_energy;             // n
    float *electron_mean_excitation_energy; // n
    float *rad_length;                      // n
    float *fX0;                             // n
    float *fX1;
    float *fD0;
    float *fC;
    float *fA;
    float *fM;
};
#endif

#ifndef VOLUME
#define VOLUME
// Volume structure data
struct Volume {
    unsigned short int *data;
    unsigned int mem_data;
    float3 size_in_mm;
    int3 size_in_vox;
    float3 voxel_size;
    int nb_voxel_volume;
    int nb_voxel_slice;
    float3 position;
};
#endif

#ifndef STACKPARTICLE
#define STACKPARTICLE
// Stack of particles, format data is defined as SoA
struct StackParticle {
	float* E;
	float* dx;
	float* dy;
	float* dz;
	float* px;
	float* py;
	float* pz;
	float* t;
    unsigned short int* type;
    unsigned int* eventID;
    unsigned int* trackID;
	unsigned int* seed;
    unsigned char* active;
	unsigned char* endsimu;
	unsigned long* table_x_brent;
	unsigned int size;
}; //
#endif

#ifndef ACTIVITIES
#define ACTIVITIES
struct Activities {
    unsigned int nb_activities;
    float tot_activity;
    unsigned int *act_index;
    float *act_cdf;
};
#endif

#ifndef PRESTEPGPU
#define PRESTEPGPU
// PresStep memory used by GPU module
struct PreStepGPU {
    float *x;
    float *y;
    float *z;
};
#endif

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
  int   type; // See G4 Particle Type
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
  float * rad_length;

  float * fX0;
  float * fX1;
  float * fD0;
  float * fC;
  float * fA;
  float * fM;

  // When the IO is used as a SourceGPUVoxellized : 
  std::vector<float> activity_data; 
  std::vector<unsigned int> activity_index;
  float tot_activity;
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
// Main function that lunch GPU calculation: Transmission Tomography Application

void GPU_GateTransTomo_init(const GateGPUIO_Input *input,
                            Materials &materials_d, Volume &phantom_d,
                            StackParticle &photons_d, StackParticle &photon_h,
                            unsigned int nb_of_particles, unsigned int seed);

void GPU_GateTransTomo(Materials &materials_d, Volume &phantom_d,
                       StackParticle &photons_d, StackParticle &photons_h,
                       unsigned int nb_of_particles);

void GPU_GateTransTomo_end(Materials &materials_d, Volume &phantom_d,
                           StackParticle &photons_d, StackParticle &photons_h);

// Main function that lunch GPU calculation: Emission Tomography Application
void GPU_GateEmisTomo_init(const GateGPUIO_Input *input,
                           Materials &materials_d, Volume &phantom_d, Activities &activities_d,
                           StackParticle &gamma1_d, StackParticle &gamma2_d,
                           StackParticle &gamma1_h, StackParticle &gamma2_h,
                           unsigned int nb_of_particles, unsigned int seed);

void GPU_GateEmisTomo(Materials &materials_d, Volume &phantom_d, Activities &activities_d,
                      StackParticle &gamma1_d, StackParticle &gamma2_d,
                      StackParticle &gamma1_h, StackParticle &gamma2_h,
                      unsigned int nb_of_particles);

void GPU_GateEmisTomo_end(Materials &materials_d, Volume &phantom_d, Activities &activities_d,
                          StackParticle &gamma1_d, StackParticle &gamma2_d,
                          StackParticle &gamma1_h, StackParticle &gamma2_h);

// Main function that lunch GPU calculation: Photon Radiation Therapy
void GPU_GatePhotRadThera_init(const GateGPUIO_Input *input, 
                                     Dosimetry &dose_d,
                                     Materials &materials_d,
                                     Volume &phantom_d,
                                     StackParticle &photons_d, StackParticle &electrons_d,
                                     StackParticle &photons_h, 
                                     unsigned int nb_of_particles, unsigned int seed);

void GPU_GatePhotRadThera(Dosimetry &dosemap_d,
                          Materials &materials_d,
                          Volume &phantom_d,
                          StackParticle &photons_d, StackParticle &electrons_d,
                          StackParticle &photons_h,
                          unsigned int nb_of_particles);

void GPU_GatePhotRadThera_end(Dosimetry &dosemap_d, 
                              Materials &materials_d, 
                              Volume &phantom_d,
                              StackParticle &photons_d, StackParticle &electrons_d,
                              StackParticle &photons_h);

// Main function that lunch GPU calculation: Optical photon
void GateOpticalBiolum_GPU(const GateGPUIO_Input * input, 
                           GateGPUIO_Output * output);
//----------------------------------------------------------

#endif
