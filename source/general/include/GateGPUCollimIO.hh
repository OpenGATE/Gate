/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATEGPUCOLLIMIO_H
#define GATEGPUCOLLIMIO_H

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

#ifndef COLLI
#define COLLI
// TODO a revoir pour le collimateur
// Volume structure data
struct Colli {
	int size_x; 
  	int size_y;
  	int size_z;
    double HexaRadius;
    double HexaHeight;
    int CubRepNumY;
    int CubRepNumZ;
  	double CubRepVecX;
  	double CubRepVecY;
  	double CubRepVecZ;
  	double LinRepVecX;
  	double LinRepVecY;
  	double LinRepVecZ;
};
#endif


#ifndef COORDHEX2
#define COORDHEX2
    struct CoordHex2 {
        double *y;
        double *z;
        unsigned int size;
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
struct GateGPUCollimIO_Particle {
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
struct GateGPUCollimIO_Input {
  typedef std::vector<GateGPUCollimIO_Particle> ParticlesList;
  ParticlesList particles;
  
  // Phantom 
  // Coordinate system : 0,0,0 corner 
  /*int phantom_size_x; 
  int phantom_size_y;
  int phantom_size_z;
  float phantom_spacing_x; // mm 
  float phantom_spacing_y; // mm 
  float phantom_spacing_z; // mm */
  
  //std::vector<unsigned short int> collim_material_data; 
  //std::vector<unsigned short int> phantom_material_data; 

  // Collimator
  int size_x;
  int size_y;
  int size_z;
  double HexaRadius;
  double HexaHeight;
  int CubRepNumY;
  int CubRepNumZ;
  double CubRepVecX;
  double CubRepVecY;
  double CubRepVecZ;
  double LinRepVecX;
  double LinRepVecY;
  double LinRepVecZ;

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
 /* std::vector<float> activity_data; 
  std::vector<unsigned int> activity_index;
  float tot_activity;*/
  // The ID of each particle will start from this number : 
  int firstInitialID; 
  double E; // MeV
  long nb_events;
};
//----------------------------------------------------------


//----------------------------------------------------------
struct GateGPUCollimIO_Output
{ 
  typedef std::vector<GateGPUCollimIO_Particle> ParticlesList;
  ParticlesList particles;
  double deltaTime;
};
//----------------------------------------------------------


//----------------------------------------------------------
typedef std::map<std::vector<int>,double> ActivityMap;
void GateGPUCollimIO_Particle_Print(const GateGPUCollimIO_Particle & p);
GateGPUCollimIO_Input* GateGPUCollimIO_Input_new();
void GateGPUCollimIO_Input_delete(GateGPUCollimIO_Input* input);
void GateGPUCollimIO_Input_Print_mat(GateGPUCollimIO_Input * input, int i);
void GateGPUCollimIO_Input_Init_Materials(GateGPUCollimIO_Input* input, 
                                    std::vector<G4Material*> & m, 
                                    G4String & name);
/*void GateGPUCollimIO_Input_parse_activities(const ActivityMap& activities, 
                                      GateGPUCollimIO_Input * input);*/
GateGPUCollimIO_Output* GateGPUCollimIO_Output_new();
void GateGPUCollimIO_Output_delete(GateGPUCollimIO_Output* output);
//----------------------------------------------------------


//----------------------------------------------------------
// Main function that lunch GPU calculation: SPECT Application

void GPU_GateSPECT_init(const GateGPUCollimIO_Input *input, Colli &colli_d, 
						CoordHex2 &centerOfHexagons_h, CoordHex2 &centerOfHexagons_d, 
						StackParticle &photons_d, StackParticle &photons_h, Materials &materials_d,
						unsigned int nb_of_particles, unsigned int nb_of_hexagons, unsigned int seed);

void GPU_GateSPECT(Colli &colli_d, CoordHex2 &centerOfHexagons_h, CoordHex2 &centerOfHexagons_d, 
					StackParticle &photons_d, StackParticle &photons_h, Materials &materials_d, 
					unsigned int nb_of_particles);
					
void GPU_GateSPECT_end(CoordHex2 &centerOfHexagons_d, StackParticle &photons_d, StackParticle &photons_h,
						Materials &materials_d);					

/*void GPU_GateTransTomo_init(const GateGPUCollimIO_Input *input,
                            Materials &materials_d, Volume &phantom_d,
                            StackParticle &photons_d, StackParticle &photon_h,
                            unsigned int nb_of_particles, unsigned int seed);

void GPU_GateTransTomo(Materials &materials_d, Volume &phantom_d,
                       StackParticle &photons_d, StackParticle &photons_h,
                       unsigned int nb_of_particles);

void GPU_GateTransTomo_end(Materials &materials_d, Volume &phantom_d,
                           StackParticle &photons_d, StackParticle &photons_h);

// Main function that lunch GPU calculation: Emission Tomography Application
void GPU_GateEmisTomo_init(const GateGPUCollimIO_Input *input,
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
void GPU_GatePhotRadThera_init(const GateGPUCollimIO_Input *input, 
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
void GateOpticalBiolum_GPU(const GateGPUCollimIO_Input * input, 
                           GateGPUCollimIO_Output * output);
//----------------------------------------------------------
*/
#endif
