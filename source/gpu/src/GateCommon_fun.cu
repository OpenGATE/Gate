#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>

/***********************************************************
 * Vars
 ***********************************************************/
__constant__ const float gpu_pi = 3.14159265358979323846;
__constant__ const float gpu_twopi = 2*gpu_pi;

/***********************************************************
 * Utils Host
 ***********************************************************/

// Get time
double time() {
	timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec / 1000000.0;
}

/***********************************************************
 * Data material structure
 ***********************************************************/

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

// Materials device allocation
void materials_device_malloc(Materials &mat, unsigned int nb_mat, unsigned int nb_elm) {
	
    mat.nb_materials = nb_mat;
    mat.nb_elements_total = nb_elm;
    
    unsigned int mem_mat_usi = nb_mat * sizeof(unsigned short int);
    unsigned int mem_mat_float = nb_mat * sizeof(float);
    unsigned int mem_elm_usi = nb_elm * sizeof(unsigned short int);
    unsigned int mem_elm_float = nb_elm * sizeof(float);
    
    cudaMalloc((void**) &mat.nb_elements, mem_mat_usi);
    cudaMalloc((void**) &mat.index, mem_mat_usi);
    cudaMalloc((void**) &mat.mixture, mem_elm_usi);
    cudaMalloc((void**) &mat.atom_num_dens, mem_elm_float);
    cudaMalloc((void**) &mat.nb_atoms_per_vol, mem_mat_float);
    cudaMalloc((void**) &mat.nb_electrons_per_vol, mem_mat_float);
    cudaMalloc((void**) &mat.electron_cut_energy, mem_mat_float);
    cudaMalloc((void**) &mat.electron_max_energy, mem_mat_float);
    cudaMalloc((void**) &mat.electron_mean_excitation_energy, mem_mat_float);
    cudaMalloc((void**) &mat.rad_length, mem_mat_float);
    cudaMalloc((void**) &mat.fX0, mem_mat_float);
    cudaMalloc((void**) &mat.fX1, mem_mat_float);
    cudaMalloc((void**) &mat.fD0, mem_mat_float);
    cudaMalloc((void**) &mat.fC, mem_mat_float);
    cudaMalloc((void**) &mat.fA, mem_mat_float);
    cudaMalloc((void**) &mat.fM, mem_mat_float);
}

// Materials free device memory
void materials_device_free(Materials &mat) {
    cudaFree(mat.nb_elements);
    cudaFree(mat.index);
    cudaFree(mat.mixture);
    cudaFree(mat.atom_num_dens);
    cudaFree(mat.nb_atoms_per_vol);
    cudaFree(mat.nb_electrons_per_vol);
    cudaFree(mat.electron_cut_energy);
    cudaFree(mat.electron_max_energy);
    cudaFree(mat.electron_mean_excitation_energy);
    cudaFree(mat.rad_length);
    cudaFree(mat.fX0);
    cudaFree(mat.fX1);
    cudaFree(mat.fD0);
    cudaFree(mat.fC);
    cudaFree(mat.fA);
    cudaFree(mat.fM);
}


// Materials host allocation
void materials_host_malloc(Materials &mat, unsigned int nb_mat, unsigned int nb_elm) {
	
    mat.nb_materials = nb_mat;
    mat.nb_elements_total = nb_elm;
    
    unsigned int mem_mat_usi = nb_mat * sizeof(unsigned short int);
    unsigned int mem_mat_float = nb_mat * sizeof(float);
    unsigned int mem_elm_usi = nb_elm * sizeof(unsigned short int);
    unsigned int mem_elm_float = nb_elm * sizeof(float);
    mat.nb_elements = (unsigned short int*)malloc(mem_mat_usi);
    mat.index = (unsigned short int*)malloc(mem_mat_usi);
    mat.mixture = (unsigned short int*)malloc(mem_elm_usi);
    mat.atom_num_dens = (float*)malloc(mem_elm_float);
    mat.nb_atoms_per_vol = (float*)malloc(mem_mat_float);
    mat.nb_electrons_per_vol = (float*)malloc(mem_mat_float);
    mat.electron_cut_energy = (float*)malloc(mem_mat_float);
    mat.electron_max_energy = (float*)malloc(mem_mat_float);
    mat.electron_mean_excitation_energy = (float*)malloc(mem_mat_float);
    mat.rad_length = (float*)malloc(mem_mat_float);
    mat.fX0 = (float*)malloc(mem_mat_float);
    mat.fX1 = (float*)malloc(mem_mat_float);
    mat.fD0 = (float*)malloc(mem_mat_float);
    mat.fC = (float*)malloc(mem_mat_float);
    mat.fA = (float*)malloc(mem_mat_float);
    mat.fM = (float*)malloc(mem_mat_float);
}

// Materials free memory
void materials_host_free(Materials &mat) {
    free(mat.nb_elements);
    free(mat.index);
    free(mat.mixture);
    free(mat.atom_num_dens);
    free(mat.nb_atoms_per_vol);
    free(mat.nb_electrons_per_vol);
    free(mat.electron_cut_energy);
    free(mat.electron_max_energy);
    free(mat.electron_mean_excitation_energy);
    free(mat.rad_length);
    free(mat.fX0);
    free(mat.fX1);
    free(mat.fD0);
    free(mat.fC);
    free(mat.fA);
    free(mat.fM);
}

/***********************************************************
 * Stack data particle structure
 ***********************************************************/

#ifndef STACKPARTICLE
#define STACKPARTICLE
// Stack of particles, format data is defined as SoA
struct StackParticle{
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

// Stack host allocation
void stack_host_malloc(StackParticle &phasespace, int stack_size) {
	phasespace.size = stack_size;
	unsigned int mem_phasespace_float = stack_size * sizeof(float);
	unsigned int mem_phasespace_uint = stack_size * sizeof(unsigned int);
	unsigned int mem_phasespace_usint = stack_size * sizeof(unsigned short int);
	unsigned int mem_phasespace_char = stack_size * sizeof(char);
	phasespace.E = (float*)malloc(mem_phasespace_float);
	phasespace.dx = (float*)malloc(mem_phasespace_float);
	phasespace.dy = (float*)malloc(mem_phasespace_float);
	phasespace.dz = (float*)malloc(mem_phasespace_float);
	phasespace.px = (float*)malloc(mem_phasespace_float);
	phasespace.py = (float*)malloc(mem_phasespace_float);
	phasespace.pz = (float*)malloc(mem_phasespace_float);
	phasespace.t = (float*)malloc(mem_phasespace_float);
	phasespace.type = (unsigned short int*)malloc(mem_phasespace_usint);
	phasespace.seed = (unsigned int*)malloc(mem_phasespace_uint);
	phasespace.eventID = (unsigned int*)malloc(mem_phasespace_uint);
    phasespace.trackID = (unsigned int*)malloc(mem_phasespace_uint);
	phasespace.endsimu = (unsigned char*)malloc(mem_phasespace_char);
	phasespace.active = (unsigned char*)malloc(mem_phasespace_char);
}

// free host mem
void stack_host_free(StackParticle &phasespace) {
	free(phasespace.E);
	free(phasespace.dx);
	free(phasespace.dy);
	free(phasespace.dz);
	free(phasespace.px);
	free(phasespace.py);
	free(phasespace.pz);
	free(phasespace.t);
	free(phasespace.type);
	free(phasespace.seed);
	free(phasespace.eventID);
	free(phasespace.trackID);
	free(phasespace.endsimu);
	free(phasespace.active);
}

// For PRNG Brent
#define UINT64 (sizeof(unsigned long)>>3)
#define UINT32 (1 - UINT64)
#define r      (4*UINT64 + 8*UINT32)
// Stack device allocation
void stack_device_malloc(StackParticle &stackpart, int stack_size) {
	stackpart.size = stack_size;
	unsigned int mem_stackpart_float = stack_size * sizeof(float);
	unsigned int mem_stackpart_uint = stack_size * sizeof(unsigned int);
	unsigned int mem_stackpart_usint = stack_size * sizeof(unsigned short int);
	unsigned int mem_stackpart_char = stack_size * sizeof(char);
	unsigned int mem_brent;
	if (r == 4) {mem_brent = stack_size * 6 * sizeof(unsigned long);}
	else {mem_brent = stack_size * 10 * sizeof(unsigned long);}

	cudaMalloc((void**) &stackpart.E, mem_stackpart_float);
	cudaMalloc((void**) &stackpart.dx, mem_stackpart_float);
	cudaMalloc((void**) &stackpart.dy, mem_stackpart_float);
	cudaMalloc((void**) &stackpart.dz, mem_stackpart_float);
	cudaMalloc((void**) &stackpart.px, mem_stackpart_float);
	cudaMalloc((void**) &stackpart.py, mem_stackpart_float);
	cudaMalloc((void**) &stackpart.pz, mem_stackpart_float);	
	cudaMalloc((void**) &stackpart.t, mem_stackpart_float);	
	cudaMalloc((void**) &stackpart.type, mem_stackpart_usint);	
	cudaMalloc((void**) &stackpart.seed, mem_stackpart_uint);
	cudaMalloc((void**) &stackpart.eventID, mem_stackpart_uint);
	cudaMalloc((void**) &stackpart.trackID, mem_stackpart_uint);
	cudaMalloc((void**) &stackpart.table_x_brent, mem_brent);
	cudaMalloc((void**) &stackpart.endsimu, mem_stackpart_char);
	cudaMalloc((void**) &stackpart.active, mem_stackpart_char);
}
#undef UINT64
#undef UINT32
#undef r

// free device mem
void stack_device_free(StackParticle &stackpart) {
	cudaFree(stackpart.E);
	cudaFree(stackpart.dx);
	cudaFree(stackpart.dy);
	cudaFree(stackpart.dz);
	cudaFree(stackpart.px);
	cudaFree(stackpart.py);
	cudaFree(stackpart.pz);
	cudaFree(stackpart.t);
	cudaFree(stackpart.type);
	cudaFree(stackpart.seed);
	cudaFree(stackpart.eventID);
	cudaFree(stackpart.trackID);
	cudaFree(stackpart.endsimu);
	cudaFree(stackpart.active);
	cudaFree(stackpart.table_x_brent);
}

/***********************************************************
 * Volume data structure
 ***********************************************************/

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

// Volume host allocation
void volume_host_malloc(Volume &vol, int nbvox) {
    vol.mem_data = nbvox * sizeof(unsigned short int);
    vol.data = (unsigned short int*)malloc(vol.mem_data);
}

// Free host memory
void volume_host_free(Volume &vol) {
    free(vol.data);
}

// Volume device allocation
void volume_device_malloc(Volume &vol, int nbvox) {
    vol.mem_data = nbvox * sizeof(unsigned short int);
	cudaMalloc((void**) &vol.data, vol.mem_data);
}

// Free device memory
void volume_device_free(Volume &vol) {
    cudaFree(vol.data);
}

/***********************************************************
 * Hexagonal hole coordinates
 ***********************************************************/

#ifndef COORDHEX2
#define COORDHEX2
// Volume structure data
struct CoordHex2 {
    double* y; 
  	double* z;
  	unsigned int size;
};
#endif


// Hexa host allocation
void Hexa_host_malloc(CoordHex2 &HexaCoord, int nbpoint) {
    HexaCoord.size = nbpoint;
    unsigned int mem_hexacoord_double = nbpoint * sizeof(double);
    HexaCoord.y = (double*)malloc(mem_hexacoord_double);
    HexaCoord.z = (double*)malloc(mem_hexacoord_double);
}

// free host mem
void Hexa_host_free(CoordHex2 &HexaCoord) {
	free(HexaCoord.y);
	free(HexaCoord.z);
}

// Hexa device allocation
void Hexa_device_malloc(CoordHex2 &HexaCoordDev, int nbpoint) {
	HexaCoordDev.size = nbpoint;
    unsigned int mem_hexacoorddev_double = nbpoint * sizeof(double);
    
    cudaMalloc((void**) &HexaCoordDev.y, mem_hexacoorddev_double);
    cudaMalloc((void**) &HexaCoordDev.z, mem_hexacoorddev_double);
}

// free device mem
void Hexa_device_free(CoordHex2 &HexaCoordDev) {
	cudaFree(HexaCoordDev.y);
	cudaFree(HexaCoordDev.z);
}

// Hexa parameters reset
void Hexa_host_reset(CoordHex2 &HexaCoord) {
    int i=0; while(i<HexaCoord.size) {
        HexaCoord.y[i] = 0.0f;
        HexaCoord.z[i] = 0.0f;
        ++i;
    }
}

/***********************************************************
 * Hexagonal hole collimator parameters
 ***********************************************************/
#ifndef COLLI
#define COLLI
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

/***********************************************************
 * Dosimetry data structure
 ***********************************************************/
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

// Dosimetry host allocation
void dosimetry_host_malloc(Dosimetry &vol, int nbvox) {
    vol.mem_data = nbvox * sizeof(float);
    vol.edep = (float*)malloc(vol.mem_data);
}

// Dosimetry free host memory
void dosimetry_host_free(Dosimetry &vol) {
    free(vol.edep);
}

// Dosimetry volume device allocation
void dosimetry_device_malloc(Dosimetry &vol, int nbvox) {
    vol.mem_data = nbvox * sizeof(float);
	cudaMalloc((void**) &vol.edep, vol.mem_data);
}

// Dosimetry free device memory
void dosimetry_device_free(Dosimetry &vol) {
    cudaFree(vol.edep);
}

// Dosimetry reset
void dosimetry_host_reset(Dosimetry &vol) {
    int i=0; while(i<vol.nb_voxel_volume) {
        vol.edep[i] = 0.0f;
        ++i;
    }
}

/***********************************************************
 * Activities structure
 ***********************************************************/

#ifndef ACTIVITIES
#define ACTIVITIES
struct Activities {
    unsigned int nb_activities;
    float tot_activity;
    unsigned int *act_index;
    float *act_cdf;
};
#endif

// Host allocation
void activities_host_malloc(Activities &act, int nbact) {
    act.act_index = (unsigned int*)malloc(nbact*sizeof(unsigned int));
    act.act_cdf = (float*)malloc(nbact*sizeof(float));
}

// Device allocation
void activities_device_malloc(Activities &act, int nbact) {
    cudaMalloc((void**) &act.act_index, nbact*sizeof(float));
    cudaMalloc((void**) &act.act_cdf, nbact*sizeof(float));
}

// Free host mem
void activities_host_free(Activities &act) {
    free(act.act_index);
    free(act.act_cdf);
}

// Free device mem
void activities_device_free(Activities &act) {
    cudaFree(act.act_index);
    cudaFree(act.act_cdf);
}

/***********************************************************
 * Copy structure functions
 ***********************************************************/

// Copy materials from host to device
void materials_copy_host2device(Materials &host, Materials &device) {
    unsigned int nb_mat = host.nb_materials;
    unsigned int nb_elm = host.nb_elements_total;
    
    unsigned int mem_mat_usi = nb_mat * sizeof(unsigned short int);
    unsigned int mem_mat_float = nb_mat * sizeof(float);
    unsigned int mem_elm_usi = nb_elm * sizeof(unsigned short int);
    unsigned int mem_elm_float = nb_elm * sizeof(float);
    
    cudaMemcpy(device.nb_elements, host.nb_elements, mem_mat_usi, cudaMemcpyHostToDevice);
    cudaMemcpy(device.index, host.index, mem_mat_usi, cudaMemcpyHostToDevice);
    cudaMemcpy(device.mixture, host.mixture, mem_elm_usi, cudaMemcpyHostToDevice);
    cudaMemcpy(device.atom_num_dens, host.atom_num_dens, mem_elm_float, cudaMemcpyHostToDevice);
    cudaMemcpy(device.nb_atoms_per_vol, host.nb_atoms_per_vol, mem_mat_float, cudaMemcpyHostToDevice);
    cudaMemcpy(device.nb_electrons_per_vol, host.nb_electrons_per_vol, mem_mat_float, cudaMemcpyHostToDevice);
    cudaMemcpy(device.electron_cut_energy, host.electron_cut_energy, mem_mat_float, cudaMemcpyHostToDevice);
    cudaMemcpy(device.electron_max_energy, host.electron_max_energy, mem_mat_float, cudaMemcpyHostToDevice);
    cudaMemcpy(device.electron_mean_excitation_energy, host.electron_mean_excitation_energy, mem_mat_float, cudaMemcpyHostToDevice);
    cudaMemcpy(device.rad_length, host.rad_length, mem_mat_float, cudaMemcpyHostToDevice);
    cudaMemcpy(device.fX0, host.fX0, mem_mat_float, cudaMemcpyHostToDevice);
    cudaMemcpy(device.fX1, host.fX1, mem_mat_float, cudaMemcpyHostToDevice);
    cudaMemcpy(device.fD0, host.fD0, mem_mat_float, cudaMemcpyHostToDevice);
    cudaMemcpy(device.fC, host.fC, mem_mat_float, cudaMemcpyHostToDevice);
    cudaMemcpy(device.fA, host.fA, mem_mat_float, cudaMemcpyHostToDevice);
    cudaMemcpy(device.fM, host.fM, mem_mat_float, cudaMemcpyHostToDevice);
}
 
// Copy stack from device to host
void stack_copy_device2host(StackParticle &stackpart, StackParticle &phasespace) {
	int stack_size = stackpart.size;
	unsigned int mem_stackpart_float = stack_size * sizeof(float);
	unsigned int mem_stackpart_char = stack_size * sizeof(char);
	unsigned int mem_stackpart_uint = stack_size * sizeof(unsigned int);
	unsigned int mem_stackpart_usint = stack_size * sizeof(unsigned short int);
	
	cudaMemcpy(phasespace.E, stackpart.E, mem_stackpart_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(phasespace.dx, stackpart.dx, mem_stackpart_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(phasespace.dy, stackpart.dy, mem_stackpart_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(phasespace.dz, stackpart.dz, mem_stackpart_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(phasespace.px, stackpart.px, mem_stackpart_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(phasespace.py, stackpart.py, mem_stackpart_float, cudaMemcpyDeviceToHost);
	cudaMemcpy(phasespace.pz, stackpart.pz, mem_stackpart_float, cudaMemcpyDeviceToHost);	
	cudaMemcpy(phasespace.t, stackpart.t, mem_stackpart_float, cudaMemcpyDeviceToHost);	
	cudaMemcpy(phasespace.type, stackpart.type, mem_stackpart_usint, cudaMemcpyDeviceToHost);	
	cudaMemcpy(phasespace.endsimu, stackpart.endsimu, mem_stackpart_char, cudaMemcpyDeviceToHost);
	cudaMemcpy(phasespace.active, stackpart.active, mem_stackpart_char, cudaMemcpyDeviceToHost);
	cudaMemcpy(phasespace.trackID, stackpart.trackID, mem_stackpart_uint, cudaMemcpyDeviceToHost);
	cudaMemcpy(phasespace.eventID, stackpart.eventID, mem_stackpart_uint, cudaMemcpyDeviceToHost);
}

// Copy stack from host to device
void stack_copy_host2device(StackParticle &phasespace, StackParticle &stackpart) {
	int stack_size = phasespace.size;
	unsigned int mem_stackpart_float = stack_size * sizeof(float);
	unsigned int mem_stackpart_char = stack_size * sizeof(char);
	unsigned int mem_stackpart_uint = stack_size * sizeof(unsigned int);
	unsigned int mem_stackpart_usint = stack_size * sizeof(unsigned short int);
	
	cudaMemcpy(stackpart.E, phasespace.E, mem_stackpart_float,   cudaMemcpyHostToDevice);
	cudaMemcpy(stackpart.dx, phasespace.dx, mem_stackpart_float, cudaMemcpyHostToDevice);
	cudaMemcpy(stackpart.dy, phasespace.dy, mem_stackpart_float, cudaMemcpyHostToDevice);
	cudaMemcpy(stackpart.dz, phasespace.dz, mem_stackpart_float, cudaMemcpyHostToDevice);
	cudaMemcpy(stackpart.px, phasespace.px, mem_stackpart_float, cudaMemcpyHostToDevice);
	cudaMemcpy(stackpart.py, phasespace.py, mem_stackpart_float, cudaMemcpyHostToDevice);
	cudaMemcpy(stackpart.pz, phasespace.pz, mem_stackpart_float, cudaMemcpyHostToDevice);	
	cudaMemcpy(stackpart.t, phasespace.t, mem_stackpart_float, cudaMemcpyHostToDevice);	
	cudaMemcpy(stackpart.type, phasespace.type, mem_stackpart_usint, cudaMemcpyHostToDevice);	
	cudaMemcpy(stackpart.endsimu, phasespace.endsimu, mem_stackpart_char, cudaMemcpyHostToDevice);
	cudaMemcpy(stackpart.active, phasespace.active, mem_stackpart_char, cudaMemcpyHostToDevice);
	cudaMemcpy(stackpart.trackID, phasespace.trackID, mem_stackpart_uint, cudaMemcpyHostToDevice);
	cudaMemcpy(stackpart.eventID, phasespace.eventID, mem_stackpart_uint, cudaMemcpyHostToDevice);
	cudaMemcpy(stackpart.seed, phasespace.seed, mem_stackpart_uint, cudaMemcpyHostToDevice);
}

// Copy volume from device to host
void volume_copy_device2host(Volume &voldevice, Volume &volhost) {
    volhost.size_in_vox = voldevice.size_in_vox;
    volhost.voxel_size = voldevice.voxel_size;
    volhost.size_in_mm = voldevice.size_in_mm;
    volhost.nb_voxel_slice = voldevice.nb_voxel_slice;
    volhost.nb_voxel_volume = voldevice.nb_voxel_volume;
    volhost.mem_data = voldevice.mem_data;
    volhost.position = voldevice.position;
	cudaMemcpy(volhost.data, voldevice.data, voldevice.mem_data, cudaMemcpyDeviceToHost);
}

// Copy volume from host to device
void volume_copy_host2device(Volume &volhost, Volume &voldevice) {
    voldevice.size_in_vox = volhost.size_in_vox;
    voldevice.voxel_size = volhost.voxel_size;
    voldevice.size_in_mm = volhost.size_in_mm;
    voldevice.nb_voxel_slice = volhost.nb_voxel_slice;
    voldevice.nb_voxel_volume = volhost.nb_voxel_volume;
    voldevice.mem_data = volhost.mem_data;
    voldevice.position = volhost.position;
	cudaMemcpy(voldevice.data, volhost.data, volhost.mem_data, cudaMemcpyHostToDevice);
}

// Copy volume from device to host
void dosimetry_copy_device2host(Dosimetry &voldevice, Dosimetry &volhost) {
    volhost.size_in_vox = voldevice.size_in_vox;
    volhost.voxel_size = voldevice.voxel_size;
    volhost.size_in_mm = voldevice.size_in_mm;
    volhost.nb_voxel_slice = voldevice.nb_voxel_slice;
    volhost.nb_voxel_volume = voldevice.nb_voxel_volume;
    volhost.mem_data = voldevice.mem_data;
    volhost.position = voldevice.position;
	cudaMemcpy(volhost.edep, voldevice.edep, voldevice.mem_data, cudaMemcpyDeviceToHost);
}

// Copy HexaCoord from host to device
void Hexa_copy_host2device(CoordHex2 &HexaCoord, CoordHex2 &HexaCoordDev) {
    int coord_size = HexaCoord.size;
	unsigned int mem_hexacoord_double = coord_size * sizeof(double);
	cudaMemcpy(HexaCoordDev.y, HexaCoord.y, mem_hexacoord_double, cudaMemcpyHostToDevice);
	cudaMemcpy(HexaCoordDev.z, HexaCoord.z, mem_hexacoord_double, cudaMemcpyHostToDevice);
}

// Copy HexaCoord from device to host
void Hexa_copy_device2host(CoordHex2 &HexaCoordDev, CoordHex2 &HexaCoord) {
    int coord_size = HexaCoordDev.size;
	unsigned int mem_hexacoorddev_double = coord_size * sizeof(double);
	cudaMemcpy(HexaCoord.y, HexaCoordDev.y, mem_hexacoorddev_double, cudaMemcpyDeviceToHost);
	cudaMemcpy(HexaCoord.z, HexaCoordDev.z, mem_hexacoorddev_double, cudaMemcpyDeviceToHost);
}

// Copy dosimetry from host to device
void dosimetry_copy_host2device(Dosimetry &volhost, Dosimetry &voldevice) {
    voldevice.size_in_vox = volhost.size_in_vox;
    voldevice.voxel_size = volhost.voxel_size;
    voldevice.size_in_mm = volhost.size_in_mm;
    voldevice.nb_voxel_slice = volhost.nb_voxel_slice;
    voldevice.nb_voxel_volume = volhost.nb_voxel_volume;
    voldevice.mem_data = volhost.mem_data;
    voldevice.position = volhost.position;
	cudaMemcpy(voldevice.edep, volhost.edep, volhost.mem_data, cudaMemcpyHostToDevice);
}

// Copy activities from host to device
void activities_copy_host2device(Activities &acthost, Activities &actdevice) {
    actdevice.nb_activities = acthost.nb_activities;
    actdevice.tot_activity = acthost.tot_activity;
    cudaMemcpy(actdevice.act_index, acthost.act_index, 
               actdevice.nb_activities*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(actdevice.act_cdf, acthost.act_cdf,
               actdevice.nb_activities*sizeof(float), cudaMemcpyHostToDevice);
}

/***********************************************************
 * Utils Device
 ***********************************************************/

// rotateUz, function from CLHEP
__device__ float3 rotateUz(float3 vector, float3 newUz) {
	float u1 = newUz.x;
	float u2 = newUz.y;
	float u3 = newUz.z;
	float up = u1*u1 + u2*u2;

	if (up>0) {
		up = sqrtf(up);
		float px = vector.x,  py = vector.y, pz = vector.z;
		vector.x = __fdividef(u1*u3*px - u2*py, up) + u1*pz;
		vector.y = __fdividef(u2*u3*px + u1*py, up) + u2*pz;
		vector.z =              -up*px +              u3*pz;
    }
	else if (u3 < 0.) { vector.x = -vector.x; vector.z = -vector.z; } // phi=0  theta=gpu_pi

	return make_float3(vector.x, vector.y, vector.z);
}

// add vector
__device__ float3 add_vector(float3 u, float3 v) {
    return make_float3(u.x+v.x, u.y+v.y, u.z+v.z);
}

// sub vector
__device__ float3 sub_vector(float3 u, float3 v) {
    return make_float3(u.x-v.x, u.y-v.y, u.z-v.z);
}

// mul a vector by a scalar
__device__ float3 scale_vector(float3 u, float a) {
    return make_float3(u.x*a, u.y*a, u.z*a);
}

// mul two vectors
__device__ float3 mul_vector(float3 u, float3 v) {
    return make_float3(u.x*v.x, u.y*v.y, u.z*v.z);
}

// div two vectors
__device__ float3 div_vector(float3 u, float3 v) {
    return make_float3(__fdividef(u.x, v.x),
                       __fdividef(u.y, v.y),
                       __fdividef(u.z, v.z));
}

// return an unitary vector
__device__ float3 unit_vector(float3 u) {
    float imag = __fdividef(1.0f, sqrtf(u.x*u.x + u.y*u.y + u.z*u.z));
    return make_float3(u.x*imag, u.y*imag, u.z*imag);
}

// return inverse vector
__device__ float3 inverse_vector(float3 u) {
    return make_float3(__fdividef(1.0f, u.x), __fdividef(1.0f, u.y), __fdividef(1.0f, u.z));
}

//// Used for the validation
__device__ float mag_vector(float3 u) {
    return sqrtf(u.x*u.x + u.y*u.y + u.z*u.z);
}
__device__ float dot_vector(float3 u, float3 v) {
    return u.x*v.x + u.y*v.y + u.z*v.z;
}

//// Return the next voxel boundary distance, it is used by the standard navigator
/*
__device__ float get_boundary_voxel_by_raycasting(int4 vox, float3 p, float3 d, float3 res) {
    
    
	float xmin, xmax, ymin, ymax, zmin, zmax;
    float3 di = inverse_vector(d);
	float tmin, tmax, tymin, tymax, tzmin, tzmax, buf;
	
    // Define the voxel bounding box
    xmin = vox.x*res.x;
    ymin = vox.y*res.y;
    zmin = vox.z*res.z;
    xmax = (d.x<0 && p.x==xmin) ? xmin-res.x : xmin+res.x;
    ymax = (d.y<0 && p.y==ymin) ? ymin-res.y : ymin+res.y;
    zmax = (d.z<0 && p.z==zmin) ? zmin-res.z : zmin+res.z;
    
    tmin = -1e9f;
    tmax = 1e9f;
    
    // on x
    if (d.x != 0.0f) {
        tmin = (xmin - p.x) * di.x;
        tmax = (xmax - p.x) * di.x;
        if (tmin > tmax) {
            buf = tmin;
            tmin = tmax;
            tmax = buf;
        }
    }
    // on y
    if (d.y != 0.0f) {
        tymin = (ymin - p.y) * di.y;
        tymax = (ymax - p.y) * di.y;
        if (tymin > tymax) {
            buf = tymin;
            tymin = tymax;
            tymax = buf;
        }
        if (tymin > tmin) {tmin = tymin;}
        if (tymax < tmax) {tmax = tymax;}
    }
    // on z
    if (d.z != 0.0f) {
        tzmin = (zmin - p.z) * di.z;
        tzmax = (zmax - p.z) * di.z;
        if (tzmin > tzmax) {
            buf = tzmin;
            tzmin = tzmax;
            tzmax = buf;
        }
        if (tzmin > tmin) {tmin = tzmin;}
        if (tzmax < tmax) {tmax = tzmax;}
    }

    return tmax;
}
*/

// Return the next voxel boundary distance, it is used by the standard navigator
__device__ float get_boundary_voxel_by_raycasting(int4 vox, float3 p, float3 d, float3 res) {


	float xmin, xmax, ymin, ymax, zmin, zmax;
    float3 di = inverse_vector(d);
	float tmin, tmax, tymin, tymax, tzmin, tzmax, buf;

	/*
    // Define the voxel bounding box
    xmin = vox.x*res.x;
    ymin = vox.y*res.y;
    zmin = vox.z*res.z;
    xmax = (d.x<0 && p.x==xmin) ? xmin-res.x : xmin+res.x;
    ymax = (d.y<0 && p.y==ymin) ? ymin-res.y : ymin+res.y;
    zmax = (d.z<0 && p.z==zmin) ? zmin-res.z : zmin+res.z;
    */

    // From Michaela
    xmin = (d.x > 0 && p.x > (vox.x+1) * res.x - EPS) ? (vox.x+1) * res.x : vox.x*res.x;
    ymin = (d.y > 0 && p.y > (vox.y+1) * res.y - EPS) ? (vox.y+1) * res.y : vox.y*res.y;
    zmin = (d.z > 0 && p.z > (vox.z+1) * res.z - EPS) ? (vox.z+1) * res.z : vox.z*res.z;

    xmax = (d.x < 0 && p.x < xmin + EPS) ? xmin-res.x : xmin+res.x;
    ymax = (d.y < 0 && p.y < ymin + EPS) ? ymin-res.y : ymin+res.y;
    zmax = (d.z < 0 && p.z < zmin + EPS) ? zmin-res.z : zmin+res.z;

    tmin = -INF;
    tmax = INF;

    // on x
    if (fabs(d.x) > EPS) {
        tmin = (xmin - p.x) * di.x;
        tmax = (xmax - p.x) * di.x;
        if (tmin > tmax) {
            buf = tmin;
            tmin = tmax;
            tmax = buf;
        }
    }
    // on y
    if (fabs(d.y) > EPS) {
        tymin = (ymin - p.y) * di.y;
        tymax = (ymax - p.y) * di.y;
        if (tymin > tymax) {
            buf = tymin;
            tymin = tymax;
            tymax = buf;
        }
        if (tymin > tmin) {tmin = tymin;}
        if (tymax < tmax) {tmax = tymax;}
    }
    // on z
    if (fabs(d.z) > EPS) {
        tzmin = (zmin - p.z) * di.z;
        tzmax = (zmax - p.z) * di.z;
        if (tzmin > tzmax) {
            buf = tzmin;
            tzmin = tzmax;
            tzmax = buf;
        }
        if (tzmin > tmin) {tmin = tzmin;}
        if (tzmax < tmax) {tmax = tzmax;}
    }

    return tmax;
}



// Check if the point is located inside a collimator hole (SPECT simulation)
__device__ bool IsInsideHex(float3 position, double radius, float cy, float cz)
{
	// Check if photon is inside an hexagon
	double dify = fabs(position.y - cy);
	double difz = fabs(position.z - cz);
	
	double horiz = radius;
	double verti = (radius * (2.0/sqrt(3.0))) / 2.0;
	
	if(difz >= 2*verti || dify >= horiz || (2*verti*horiz - verti*dify - horiz*difz) <= 0.0 )
		return false;
	
	return true;
}

__device__ int GetHexIndex(float3 position, Colli colli, CoordHex2 centerOfHexagons)
{
  	int col, raw, hex, temp, new_raw, min, max;
  	
	// Define hexagon index
    
    // Find the column in the array of hexagons

  	col = round(((colli.CubRepVecY * (( colli.CubRepNumY - 1 ) / 2.0)) - position.y) 
  						/ colli.CubRepVecY);
  				
  	// if the photon is too close to external frame, col value is incorrect			
  	if (col < 0.0)
  			col = 0.0;
 	else if (col > (colli.CubRepNumY - 1))
    		col = colli.CubRepNumY - 1;
    
  	// Find the raw in the array of hexagons
     
 	raw = round((colli.LinRepVecZ * (colli.CubRepNumZ - 1.0) - position.z) 
  					/ colli.LinRepVecZ);

  	// if the photon is too close to external frame, raw value is incorrect
  	if (raw < 0.0)
  			raw = 0.0;
  	else if (raw > (colli.CubRepNumZ - 1.0) * 2.0 )
    		raw = (colli.CubRepNumZ - 1.0) * 2.0;
  
  	// Find the hexagon index
  
  	// Even raw 
  	if ( raw % 2 == 0.0 ) {
			hex = (raw / 2.0) * ((2.0 * colli.CubRepNumY) - 1.0) + col;
			// Test centered hexagon
			if (IsInsideHex(position, colli.HexaRadius, centerOfHexagons.y[hex], centerOfHexagons.z[hex]))
				return hex;
			else {
				if (raw - 1 >= 0) {
					new_raw = raw - 1;
					min = new_raw * colli.CubRepNumY - ((new_raw - 1)/2);
					max = min + colli.CubRepNumY - 1;
					temp = hex - colli.CubRepNumY - 1;
					
					if(temp >= min)
						// Test top left hexagon
						if(IsInsideHex(position, colli.HexaRadius, centerOfHexagons.y[temp], centerOfHexagons.z[temp]))
							return temp;
			
					temp = hex + colli.CubRepNumY;
				
					if(temp < max)
						// Test top right hexagon
						if(IsInsideHex(position, colli.HexaRadius, centerOfHexagons.y[temp], centerOfHexagons.z[temp]))
							return temp;
				}
				
				if (raw + 1 < colli.CubRepNumY * 2 - 1) {
					new_raw = raw + 1;
					min = new_raw * colli.CubRepNumY - ((new_raw - 1)/2);
					max = min + colli.CubRepNumY - 1;
					temp = hex + colli.CubRepNumY - 1;
					
					if(temp >= min)
						// Test bottom left hexagon
						if(IsInsideHex(position, colli.HexaRadius, centerOfHexagons.y[temp], centerOfHexagons.z[temp]))
							return temp;
					
					temp = hex + colli.CubRepNumY;
				    
					if(temp < max)
						// Test bottom right hexagon
						if(IsInsideHex(position, colli.HexaRadius, centerOfHexagons.y[temp], centerOfHexagons.z[temp]))
							return temp;
				}
			}
	}
	// Odd raw
  	else {
  			hex = ((raw + 1.0)/ 2.0) * colli.CubRepNumY + ((raw - 1.0)/ 2.0) * (colli.CubRepNumY - 1.0) + col;
  			
  			min = raw * colli.CubRepNumY - ((raw - 1)/2);
			max = min + colli.CubRepNumY - 1;
			
			if(hex < max)			
				// Test right hexagon
				if (IsInsideHex(position, colli.HexaRadius, centerOfHexagons.y[hex], centerOfHexagons.z[hex]))
					return hex;
				
  			temp = hex - 1; 
				
			if(temp >= min)
				// Test left hexagon
				if(IsInsideHex(position, colli.HexaRadius, centerOfHexagons.y[temp], centerOfHexagons.z[temp]))
						return temp;
				
  			temp = hex - colli.CubRepNumY;
  			
			// Test top hexagon
			if(IsInsideHex(position, colli.HexaRadius, centerOfHexagons.y[temp], centerOfHexagons.z[temp]))
				return temp;
			
  			temp = hex + colli.CubRepNumY - 1;
  			
  			// Test bottom hexagon
			if(IsInsideHex(position, colli.HexaRadius, centerOfHexagons.y[temp], centerOfHexagons.z[temp]))
				return temp;
  	}
  		
	return -1;
}

// Return the next edge distance inside a hole (for SPECT simulation)
__device__ double get_hexagon_boundary_by_raycasting(float3 p, float3 preel, float3 d, int size_x, 
													double radius, Colli colli, CoordHex2 centerOfHexagons) {
	
	float xmin, xmax, ymin, ymax, e1min, e1max, e2min, e2max;
	float txmin, txmax, tmin, tmax, tymin, tymax, tzmin, tzmax, te1min, te1max, te2min, te2max, buf;
	
	//printf("position %f %f %f \n", p.x, p.y, p.z);
	//printf("direction %f %f %f \n", d.x, d.y, d.z);
	
	xmin = -(size_x / 2.0);
	xmax = size_x / 2.0;
	
	ymin = e1min = e2min = -radius;
	ymax = e1max = e2max = radius;
	
	tmin = -INF;
    tmax = INF;
    
    int w;
	
	float3 di = inverse_vector(d);
	
	// on x
    if (fabs(d.x) < EPS) {
    	if (p.x < xmin || p.x > xmax) {return 0;}
    }
    else {
    	w = 0;
        tmin = txmin = (xmin - p.x) * di.x;
        tmax = txmax = (xmax - p.x) * di.x;
        //printf("on x: %f %f - %f %f - %f %f \n", xmin, xmax, p.x, di.x, tmin, tmax);
        if (tmin > tmax) {
            buf = tmin;
            tmin = tmax;
            tmax = buf;
        }
        if (tmin > tmax) {return 0.0;}
    }
    
    // on y
    if (fabs(d.y) < EPS) {
    	if (p.y < ymin || p.y > ymax) {return 0;}
    }
    else {
        tymin = (ymin - p.y) * di.y;
        tymax = (ymax - p.y) * di.y;
        //printf("on y: %f %f - %f %f - %f %f \n", ymin, ymax, p.y, di.y, tymin, tymax);
        if (tymin > tymax) {
            buf = tymin;
            tymin = tymax;
            tymax = buf;
        }
        if (tymin > tmin) {tmin = tymin;}
        if (tymax < tmax) {tmax = tymax; w = 1;}
        if (tmin > tmax) {return 0.0;}
    }
    
    // on e1  (changement de referentiel dans le plan yz, rotation de -60°) 
    
    //float p1y = ((p.y - cy) * cos( -M_PI / 3.0 )) + ((p.z - cz) * sin ( -M_PI / 3.0 ));
    //p1y += cy;
    
    float p1y = (p.y * cos( -M_PI / 3.0 )) + (p.z * sin ( -M_PI / 3.0 ));
    
    float d1y = d.y * cos( -M_PI / 3.0 ) + d.z * sin ( -M_PI / 3.0 );

	float di1y;
	
	if (fabs(d1y) < EPS) {
    	if (p1y < e1min || p1y > e1max) {return 0;}
    }
    else {
		di1y = 1.0f / d1y;
        te1min = (e1min - p1y) * di1y;
        te1max = (e1max - p1y) * di1y;
        //printf("on e1: %f %f - %f %f - %f %f \n", e1min, e1max, p1y, d1y, te1min, te1max);
        if (te1min > te1max) {
            buf = te1min;
            te1min = te1max;
            te1max = buf;
        }
        if (te1min > tmin) {tmin = te1min;}
        if (te1max < tmax) {tmax = te1max; w = 2;}
        if (tmin > tmax) {return 0.0;}
    }

	// on e2 (changement de referentiel dans le plan yz, rotation de +60°) 
	    
    //float p2y = ((p.y - cy) * cos( M_PI / 3.0 )) + ((p.z - cz) * sin ( M_PI / 3.0 ));
    //p2y += cy;
     
    float p2y = (p.y * cos( M_PI / 3.0 )) + (p.z * sin ( M_PI / 3.0 )); 
     
    float d2y = d.y * cos( M_PI / 3.0 ) + d.z * sin ( M_PI / 3.0 );

	float di2y;
	
	if (fabs(d2y) < EPS) {
    	if (p2y < e2min || p2y > e2max) {return 0;}
    }
    else {
		di2y = 1.0f / d2y;
        te2min = (e2min - p2y) * di2y;
        te2max = (e2max - p2y) * di2y;
        //printf("on e2: %f %f - %f %f - %f %f \n", e2min, e2max, p2y, d2y, te2min, te2max);
        if (te2min > te2max) {
            buf = te2min;
            te2min = te2max;
            te2max = buf;
        }
        if (te2min > tmin) {tmin = te2min;}
        if (te2max < tmax) {tmax = te2max; w = 3;}
        if (tmin > tmax) {return 0.0;}
    }

	//if(fabs(tmax)>EPS) 
	
	//printf("final %d //// %f %f / %f %f / %f %f / %f %f //// direction %f %f %f \n", 
		//			w, tmin, tmax, tymin, tymax, te1min, te1max, te2min, te2max, d.x, d.y, d.z);

	/*
	float3 pos_test;
	
	pos_test.x = preel.x + (d.x * (tmax + 1.0e-03f));
    pos_test.y = preel.y + (d.y * (tmax + 1.0e-03f));
    pos_test.z = preel.z + (d.z * (tmax + 1.0e-03f));
    		 
	int hex2 = GetHexIndex(pos_test, colli, centerOfHexagons);
    	
    if(hex2 >= 0) {
    		printf("HOLE dist %f hex %d - pos %f %f - pos_test %f %f %f - center %f %f - dir %f %f %f - diff %f - r %f \n", tmax, hex2, 
    				p.y, p.z, pos_test.x, pos_test.y, pos_test.z, centerOfHexagons.y[hex2], centerOfHexagons.z[hex2], d.x, d.y, d.z,
    				fabs(pos_test.y - centerOfHexagons.y[hex2]), colli.HexaRadius);
   	 } 
   	 */

    return tmax;
}

// Binary search
__device__ int binary_search(float *val, float key, int n) {
    int min=0, max=n, mid;
    while (min < max) {
        mid = (min + max) >> 1;
        if (key > val[mid]) {
            min = mid + 1;
        } else {
            max = mid;
        }
    }
    return min; 
}

void dosimetry_dump(Dosimetry dosemap) {
    // first write te header
    FILE *pfile = fopen("dosemap.mhd", "w");
    fprintf(pfile, "ObjectType = Image \n");
    fprintf(pfile, "NDims = 3 \n");
    fprintf(pfile, "BinaryData = True \n");
    fprintf(pfile, "BinaryDataOrderMDB = False \n");
    fprintf(pfile, "CompressedData = False \n");
    fprintf(pfile, "ElementSpacing = %f %f %f \n", dosemap.voxel_size.x, 
                                                   dosemap.voxel_size.y, 
                                                   dosemap.voxel_size.z);
    fprintf(pfile, "DimSize = %i %i %i \n", dosemap.size_in_vox.x, 
                                            dosemap.size_in_vox.y, 
                                            dosemap.size_in_vox.z);
    fprintf(pfile, "ElementType = MET_FLOAT \n");
    
    fprintf(pfile, "ElementDataFile = dosemap.raw\n");
    fclose(pfile);

    // then export data
    pfile = fopen("dosemap.raw", "wb");
    fwrite(dosemap.edep, dosemap.nb_voxel_volume, sizeof(float), pfile);
    fclose(pfile);

}



/***********************************************************
 * PRNG Brent xor256
 ***********************************************************/

// Brent PRNG integer version
__device__ unsigned long weyl;
__device__ unsigned long brent_int(unsigned int index, unsigned long *device_x_brent, unsigned long seed)

{
	
#define UINT64 (sizeof(unsigned long)>>3)
#define UINT32 (1 - UINT64) 
#define wlen (64*UINT64 +  32*UINT32)
#define r    (4*UINT64 + 8*UINT32)
#define s    (3*UINT64 +  3*UINT32)
#define a    (37*UINT64 +  18*UINT32)
#define b    (27*UINT64 +  13*UINT32)
#define c    (29*UINT64 +  14*UINT32)
#define d    (33*UINT64 +  15*UINT32)
#define ws   (27*UINT64 +  16*UINT32) 

	int z, z_w, z_i_brent;	
	if (r==4){
		z=6; z_w=4; z_i_brent=5;}
	else{
		z=10; z_w=8; z_i_brent=9;}
	
	unsigned long w = device_x_brent[z*index + z_w];
	unsigned long i_brent = device_x_brent[z*index + z_i_brent];
	unsigned long zero = 0;
	unsigned long t, v;
	int k;
	
	if (seed != zero) { // Initialisation necessary
		// weyl = odd approximation to 2**wlen*(3-sqrt(5))/2.
		if (UINT32) 
			weyl = 0x61c88647;
		else 
			weyl = ((((unsigned long)0x61c88646)<<16)<<16) + (unsigned long)0x80b583eb;
		
		v = (seed!=zero)? seed:~seed;  // v must be nonzero
		
		for (k = wlen; k > 0; k--) {   // Avoid correlations for close seeds
			v ^= v<<10; v ^= v>>15;    // Recurrence has period 2**wlen-1
			v ^= v<<4;  v ^= v>>13;    // for wlen = 32 or 64
		}
		for (w = v, k = 0; k < r; k++) { // Initialise circular array
			v ^= v<<10; v ^= v>>15; 
			v ^= v<<4;  v ^= v>>13;
			device_x_brent[k + z*index] = v + (w+=weyl);              
		}
		for (i_brent = r-1, k = 4*r; k > 0; k--) { // Discard first 4*r results
			t = device_x_brent[(i_brent = (i_brent+1)&(r-1)) + z*index];   t ^= t<<a;  t ^= t>>b;			
			v = device_x_brent[((i_brent+(r-s))&(r-1)) + z*index];	v ^= v<<c;  v ^= v>>d;       
			device_x_brent[i_brent + z*index] = t^v;  
		}
    }
    
	// Apart from initialisation (above), this is the generator
	t = device_x_brent[(i_brent = (i_brent+1)&(r-1)) + z*index]; // Assumes that r is a power of two
	v = device_x_brent[((i_brent+(r-s))&(r-1)) + z*index];       // Index is (i-s) mod r
	t ^= t<<a;  t ^= t>>b;                                       // (I + L^a)(I + R^b)
	v ^= v<<c;  v ^= v>>d;                                       // (I + L^c)(I + R^d)
	device_x_brent[i_brent + z*index] = (v ^= t); 				 // Update circular array                 
	w += weyl;                                                   // Update Weyl generator
	
	device_x_brent[z*index + z_w] = w;
	device_x_brent[z*index + z_i_brent] = i_brent;
	
	return (v + (w^(w>>ws)));  // Return combination
	
#undef UINT64
#undef UINT32
#undef wlen
#undef r
#undef s
#undef a
#undef b
#undef c
#undef d
#undef ws 
}	

// Brent PRNG real version
__device__ double Brent_real(int index, unsigned long *device_x_brent, unsigned long seed)

{
	
#define UINT64 (sizeof(unsigned long)>>3)
#define UINT32 (1 - UINT64) 
#define UREAL64 (sizeof(double)>>3)
#define UREAL32 (1 - UREAL64)
	
	// sr = number of bits discarded = 11 for double, 40 or 8 for float
	
#define sr (11*UREAL64 +(40*UINT64 + 8*UINT32)*UREAL32)
	
	// ss (used for scaling) is 53 or 21 for double, 24 for float
	
#define ss ((53*UINT64 + 21*UINT32)*UREAL64 + 24*UREAL32)
	
	// SCALE is 0.5**ss, SC32 is 0.5**32
	
#define SCALE ((double)1/(double)((unsigned long)1<<ss)) 
#define SC32  ((double)1/((double)65536*(double)65536)) 
	
	double res;
	
	res = (double)0; 
	while (res == (double)0)  // Loop until nonzero result.
    {   // Usually only one iteration.
		res = (double)(brent_int(index, device_x_brent, seed)>>sr);     // Discard sr random bits.
		seed = (unsigned long)0;                                        // Zero seed for next time.
		if (UINT32 && UREAL64)                                          // Need another call to xor4096i.
			res += SC32*(double)brent_int(index, device_x_brent, seed); // Add low-order 32 bits.
    }
	return (SCALE*res); // Return result in (0.0, 1.0).
	
#undef UINT64
#undef UINT32
#undef UREAL64
#undef UREAL32
#undef SCALE
#undef SC32
#undef sr
#undef ss
}

// Init Brent seed
__global__ void kernel_brent_init(StackParticle stackpart) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id < stackpart.size) {
		unsigned int seed = stackpart.seed[id];
		float dummy = brent_int(id, stackpart.table_x_brent, seed);
	}
}

/***********************************************************
 * Particles source
 ***********************************************************/

// Voxelized back2back source
__global__ void kernel_voxelized_source_b2b(StackParticle g1, StackParticle g2, Activities act,
                                            float E, int3 size_in_vox, float3 voxel_size) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (id >= g1.size) return;

    float jump = (float)(size_in_vox.x * size_in_vox.y);
    float ind, x, y, z;
    
    float rnd = Brent_real(id, g1.table_x_brent, 0);
    int pos = binary_search(act.act_cdf, rnd, act.nb_activities);
    
    // get the voxel position (x, y, z)
    ind = (float)act.act_index[pos];
    z = floor(ind / jump);
    ind -= (z * jump);
    y = floor(ind / (float)(size_in_vox.x));
    x = ind - y*size_in_vox.x;

    // random position inside the voxel
    x += Brent_real(id, g1.table_x_brent, 0);
    y += Brent_real(id, g1.table_x_brent, 0);
    z += Brent_real(id, g1.table_x_brent, 0);

    // must be in mm
    x *= voxel_size.x;
    y *= voxel_size.y;
    z *= voxel_size.z;

    // random orientation
    float phi   = Brent_real(id, g1.table_x_brent, 0);
    float theta = Brent_real(id, g1.table_x_brent, 0);
    phi   = gpu_twopi * phi;
    theta = acosf(1.0f - 2.0f*theta);
    
    // convert to cartesian
    float dx = __cosf(phi)*__sinf(theta);
    float dy = __sinf(phi)*__sinf(theta);
    float dz = __cosf(theta);

    // first gamma
    g1.dx[id] = dx;
    g1.dy[id] = dy;
    g1.dz[id] = dz;
    g1.E[id] = E;
    g1.px[id] = x;
    g1.py[id] = y;
    g1.pz[id] = z;
    g1.t[id] = 0.0f;
    g1.active[id] = 1;
    g1.endsimu[id] = 0;
    g1.type[id] = GAMMA;

    // second gamma
    g2.dx[id] = -dx;
    g2.dy[id] = -dy;
    g2.dz[id] = -dz;
    g2.E[id] = E;
    g2.px[id] = x;
    g2.py[id] = y;
    g2.pz[id] = z;
    g2.t[id] = 0.0f;
    g2.active[id] = 1;
    g2.endsimu[id] = 0;
    g2.type[id] = GAMMA;
}

/***********************************************************
 * Photons Physics Effects
 ***********************************************************/

//// Comptons Standard //////////////////////////////////////

// Compton Cross Section Per Atom (Standard - Klein-Nishina)
__device__ float Compton_CSPA_Standard(float E, unsigned short int Z) {
	float CrossSection = 0.0;
	if (Z<1 || E < 1e-4f) {return CrossSection;}

	float p1Z = Z*(2.7965e-23f + 1.9756e-27f*Z + -3.9178e-29f*Z*Z);
	float p2Z = Z*(-1.8300e-23f + -1.0205e-24f*Z + 6.8241e-27f*Z*Z);
	float p3Z = Z*(6.7527e-22f + -7.3913e-24f*Z + 6.0480e-27f*Z*Z);
	float p4Z = Z*(-1.9798e-21f + 2.7079e-24f*Z + 3.0274e-26f*Z*Z);
	float T0 = (Z < 1.5f)? 40.0e-3f : 15.0e-3f;
	float d1, d2, d3, d4, d5;

	d1 = __fdividef(fmaxf(E, T0), 0.510998910f); // X
	CrossSection = __fdividef(p1Z*__logf(1.0f+2.0f*d1), d1) + __fdividef(p2Z + p3Z*d1 + p4Z*d1*d1, 1.0f + 20.0f*d1 + 230.0f*d1*d1 + 440.0f*d1*d1*d1);

	if (E < T0) {
		d1 = __fdividef(T0+1.0e-3f, 0.510998910f); // X
		d2 = __fdividef(p1Z*__logf(1.0f+2.0f*d1), d1) + __fdividef(p2Z + p3Z*d1 + p4Z*d1*d1, 1.0f + 20.0f*d1 + 230.0f*d1*d1 + 440.0f*d1*d1*d1); // sigma
		d3 = __fdividef(-T0 * (d2 - CrossSection), CrossSection*1.0e-3f); // c1
		d4 = (Z > 1.5f)? 0.375f-0.0556f*__logf(Z) : 0.15f; // c2
		d5 = __logf(__fdividef(E, T0)); // y
		CrossSection *= __expf(-d5 * (d3 + d4*d5));
	}
	
    return CrossSection;
}

// Compute the total Compton cross section for a given material
__device__ float Compton_CS_Standard(Materials materials, unsigned int mat, float E) {
	float CS = 0.0f;
	int i;
	int index = materials.index[mat];
	// Model standard
	for (i = 0; i < materials.nb_elements[mat]; ++i) {
		CS += (materials.atom_num_dens[index+i] * Compton_CSPA_Standard(E, materials.mixture[index+i]));
	}
	return CS;
}

// Compton Scatter (Standard - Klein-Nishina) without secondary
__device__ float Compton_Effect_Standard_NoSec(StackParticle photons, 
                                               unsigned int id,
                                               int* count_d) {
	float gamE0 = photons.E[id];

	float E0 = __fdividef(gamE0, 0.510998910f);
    float3 gamDir0 = make_float3(photons.dx[id], photons.dy[id], photons.dz[id]);

    // sample the energy rate of the scattered gamma

	float epszero = __fdividef(1.0f, (1.0f + 2.0f * E0));
	float eps02 = epszero*epszero;
	float a1 = -__logf(epszero);
	float a2 = __fdividef(a1, (a1 + 0.5f*(1.0f-eps02)));

	float greject, onecost, eps, eps2, sint2, cosTheta, sinTheta, phi;
	do {
		if (a2 > Brent_real(id, photons.table_x_brent, 0)) {
			eps = __expf(-a1 * Brent_real(id, photons.table_x_brent, 0));
			eps2 = eps*eps;
		} else {
			eps2 = eps02 + (1.0f - eps02) * Brent_real(id, photons.table_x_brent, 0);
			eps = sqrt(eps2);
		}
		onecost = __fdividef(1.0f - eps, eps * E0);
		sint2 = onecost * (2.0f - onecost);
		greject = 1.0f - eps * __fdividef(sint2, 1.0f + eps2);
	} while (greject < Brent_real(id, photons.table_x_brent, 0));

    // scattered gamma angles

    if (sint2 < 0.0f) {sint2 = 0.0f;}
	cosTheta = 1.0f - onecost;
	sinTheta = sqrt(sint2);
	phi = Brent_real(id, photons.table_x_brent, 0) * gpu_twopi;

    // update the scattered gamma

    float3 gamDir1 = make_float3(sinTheta*__cosf(phi), sinTheta*__sinf(phi), cosTheta);
    gamDir1 = rotateUz(gamDir1, gamDir0);
    photons.dx[id] = gamDir1.x;
    photons.dy[id] = gamDir1.y;
    photons.dz[id] = gamDir1.z;
    float gamE1  = gamE0 * eps;
    if (gamE1 > 1.0e-06f) {photons.E[id] = gamE1;}
    else {
        photons.endsimu[id] = 1; // stop this particle
        photons.active[id]  = 0; // this particle is absorbed
        atomicAdd(count_d, 1);   // count simulated primaries
        return gamE1;            // Local energy deposit
    }

    return 0.0f;
}

// Compton Scatter (Standard - Klein-Nishina) with secondary (e-)
__device__ float Compton_Effect_Standard_WiSec(StackParticle photons, 
                                               StackParticle electrons,
                                               float cutE,
                                               unsigned int id,
                                               int* count_d) {
	float gamE0 = photons.E[id];
	float E0 = __fdividef(gamE0, 0.510998910f);
    float3 gamDir0 = make_float3(photons.dx[id], photons.dy[id], photons.dz[id]);

    // sample the energy rate pf the scattered gamma

	float epszero = __fdividef(1.0f, (1.0f + 2.0f * E0));
	float eps02 = epszero*epszero;
	float a1 = -__logf(epszero);
	float a2 = __fdividef(a1, (a1 + 0.5f*(1.0f-eps02)));

	float greject, onecost, eps, eps2, sint2, cosTheta, sinTheta, phi;
	do {
		if (a2 > Brent_real(id, photons.table_x_brent, 0)) {
			eps = __expf(-a1 * Brent_real(id, photons.table_x_brent, 0));
			eps2 = eps*eps;
		} else {
			eps2 = eps02 + (1.0f - eps02) * Brent_real(id, photons.table_x_brent, 0);
			eps = sqrt(eps2);
		}
		onecost = __fdividef(1.0f - eps, eps * E0);
		sint2 = onecost * (2.0f - onecost);
		greject = 1.0f - eps * __fdividef(sint2, 1.0f + eps2);
	} while (greject < Brent_real(id, photons.table_x_brent, 0));

    // scattered gamma angles

    if (sint2 < 0.0f) {sint2 = 0.0f;}
	cosTheta = 1.0f - onecost;
	sinTheta = sqrt(sint2);
	phi = Brent_real(id, photons.table_x_brent, 0) * gpu_twopi;

    // update the scattered gamma

    float3 gamDir1 = make_float3(sinTheta*__cosf(phi), sinTheta*__sinf(phi), cosTheta);
    gamDir1 = rotateUz(gamDir1, gamDir0);
    photons.dx[id] = gamDir1.x;
    photons.dy[id] = gamDir1.y;
    photons.dz[id] = gamDir1.z;
    float gamE1  = gamE0 * eps;
    if (gamE1 > 1.0e-06f) {photons.E[id] = gamE1;}
    else {
        //printf("Compton => X\n");
        photons.endsimu[id] = 1; // absorbed this particle
        photons.active[id]  = 0;
        atomicAdd(count_d, 1);   // count simulated primaries
        return gamE1;            // Local energy deposit
    }

    // kinematic of the scattered electron
   
    float eKinE = gamE0 - gamE1;
    //          DBL_MIN             cut production
    if (eKinE > 1.0e-38f && eKinE > cutE) {
        float3 eDir = sub_vector(scale_vector(gamDir0, gamE0), scale_vector(gamDir1, gamE1));
        eDir = unit_vector(eDir);
        electrons.dx[id] = eDir.x;
        electrons.dy[id] = eDir.y;
        electrons.dz[id] = eDir.z;
        electrons.E[id]  = eKinE;
        electrons.px[id] = photons.px[id];
        electrons.py[id] = photons.py[id];
        electrons.pz[id] = photons.pz[id];
        electrons.endsimu[id] = 0;
        // Now start to track this electron and freeze the photon tracking
        photons.active[id] = 0;
        electrons.active[id] = 1;
        //printf("Compton => e- cutE %e\n", cutE);
        return 0.0f;
    } 
        
    //printf("Compton => / cutE %e\n", cutE);

    return eKinE;
}

//// PhotoElectric Standard //////////////////////////////////////

// Compute Theta distribution of the emitted electron, with respect to the incident Gamma
// The Sauter-Gavrila distribution for the K-shell is used
__device__ float PhotoElec_ElecCosThetaDistribution(StackParticle part,
                                                    unsigned int id,
                                                    float kineEnergy) {
    float costeta = 1.0f;
	float gamma = kineEnergy * 1.9569513367f + 1.0f;  // 1/electron_mass_c2
    if (gamma > 5.0f) {return costeta;}
    float beta = __fdividef(sqrtf(gamma*gamma - 1.0f), gamma);
    float b    = 0.5f*gamma*(gamma - 1.0f)*(gamma - 2.0f);

    float rndm, term, greject, grejsup;
    if (gamma < 2.0f) {grejsup = gamma*gamma*(1.0f + b - beta*b);}
    else              {grejsup = gamma*gamma*(1.0f + b + beta*b);}

    do {
        rndm = 1.0f - 2.0f*Brent_real(id, part.table_x_brent, 0);
        costeta = __fdividef(rndm + beta, rndm*beta + 1.0f);
        term = 1.0f - beta*costeta;
        greject = __fdividef((1.0f - costeta*costeta)*(1.0f + b*term), term*term);
    } while(greject < Brent_real(id, part.table_x_brent, 0)*grejsup);

    return costeta;
}

// PhotoElectric Cross Section Per Atom (Standard)
__device__ float PhotoElec_CSPA_Standard(float E, unsigned short int Z) {
	 // from Sandia, the same for all Z
	float Emin = fmax(PhotoElec_std_IonizationPotentials[Z]*1e-6f, 0.01e-3f);
	if (E < Emin) {return 0.0f;}
	
	int start = PhotoElec_std_CumulIntervals[Z-1];
	int stop = start + PhotoElec_std_NbIntervals[Z];
	int pos=stop;
	while (E < PhotoElec_std_SandiaTable[pos][0]*1.0e-3f){--pos;}
	float AoverAvo = 0.0103642688246f * __fdividef((float)Z, PhotoElec_std_ZtoAratio[Z]);
	float rE = __fdividef(1.0f, E);
	float rE2 = rE*rE;

	return rE * PhotoElec_std_SandiaTable[pos][1] * AoverAvo * 0.160217648e-22f
		+ rE2 * PhotoElec_std_SandiaTable[pos][2] * AoverAvo * 0.160217648e-25f
		+ rE * rE2 * PhotoElec_std_SandiaTable[pos][3] * AoverAvo * 0.160217648e-28f
		+ rE2 * rE2 * PhotoElec_std_SandiaTable[pos][4] * AoverAvo * 0.160217648e-31f;
}

// Compute the total Compton cross section for a given material
__device__ float PhotoElec_CS_Standard(Materials materials, unsigned int mat, float E) {
	float CS = 0.0f;
	int i;
	int index = materials.index[mat];
	// Model standard
	for (i = 0; i < materials.nb_elements[mat]; ++i) {
		CS += (materials.atom_num_dens[index+i] * PhotoElec_CSPA_Standard(E, materials.mixture[index+i]));
	}
	return CS;
}

// PhotoElectric effect (Standard) without seconday
__device__ float PhotoElec_Effect_Standard_NoSec(StackParticle photons,
                                                 unsigned int id,
                                                 int* count_d) {
    // Absorbed the photon
    photons.endsimu[id] = 1; // stop the simulation
    photons.active[id] = 0;  // this particle is absorbed
    atomicAdd(count_d, 1);   // count simulated primaries

    return 0.0f;      
}

// PhotoElectric effect (Standard) with seconday (e-)
__device__ float PhotoElec_Effect_Standard_WiSec(StackParticle photons,
                                                 StackParticle electrons,
                                                 Materials mat,
                                                 float cutE,
                                                 unsigned int matindex,
                                                 unsigned int id,
                                                 int* count_d) {

    float energy = photons.E[id];
    float3 PhotonDirection = make_float3(photons.dx[id], photons.dy[id], photons.dz[id]);

    // Select randomly one element constituing the material
    unsigned int n = mat.nb_elements[matindex]-1;
    unsigned int index = mat.index[matindex];
    unsigned int Z = mat.mixture[index+n];
    unsigned int i = 0;
    if (n > 0) {
        float x = Brent_real(id, photons.table_x_brent, 0) * 
                  PhotoElec_CS_Standard(mat, matindex, energy);
        float xsec = 0.0f;
        for (i=0; i<n; ++i) {
            xsec += mat.atom_num_dens[index+i] * 
                    PhotoElec_CSPA_Standard(energy, mat.mixture[index+i]);
            if (x <= xsec) {
                Z = mat.mixture[index+i];
                break;
            }
        }

    }

    //// Photo electron

    // Select atomic shell
    unsigned short int nShells = atom_NumberOfShells[Z];
    index = atom_IndexOfShells[Z];
    float bindingEnergy = atom_BindingEnergies[index]*1.0e-06f; // in eV
    i=0; while (i < nShells && energy < bindingEnergy) {
        ++i;
        bindingEnergy = atom_BindingEnergies[index + i]*1.0e-06f; // in ev
    }
        
    // no shell available
    if (i == nShells) {return 0.0f;}
    float ElecKineEnergy = energy - bindingEnergy;

    float cosTeta = 0.0f;
    //                   1 eV                         cut production
    if (ElecKineEnergy > 1.0e-06f && ElecKineEnergy > cutE) {
        // direction of the photo electron
        cosTeta = PhotoElec_ElecCosThetaDistribution(photons, id, ElecKineEnergy);
        float sinTeta = sqrtf(1.0f - cosTeta*cosTeta);
        float Phi = gpu_twopi * Brent_real(id, photons.table_x_brent, 0);
        float3 ElecDirection = make_float3(sinTeta*cos(Phi), sinTeta*sin(Phi), cosTeta);
        ElecDirection = rotateUz(ElecDirection, PhotonDirection);
        // Create an electron
        electrons.dx[id] = ElecDirection.x;
        electrons.dy[id] = ElecDirection.y;
        electrons.dz[id] = ElecDirection.z;
        electrons.E[id]  = ElecKineEnergy;
        electrons.px[id] = photons.px[id];
        electrons.py[id] = photons.py[id];
        electrons.pz[id] = photons.pz[id];
        electrons.endsimu[id] = 0;
        // Start to track this electron 
        electrons.active[id] = 1;
        //printf("PE => e-\n");
        return bindingEnergy;
    }
    
    // Absorbed the photon
    photons.endsimu[id] = 1; // stop the simulation
    photons.active[id]  = 0;
    atomicAdd(count_d, 1);   // count simulated primaries
        
    // LocalEnergy Deposit
    return bindingEnergy+ElecKineEnergy;
}

/***********************************************************
 * Electrons Physics Effects
 ***********************************************************/

// eIonisation Cross Section Per Atom (Möller model)
__device__ float eIonisation_CSPA_Standard(float E, unsigned short int Z, 
                                           float cutE, float maxE) {
	float CS = 0.0f;
	float xmin = __fdividef(cutE, E);
    float tmax = fmin(maxE, 0.5f*E);
	float xmax = __fdividef(tmax, E);
	float gam = E * 1.9569513367f + 1.0f;  // 1/electron_mass_c2
	float igam2 = __fdividef(1.0f, gam*gam);
	float ibeta2 = __fdividef(1.0f, 1.0f - igam2);
	float g = (2.0f*gam - 1.0f)*igam2;
	
    if (cutE < tmax) {
        // Cross Section per e-
	    CS = ((xmax-xmin) * (1.0f-g + __fdividef(1.0, (xmin*xmax)) + __fdividef(1.0f, (1.0f-xmin)*(1.0f-xmax))) - g*__logf( __fdividef(xmax*(1.0 - xmin), xmin*(1.0 - xmax)))) * ibeta2;
    	CS *= (__fdividef(2.549549299e-23f, E)); // gpu_twopi_mc2_rcl2
    	CS *= (float)Z;
    }

    return CS;
}

// Compute the total eIonisation cross section for a given material
__device__ float eIonisation_CS_Standard(Materials materials, unsigned int mat, float E) {
	float CS = 0.0f;
	int i;
	int index = materials.index[mat];
    float cutE = materials.electron_cut_energy[mat];
    float maxE = materials.electron_max_energy[mat];
	// Model standard
	for (i = 0; i < materials.nb_elements[mat]; ++i) {
        CS += (materials.atom_num_dens[index+i] * 
                eIonisation_CSPA_Standard(E, materials.mixture[index+i], cutE, maxE));
	}

	return CS;
}

// Compute the dE/dx due to the ionization
__device__ float eIonisation_dedx_Standard(Materials materials, unsigned int mat, float E) {

    float meanExcitationEnergy = materials.electron_mean_excitation_energy[mat];
    float cutE = materials.electron_cut_energy[mat];

    float electronDensity = materials.nb_electrons_per_vol[mat];
    float Natm = materials.nb_atoms_per_vol[mat];
    float Zeff = __fdividef(electronDensity, Natm);
    float th = 0.25f*sqrtf(Zeff) * 0.001f; // keV
    unsigned short int flag_low_E = 0;
    float tkin = E;
    if (tkin < th) {tkin = th; flag_low_E = 1;};
    float tau = tkin * 1.9569513367f; // 1/electron_mass_c2
    float gam = tau + 1.0f;
    float gam2 = gam*gam;
    float beta2 = 1.0f - __fdividef(1.0f, gam2);
    float eexc2 = meanExcitationEnergy * 1.9569513367f; // 1/electron_mass_c2
    eexc2 = eexc2 * eexc2;

    float d = (cutE < tkin*0.5f)? cutE : tkin*0.5f;
    d = d * 1.9569513367f; // 1/electron_mass_c2

    float dedx = __logf(2.0f * __fdividef(tau+2.0f, eexc2)) - 1.0f - beta2 + __logf((tau-d)*d) + __fdividef(tau, tau-d) + __fdividef(0.5f*d*d + (2.0f*tau + 1.0f) * __logf(1.0f - __fdividef(d, tau)), gam2);
    
    // Density correction
    float twoln10 = 2.0f*__logf(10.0f);
    float x = __fdividef(__logf(beta2*gam2), twoln10);
    float y = 0.0f;
    if (x < materials.fX0[mat]) {
        if (materials.fD0[mat] > 0.0f) {
            y = materials.fD0[mat]*__powf(10.0f, 2.0f*(x-materials.fX0[mat]));
        }
    } else if (x >= materials.fX1[mat]) {
        y = twoln10*x - materials.fC[mat];
    } else {
        y = twoln10*x - materials.fC[mat] + materials.fA[mat]
            * __powf(materials.fX1[mat]-x, materials.fM[mat]);
    }
    dedx -= y;
   
    // Total ionization loss
    //                 gpu_twopi_mc2_rcl2
    dedx *= __fdividef(2.549549299e-23f*electronDensity, beta2);
    if (dedx < 0.0f) {dedx = 0.0f;};
    
    // Low energy extrapolation
    if (flag_low_E) {
        //       200 eV
        if (E >= 200.0e-06f) {dedx *= sqrtf( __fdividef(tkin, E));}
        else {dedx *= __fdividef(sqrtf(tkin*E), 200.0e-06f);} // 200 eV       
    }    

    return dedx;
}

// Compute the scattering due to the ionization
__device__ float eIonisation_Effect_Standard_NoSec(StackParticle electrons,
                                                   StackParticle photons, 
                                                   float tmin, float maxE, // tmin=cutE
                                                   unsigned int id, int *count_d) {
    float E = electrons.E[id];
    float tmax = E * 0.5f;
    if (maxE < tmax) {tmax = maxE;};
    if (tmin >= tmax) { // tmin is the same that cutE
        // stop the simulation for this one
        electrons.endsimu[id] = 1;
        // Unfreeze the photon tracking
        electrons.active[id] = 0;
        photons.active[id] = 1;
        atomicAdd(count_d, 1);   // count simulated secondaries
        return E;
    }

    float energy = E + 0.510998910f; // electron_mass_c2
    float totalMomentum = sqrtf(E * (energy + 0.510998910f));
    float xmin = __fdividef(tmin, E);
    float xmax = __fdividef(tmax, E);
    float gam = energy * 1.9569513367f; // 1/electron_mass_c2
    float gamma2 = gam*gam;
    float beta2 = 1.0f - __fdividef(1.0f, gamma2);

    // GetMomentumDirection
    float3 direction = make_float3(electrons.dx[id], electrons.dy[id], electrons.dz[id]);

    // Moller (e-e-) scattering
    float g = __fdividef(2.0f*gam - 1.0f, gamma2);
    float y = 1.0f - xmax;
    float grej = 1.0f - g*xmax + xmax*xmax*(1.0f - g + __fdividef(1.0f - g*y, y*y));
    float x, z, q;
    do {
        q = Brent_real(id, electrons.table_x_brent, 0);
        x = __fdividef(xmin*xmax, xmin*(1.0f - q) + xmax*q);
        y = 1.0f - x;
        z = 1.0f - g*x + x*x*(1.0f - g + __fdividef(1.0f - g*y, y*y));
    } while(grej * Brent_real(id, electrons.table_x_brent, 0) > z);
    
    float deltaKinEnergy = x * E;
    float deltaMomentum = sqrtf(deltaKinEnergy * (deltaKinEnergy + 2.0f*0.510998910f)); // electron_mass_c2
    float cost = deltaKinEnergy * __fdividef(energy + 0.510998910f, deltaMomentum * totalMomentum);
    float sint = 1.0f - cost*cost;
    if (sint > 0.0f) {sint = sqrtf(sint);};
    float phi = gpu_twopi * Brent_real(id, electrons.table_x_brent, 0);
    float3 deltaDirection = make_float3(sint*__cosf(phi), sint*__sinf(phi), cost);
    deltaDirection = rotateUz(deltaDirection, direction);
    electrons.E[id] = E - deltaKinEnergy;
    float3 dir = sub_vector(scale_vector(direction, totalMomentum), 
                            scale_vector(deltaDirection, deltaMomentum));
    dir = unit_vector(dir);
    electrons.dx[id] = dir.x;
    electrons.dy[id] = dir.y;
    electrons.dz[id] = dir.z;

    return deltaKinEnergy;
}

// Multiple Scattering 
__device__ float MSC_CSPA(float E, unsigned short int Z) {

    float Z23 = __expf( 0.666666666666f*__logf((float)Z) ); 

    float eTotalEnergy = E + 0.51099891f;
    float beta2 = E * __fdividef(eTotalEnergy+0.51099891f, eTotalEnergy*eTotalEnergy);
    double bg2  = E * __fdividef(eTotalEnergy+0.51099891f, 0.26111988f); // e_mass_c2*e_mass_c2

    float eps = 37557.7634f * __fdividef(bg2, Z23); // epsfactor
    float epsmin = 1.0e-04f;
    float epsmax = 1.0e+10f;
    float sigma;
    if     (eps<epsmin)  sigma = 2.0f*eps*eps;
    else if(eps<epsmax)  sigma = __logf(1.0f+2.0f*eps) - 2.0f*__fdividef(eps, (1.0f+2.0f*eps));
    else                 sigma = __logf(2.0f*eps) - 1.0f+__fdividef(1.0f, eps);
    sigma *= __fdividef(Z*Z, (beta2*bg2));

    // get bin number in Z
    int iZ = 14;
    while ((iZ >= 0) && (Zdat[iZ] >= Z)) iZ -= 1;
    if (iZ == 14)                        iZ  = 13;
    if (iZ == -1)                        iZ  = 0 ;

    float Z1 = Zdat[iZ];
    float Z2 = Zdat[iZ+1];
    float ratZ = __fdividef((Z-Z1)*(Z+Z1), (Z2-Z1)*(Z2+Z1));

    float c1, c2;
    if(E <= 10.0f) { // Tlim = 10 MeV
        // get bin number in T (beta2)
        int iT = 21;
        while ((iT >= 0) && (Tdat[iT] >= E)) iT -= 1;
        if (iT == 21)                        iT  = 20;
        if (iT == -1)                        iT  = 0 ;

        //  calculate betasquare values
        float T  = Tdat[iT];   
        float EE = T + 0.51099891f;
        float b2small = T * __fdividef(EE + 0.51099891f, EE*EE);

        T = Tdat[iT+1]; 
        EE = T + 0.51099891f;
        float b2big = T * __fdividef(EE + 0.51099891f, EE*EE);
        float ratb2 = __fdividef(beta2-b2small, b2big-b2small);

        c1 = celectron[iZ][iT];
        c2 = celectron[iZ+1][iT];
        float cc1 = c1 + ratZ*(c2-c1);
        
        c1 = celectron[iZ][iT+1];
        c2 = celectron[iZ+1][iT+1];
        float cc2 = c1 + ratZ*(c2-c1);

        sigma *= __fdividef(4.98934390e-23f, cc1 + ratb2*(cc2-cc1)); // sigmafactor

    } else {
        //   bg2lim                                                       beta2lim
        c1 = 422.104880f*sig0[iZ]   * __fdividef(1.0f+hecorr[iZ]  *(beta2-0.997636519f), bg2);
        c2 = 422.104880f*sig0[iZ+1] * __fdividef(1.0f+hecorr[iZ+1]*(beta2-0.997636519f), bg2);

        if ((Z >= Z1) && (Z <= Z2)) {
            sigma = c1 + ratZ*(c2-c1);

        } else if(Z < Z1) {
            sigma = Z*Z*__fdividef(c1, (Z1*Z1));

        } else if(Z > Z2) {
            sigma = Z*Z*__fdividef(c2, (Z2*Z2));
        }
    }

    return sigma;
}

// Compute the total MSC cross section for a given material
__device__ float MSC_CS(Materials materials, unsigned int mat, float E) {
	float CS = 0.0f;
	int i;
	int index = materials.index[mat];
	for (i = 0; i < materials.nb_elements[mat]; ++i) {
        CS += (materials.atom_num_dens[index+i] * 
                MSC_CSPA(E, materials.mixture[index+i]));
	}
    return CS;
}

// Multiple Scattering effect
__device__ float MSC_Effect(StackParticle electrons, Materials materials, float trueStepLength, 
                            unsigned int mat, unsigned int id) {

    // double betacp = sqrt(currentKinEnergy*(currentKinEnergy+2.*mass)*KineticEnergy*(KineticEnergy+2.*mass)/((currentKinEnergy+mass)*(KineticEnergy+mass)));
   
    //E = 1.0f;

    float E = electrons.E[id];

    // !!!! Approx Seb :   currentKinEnergy = KineticEnergy
    float betacp = E * __fdividef(E+1.02199782f, E+0.51099891f);
    float y = __fdividef(trueStepLength, materials.rad_length[mat]);
    float theta = 13.6f * __fdividef(__powf(y, 0.5f), betacp);
    y = __logf(y);

    // correction in theta formula
    float Zeff = __fdividef(materials.nb_electrons_per_vol[mat], 
                            materials.nb_atoms_per_vol[mat]);
    float lnZ = __logf(Zeff);
    float coeffth1 = (1.0f - __fdividef(8.7780e-2f, Zeff)) * (0.87f + 0.03f*lnZ);
    float coeffth2 = (4.0780e-2f + 1.7315e-4f*Zeff) * (0.87f + 0.03f*lnZ);
    float corr = coeffth1 + coeffth2 * y;
    theta *= corr ;

    float phi = gpu_twopi * Brent_real(id, electrons.table_x_brent, 0);
    
    float3 direction = make_float3(electrons.dx[id], electrons.dy[id], electrons.dz[id]);
    float3 deltaDirection = make_float3(__cosf(phi)*__sinf(theta),
                                        __sinf(phi)*__sinf(theta),
                                        __cosf(theta));
    direction = rotateUz(deltaDirection, direction);
    electrons.dx[id] = direction.x;
    electrons.dy[id] = direction.y;
    electrons.dz[id] = direction.z;

    return 0.0f;
}

/***********************************************************
 * Navigator
 ***********************************************************/

// Regular Navigator with voxelized phantom for photons without secondary
__global__ void kernel_NavRegularPhan_Photon_NoSec(StackParticle photons,
                                                   Volume phantom,
                                                   Materials materials,
                                                   int* count_d) {
    unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (id >= photons.size) return;
    if (photons.endsimu[id]) return;
    if (!photons.active[id]) return;

    //// Init ///////////////////////////////////////////////////////////////////

    // Read position
    float3 position; // mm
    position.x = photons.px[id];
    position.y = photons.py[id];
    position.z = photons.pz[id];
    
    // Defined index phantom
    int4 index_phantom;
    float3 ivoxsize = inverse_vector(phantom.voxel_size);
    index_phantom.x = int(position.x * ivoxsize.x);
    index_phantom.y = int(position.y * ivoxsize.y);
    index_phantom.z = int(position.z * ivoxsize.z);
    index_phantom.w = index_phantom.z*phantom.nb_voxel_slice
                     + index_phantom.y*phantom.size_in_vox.x
                     + index_phantom.x; // linear index

    // Read direction
    float3 direction;
    direction.x = photons.dx[id];
    direction.y = photons.dy[id];
    direction.z = photons.dz[id];

    // Get energy
    float energy = photons.E[id];

    // Get material
    unsigned short int mat = phantom.data[index_phantom.w];

    //// Find next discrete interaction ///////////////////////////////////////

    // Find next discrete interaction, total_dedx and next discrete intraction distance
    float next_interaction_distance =  FLT_MAX;
    unsigned char next_discrete_process = 0; 
    float interaction_distance;
    float cross_section;

    // Photoelectric
    cross_section = PhotoElec_CS_Standard(materials, mat, energy);
    interaction_distance = __fdividef(-__logf(Brent_real(id, photons.table_x_brent, 0)),
                                     cross_section);
    if (interaction_distance < next_interaction_distance) {
        next_interaction_distance = interaction_distance;
        next_discrete_process = PHOTON_PHOTOELECTRIC;
    }

    // Compton
    cross_section = Compton_CS_Standard(materials, mat, energy);
    interaction_distance = __fdividef(-__logf(Brent_real(id, photons.table_x_brent, 0)),
                                     cross_section);
    
    if (interaction_distance < next_interaction_distance) {
        next_interaction_distance = interaction_distance;
        next_discrete_process = PHOTON_COMPTON;
    }

    // Distance to the next voxel boundary (raycasting)
    float interaction_distance2 = get_boundary_voxel_by_raycasting(index_phantom, position, 
                                                            direction, phantom.voxel_size);
    if (interaction_distance2 < next_interaction_distance) {
        // overshoot the distance of 1 um to be inside the next voxel
        next_interaction_distance = interaction_distance2 + 1.0e-03f;
        next_discrete_process = PHOTON_BOUNDARY_VOXEL;
    }

    //// Move particle //////////////////////////////////////////////////////

    position.x += direction.x * next_interaction_distance;
    position.y += direction.y * next_interaction_distance;
    position.z += direction.z * next_interaction_distance;
	photons.t[id] += (3.33564095198e-03f * next_interaction_distance);
    photons.px[id] = position.x;
    photons.py[id] = position.y;
    photons.pz[id] = position.z;

    // Stop simulation if out of phantom or no more energy
    if (   position.x <= 0 || position.x >= phantom.size_in_mm.x
        || position.y <= 0 || position.y >= phantom.size_in_mm.y 
        || position.z <= 0 || position.z >= phantom.size_in_mm.z ) {
        photons.endsimu[id] = 1;                     // stop the simulation
        atomicAdd(count_d, 1);                       // count simulated primaries
        return;
    }

    //// Resolve discrete processe //////////////////////////////////////////

    // Resolve discrete processes
    if (next_discrete_process == PHOTON_PHOTOELECTRIC) {
        float discrete_loss = PhotoElec_Effect_Standard_NoSec(photons, id, count_d);
    }

    if (next_discrete_process == PHOTON_COMPTON) {
        float discrete_loss = Compton_Effect_Standard_NoSec(photons, id, count_d);
    }
}

// Navigator with hexagonal hole collimator for photons without secondary 
__global__ void kernel_NavHexaColli_Photon_NoSec(StackParticle photons, Colli colli, 
												CoordHex2 centerOfHexagons, Materials materials,
												int* count_d) {
												
    unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (id >= photons.size) return;
    if (photons.endsimu[id]) return;
	if (!photons.active[id]) return;
    

    //// Init ///////////////////////////////////////////////////////////////////

    // Read position
    float3 position; // mm
    position.x = photons.px[id];
    position.y = photons.py[id];
    position.z = photons.pz[id];
    
    //printf("position %f %f %f \n", position.x, position.y, position.z);
    
    
	
    // Read direction
    float3 direction;
    direction.x = photons.dx[id];
    direction.y = photons.dy[id];
    direction.z = photons.dz[id];
    
    // Get energy
    float energy = photons.E[id];
    
    //// Find next discrete interaction ///////////////////////////////////////
    
    // Find next discrete interaction, total_dedx and next discrete intraction distance
    float next_interaction_distance =  FLT_MAX;
    unsigned char next_discrete_process = 0; 
    float interaction_distance;
   	double interaction_distance2 = 0.0;
   	double interaction_distance3 = FLT_MAX;
    float cross_section;
      
    unsigned short int mat; 
    
    double half_colli_size_x = colli.size_x / 2.0;
  	double half_colli_size_y = colli.size_y / 2.0;
  	double half_colli_size_z = colli.size_z / 2.0;
    
    float3 pos_test, inv_dir, temp;
    double dist;
    int hex;
    
	// If photon is outside an hexagonal hole
	if(GetHexIndex(position, colli, centerOfHexagons)<0)
    {
    	mat = 0; //lead
    
    	// Photoelectric
    	cross_section = PhotoElec_CS_Standard(materials, mat, energy); 
    	interaction_distance = __fdividef(-__logf(Brent_real(id, photons.table_x_brent, 0)),
                                     cross_section);
    	if (interaction_distance < next_interaction_distance) {
        	next_interaction_distance = interaction_distance;
       		next_discrete_process = PHOTON_PHOTOELECTRIC;
    	}
    	
    	//printf("PE %f %f\n", interaction_distance, cross_section);

    	// Compton
    	cross_section = Compton_CS_Standard(materials, mat, energy);
    	interaction_distance = __fdividef(-__logf(Brent_real(id, photons.table_x_brent, 0)),
        	                             cross_section);
    
    	if (interaction_distance < next_interaction_distance) {
        	next_interaction_distance = interaction_distance;
        	next_discrete_process = PHOTON_COMPTON;
    	}
    	
    	//printf("C %f %f\n", interaction_distance, cross_section);
    	
    	//  Hexagon edge
    	
    	pos_test = position;
    		
    	// Search for the next position inside an hexagon
    	while(GetHexIndex(pos_test, colli, centerOfHexagons)<0
    		&& fabs(pos_test.x) < half_colli_size_x
        	&& fabs(pos_test.y) < half_colli_size_y
        	&& fabs(pos_test.z) < half_colli_size_z) {
        		
        	pos_test.x += direction.x * 0.1;
    		pos_test.y += direction.y * 0.1;
    		pos_test.z += direction.z * 0.1;	
        }
    		
    	hex = GetHexIndex(pos_test, colli, centerOfHexagons);	
    		
    	if (hex>=0) {
        		
			temp.x = pos_test.x;
			temp.y = pos_test.y - centerOfHexagons.y[hex];
			temp.z = pos_test.z - centerOfHexagons.z[hex];
        	
        	// Inverse the direction to find the septa position entrance
        	inv_dir.x = -direction.x;
        	inv_dir.y = -direction.y;
        	inv_dir.z = -direction.z;
        	
        	interaction_distance2 = get_hexagon_boundary_by_raycasting(temp, position, inv_dir, colli.size_x, 
																	colli.HexaRadius, colli, centerOfHexagons);
																
			// compute the distance from the initial position to deduce the next hole distance													
			dist = sqrt((pos_test.x - position.x)*(pos_test.x - position.x) 
    					+ (pos_test.y - position.y)*(pos_test.y - position.y) 
    					+ (pos_test.z - position.z)*(pos_test.z - position.z));
    	
    		interaction_distance3 = dist - interaction_distance2 + 1.0e-03f;
    	}
        
        if (interaction_distance3 < next_interaction_distance) {
        	// overshoot the distance of 1 um to be inside the next septa
        	next_interaction_distance = interaction_distance2 + 1.0e-03f;
        	next_discrete_process = PHOTON_BOUNDARY_HOLE;	
        	//printf("boundary septa %f \n", interaction_distance3);
    	}		
    	
    }
    else   // photon is inside an hexagonal hole 
    {
		mat = 1; //air
		
		// Photoelectric
    	cross_section = PhotoElec_CS_Standard(materials, mat, energy); 
    	interaction_distance = __fdividef(-__logf(Brent_real(id, photons.table_x_brent, 0)),
                                     cross_section);
    	if (interaction_distance < next_interaction_distance) {
        	next_interaction_distance = interaction_distance;
       		next_discrete_process = PHOTON_PHOTOELECTRIC;
    	}
    	
    	//printf("PE %f %f\n", interaction_distance, cross_section);

    	// Compton
    	cross_section = Compton_CS_Standard(materials, mat, energy);
    	interaction_distance = __fdividef(-__logf(Brent_real(id, photons.table_x_brent, 0)),
        	                             cross_section);
    
    	if (interaction_distance < next_interaction_distance) {
        	next_interaction_distance = interaction_distance;
        	next_discrete_process = PHOTON_COMPTON;
    	}
    	
    	//printf("C %f %f\n", interaction_distance, cross_section);
		
		//  Hexagon edge
		
		// Mettre dans le referentiel de l'hexagone
		float3 temp;
		temp.x = position.x;
		temp.y = position.y - centerOfHexagons.y[hex];
		temp.z = position.z - centerOfHexagons.z[hex];
		
		interaction_distance2 = get_hexagon_boundary_by_raycasting(temp, position, direction, colli.size_x, 
																	colli.HexaRadius, colli, centerOfHexagons);														
																				
    	if (interaction_distance2 < next_interaction_distance) {
        	// overshoot the distance of 1 um to be inside the next septa
        	next_interaction_distance = interaction_distance2 + 1.0e-03f;
        	next_discrete_process = PHOTON_BOUNDARY_HOLE;	
    	}					
    }
    							
    							    
    //// Move particle //////////////////////////////////////////////////////

    position.x += direction.x * next_interaction_distance;
    position.y += direction.y * next_interaction_distance;
    position.z += direction.z * next_interaction_distance;
	photons.t[id] += (3.33564095198e-03f * next_interaction_distance);
	
	//printf("position init %f %f %f //// next %f %f %f \n", photons.px[id], photons.py[id], 
		//		photons.pz[id], position.x, position.y, position.z);
	
    photons.px[id] = position.x;
    photons.py[id] = position.y;
    photons.pz[id] = position.z;
    
	// Stop simulation if out of collimator
		
	if ( fabs(position.x) > half_colli_size_x
        || fabs(position.y) > half_colli_size_y
        || fabs(position.z) > half_colli_size_z ) {
        photons.endsimu[id] = 1;                     // stop the simulation
        atomicAdd(count_d, 1);                       // count simulated primaries
        //printf("effect %d end position: %f %f %f \n", next_discrete_process, position.x, position.y, position.z);
        return;
    }
   
    
    //// Resolve discrete processe //////////////////////////////////////////
    
    // Resolve discrete processes

   if (next_discrete_process == PHOTON_PHOTOELECTRIC) {
    	//printf("photoElec \n");
        float discrete_loss = PhotoElec_Effect_Standard_NoSec(photons, id, count_d);
    }

    if (next_discrete_process == PHOTON_COMPTON) {
    	//printf("Scatter \n");
        float discrete_loss = Compton_Effect_Standard_NoSec(photons, id, count_d);
    }
	
}


// Regular Navigator with voxelized phantom for photons with secondary
__global__ void kernel_NavRegularPhan_Photon_WiSec(StackParticle photons,
                                                   StackParticle electrons,
                                                   Volume phantom,
                                                   Materials materials,
                                                   Dosimetry dosemap,
                                                   int* count_d, float step_limiter) {
    unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    
    if (id >= photons.size) return;
    //printf("ID %i Nav gamma endsimu %i active %i\n", 
    //          id, photons.endsimu[id], photons.active[id]);
    if (photons.endsimu[id]) return;
    if (!photons.active[id]) return;

    //// Init ///////////////////////////////////////////////////////////////////

    // Read position
    float3 position; // mm
    position.x = photons.px[id];
    position.y = photons.py[id];
    position.z = photons.pz[id];

    // Defined index phantom
    int4 index_phantom;
    float3 ivoxsize = inverse_vector(phantom.voxel_size);
    index_phantom.x = int(position.x * ivoxsize.x);
    index_phantom.y = int(position.y * ivoxsize.y);
    index_phantom.z = int(position.z * ivoxsize.z);
    index_phantom.w = index_phantom.z*phantom.nb_voxel_slice
                       + index_phantom.y*phantom.size_in_vox.x
                       + index_phantom.x; // linear index
    
    // Read direction
    float3 direction;
    direction.x = photons.dx[id];
    direction.y = photons.dy[id];
    direction.z = photons.dz[id];

    // Get energy
    float energy = photons.E[id];

    // Get material
    unsigned short int mat = phantom.data[index_phantom.w];

    /// Debug ///
    //printf("gamma %i E %e pos %.2f %.2f %.2f mat %i\n", id, energy, position.x, position.y, position.z, mat);

    //// Find next discrete interaction ///////////////////////////////////////

    // Find next discrete interaction, total_dedx and next discrete intraction distance
    float next_interaction_distance =  FLT_MAX;
    unsigned char next_discrete_process = 0; 
    float interaction_distance;
    float cross_section;

    // Photoelectric
    cross_section = PhotoElec_CS_Standard(materials, mat, energy);
    interaction_distance = __fdividef(-__logf(Brent_real(id, photons.table_x_brent, 0)),
                                     cross_section);
    //printf("PE CS %e PIL %e\n", cross_section, interaction_distance);
    if (interaction_distance < next_interaction_distance) {
        next_interaction_distance = interaction_distance;
        next_discrete_process = PHOTON_PHOTOELECTRIC;
    }

    // Compton
    cross_section = Compton_CS_Standard(materials, mat, energy);
    interaction_distance = __fdividef(-__logf(Brent_real(id, photons.table_x_brent, 0)),
                                     cross_section);
    //printf("Cpt CS %e PIL %e\n", cross_section, interaction_distance);
    if (interaction_distance < next_interaction_distance) {
        next_interaction_distance = interaction_distance;
        next_discrete_process = PHOTON_COMPTON;
    }

    // Distance to the next voxel boundary (raycasting)
    interaction_distance = get_boundary_voxel_by_raycasting(index_phantom, position, 
                                                            direction, phantom.voxel_size);
    //printf("Boundary PIL %e\n", interaction_distance);
    if (interaction_distance < next_interaction_distance) {
        // overshoot the distance of 1 um to be inside the next voxel
        next_interaction_distance = interaction_distance+1.0e-03f;
        next_discrete_process = PHOTON_BOUNDARY_VOXEL;
    }
    
    // step limiter
    if (step_limiter < next_interaction_distance) {
        next_interaction_distance = step_limiter;
        next_discrete_process = PHOTON_STEP_LIMITER;
    }
    
    //// Move particle //////////////////////////////////////////////////////

    position.x += direction.x * next_interaction_distance;
    position.y += direction.y * next_interaction_distance;
    position.z += direction.z * next_interaction_distance;
	photons.t[id] += (3.33564095198e-03f * next_interaction_distance);
    photons.px[id] = position.x;
    photons.py[id] = position.y;
    photons.pz[id] = position.z;

    // Stop simulation if out of phantom
    if (   position.x <= 0 || position.x >= phantom.size_in_mm.x
        || position.y <= 0 || position.y >= phantom.size_in_mm.y 
        || position.z <= 0 || position.z >= phantom.size_in_mm.z ) {
        photons.endsimu[id] = 1;                     // stop the simulation
        atomicAdd(count_d, 1);                       // count simulated primaries
        return;
    }

    //// Resolve discrete processe //////////////////////////////////////////

    float discrete_loss = 0.0f;
    if (next_discrete_process == PHOTON_BOUNDARY_VOXEL ||
        next_discrete_process == PHOTON_STEP_LIMITER) { 
        //printf("boundary || step limiter\n");
        return;
    }
    
    if (next_discrete_process == PHOTON_PHOTOELECTRIC) {
        //printf("PE\n");
        discrete_loss = PhotoElec_Effect_Standard_WiSec(photons, electrons, materials,
                                                        materials.electron_cut_energy[mat], 
                                                        mat,  id, count_d);
    }
    
    if (next_discrete_process == PHOTON_COMPTON) {
        //printf("Compton\n");
        discrete_loss = Compton_Effect_Standard_WiSec(photons, electrons,
                                                      materials.electron_cut_energy[mat],
                                                      id, count_d);
        //printf("energy deposit %e\n", discrete_loss);
    }
  
    // Dosemap scoring
    ivoxsize = inverse_vector(dosemap.voxel_size);
    index_phantom.x = int(position.x * ivoxsize.x);
    index_phantom.y = int(position.y * ivoxsize.y);
    index_phantom.z = int(position.z * ivoxsize.z);
    index_phantom.w = index_phantom.z*dosemap.nb_voxel_slice
                       + index_phantom.y*dosemap.size_in_vox.x
                       + index_phantom.x; // linear index
    //printf("index dosemap %i\n", index_phantom.w);
    atomicAdd(&dosemap.edep[index_phantom.w], discrete_loss);
}

// Regular Navigator with voxelized phantom for electrons bind with a photon
__global__ void kernel_NavRegularPhan_Electron_BdPhoton(StackParticle electrons,
                                                        StackParticle photons,
                                                        Volume phantom,
                                                        Materials materials,
                                                        Dosimetry dosemap,
                                                        int* count_d, float step_limiter) {
    unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (id >= electrons.size)  return;
    //printf("\nNav e- endsimu %i active %i\n", electrons.endsimu[id], electrons.active[id]);
    if (electrons.endsimu[id]) return;
    if (!electrons.active[id]) return;

    //// Init ///////////////////////////////////////////////////////////////////

    // Read position
    float3 position; // mm
    position.x = electrons.px[id];
    position.y = electrons.py[id];
    position.z = electrons.pz[id];

    // Defined index phantom
    int4 index_phantom;
    float3 ivoxsize = inverse_vector(phantom.voxel_size);
    index_phantom.x = int(position.x * ivoxsize.x);
    index_phantom.y = int(position.y * ivoxsize.y);
    index_phantom.z = int(position.z * ivoxsize.z);
    index_phantom.w = index_phantom.z*phantom.nb_voxel_slice
                     + index_phantom.y*phantom.size_in_vox.x
                     + index_phantom.x; // linear index
    
    // Read direction
    float3 direction;
    direction.x = electrons.dx[id];
    direction.y = electrons.dy[id];
    direction.z = electrons.dz[id];

    // Get energy
    float energy = electrons.E[id];

    // Get material
    unsigned short int mat = phantom.data[index_phantom.w];
    
    /// Debug ///
    //printf("e- %i E %e pos %.2f %.2f %.2f\n", id, energy, position.x, position.y, position.z);

    //// Find next discrete interaction ///////////////////////////////////////

    // Find next discrete interaction, total_dedx and next discrete intraction distance
    float next_interaction_distance =  FLT_MAX;
    float total_dedx = 0.0f;
    unsigned char next_discrete_process = 0; 
    float interaction_distance;
    float cross_section;
    float probe = 0.0f; // DEBUG

    // eIonisation
    cross_section = eIonisation_CS_Standard(materials, mat, energy);
    interaction_distance = __fdividef(-__logf(Brent_real(id, electrons.table_x_brent, 0)),
                                      cross_section);
    total_dedx += eIonisation_dedx_Standard(materials, mat, energy);
    if (interaction_distance < next_interaction_distance) {
        next_interaction_distance = interaction_distance;
        next_discrete_process = ELECTRON_EIONISATION;
    }
    
    // Multiple Scattering
    cross_section = MSC_CS(materials, mat, energy);
    interaction_distance = __fdividef(-__logf(Brent_real(id, electrons.table_x_brent, 0)),
                                      cross_section);
    // dedx = 0.0
    if (interaction_distance < next_interaction_distance) {
        next_interaction_distance = interaction_distance;
        next_discrete_process = ELECTRON_MSC;
    }

    // Distance to the next voxel boundary (raycasting)
    interaction_distance = get_boundary_voxel_by_raycasting(index_phantom, position, 
                                                            direction, phantom.voxel_size);
    //printf("Boundary PIL %e\n", interaction_distance);
    if (interaction_distance < next_interaction_distance) {
        // overshoot the distance of 1 um to be inside the next voxel
        next_interaction_distance = interaction_distance+1.0e-03f;
        next_discrete_process = ELECTRON_BOUNDARY_VOXEL;
    }
   
    // FIXME STEP LIMITER was not valided yet!
    // step limiter
    if (step_limiter < next_interaction_distance) {
        next_interaction_distance = step_limiter;
        next_discrete_process = PHOTON_STEP_LIMITER;
    }

    //printf("E %e dist %e\n", energy, next_interaction_distance);

    //// Resolve continuous processes ///////////////////////////////////////

    float safety_distance = __fdividef(energy, total_dedx);
    float continuous_loss = 0.0f;
    //printf("Safety PIL %e\n", safety_distance);
    if (safety_distance < next_interaction_distance) {
        next_interaction_distance = safety_distance;
        next_discrete_process = ELECTRON_SAFETY;
        continuous_loss = energy;
    } else {
        continuous_loss = total_dedx * next_interaction_distance;
        energy -= continuous_loss;
        if (energy < 0.0f) energy = 0.0f;
        electrons.E[id] = energy;
    }
    
    // continuous loss should be at random point along step
    float rnd_dist = next_interaction_distance * Brent_real(id, electrons.table_x_brent, 0);
    float3 rnd_pos;
    rnd_pos.x = position.x - direction.x * rnd_dist;
    rnd_pos.y = position.y - direction.y * rnd_dist;
    rnd_pos.z = position.z - direction.z * rnd_dist;
    if (   rnd_pos.x <= 0 || rnd_pos.x >= dosemap.size_in_mm.x
        || rnd_pos.y <= 0 || rnd_pos.y >= dosemap.size_in_mm.y 
        || rnd_pos.z <= 0 || rnd_pos.z >= dosemap.size_in_mm.z ) {
        rnd_pos = position;
    }
    ivoxsize = inverse_vector(dosemap.voxel_size);
    index_phantom.x = int(rnd_pos.x * ivoxsize.x);
    index_phantom.y = int(rnd_pos.y * ivoxsize.y);
    index_phantom.z = int(rnd_pos.z * ivoxsize.z);
    index_phantom.w = index_phantom.z*dosemap.nb_voxel_slice
                      + index_phantom.y*dosemap.size_in_vox.x
                      + index_phantom.x; // linear index
    atomicAdd(&dosemap.edep[index_phantom.w], continuous_loss);

    //// Move particle //////////////////////////////////////////////////////

    //printf("E %e dist %e\n", energy, next_interaction_distance);
    
    position.x += direction.x * next_interaction_distance;
    position.y += direction.y * next_interaction_distance;
    position.z += direction.z * next_interaction_distance;
	electrons.t[id] += (3.33564095198e-03f * next_interaction_distance);
    electrons.px[id] = position.x;
    electrons.py[id] = position.y;
    electrons.pz[id] = position.z;

    // Stop simulation if out of phantom 
    if (   position.x <= 0 || position.x >= phantom.size_in_mm.x
        || position.y <= 0 || position.y >= phantom.size_in_mm.y 
        || position.z <= 0 || position.z >= phantom.size_in_mm.z ) {
        electrons.endsimu[id] = 1;   // stop the simulation
        electrons.active[id]  = 0;
        photons.active[id]    = 1;   // unfreeze the photon tracking

        atomicAdd(count_d, 1);       // count simulated secondaries
        return;
    }

    //// Resolve discrete processe //////////////////////////////////////////

    float discrete_loss = 0.0f;

    if (next_discrete_process == ELECTRON_BOUNDARY_VOXEL ||
        next_discrete_process == ELECTRON_STEP_LIMITER) {
        //printf("Boundary || step limiter\n");
        return;
    }

    if (next_discrete_process == ELECTRON_SAFETY) {
        //printf("Safety\n");
        electrons.endsimu[id] = 1;   // stop the simulation
        electrons.active[id]  = 0;
        photons.active[id]    = 1;   // unfreeze the photon tracking

        atomicAdd(count_d, 1);       // count simulated secondaries
        return; 
    }

    if (next_discrete_process == ELECTRON_EIONISATION) {
        //printf("eIonisation\n");
        discrete_loss = eIonisation_Effect_Standard_NoSec(electrons, photons, 
                                          materials.electron_cut_energy[mat], 
                                          materials.electron_max_energy[mat],
                                          id, count_d);
    }
    
    if (next_discrete_process == ELECTRON_MSC) {
        //printf("MSC\n");
        // FIXME trueStepLength = next_interaction_distance?!
        discrete_loss = MSC_Effect(electrons, materials, next_interaction_distance, mat, id);
    }
    
    // Dosemap scoring
    ivoxsize = inverse_vector(dosemap.voxel_size);
    index_phantom.x = int(position.x * ivoxsize.x);
    index_phantom.y = int(position.y * ivoxsize.y);
    index_phantom.z = int(position.z * ivoxsize.z);
    index_phantom.w = index_phantom.z*dosemap.nb_voxel_slice
                       + index_phantom.y*dosemap.size_in_vox.x
                       + index_phantom.x; // linear index
    //printf("index dosemap %i\n", index_phantom.w);
    atomicAdd(&dosemap.edep[index_phantom.w], discrete_loss);

}
