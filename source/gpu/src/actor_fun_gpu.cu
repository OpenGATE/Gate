#include "actor_cst_gpu.cu"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>

/***********************************************************
 * Vars
 ***********************************************************/
texture<unsigned short int, 1, cudaReadModeElementType> tex_phantom;

__constant__ const float gpu_pi = 3.14159265358979323846;
__constant__ const float gpu_twopi = 2*gpu_pi;

/***********************************************************
 * Stack data strucutre
 ***********************************************************/

// Stack of gamma particles, format data is defined as SoA
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
	unsigned char* interaction;
    unsigned char* active;
	unsigned char* endsimu;
	unsigned long* table_x_brent;
	unsigned int size;
}; //

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
	phasespace.interaction = (unsigned char*)malloc(mem_phasespace_char);	
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
	free(phasespace.interaction);
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
	cudaMalloc((void**) &stackpart.interaction, mem_stackpart_char);	
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
	cudaFree(stackpart.interaction);
	cudaFree(stackpart.active);
	cudaFree(stackpart.table_x_brent);
}

/***********************************************************
 * Volume data strucutre
 ***********************************************************/

// Volume structure data
template <typename T>
struct Volume {
    T* data;
    T most_att_data;
    unsigned int mem_data;
    float3 size_in_mm;
    int3 size_in_vox;
    float3 voxel_size;
    int nb_voxel_volume;
    int nb_voxel_slice;
    float3 position;
};

// Volume host allocation
template <typename T>
void volume_host_malloc(Volume<T> &vol, int nbvox) {
    vol.mem_data = nbvox * sizeof(T);
    vol.data = (T*)malloc(vol.mem_data);
}

// Free host memory
template <typename T>
void volume_host_free(Volume<T> &vol) {
    free(vol.data);
}

// Volume device allocation
template <typename T>
void volume_device_malloc(Volume<T> &vol, int nbvox) {
    vol.mem_data = nbvox * sizeof(T);
	cudaMalloc((void**) &vol.data, vol.mem_data);
}

// Free device memory
template <typename T>
void volume_device_free(Volume<T> &vol) {
    cudaFree(vol.data);
}

/***********************************************************
 * Copy structure functions
 ***********************************************************/

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
	cudaMemcpy(phasespace.interaction, stackpart.interaction, mem_stackpart_char, cudaMemcpyDeviceToHost);
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
	cudaMemcpy(stackpart.interaction, phasespace.interaction, mem_stackpart_char, cudaMemcpyHostToDevice);
	cudaMemcpy(stackpart.endsimu, phasespace.endsimu, mem_stackpart_char, cudaMemcpyHostToDevice);
	cudaMemcpy(stackpart.active, phasespace.active, mem_stackpart_char, cudaMemcpyHostToDevice);
	cudaMemcpy(stackpart.trackID, phasespace.trackID, mem_stackpart_uint, cudaMemcpyHostToDevice);
	cudaMemcpy(stackpart.eventID, phasespace.eventID, mem_stackpart_uint, cudaMemcpyHostToDevice);
}

// Copy volume from device to host
template <typename T>
void volume_copy_device2host(Volume<T> &voldevice, Volume<T> &volhost) {
    volhost.size_in_vox = voldevice.size_in_vox;
    volhost.voxel_size = voldevice.voxel_size;
    volhost.size_in_mm = voldevice.size_in_mm;
    volhost.nb_voxel_slice = voldevice.nb_voxel_slice;
    volhost.nb_voxel_volume = voldevice.nb_voxel_volume;
    volhost.mem_data = voldevice.mem_data;
    volhost.most_att_data = voldevice.most_att_data;
    volhost.position = voldevice.position;
	cudaMemcpy(volhost.data, voldevice.data, voldevice.mem_data, cudaMemcpyDeviceToHost);
}

// Copy volume from host to device
template <typename T>
void volume_copy_host2device(Volume<T> &volhost, Volume<T> &voldevice) {
    voldevice.size_in_vox = volhost.size_in_vox;
    voldevice.voxel_size = volhost.voxel_size;
    voldevice.size_in_mm = volhost.size_in_mm;
    voldevice.nb_voxel_slice = volhost.nb_voxel_slice;
    voldevice.nb_voxel_volume = volhost.nb_voxel_volume;
    voldevice.mem_data = volhost.mem_data;
    voldevice.most_att_data = volhost.most_att_data;
    voldevice.position = volhost.position;
	cudaMemcpy(voldevice.data, volhost.data, volhost.mem_data, cudaMemcpyHostToDevice);
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
__device__ float Compton_CS_Standard(int mat, float E) {
	float CS = 0.0f;
	int i;
	int index = mat_index[mat];
	// Model standard
	for (i = 0; i < mat_nb_elements[mat]; ++i) {
		CS += (mat_atom_num_dens[index+i] * Compton_CSPA_Standard(E, mat_mixture[index+i]));
	}
	return CS;
}

// Compton Scatter (Standard - Klein-Nishina)
__device__ float Compton_SampleSecondaries_Standard(StackParticle photons, 
                                                    StackParticle electrons,
                                                    float cutE,
                                                    unsigned int id,
                                                    int* count_d) {
	float gamE0 = photons.E[id];

    //if (gamE0 <= 1.0e-6f) { // 1 eV
    //    photons.endsimu[id] = 1; // stop this particle
    //    photons.active[id] = 0;
    //    atomicAdd(count_d, 1); // count simulated primaries
    //    return gamE0; // Local Energy Deposit
    //}

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
        photons.endsimu[id] = 1; // stop this particle
        photons.active[id] = 0;
        atomicAdd(count_d, 1); // count simulated primaries
        return gamE1; // Local energy deposit
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
        electrons.interaction[id] = 0;
        // Now track this electron before going back to the photon tracking
        photons.active[id] = 0;
        electrons.active[id] = 1;
    } 

    return 0.0f;
}

//// PhotoElectric Standard //////////////////////////////////////

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
__device__ float PhotoElec_CS_Standard(unsigned int mat, float E) {
	float CS = 0.0f;
	int i;
	int index = mat_index[mat];
	// Model standard
	for (i = 0; i < mat_nb_elements[mat]; ++i) {
		CS += (mat_atom_num_dens[index+i] * PhotoElec_CSPA_Standard(E, mat_mixture[index+i]));
	}
	return CS;
}

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

// Compute secondaries particles
__device__ float PhotoElec_SampleSecondaries_Standard(StackParticle photons,
                                                      StackParticle electrons,
                                                      float cutE,
                                                      unsigned int mat,
                                                      unsigned int id,
                                                      int* count_d) {

    float energy = photons.E[id];
    float3 PhotonDirection = make_float3(photons.dx[id], photons.dy[id], photons.dz[id]);

    // Select randomly one element constituing the material
    unsigned int n = mat_nb_elements[mat]-1;
    unsigned int index = mat_index[mat];
    unsigned int Z = mat_mixture[index+n];
    unsigned int i = 0;
    if (n > 0) {
        float x = Brent_real(id, photons.table_x_brent, 0) * PhotoElec_CS_Standard(mat, energy);
        float xsec = 0.0f;
        for (i=0; i<n; ++i) {
            xsec += mat_atom_num_dens[index+i] * 
                    PhotoElec_CSPA_Standard(energy, mat_mixture[index+i]);
            if (x <= xsec) {
                Z = mat_mixture[index+i];
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
        electrons.interaction[id] = 0;
        // Now track this electron 
        electrons.active[id] = 1;
    }
    
    // Absorbed the photon
    photons.endsimu[id] = 1; // stop the simulation
    photons.active[id] = 0;
    atomicAdd(count_d, 1); // count simulated primaries
        
    // LocalEnergy Deposit
    return bindingEnergy;
}

/***********************************************************
 * Electrons Physics Effects
 ***********************************************************/

// eIonisation Cross Section Per Atom (Möller model)
__device__ float eIonisation_CSPA_Standard(float E, unsigned short int Z) {
    float cutE = electron_cut_energy[1]; 
    float maxE = electron_max_energy[1];

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
__device__ float eIonisation_CS_Standard(int mat, float E) {
	float CS = 0.0f;
	int i;
	int index = mat_index[mat];

    // TODO There are a problem with CS for a low energy //////////////
    // !!!! This is for testing only !!!!
    if (E < 0.5f) { // 500 keV
        return 1/(2*E); // linear approximation of the CS
    }
    ///////////////////////////////////////////////////////////////////

	// Model standard
	for (i = 0; i < mat_nb_elements[mat]; ++i) {
        CS += (mat_atom_num_dens[index+i] * eIonisation_CSPA_Standard(E, mat_mixture[index+i]));
	}

	return CS;
}

// Compute the dE/dx due to the ionization
__device__ float eIonisation_dedx_Standard(int mat, float E, float cutE) {

    float meanExcitationEnergy = electron_mean_excitation_energy[mat];

    float electronDensity = mat_nb_electrons_per_vol[mat];
    float Natm = mat_nb_atoms_per_vol[mat];
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
    
    // Density coorection
    float twoln10 = 2.0f*__logf(10.0f);
    float x = __fdividef(__logf(beta2*gam2), twoln10);
    float y = 0.0f;
    if (x < fX0[mat]) {
        if (fD0[mat] > 0.0f) {y = fD0[mat]*__powf(10.0f, 2.0f*(x-fX0[mat]));}
    } else if (x >= fX1[mat]) {
        y = twoln10*x - fC[mat];
    } else {
        y = twoln10*x - fC[mat] + fA[mat]*__powf(fX1[mat]-x, fM[mat]);
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
__device__ void eIonisation_SampleSecondaries_Standard(StackParticle photons, 
                                                       StackParticle electrons,
                                                       float tmin, float maxE,
                                                       unsigned int id) {
    float E = electrons.E[id];
    float tmax = E * 0.5f;
    if (maxE < tmax) {tmax = maxE;};
    if (tmin >= tmax) { // tmin is the same that cutE
        // stop the simulation for this one
        electrons.endsimu[id] = 1;
        electrons.interaction[id] = 0;
        // Continue the photon tracking
        electrons.active[id] = 0;
        photons.active[id] = 1;
        return;
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
}


/***********************************************************
 * Sources
 ***********************************************************/

// Particle gun
__global__ void kernel_photons_gun_source(StackParticle photons, StackParticle electrons,
                                          float E, float posx, float posy, float posz,
                                          int* count_p) {

	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (id < photons.size) {
        *count_p = 0;
        // Photons
        photons.dx[id] = 0.0f;
        photons.dy[id] = 0.0f;
		photons.dz[id] = 1.0f;
		photons.E[id] = E;
		photons.px[id] = posx;
		photons.py[id] = posy;
		photons.pz[id] = posz;
		photons.endsimu[id] = 0;
		photons.interaction[id] = 0;
        photons.active[id] = 1;
        // Electrons
        electrons.dx[id] = 0.0f;
        electrons.dy[id] = 0.0f;
		electrons.dz[id] = 0.0f;
		electrons.E[id] = 0.0f;
		electrons.px[id] = 0.0f;
		electrons.py[id] = 0.0f;
		electrons.pz[id] = 0.0f;
		electrons.endsimu[id] = 1;
		electrons.interaction[id] = 0;
        electrons.active[id] = 0;
	}

}

/*
// Iso source of e- (uniform E betwen 0 to 10 MeV)
__global__ void kernel_iso_source_e(StackParticle stackpart, float posx, float posy, float posz) {

	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    if (id < stackpart.size) {

        float phi   = Brent_real(id, stackpart.table_x_brent, 0);
        float theta = Brent_real(id, stackpart.table_x_brent, 0);
        phi   = gpu_twopi * phi;
        theta = acosf(1.0f - 2.0f*theta);
        
        // convert to cartesian
        float x = __cosf(phi)*__sinf(theta);
        float y = __sinf(phi)*__sinf(theta);
        float z = __cosf(theta);
		
        stackpart.dx[id] = x;
        stackpart.dy[id] = y;
		stackpart.dz[id] = z;
		stackpart.E[id] = 8.0f * Brent_real(id, stackpart.table_x_brent, 0);
		stackpart.px[id] = posx;
		stackpart.py[id] = posy;
		stackpart.pz[id] = posz;
		stackpart.endsimu[id] = 0;
		stackpart.interaction[id] = 0;
	}

}
*/

/***********************************************************
 * Tracking Kernel
 ***********************************************************/

// Photons - Fictitious tracking (or delta-tracking)
template <typename T>
__global__ void kernel_photons_woodcock(StackParticle photons, Volume<T> phantom, int* count_d) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	float3 ivoxsize = inverse_vector(phantom.voxel_size);
	
	float3 p, delta;
	float CS_pe, CS_cpt, CS_sum;
	int3 vox;
	float path, rec_mu_maj, E;
	unsigned short int mat;
        
	/*    if (id < photons.size) {
        printf(">>> Woodcock ID %i endsimu %i active %i\n", id, photons.endsimu[id], photons.active[id]);
	}*/
	
	if (id < photons.size && !photons.endsimu[id] && photons.active[id]) {
		p.x = photons.px[id];
		p.y = photons.py[id];
		p.z = photons.pz[id];
		delta.x = photons.dx[id];
		delta.y = photons.dy[id];
		delta.z = photons.dz[id];
		E = photons.E[id];

		//printf(">>> ID %i p %e %e %e d %e %e %e E %e\n", id, p.x, p.y, p.z, delta.x, delta.y, delta.z, E);

		// CS from the most attenuate material
		CS_pe  = PhotoElec_CS_Standard(phantom.most_att_data, E);
		CS_cpt = Compton_CS_Standard(phantom.most_att_data, E);
		rec_mu_maj = __fdividef(1.0f, CS_pe + CS_cpt);

		//        printf(">>> rec mumaj %e\n", rec_mu_maj);

		while (1) {
			// get mean path from the most attenuate material
			path = -__logf(Brent_real(id, photons.table_x_brent, 0)) * rec_mu_maj; // mm

            // fly along the path
			p.x = p.x + delta.x * path;
			p.y = p.y + delta.y * path;
			p.z = p.z + delta.z * path;

			// still inside the volume?
			if (p.x < 0 || p.y < 0 || p.z < 0
				|| p.x >= phantom.size_in_mm.x || p.y >= phantom.size_in_mm.y 
                || p.z >= phantom.size_in_mm.z) {
				photons.endsimu[id] = 1; // stop simulation for this one
				photons.interaction[id] = 0;
				photons.active[id] = 0;
				photons.px[id] = p.x;
				photons.py[id] = p.y;
				photons.pz[id] = p.z;
                atomicAdd(count_d, 1);
		//                printf(">>> OUT - p %e %e %e - path %e\n", p.x, p.y, p.z, path);
				return;
			}
		
			// which voxel?
			vox.x = int(p.x * ivoxsize.x);
			vox.y = int(p.y * ivoxsize.y);
			vox.z = int(p.z * ivoxsize.z);
			
			// get mat
            mat = phantom.data[vox.z*phantom.nb_voxel_slice + 
                               vox.y*phantom.size_in_vox.x + vox.x];
	    //            printf(">>> ID %i mat %i\n", id, mat);
		    CS_pe  = PhotoElec_CS_Standard(mat, E);
    		CS_cpt = Compton_CS_Standard(mat, E);
            CS_sum = CS_pe + CS_cpt;
            
            // Does the interaction is real?
			if (CS_sum * rec_mu_maj > Brent_real(id, photons.table_x_brent, 0)) {break;}

		} // While 

        // update position
		photons.px[id] = p.x;
		photons.py[id] = p.y;
		photons.pz[id] = p.z;

        // select an interaction
        CS_pe = __fdividef(CS_pe, CS_sum);
        if (CS_pe > Brent_real(id, photons.table_x_brent, 0)) {
            // photoelectric
            photons.interaction[id] = 1;
        } else {
            // Compton
            photons.interaction[id] = 2;
        }

	} // if ID

}

// Photons - classical tracking
#define PHOTON_NO_PROCESS 0
#define PHOTON_PHOTOELECTRIC 1
#define PHOTON_COMPTON 2
#define PHOTON_STEP_LIMITER 3
// FIXME in case of dosemap
//template <typename T1, typename T2>
//__global__ void kernel_photons_classical(StackParticle photons, StackParticle electrons,
//                                           Volume<T1> phantom, Volume<T2> dosemap,
//                                           int* count_d) {
template <typename T1>
__global__ void kernel_photons_classical(StackParticle photons, StackParticle electrons,
                                           Volume<T1> phantom,
                                           int* count_d) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    
    if (id >= photons.size) return;
    if (photons.endsimu[id]) return;
    if (!photons.active[id]) return;

    float3 position; // mm
    {
        position.x = photons.px[id];
        position.y = photons.py[id];
        position.z = photons.pz[id];
    }

	int4 index_phantom;
    {
        float3 ivoxsize = inverse_vector(phantom.voxel_size);
        index_phantom.x = int(position.x * ivoxsize.x);
        index_phantom.y = int(position.y * ivoxsize.y);
        index_phantom.z = int(position.z * ivoxsize.z);
        index_phantom.w = index_phantom.z*phantom.nb_voxel_slice
            + index_phantom.y*phantom.size_in_vox.x
            + index_phantom.x; // linear index
    }

    // FIXME add dosemap
    /*
    int4 index_dosemap;
    {
        float3 ivoxsize = inverse_vector(dosemap.voxel_size);
        index_dosemap.x = int(position.x * ivoxsize.x);
        index_dosemap.y = int(position.y * ivoxsize.y);
        index_dosemap.z = int(position.z * ivoxsize.z);
        index_dosemap.w = index_dosemap.z*dosemap.nb_voxel_slice
            + index_dosemap.y*dosemap.size_in_vox.x
            + index_dosemap.x; // linear index
    }
    */

    float3 direction;
    {
        direction.x = photons.dx[id];
        direction.y = photons.dy[id];
        direction.z = photons.dz[id];
    }

    float energy = photons.E[id];

    T1 material = phantom.data[index_phantom.w];

    //float tracking_energy = 0;

    // find next discrete interaction, total_dedx and next discrete intraction distance
    float next_interaction_distance =  FLT_MAX;
    unsigned char next_discrete_process = 0; 

    //printf("ID %i process %i\n", id, next_discrete_process);

    { // photoelectric
		float cross_section = PhotoElec_CS_Standard(material, energy);
        float interaction_distance = __fdividef(
                -__logf(Brent_real(id, photons.table_x_brent, 0)),
                cross_section);
        //printf("ID %i PE distance %e\n", id, interaction_distance);
        if (interaction_distance < next_interaction_distance)
        {
            next_interaction_distance = interaction_distance;
            next_discrete_process = PHOTON_PHOTOELECTRIC;
        }
    }

    { // compton
		float cross_section = Compton_CS_Standard(material, energy);
        float interaction_distance = __fdividef(
                -__logf(Brent_real(id, photons.table_x_brent, 0)),
                cross_section);
        //printf("ID %i Compton distance %e\n", id, interaction_distance);
        if (interaction_distance < next_interaction_distance)
        {
            next_interaction_distance = interaction_distance;
            next_discrete_process = PHOTON_COMPTON;
        }
    }

    { // step limiter
        float interaction_distance = .5f; // FIXME step limiter
        //printf("ID %i STEP distance %e\n", id, interaction_distance);
        if (interaction_distance < next_interaction_distance)
        {
            next_interaction_distance = interaction_distance;
            next_discrete_process = PHOTON_STEP_LIMITER;
        }
    }

    //printf(">>> ID %i process %i\n", id, next_discrete_process);

    { // move particle
        position.x += direction.x * next_interaction_distance;
        position.y += direction.y * next_interaction_distance;
        position.z += direction.z * next_interaction_distance;
        photons.px[id] = position.x;
        photons.py[id] = position.y;
        photons.pz[id] = position.z;
    }

    //printf("photon %i - p %e %e %e\n", id, photons.px[id], photons.py[id], photons.pz[id]);

    // stop simulation if out of phantom or no more energy
    if ( position.x < 0 || position.x >= phantom.size_in_mm.x
      || position.y < 0 || position.y >= phantom.size_in_mm.y 
      || position.z < 0 || position.z >= phantom.size_in_mm.z )
    {
        photons.endsimu[id] = 1;
        photons.interaction[id] = PHOTON_NO_PROCESS;
        atomicAdd(count_d, 1);
        photons.active[id] = 0;
        //printf("OUT\n");
        return;
    }

    // resolve discrete processes
    if (next_discrete_process == PHOTON_PHOTOELECTRIC)
    {
        //printf("PE\n");
        float ecut = electron_cut_energy[material];
        float discrete_loss = PhotoElec_SampleSecondaries_Standard(photons, electrons, ecut,
                                                          material, id, count_d);

        // FIXME add dosemap
        /*
        float3 ivoxsize = inverse_vector(dosemap.voxel_size);
        index_dosemap.x = int(position.x * ivoxsize.x);
        index_dosemap.y = int(position.y * ivoxsize.y);
        index_dosemap.z = int(position.z * ivoxsize.z);
        index_dosemap.w = index_dosemap.z*dosemap.nb_voxel_slice
            + index_dosemap.y*dosemap.size_in_vox.x
            + index_dosemap.x; // linear index

        atomicAdd(&dosemap.data[index_dosemap.w], discrete_loss);
        */

        //++photons.ct_photoelectric[id];
    }

    if (next_discrete_process == PHOTON_COMPTON)
    {
        //printf("Compton\n");
        float ecut = electron_cut_energy[material];
        float discrete_loss = Compton_SampleSecondaries_Standard(photons, electrons, ecut,
                                                        id, count_d);
        
        // FIXME add dosemap
        /*
        float3 ivoxsize = inverse_vector(dosemap.voxel_size);
        index_dosemap.x = int(position.x * ivoxsize.x);
        index_dosemap.y = int(position.y * ivoxsize.y);
        index_dosemap.z = int(position.z * ivoxsize.z);
        index_dosemap.w = index_dosemap.z*dosemap.nb_voxel_slice
            + index_dosemap.y*dosemap.size_in_vox.x
            + index_dosemap.x; // linear index

        atomicAdd(&dosemap.data[index_dosemap.w], discrete_loss);
        */

        //++photons.ct_compton[id];
    }
}

// Electrons - Fictitious tracking (or delta-tracking)
template <typename T>
__global__ void kernel_electrons_woodcock(StackParticle photons, StackParticle electrons,
                                          Volume<T> phantom) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    float3 ivoxsize = inverse_vector(phantom.voxel_size);

	float3 p, delta;
	float CS_eio, CS_sum;
	int3 vox;
	float path, rec_mu_maj, E;
	unsigned short int mat;
	
	if (id < electrons.size && !electrons.endsimu[id] && electrons.active[id]) {
		p.x = electrons.px[id];
		p.y = electrons.py[id];
		p.z = electrons.pz[id];
		delta.x = electrons.dx[id];
		delta.y = electrons.dy[id];
		delta.z = electrons.dz[id];
		E = electrons.E[id];
		
        // CS from the most attenuate material
		CS_eio  = eIonisation_CS_Standard(phantom.most_att_data, E);
        rec_mu_maj = __fdividef(1.0f, CS_eio);

		while (1) {
			// get mean path from the most attenuate material
			path = -__logf(Brent_real(id, electrons.table_x_brent, 0)) * rec_mu_maj; // mm

            // fly along the path
			p.x = p.x + delta.x * path;
			p.y = p.y + delta.y * path;
			p.z = p.z + delta.z * path;

			// still inside the volume?
			if (p.x < 0 || p.y < 0 || p.z < 0
				|| p.x >= phantom.size_in_mm.x || p.y >= phantom.size_in_mm.y 
                || p.z >= phantom.size_in_mm.z) {
				electrons.endsimu[id] = 1; // stop simulation for this one
				electrons.interaction[id] = 0;
				electrons.px[id] = p.x;
				electrons.py[id] = p.y;
				electrons.pz[id] = p.z;
                // Continue the photon tracking
                electrons.active[id] = 0;
                photons.active[id] = 1;
				return;
			}
		
			// which voxel?
			vox.x = int(p.x * ivoxsize.x);
			vox.y = int(p.y * ivoxsize.y);
			vox.z = int(p.z * ivoxsize.z);
			
			// get mat
            mat = phantom.data[vox.z*phantom.nb_voxel_slice +
                               vox.y*phantom.size_in_vox.x + vox.x];
		    CS_eio = eIonisation_CS_Standard(mat, E);
            CS_sum = CS_eio;
            
            // Does the interaction is real?
			if (CS_sum * rec_mu_maj > Brent_real(id, electrons.table_x_brent, 0)) {break;}

		} // While 

        // update position
		electrons.px[id] = p.x;
		electrons.py[id] = p.y;
		electrons.pz[id] = p.z;

        // select an interaction
        // eIonisation
        electrons.interaction[id] = 1;

	} // if ID

}

#define ELECTRON_NO_PROCESS 0
#define ELECTRON_IONISATION 1
#define ELECTRON_STEP_LIMITER 2
// Electrons - classical tracking
// FIXME in case of dosemap
//template <typename T1, typename T2>
//__global__ void kernel_electrons_classical(StackParticle photons, StackParticle electrons,
//                                           Volume<T1> phantom, Volume<T2> dosemap) {
template <typename T1>
__global__ void kernel_electrons_classical(StackParticle photons, StackParticle electrons,
                                           Volume<T1> phantom) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (id >= electrons.size) return;
    if (electrons.endsimu[id]) return;
    if (!electrons.active[id]) return;


    float3 position; // mm
    {
        position.x = electrons.px[id];
        position.y = electrons.py[id];
        position.z = electrons.pz[id];
    }

	int4 index_phantom;
    {
        float3 ivoxsize = inverse_vector(phantom.voxel_size);
        index_phantom.x = int(position.x * ivoxsize.x);
        index_phantom.y = int(position.y * ivoxsize.y);
        index_phantom.z = int(position.z * ivoxsize.z);
        index_phantom.w = index_phantom.z*phantom.nb_voxel_slice
            + index_phantom.y*phantom.size_in_vox.x
            + index_phantom.x; // linear index
    }

    // FIXME add dosemap
    /*
    int4 index_dosemap;
    {
        float3 ivoxsize = inverse_vector(dosemap.voxel_size);
        index_dosemap.x = int(position.x * ivoxsize.x);
        index_dosemap.y = int(position.y * ivoxsize.y);
        index_dosemap.z = int(position.z * ivoxsize.z);
        index_dosemap.w = index_dosemap.z*dosemap.nb_voxel_slice
            + index_dosemap.y*dosemap.size_in_vox.x
            + index_dosemap.x; // linear index
    }
    */

    float3 direction;
    {
        direction.x = electrons.dx[id];
        direction.y = electrons.dy[id];
        direction.z = electrons.dz[id];
    }

    float energy = electrons.E[id];

    T1 material = phantom.data[index_phantom.w];

    float tracking_energy = 0;

    // find next discrete interaction, total_dedx and next discrete intraction distance
    float next_interaction_distance =  FLT_MAX;
    float total_dedx = 0;
    unsigned char next_discrete_process = 0; 

    { // ionisation
        float ecut = electron_cut_energy[material];
		float cross_section = eIonisation_CS_Standard(material, energy);
        float interaction_distance = __fdividef(
                -__logf(Brent_real(id, electrons.table_x_brent, 0)),
                cross_section);
        total_dedx += eIonisation_dedx_Standard(material, energy, ecut);
        if (interaction_distance < next_interaction_distance)
        {
            next_interaction_distance = interaction_distance;
            next_discrete_process = ELECTRON_IONISATION;
        }
    }

    { // step limiter
        float interaction_distance = .5f; // FIXME step limiter
        if (interaction_distance < next_interaction_distance)
        {
            next_interaction_distance = interaction_distance;
            next_discrete_process = ELECTRON_STEP_LIMITER;
        }
    }

    { // resolve continuous processes
        float continuous_loss = total_dedx * next_interaction_distance;
        // FIXME dirty fmin hack
        if (continuous_loss > energy)
            continuous_loss = energy;

        // FIXME continouous loss should be at random point along the step
        // FIXME add dosemap
        //atomicAdd(&dosemap.data[index_dosemap.w], continuous_loss);

        energy -= continuous_loss;
        electrons.E[id]  = energy;
    }
    
    { // move particle
        position.x += direction.x * next_interaction_distance;
        position.y += direction.y * next_interaction_distance;
        position.z += direction.z * next_interaction_distance;
        electrons.px[id] = position.x;
        electrons.py[id] = position.y;
        electrons.pz[id] = position.z;
    }
    
    //printf("electron %i - p %e %e %e - process %i - dist %e\n", id, electrons.px[id], electrons.py[id], electrons.pz[id], next_discrete_process, next_interaction_distance);

    // stop simulation if out of phantom or no more energy
    if (energy <= tracking_energy
            || position.x < 0 || position.x >= phantom.size_in_mm.x
            || position.y < 0 || position.y >= phantom.size_in_mm.y 
            || position.z < 0 || position.z >= phantom.size_in_mm.z )
    {
        electrons.endsimu[id] = 1;
        electrons.interaction[id] = ELECTRON_NO_PROCESS;

        // Continue the photon tracking
        electrons.active[id] = 0;
        photons.active[id] = 1;
        return;
    }

    // resolve discrete processes
    if (next_discrete_process == ELECTRON_IONISATION)
    {
        float ecut = electron_cut_energy[material];
        float emax = electron_max_energy[material];
        eIonisation_SampleSecondaries_Standard(photons, electrons, ecut, emax, id);

        float discrete_loss = energy - electrons.E[id];

        // FIXME add dosemap
        /*
        float3 ivoxsize = inverse_vector(dosemap.voxel_size);
        index_dosemap.x = int(position.x * ivoxsize.x);
        index_dosemap.y = int(position.y * ivoxsize.y);
        index_dosemap.z = int(position.z * ivoxsize.z);
        index_dosemap.w = index_dosemap.z*dosemap.nb_voxel_slice
            + index_dosemap.y*dosemap.size_in_vox.x
            + index_dosemap.x; // linear index

        atomicAdd(&dosemap.data[index_dosemap.w], discrete_loss);
        */

        //++electrons.ct_eionization[id];
    }
}




/***********************************************************
 * Interactions
 ***********************************************************/

// Kernel interactions
template <typename T1>
__global__ void kernel_photons_interactions(StackParticle photons, StackParticle electrons,
                                            Volume<T1> phantom, 
                                            int* count_d) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

	/*    if (id<photons.size) {
        printf(">>> ID %i endsimu %i active %i\n", id, photons.endsimu[id], photons.active[id]);
	}*/

	if (id < photons.size && !photons.endsimu[id] && photons.active[id]) {
        // get partile position
        float3 p;
        p.x = photons.px[id];
        p.y = photons.py[id];
        p.z = photons.pz[id];
        // get material
        int3 pos_phan;
        pos_phan.x = (int)(__fdividef(p.x, phantom.voxel_size.x));
        pos_phan.y = (int)(__fdividef(p.y, phantom.voxel_size.y));
        pos_phan.z = (int)(__fdividef(p.z, phantom.voxel_size.z));
        int ind_phan = pos_phan.z * phantom.nb_voxel_slice +
                       pos_phan.y * phantom.size_in_vox.x +
                       pos_phan.x;
        T1 mat = phantom.data[ind_phan]; 
	//        printf(">>> p %e %e %e ind %i mat %i\n", p.x, p.y, p.z, ind_phan, mat);
        /*
        // get dosemap index
        int3 pos_map;
        pos_map.x = (int)(__fdividef(p.x-dosemap.position.x, dosemap.voxel_size.x));
        pos_map.y = (int)(__fdividef(p.y-dosemap.position.y, dosemap.voxel_size.y));
        pos_map.z = (int)(__fdividef(p.z-dosemap.position.z, dosemap.voxel_size.z));
        int ind_map;
        if (pos_map.x >= 0 && pos_map.x < dosemap.size_in_vox.x &&
            pos_map.y >= 0 && pos_map.y < dosemap.size_in_vox.y &&
            pos_map.z >= 0 && pos_map.z < dosemap.size_in_vox.z) {
            
            ind_map = pos_map.z * dosemap.nb_voxel_slice +
                      pos_map.y * dosemap.size_in_vox.x +
                      pos_map.x;
        } else {
            ind_map = -1;
        }
        */
        // get cut production for a given material
        float cutE = electron_cut_energy[mat];

        // apply an interaction
        float dose;
		switch (photons.interaction[id]) {
		case 0:
			// do nothing and release the thread
			return;
		case 1: 
            // Photoelectric
		  //            printf(">>>>>> PE\n");
            dose = PhotoElec_SampleSecondaries_Standard(photons, electrons, cutE,
                                                        mat, id, count_d);
            //if (ind_map != -1) {atomicAdd(&dosemap.data[ind_map], dose);}
            return;
        case 2:
            // Compton
	  //            printf(">>>>>> Compton\n");
            dose = Compton_SampleSecondaries_Standard(photons, electrons, cutE,
                                                      id, count_d);
            //if (ind_map != -1) {atomicAdd(&dosemap.data[ind_map], dose);}
            return;
		}
		
	} // if ID
}

// Kernel interactions
template <typename T1>
__global__ void kernel_electrons_interactions(StackParticle photons, StackParticle electrons, 
                                              Volume<T1> phantom) {
	unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
            
    if (id < electrons.size && !electrons.endsimu[id] && electrons.active[id]) {
        // get partile position
        float3 p;
        p.x = electrons.px[id];
        p.y = electrons.py[id];
        p.z = electrons.pz[id];
        // get material
        int3 pos_phan;
        pos_phan.x = (int)(__fdividef(p.x, phantom.voxel_size.x));
        pos_phan.y = (int)(__fdividef(p.y, phantom.voxel_size.y));
        pos_phan.z = (int)(__fdividef(p.z, phantom.voxel_size.z));
        int ind_phan = pos_phan.z * phantom.nb_voxel_slice +
                       pos_phan.y * phantom.size_in_vox.x +
                       pos_phan.x;
        T1 mat = phantom.data[ind_phan]; 
        /*
        // get dosemap index
        int3 pos_map;
        pos_map.x = (int)(__fdividef(p.x-dosemap.position.x, dosemap.voxel_size.x));
        pos_map.y = (int)(__fdividef(p.y-dosemap.position.y, dosemap.voxel_size.y));
        pos_map.z = (int)(__fdividef(p.z-dosemap.position.z, dosemap.voxel_size.z));
        //printf("p %e %e %e map %i %i %i\n", p.x, p.y, p.z, pos_map.x, pos_map.y, pos_map.z);
        //printf("nb vox / slice %i size x %i\n", dosemap.nb_voxel_slice, dosemap.size_in_vox.x);
        // does the position is inside the dosemap
        int ind_map;
        if (pos_map.x >= 0 && pos_map.x < dosemap.size_in_vox.x &&
            pos_map.y >= 0 && pos_map.y < dosemap.size_in_vox.y &&
            pos_map.z >= 0 && pos_map.z < dosemap.size_in_vox.z) {
            
            ind_map = pos_map.z * dosemap.nb_voxel_slice +
                      pos_map.y * dosemap.size_in_vox.x +
                      pos_map.x;
        } else {
            ind_map = -1;
        }
        */
        // get cut production for a given material
        float cutE = electron_cut_energy[mat];
        float maxE = electron_max_energy[mat];

        // apply an interaction
        float dedx;
        switch (electrons.interaction[id]) {
        case 0:
            // do nothing and release the thread
            return;
        case 1:
            // eIonisation
			eIonisation_SampleSecondaries_Standard(photons, electrons, cutE, maxE, id);
            dedx = eIonisation_dedx_Standard(mat, electrons.E[id], cutE);
            //if (ind_map != -1) {atomicAdd(&dosemap.data[ind_map], dedx);}
            return;
		}
		
	} // if ID
}

/***********************************************************
 * Utils Host
 ***********************************************************/

// Build and load a water box as phantom in the texture mem
void load_water_box_in_tex(int3 dim_phantom) {
	// Build the water box
	int nb = dim_phantom.z * dim_phantom.y * dim_phantom.x;
	unsigned int mem_phantom = nb * sizeof(unsigned short int);
	unsigned short int* phantom = (unsigned short int*)malloc(mem_phantom);
	// Fill phantom with water
	int n=0; while (n<nb) {phantom[n]=1; ++n;}; // 1-water  3-lung
	// Load phantom to texture
	unsigned short int* dphantom;
	cudaMalloc((void**) &dphantom, mem_phantom);
	cudaMemcpy(dphantom, phantom, mem_phantom, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_phantom, dphantom, mem_phantom);
	free(phantom);
}

// Load phantom in the tex mem
void load_phantom_in_tex(const char* filename, int3 dim_phantom) {
	int nb = dim_phantom.z * dim_phantom.y * dim_phantom.x;
	unsigned int mem_phantom = nb * sizeof(unsigned short int);
	unsigned short int* phantom = (unsigned short int*)malloc(mem_phantom);
	// Read data
	FILE * pfile = fopen(filename, "rb");
	fread(phantom, sizeof(unsigned short int), nb, pfile);
	fclose(pfile);
	// Load phantom to texture
	unsigned short int* dphantom;
	cudaMalloc((void**) &dphantom, mem_phantom);
	cudaMemcpy(dphantom, phantom, mem_phantom, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_phantom, dphantom, mem_phantom);
	free(phantom);
}

// Get time
double time() {
	timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// Dump phasespace
void dump_particle_to_phspfile(StackParticle &phasespace, int nb_particle, const char* phsp_name) {
	FILE* pfile = fopen(phsp_name, "wb");
	
	int i = 0;
	while (i<nb_particle) {
		fwrite(&phasespace.E[i], sizeof(float), 1, pfile);		
		fwrite(&phasespace.px[i], sizeof(float), 1, pfile);
		fwrite(&phasespace.py[i], sizeof(float), 1, pfile);
		fwrite(&phasespace.pz[i], sizeof(float), 1, pfile);
		fwrite(&phasespace.dx[i], sizeof(float), 1, pfile);
		fwrite(&phasespace.dy[i], sizeof(float), 1, pfile);
		fwrite(&phasespace.dz[i], sizeof(float), 1, pfile);
		++i;
	}

	fclose(pfile);
}

// Export stack to text file
void stack_host_export_txt(StackParticle &part, const char* name) {
    FILE* pfile;
    pfile = fopen(name, "a");
    int i=0; while (i<part.size) {
        if (part.active[i]) {
            fprintf(pfile, "track %i subtrack %i pos %f %f %f E %e endsimu %i\n", i, 
                    0, 
                    part.px[i], part.py[i], part.pz[i], part.E[i], part.endsimu[i]);
        }
        ++i;
    }
    fclose(pfile);
}

// Build water box
template <typename T>
void build_water_box(Volume<T> &phantom, 
                     int nz, int ny, int nx, 
                     float voxsizez, float voxsizey, float voxsizex) { 
    phantom.size_in_vox = make_int3(nx, ny, nz);
    phantom.voxel_size = make_float3(voxsizex, voxsizey, voxsizez);
    phantom.size_in_mm = make_float3(nx*voxsizex, ny*voxsizey, nz*voxsizez);
    phantom.position = make_float3(0.0f, 0.0f, 0.0f);
    phantom.nb_voxel_slice = ny*nx;
    phantom.nb_voxel_volume = phantom.nb_voxel_slice * nz;
    phantom.mem_data = phantom.nb_voxel_volume * sizeof(T);
    phantom.data = (T*)malloc(phantom.mem_data);
	// Fill phantom with water
	int n=0; while (n<phantom.nb_voxel_volume) {phantom.data[n]=(T)1; ++n;}; // 1-water
    phantom.most_att_data = (T)1; // water
}

// Build empty box
template <typename T>
void build_empty_box(Volume<T> &phantom, 
                     int nz, int ny, int nx, 
                     float voxsizez, float voxsizey, float voxsizex,
                     float posx, float posy, float posz) { 
    phantom.size_in_vox = make_int3(nx, ny, nz);
    phantom.voxel_size = make_float3(voxsizex, voxsizey, voxsizez);
    phantom.size_in_mm = make_float3(nx*voxsizex, ny*voxsizey, nz*voxsizez);
    phantom.position = make_float3(posx, posy, posz);
    phantom.nb_voxel_slice = ny*nx;
    phantom.nb_voxel_volume = phantom.nb_voxel_slice * nz;
    phantom.mem_data = phantom.nb_voxel_volume * sizeof(T);
    phantom.data = (T*)malloc(phantom.mem_data);
	// Fill phantom with water
	int n=0; while (n<phantom.nb_voxel_volume) {phantom.data[n]=(T)0; ++n;}; // 1-water
    phantom.most_att_data = (T)0; // water
}

// Export volume in .vol format
template <typename T>
void volume_host_export_vol(Volume<T> &dosemap, const char* name) {
    FILE* pfile;
    pfile = fopen(name, "wb");
    T nz = T(dosemap.size_in_vox.z);
    T ny = T(dosemap.size_in_vox.y);
    T nx = T(dosemap.size_in_vox.x);
    fwrite(&nz, sizeof(T), 1, pfile);
    fwrite(&ny, sizeof(T), 1, pfile);
    fwrite(&nx, sizeof(T), 1, pfile);
    fwrite(dosemap.data, sizeof(T), dosemap.nb_voxel_volume, pfile);
    fclose(pfile);
}
