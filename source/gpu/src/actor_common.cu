#include "actor_cst.cu"
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
	cudaMemcpy(stackpart.seed, phasespace.seed, mem_stackpart_uint, cudaMemcpyHostToDevice);
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

//// Return the next voxel boundary distance, it is used by the standard navigator
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
