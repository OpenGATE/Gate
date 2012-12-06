#include "actor_common.cu"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>

/***********************************************************
 * Photons Physics Effects
 ***********************************************************/

//// Comptons Standard //////////////////////////////////////

// Compton Scatter (Standard - Klein-Nishina)
__device__ float Compton_ct_SampleSecondaries_Standard(StackParticle photons, 
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
        photons.active[id] = 0;  // this particle is absorbed
        atomicAdd(count_d, 1);   // count simulated primaries
        return gamE1;            // Local energy deposit
    }

    return 0.0f;
}

//// PhotoElectric Standard //////////////////////////////////////

// Compute secondaries particles
__device__ float PhotoElec_ct_SampleSecondaries_Standard(StackParticle photons,
                                                         unsigned int id,
                                                         int* count_d) {
    // Absorbed the photon
    photons.endsimu[id] = 1; // stop the simulation
    photons.active[id] = 0;  // this particle is absorbed
    atomicAdd(count_d, 1);   // count simulated primaries

    return 0.0f;
        
}

/***********************************************************
 * Tracking Kernel
 ***********************************************************/

// Photons - regular tracking
#define PHOTON_PHOTOELECTRIC 1
#define PHOTON_COMPTON 2
#define PHOTON_STEP_LIMITER 3
#define PHOTON_BOUNDARY_VOXEL 4
template <typename T1>
__global__ void kernel_ct_navigation_regular(StackParticle photons,
                                             Volume<T1> phantom,
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
    T1 mat = phantom.data[index_phantom.w];

    /*
	int index = materials.index[mat];
    printf("nb_mat %i mat %i index %i nb_elts %i\n", materials.nb_materials, mat, index, materials.nb_elements[mat]);

    int toto=0;
    while (toto<2) {
        printf("mixture: %i num_dens %e\n", materials.mixture[index+toto], materials.atom_num_dens[index+toto]);
        ++toto;
    }

    toto=0;
    while(toto<materials.nb_elements_total) {
        printf("elts %i\n", materials.mixture[toto]);
        ++toto;
    }
    */

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

    // Step limiter
    interaction_distance = 10.0f; // FIXME step limiter
    if (interaction_distance < next_interaction_distance) {
       next_interaction_distance = interaction_distance;
       next_discrete_process = PHOTON_STEP_LIMITER;
    }

    // Distance to the next voxel boundary (raycasting)
    interaction_distance = get_boundary_voxel_by_raycasting(index_phantom, position, 
                                                            direction, phantom.voxel_size);
    if (interaction_distance < next_interaction_distance) {
      next_interaction_distance = interaction_distance;
      next_discrete_process = PHOTON_BOUNDARY_VOXEL;
    }


    //// Move particle //////////////////////////////////////////////////////

    position.x += direction.x * next_interaction_distance;
    position.y += direction.y * next_interaction_distance;
    position.z += direction.z * next_interaction_distance;
    // Dirty part FIXME
    //   apply "magnetic grid" on the particle position due to aproximation 
    //   from the GPU (on the next_interaction_distance).
    float eps = 1.0e-6f; // 1 um
    float res_min, res_max, grid_pos_min, grid_pos_max;
    index_phantom.x = int(position.x * ivoxsize.x);
    index_phantom.y = int(position.y * ivoxsize.y);
    index_phantom.z = int(position.z * ivoxsize.z);
    // on x 
    grid_pos_min = index_phantom.x * phantom.voxel_size.x;
    grid_pos_max = (index_phantom.x+1) * phantom.voxel_size.x;
    res_min = position.x - grid_pos_min;
    res_max = position.x - grid_pos_max;
    if (res_min < eps) {position.x = grid_pos_min;}
    if (res_max > eps) {position.x = grid_pos_max;}
    // on y
    grid_pos_min = index_phantom.y * phantom.voxel_size.y;
    grid_pos_max = (index_phantom.y+1) * phantom.voxel_size.y;
    res_min = position.y - grid_pos_min;
    res_max = position.y - grid_pos_max;
    if (res_min < eps) {position.y = grid_pos_min;}
    if (res_max > eps) {position.y = grid_pos_max;}
    // on z
    grid_pos_min = index_phantom.z * phantom.voxel_size.z;
    grid_pos_max = (index_phantom.z+1) * phantom.voxel_size.z;
    res_min = position.z - grid_pos_min;
    res_max = position.z - grid_pos_max;
    if (res_min < eps) {position.z = grid_pos_min;}
    if (res_max > eps) {position.z = grid_pos_max;}

    photons.px[id] = position.x;
    photons.py[id] = position.y;
    photons.pz[id] = position.z;

    // Stop simulation if out of phantom or no more energy
    if ( position.x <= 0 || position.x >= phantom.size_in_mm.x
     || position.y <= 0 || position.y >= phantom.size_in_mm.y 
     || position.z <= 0 || position.z >= phantom.size_in_mm.z ) {
       photons.endsimu[id] = 1;                     // stop the simulation
       atomicAdd(count_d, 1);                       // count simulated primaries
       return;
    }

    //// Resolve discrete processe //////////////////////////////////////////

    // Resolve discrete processes
    if (next_discrete_process == PHOTON_PHOTOELECTRIC) {
       float discrete_loss = PhotoElec_ct_SampleSecondaries_Standard(photons, id, count_d);
       //printf("id %i PE\n", id);
    }

    if (next_discrete_process == PHOTON_COMPTON) {
       float discrete_loss = Compton_ct_SampleSecondaries_Standard(photons, id, count_d);
       //printf("id %i Compton\n", id);
    }
}
#undef PHOTON_PHOTOELECTRIC
#undef PHOTON_COMPTON
#undef PHOTON_STEP_LIMITER
#undef PHOTON_BOUNDARY_VOXEL


