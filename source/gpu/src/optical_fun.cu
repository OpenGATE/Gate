#include "actor_common.cu"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>

/***********************************************************
 * Photons Physics Effects
 ***********************************************************/



/***********************************************************
 * Source
 ***********************************************************/

template <typename T1, typename T2>
__global__ void kernel_optical_voxelized_source(StackParticle photons, 
                                                Volume<T1> phantom_act, 
                                                Volume<T2> phantom_ind, float E) {

    unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (id >= photons.size) return;
		
    float ind, x, y, z;
    
    float rnd = Brent_real(id, photons.table_x_brent, 0);
    int pos = 0;
    while (phantom_act.data[pos] < rnd) {++pos;};
    
    // get the voxel position (x, y, z)
    ind = (float)(phantom_ind.data[pos]);
    z = floor(ind / (float)phantom_act.nb_voxel_slice);
    ind -= (z * (float)phantom_act.nb_voxel_slice);
    y = floor(ind / (float)(phantom_act.size_in_vox.x));
    x = ind - y * (float)phantom_act.size_in_vox.x;

    // random position inside the voxel
    x += Brent_real(id, photons.table_x_brent, 0);
    y += Brent_real(id, photons.table_x_brent, 0);
    z += Brent_real(id, photons.table_x_brent, 0);

    // must be in mm
    x *= phantom_act.voxel_size.x;
    y *= phantom_act.voxel_size.y;
    z *= phantom_act.voxel_size.z;

    // random orientation
    float phi   = Brent_real(id, photons.table_x_brent, 0);
    float theta = Brent_real(id, photons.table_x_brent, 0);
    phi   = gpu_twopi * phi;
    theta = acosf(1.0f - 2.0f*theta);
    
    // convert to cartesian
    float dx = __cosf(phi)*__sinf(theta);
    float dy = __sinf(phi)*__sinf(theta);
    float dz = __cosf(theta);

    // first gamma
    photons.dx[id] = dx;
    photons.dy[id] = dy;
    photons.dz[id] = dz;
    photons.E[id] = E;
    photons.px[id] = x;
    photons.py[id] = y;
    photons.pz[id] = z;
    photons.t[id] = 0.0f;
    photons.endsimu[id] = 0;
    photons.interaction[id] = 0;
    photons.type[id] = OPTICALPHOTON;
    photons.active[id] = 1;
}


/***********************************************************
 * Tracking Kernel
 ***********************************************************/

/*

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

*/
