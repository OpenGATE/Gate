#include "actor_cst_gpu.cu"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>

/***********************************************************
 * Photons Physics Effects
 ***********************************************************/

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

