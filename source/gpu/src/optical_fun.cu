#include "actor_common.cu"
#include "optical_cst.cu"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>

// vesna - for ROOT output
#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TSystem.h>
#include <TPluginManager.h>
// vesna - for ROOT output

__device__ float loglog_interpolation(float x, float x0, float y0, float x1, float y1) {
	if (x < x0) {return y0;}
	if (x > x1) {return y1;}
	x0 = __fdividef(1.0f, x0);
	return __powf(10.0f, __log10f(y0) + __log10f(__fdividef(y1, y0)) *
		__fdividef(__log10f(x * x0), __log10f(x1 * x0)));
}

__device__ float lin_interpolation(float x, float x0, float y0, float x1, float y1) {
	if (x < x0) {return y0;}
	if (x > x1) {return y1;}
	return y0 + (y1 - y0) * __fdividef(x - x0, x1 - x0);
}

/***********************************************************
 * Photons Physics Effects
 ***********************************************************/


// vesna - Compute the total Mie cross section for a given material
__device__ float Mie_CS(int mat, float E) {

	int start = 0;
	int stop  = start +5; 
	int pos;

	for (pos=start; pos<stop; pos+=2) {
		if (Mie_scatteringlength_Table[mat][pos] >= E) {break;}
	}

      if (pos == 0) {
      return __fdividef(1.0f, Mie_scatteringlength_Table[mat][pos+1]);
      }
      else{
		return __fdividef(1.0f, loglog_interpolation(E, Mie_scatteringlength_Table[mat][pos-2], 
                                                    Mie_scatteringlength_Table[mat][pos-1], 
                                                    Mie_scatteringlength_Table[mat][pos], 
                                                    Mie_scatteringlength_Table[mat][pos+1]));
    }

}  // vesna - Compute the total Mie cross section for a given material


// vesna - Mie Scatter (Henyey-Greenstein approximation)
__device__ float3 Mie_scatter(StackParticle stack, unsigned int id, int mat) { 

      float forward_g = mat_anisotropy[mat];
      float backward_g = mat_anisotropy[mat];
      float ForwardRatio = 1.0f;
      unsigned char direction=0; 
      float g;
      
      if (Brent_real(id, stack.table_x_brent, 0)<= ForwardRatio) {
      	g = forward_g;
      }
      else {
      	g = backward_g;
      	direction = 1; 
      }

	float r = Brent_real(id, stack.table_x_brent, 0);	
      	float theta;
      	if(g == 0.0f) {	
		theta = acosf(2.0f * r - 1.0f); 
		}else {
        float val_in_acos = __fdividef(2.0f*r*(1.0f + g)*(1.0f + g)*(1.0f - g + g * r),(1.0f - g + 2.0f*g*r)*(1.0f - g + 2.0f*g*r))- 1.0f; 
        val_in_acos = fmin(val_in_acos, 1.0f); 
		theta = acosf(val_in_acos); 
		}
		
	float costheta, sintheta, phi;	
		
	costheta = cosf(theta);	
	sintheta = sqrt(1.0f - costheta*costheta);
	phi = Brent_real(id, stack.table_x_brent, 0) * gpu_twopi;
	
	if (direction) theta = gpu_pi - theta;

    float3 Dir1 = make_float3(sintheta*__cosf(phi), sintheta*__sinf(phi), costheta);
    Dir1 = rotateUz(Dir1, make_float3(stack.dx[id], stack.dy[id], stack.dz[id]));
    stack.dx[id] = Dir1.x;
    stack.dy[id] = Dir1.y;
    stack.dz[id] = Dir1.z;
}  // vesna - Mie Scatter (Henyey-Greenstein approximation)


// vesna - Surface effects

// Compute the Fresnel reflectance (MCML code)
__device__ float2 RFresnel(float n_incident, /* incident refractive index.*/
				           float n_transmit, /* transmit refractive index.*/
				           float c_incident_angle) 
                           /* cosine of the incident angle. 0<a1<90 degrees. */
{
  float r;
  float c_transmission_angle;
  
  if(n_incident==n_transmit) {			/** matched boundary. **/
    c_transmission_angle = c_incident_angle;
    r = 0.0f;
    //printf("case 1\n");
  }
  else if(c_incident_angle > COSZERO) {	/** normal incident. **/
    c_transmission_angle = c_incident_angle;
    r = __fdividef(n_transmit-n_incident, n_transmit+n_incident);
    r *= r;
    //printf("case 2\n");
  }
  else if(c_incident_angle < COS90D)  {	/** very slant. **/
    c_transmission_angle = 0.0f;
    r = 1.0f;
    //printf("case 3\n");
  }
  else  {		/** general. **/
    float sa1, sa2;	/* sine of the incident and transmission angles. */
    float ca2;
    
    sa1 = sqrtf(1.0f-c_incident_angle*c_incident_angle);
    sa2 = __fdividef(n_incident*sa1, n_transmit);
    if(sa2 >= 1.0f) { 	/* double check for total internal reflection. */
      c_transmission_angle = 0.0f;
      r = 1.0f;
    //printf("case 4\n");
    }
    else  {
      float cap, cam;	/* cosines of the sum ap or difference am of the two */
			            /* angles. ap = a_incident+a_transmit am = a_incident - a_transmit. */
      float sap, sam;	/* sines. */
      c_transmission_angle = sqrtf(1.0f-sa2*sa2);
      ca2 = c_transmission_angle;
      cap = c_incident_angle*ca2 - sa1*sa2; /* c+ = cc - ss. */
      cam = c_incident_angle*ca2 + sa1*sa2; /* c- = cc + ss. */
      sap = sa1*ca2 + c_incident_angle*sa2; /* s+ = sc + cs. */
      sam = sa1*ca2 - c_incident_angle*sa2; /* s- = sc - cs. */
      r = __fdividef(0.5f*sam*sam*(cam*cam+cap*cap), sap*sap*cam*cam); 
      //printf("case 5\n");
    }
  }

  return make_float2(r, c_transmission_angle);
}
// Fresnel Reflectance

// Fresnel Processes
__device__ float3 Fresnel_process(StackParticle photon, unsigned int id, 
                                  unsigned short int mat_i, unsigned short int mat_t) { 

  //float ux = photon.dx[id];
  //float uy = photon.dy[id];
  float uz = photon.dz[id]; /* z directional cosine. */
  float ni = mat_Rindex[mat_i];
  float nt = mat_Rindex[mat_t];

  // DEBUG
  //ux = 0.0f;
  //uy = 0.0f;
  //uz = 1.0f;
  //ni = 1.2f;
  //nt = 1.4f;
 
  //printf("before ux %f uy %f uz %f\n", ux, uy, uz);
  
  /* Get r. */
  //  if( uz <= 0.7) /* 0.7 is the cosine of the critical angle of total internal reflection */
  //   r=1.0;		/* total internal reflection. */
  //  else r = RFresnel(ni, nt, uz, &uz1);
 
  //r = RFresnel(ni, nt, uz, &uz1);
  float2 res = RFresnel(ni, nt, uz); // res.x=r  res.y=uz1

  //printf("ni %f nt %f r %f uz1 %f\n", ni, nt, res.x, res.y);

  if (Brent_real(id, photon.table_x_brent, 0) > res.x) {	/* transmitted */
      photon.dx[id] *= __fdividef(ni, nt);
      photon.dy[id] *= __fdividef(ni, nt);
      photon.dz[id] = res.y;
  }
  else {						/* reflected. */
      photon.dz[id] = -uz;
  }
  
  //printf("(ux, uy, uz) = %f %f %f %f %f\n", ux, uy, uz, res.x, res.y);

  //float3 Dir1 = make_float3(ux, uy, uz);
  //Dir1 = rotateUz(Dir1, make_float3(photon.dx[id], photon.dy[id], photon.dz[id]));
  //photon.dx[id] = Dir1.x;
  //photon.dy[id] = Dir1.y;
  //photon.dz[id] = Dir1.z;

  //printf("after ux %f uy %f uz %f\n", Dir1.x, Dir1.y, Dir1.z);


}  // vesna - Fresnel Processes

/***********************************************************
 * Source
 ***********************************************************/

template <typename T1>
__global__ void kernel_optical_voxelized_source(StackParticle photons, 
                                                Volume<T1> phantom_mat,
                                                float *phantom_act,
                                                unsigned int *phantom_ind, float E) {

    unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (id >= photons.size) return;
		
    float ind, x, y, z;
    
    float rnd = Brent_real(id, photons.table_x_brent, 0);
    int pos = 0;
    while (phantom_act[pos] < rnd) {++pos;};
    
    // get the voxel position (x, y, z)
    ind = (float)(phantom_ind[pos]);
    //float debug = phantom_act.data[10];
    
    z = floor(ind / (float)phantom_mat.nb_voxel_slice);
    ind -= (z * (float)phantom_mat.nb_voxel_slice);
    y = floor(ind / (float)(phantom_mat.size_in_vox.x));
    x = ind - y * (float)phantom_mat.size_in_vox.x;


    // random position inside the voxel
    x += Brent_real(id, photons.table_x_brent, 0);
    y += Brent_real(id, photons.table_x_brent, 0);
    z += Brent_real(id, photons.table_x_brent, 0);

    // must be in mm
    x *= phantom_mat.voxel_size.x;
    y *= phantom_mat.voxel_size.y;
    z *= phantom_mat.voxel_size.z;

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
    photons.eventID[id] = id;
    photons.trackID[id] = 0;
}


/***********************************************************
 * Tracking Kernel
 ***********************************************************/

// Optical Photons - regular tracking
template <typename T1>
__global__ void kernel_optical_navigation_regular(StackParticle photons,
                                                  Volume<T1> phantom,
                                                  int* count_d) {
    unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (id >= photons.size) return;
    if (photons.endsimu[id]) return;

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


    //// Find next discrete interaction ///////////////////////////////////////

    // Find next discrete interaction, total_dedx and next discrete intraction distance
    float next_interaction_distance =  FLT_MAX;
    unsigned char next_discrete_process = 0; 
    float interaction_distance;
    float cross_section;

    // Mie
    cross_section = Mie_CS(mat, energy); 
    interaction_distance = __fdividef(-__logf(Brent_real(id, photons.table_x_brent, 0)),
                                     cross_section);
    if (interaction_distance < next_interaction_distance) {
       next_interaction_distance = interaction_distance;
       next_discrete_process = OPTICALPHOTON_MIE;
    }

    // Distance to the next voxel boundary (raycasting)
    interaction_distance = get_boundary_voxel_by_raycasting(index_phantom, position, 
                                                            direction, phantom.voxel_size);
    // Overshoot the distance to the particle inside the next voxel
    interaction_distance += 1.0e-06f; // 1um
    if (interaction_distance < next_interaction_distance) {
      next_interaction_distance = interaction_distance;
      next_discrete_process = OPTICALPHOTON_BOUNDARY_VOXEL;
    }

    int3 old_ind;
    old_ind.x=index_phantom.x;
    old_ind.y=index_phantom.y;
    old_ind.z=index_phantom.z;

    //// Move particle //////////////////////////////////////////////////////

    position.x += direction.x * next_interaction_distance;
    position.y += direction.y * next_interaction_distance;
    position.z += direction.z * next_interaction_distance;
    

    // Dirty part FIXME
    //   apply "magnetic grid" on the particle position due to aproximation 
    //   from the GPU (on the next_interaction_distance).
    /*
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
    */

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

    /*
        index_phantom.x = int(position.x * ivoxsize.x);
        index_phantom.y = int(position.y * ivoxsize.y);
        index_phantom.z = int(position.z * ivoxsize.z);
        index_phantom.w = index_phantom.z*phantom.nb_voxel_slice
                     + index_phantom.y*phantom.size_in_vox.x
                     + index_phantom.x; // linear index

        T1 old_mat = mat;
        mat = phantom.data[index_phantom.w];
   
    printf("%i %f %f %f next %i %f - %i %i %i - %i %i %i - %i %i\n", id,
            position.x, position.y, position.z, 
            next_discrete_process, next_interaction_distance,
            old_ind.x, old_ind.y, old_ind.z,
            index_phantom.x, index_phantom.y, index_phantom.z,
            old_mat, mat);
    */
    
    // Resolve discrete processes
    if (next_discrete_process == OPTICALPHOTON_BOUNDARY_VOXEL) {
        
        // Check the change of material for Fresnel
        index_phantom.x = int(position.x * ivoxsize.x);
        index_phantom.y = int(position.y * ivoxsize.y);
        index_phantom.z = int(position.z * ivoxsize.z);
        index_phantom.w = index_phantom.z*phantom.nb_voxel_slice
                     + index_phantom.y*phantom.size_in_vox.x
                     + index_phantom.x; // linear index

        T1 old_mat = mat;
        mat = phantom.data[index_phantom.w];
    
        if (old_mat != mat) {
            // Fresnel
            //printf("%i FRESNEL\n", id);
            Fresnel_process(photons, id, old_mat, mat);
            /*
		float3 dir = Fresnel_process(stackgamma, id, &mat_initial, mat);
		float3 toto = make_float3(stackgamma.dx[id], stackgamma.dy[id], stackgamma.dz[id]);
		toto = deflect_particle(toto, dir);
		stackgamma.dx[id] = toto.x;
		stackgamma.dy[id] = toto.y;
		stackgamma.dz[id] = toto.z;*/

        }
    }

    if (next_discrete_process == OPTICALPHOTON_MIE) {
        // Mie scattering
        Mie_scatter(photons, id, mat);
    }
}


