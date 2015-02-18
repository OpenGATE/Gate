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
 * Optical Photons Physics Effects
 ***********************************************************/

// Compute the total Mie cross section for a given material
// !!! This code works if we have only 3 [energy,scattering-length] couples in the Gateoptical_cst.cu file !!!

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

}  // Compute the total Mie cross section for a given material


// Mie Scatter (Henyey-Greenstein approximation)
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
}  // Mie Scatter (Henyey-Greenstein approximation)


// Surface effects - Fresnel

// Compute the Fresnel reflectance (from MCML code)
__device__ float2 RFresnel(float n_incident, /* incident refractive index.*/
		           float n_transmit, /* transmit refractive index.*/
			   float c_incident_angle) /* cosine of the incident angle. 0<a1<90 degrees. */
{
  float r;
  float c_transmission_angle;
  
  if(n_incident==n_transmit) {			/** matched boundary. **/
    c_transmission_angle = c_incident_angle;
    r = 0.0f;
    // printf("case (n_incident==n_transmit): reflectance= %f and c_transmission_angle= %f \n", r, c_transmission_angle);
  }

  else if(c_incident_angle > COSZERO) {	/** normal incident. **/
    c_transmission_angle = c_incident_angle;
    r = __fdividef(n_transmit-n_incident, n_transmit+n_incident);
    r *= r;
    // printf("case (normal incident): reflectance= %f and c_transmission_angle= %f \n", r, c_transmission_angle);
  }
  else if(c_incident_angle < COS90D)  {	/** very slant. **/
    c_transmission_angle = 0.0f;
    r = 1.0f;
    // printf("case (very slant): reflectance= %f and c_transmission_angle= %f \n", r, c_transmission_angle);
  }

  else  {		/** general. **/
    float sa1, sa2;	/* sine of the incident and transmission angles. */
    float ca2;
    
    sa1 = sqrtf(1.0f-c_incident_angle*c_incident_angle);
    sa2 = __fdividef(n_incident*sa1, n_transmit);
    if(sa2 >= 1.0f) { 	/* double check for total internal reflection. */
      c_transmission_angle = 0.0f;
      r = 1.0f;
    // printf("case (total internal reflection): reflectance= %f and c_transmission_angle= %f \n", r, c_transmission_angle);
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

    // printf("case (general case): reflectance= %f and c_transmission_angle= %f \n", r, c_transmission_angle);
    }
  }


   //printf("RFresnel result: ni= %f nt= %f r= %f c_transmission_angle= %f\n", n_incident, n_transmit, r, c_transmission_angle);
  
   return make_float2(r, c_transmission_angle);
}
// Fresnel Reflectance

// Fresnel Processes
// !!! This code works when the surface Normal is the y-axis !!!
__device__ float3 Fresnel_process(StackParticle photon, unsigned int id, 
                                  unsigned short int mat_i, unsigned short int mat_t) { 

  float uy = photon.dy[id]; /* !!!! y !!!! directional cosine. */

/* S.JAN - DEBUG july 2014: reverse the material Rindex value to be coherent with the Geant4 convention */
  float ni= mat_Rindex[mat_t];
  float nt= mat_Rindex[mat_i];
  float2 res;

  // printf("step1 of Fresnel_process: ni= %f nt= %f y directional cosine = %f\n", ni, nt, uy);

  /* Get reflectance */
  if (uy>0.0) {
  res = RFresnel(ni, nt, uy); // res.x=reflectance  res.y= y-directional cosine of transmission angle 
   }
   else {
  res = RFresnel(ni, nt, -uy);
  }

  // printf("step2 of Fresnel_process: Get RFresnel value =  %f and photon transmission angle= %f\n", res.x, res.y);

  if (Brent_real(id, photon.table_x_brent, 0) > res.x) {	/* transmitted */
      		photon.dx[id] *= __fdividef(ni, nt);
      		photon.dz[id] *= __fdividef(ni, nt);

		if (uy>0.0) {
			photon.dy[id] = res.y;
		}
		else {
			photon.dy[id] = -res.y;
		}

  // printf("step3 of Fresnel_process - photon transmitted (dx,dy,dz) =  %f %f %f\n", photon.dx[id], photon.dy[id], photon.dz[id]);
  }
  else {						/* reflected. */
      photon.dy[id] = -uy;

  // printf("step4 of Fresnel_process - else photon reflected (dx,dy,dz) =  %f %f %f\n", photon.dx[id], photon.dy[id], photon.dz[id]);

  }

}  // Fresnel Processes

/***********************************************************
 * Source
 ***********************************************************/

__global__ void kernel_optical_voxelized_source(StackParticle photons, 
                                                Volume phantom_mat,
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
    photons.type[id] = OPTICALPHOTON;
    photons.active[id] = 1;
    photons.eventID[id] = id;
    photons.trackID[id] = 0;
}


/***********************************************************
 * Tracking Kernel
 ***********************************************************/

// Optical Photons - regular tracking
// This is the Regular Navigator - same as in GateCommon_fun.cu (kernel_NavRegularPhan_Photon_NoSec)
// Regular Navigator with voxelized phantom for photons without secondary

__global__ void kernel_optical_navigation_regular(StackParticle photons, Volume phantom, int* count_d) {

    unsigned int id = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (id >= photons.size) return;
    if (photons.endsimu[id]) return;

    //// Init ///////////////////////////////////////////////////////////////////

    // Read position
    float3 position; // mm
    position.x = photons.px[id];
    position.y = photons.py[id];
    position.z = photons.pz[id];

  // printf("step0 of Navigator particle id = %i and  position (x,y,z) =  %f %f %f\n", id, photons.px[id], photons.py[id], photons.pz[id]);

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

  // printf("step1 of Navigator particle id = %i and direction (dx,dy,dz) =  %f %f %f\n", id, photons.dx[id], photons.dy[id], photons.dz[id]);


    // Get energy
    float energy = photons.E[id];

    // Get material
    unsigned short int mat = phantom.data[index_phantom.w];

  // printf("step2 of Navigator Get id %i and Material %i\n", id, mat);

    //// Find next discrete interaction ///////////////////////////////////////

    // Find next discrete interaction, total_dedx and next discrete intraction distance
    float next_interaction_distance =  FLT_MAX;
    unsigned char next_discrete_process = 0; 
    float interaction_distance;
    float cross_section;

    // Mie scattering 
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
  // printf("step3 of Navigator Distance to next boundary =  %f\n", interaction_distance);

    // Overshoot the distance to the particle inside the next voxel
    if (interaction_distance < next_interaction_distance) {
      next_interaction_distance = interaction_distance+1.0e-04f;

      
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

  // printf("step4 of Navigator Move particle id = %i and (x,y,z) =  %f %f %f and (dx,dy,dz) = %f %f %f \n", id, photons.px[id], photons.py[id], photons.pz[id], direction.x, direction.y, direction.z );

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
   
    // printf("%i %f %f %f next %i %f - %i %i %i - %i %i %i - %i %i\n", id,
            position.x, position.y, position.z, 
            next_discrete_process, next_interaction_distance,
            old_ind.x, old_ind.y, old_ind.z,
            index_phantom.x, index_phantom.y, index_phantom.z,
            old_mat, mat);
    */
    
 /*DEBUG SEB begin----*/   if (next_discrete_process == OPTICALPHOTON_BOUNDARY_VOXEL) {
        
        // Check the change of material for Fresnel
        index_phantom.x = int(position.x * ivoxsize.x);
        index_phantom.y = int(position.y * ivoxsize.y);
        index_phantom.z = int(position.z * ivoxsize.z);
        index_phantom.w = index_phantom.z*phantom.nb_voxel_slice
                     + index_phantom.y*phantom.size_in_vox.x
                     + index_phantom.x; // linear index

        unsigned short int old_mat = mat;
        mat = phantom.data[index_phantom.w];

  // printf("step5 of Navigator: OPTICALPHOTON_BOUNDARY_VOXEL - Get id %i and New Material %i\n", id, mat);

        if (old_mat != mat) {

            Fresnel_process(photons, id, old_mat, mat);

  // printf("step6 of Navigator: OPTICALPHOTON_BOUNDARY_VOXEL - Fresnel_process done for id= %i : (dx,dy,dz) =  %f %f %f \n", id, photons.dx[id], photons.dy[id], photons.dz[id]);

        }
    }/*----end DEBUG SEB*/ // endif next_discrete_process == OPTICALPHOTON_BOUNDARY_VOXEL

    if (next_discrete_process == OPTICALPHOTON_MIE) {
 
        Mie_scatter(photons, id, mat);
    }

}


// Ray casting: mapping particles on the phantom

float3 back_raytrace_particle(float xi1, float yi1, float zi1, 
                              float xd, float yd, float zd) {


	float xmin, xmax, ymin, ymax, zmin, zmax;

// This is hard coded for the validation phantom that was used:
	xmin = -5.0;
	xmax = 5.0;
	ymin = -5.0;
	ymax = 5.0;
	zmin = -5.0;
	zmax = 5.0;
// hard coded.

	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	float xdi, ydi, zdi;
	float buf;

    xd = -xd;
    yd = -yd;
    zd = -zd;

    tmin = -1e9f;
    tmax = 1e9f;
    
    // on x
    if (xd != 0.0f) {
        xdi = 1.0f / xd;
        tmin = (xmin - xi1) * xdi;
        tmax = (xmax - xi1) * xdi;
        if (tmin > tmax) {
            buf = tmin;
            tmin = tmax;
            tmax = buf;
        }
    }
    // on y
    if (yd != 0.0f) {
        ydi = 1.0f / yd;
        tymin = (ymin - yi1) * ydi;
        tymax = (ymax - yi1) * ydi;
        if (tymin > tymax) {
            buf = tymin;
            tymin = tymax;
            tymax = buf;
        }
        if (tymin > tmin) {tmin = tymin;}
        if (tymax < tmax) {tmax = tymax;}
    }
    // on z
    if (zd != 0.0f) {
        zdi = 1.0f / zd;
        tzmin = (zmin - zi1) * zdi;
        tzmax = (zmax - zi1) * zdi;
        if (tzmin > tzmax) {
            buf = tzmin;
            tzmin = tzmax;
            tzmax = buf;
        }
        if (tzmin > tmin) {tmin = tzmin;}
        if (tzmax < tmax) {tmax = tzmax;}
    }

    return make_float3(xi1+xd*tmin, yi1+yd*tmin, zi1+zd*tmin);

}


