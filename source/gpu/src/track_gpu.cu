#include "fun_gpu.cu"
#include "GateSourceGPUVoxellizedIO.hh"

#include <vector>

void GateGPUGeneratePrimaries(const GateSourceGPUVoxellizedInput * input, 
                              GateSourceGPUVoxellizedOutput & output) {

  /*
  //--------------------------------------------------------------------------
  // BEGIN TEST
  // DS FIXME --> test because GPU do not work on my computer

  int initialID = input->firstInitialID;

  // Send n times the same particles
  for(int n=0; n<input->nb_events/6; n++) {
    GateSourceGPUVoxellizedOutputParticle particle;
    particle.E = 0.511;
    particle.px = 256+133.46; particle.py = 126 + 81.14; particle.pz = 92 + 131.289; 
    double nn = sqrt(pow(0.0507817, 2) + pow(-0.152974,2) + pow(-0.484913,2));
    particle.dx = 0.0507817/nn; particle.dy = -0.152974/nn; particle.dz = -0.484913/nn;
    particle.t = 0.789737;  // express in ns ==> this is the particle TOF (from 0)
    particle.initialID = initialID; // First pair of particle (first photon)
    output.particles.push_back(particle);
    
    particle.E = 0.511;
    particle.px = 256+133.46; particle.py = 126 + 81.14; particle.pz = 92 + 131.289; 
    particle.dx = -0.0507817/nn; particle.dy = 0.152974/nn; particle.dz = 0.484913/nn;
    particle.t = 0.846211;
    particle.initialID = initialID;  // First pair of particle (second photon)
    output.particles.push_back(particle);
    
    particle.E = 0.511;
    particle.px = 256+186.792; particle.py = 126 + 120.485; particle.pz = 92 + 111.315; 
    nn = sqrt(pow(0.00673269, 2) + pow(0.508965,2) + pow(0.045058,2));
    particle.dx = 0.00673269/nn; particle.dy = 0.508965/nn; particle.dz = 0.045058/nn;
    particle.t = 1;
    particle.initialID = initialID+1; // Next pair of particle (first photon)
    output.particles.push_back(particle);
    
    particle.E = 0.511;
    particle.px = 256+186.792; particle.py = 126 + 120.485; particle.pz = 92 + 111.315; 
    particle.dx = -0.00673269/nn; particle.dy = -0.508965/nn; particle.dz = -0.045058/nn;
    particle.t = 2;
    particle.initialID = initialID+1;// Next pair of particle (second photon)
    output.particles.push_back(particle);
    
    particle.E = 0.511;
    particle.px = 256+186.792; particle.py = 126 + 120.485; particle.pz = 92 + 111.315; 
    nn = sqrt(pow(0.00673269, 2) + pow(0.508965,2) + pow(0.045058,2));
    particle.dx = 0.00673269/nn; particle.dy = 0.508965/nn; particle.dz = 0.045058/nn;
    particle.t = 1;
    particle.initialID = initialID+3; // Another pair of particle (first photon), let suppose that the number 2 does not go out of the volume
    output.particles.push_back(particle);
    
    particle.E = 0.511;
    particle.px = 256+186.792; particle.py = 126 + 120.485; particle.pz = 92 + 111.315; 
    particle.dx = -0.00673269/nn; particle.dy = -0.508965/nn; particle.dz = -0.045058/nn;
    particle.t = 2;
    particle.initialID = initialID+4; // Another pair of particle (first photon), let suppose that the number 3/second photon does not go out of the volume
    output.particles.push_back(particle);
    
  }

  return;
  // END TEST 
  //--------------------------------------------------------------------------
  */

	int positron = input->nb_events / 2.0f; // positron generated (nb gamma = 2*ptot) 
    unsigned short int most_att_mat = 7; // 1 Water  -  7 RibBone  FIXME add most att mat selector

	// T0 run
    int firstInitialID = input->firstInitialID;

	// Energy
	float E = input->E; // 511 keV
	long seed = input->seed + firstInitialID; // Avoid to use the same seed each time
	
    // Define phantom
	int3 dim_phantom;
	dim_phantom.z = input->phantom_size_z; // vox
	dim_phantom.y = input->phantom_size_y; // vox
	dim_phantom.x = input->phantom_size_x; // vox
	float size_voxel = input->phantom_spacing; // mm

	// Select a GPU
	cudaSetDevice(1);

	// Vars
	int gamma_sim = 0;	
	int gamma_max_sim = 2 * positron; // maximum number of particles simulated (2 gammas per positron)

	// Defined Stacks
	StackGamma stackgamma1, stackgamma2;
	init_device_stackgamma(stackgamma1, positron);
	init_device_stackgamma(stackgamma2, positron);
	StackGamma phasespace1, phasespace2;
	init_host_stackgamma(phasespace1, positron);
	init_host_stackgamma(phasespace2, positron);
	
	// Kernel vars
	dim3 threads, grid;
	int block_size = 256;
	int grid_size = (positron + block_size - 1) / block_size;
	threads.x = block_size;
	grid.x = grid_size;
	
	// Init random
	int* tmp = (int*)malloc(positron * sizeof(int));
	srand(seed);
	int n=0; while (n<positron) {tmp[n] = rand(); ++n;};
	cudaMemcpy(stackgamma1.seed, tmp, positron * sizeof(int), cudaMemcpyHostToDevice);
	n=0; while (n<positron) {tmp[n] = rand(); ++n;};
	cudaMemcpy(stackgamma2.seed, tmp, positron * sizeof(int), cudaMemcpyHostToDevice);
	free(tmp);
	kernel_brent_init<<<grid, threads>>>(stackgamma1);
	kernel_brent_init<<<grid, threads>>>(stackgamma2);

	// Load phantom in texture mem
	int nb = dim_phantom.z * dim_phantom.y * dim_phantom.x;
	unsigned int mem_phantom = nb * sizeof(unsigned short int);
	unsigned short int* dphantom;
	cudaMalloc((void**) &dphantom, mem_phantom);
	cudaMemcpy(dphantom, &(input->phantom_material_data[0]), mem_phantom, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_phantom, dphantom, mem_phantom);

	// load activities in texture mem
	int nb_act = input->activity_data.size();
	unsigned int mem_act_f = nb_act * sizeof(float);
	float* dactivities;
	cudaMalloc((void**) &dactivities, mem_act_f);
	cudaMemcpy(dactivities, &(input->activity_data[0]), mem_act_f, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_act_val, dactivities, mem_act_f);

	// load activities indices in the texture mem
	unsigned int mem_act_i = nb_act * sizeof(unsigned int);
	unsigned int* dindex;
	cudaMalloc((void**) &dindex, mem_act_i);
	cudaMemcpy(dindex, &(input->activity_index[0]), mem_act_i, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_act_ind, dindex, mem_act_i);
	cudaThreadSynchronize();
	
	// Generation
	kernel_voxelized_source_b2b<<<grid, threads>>>(stackgamma1, stackgamma2, dim_phantom, E, size_voxel);
	cudaThreadSynchronize();
	
	// Main loop
	while (gamma_sim < gamma_max_sim) {
		
		// Navigation Standard model
		kernel_woodcock_Standard<<<grid, threads>>>(dim_phantom, stackgamma1, size_voxel,
													most_att_mat);
		kernel_woodcock_Standard<<<grid, threads>>>(dim_phantom, stackgamma2, size_voxel,
													most_att_mat);
		cudaThreadSynchronize();
		
		// Interaction
		kernel_interactions<<<grid, threads>>>(stackgamma1, dim_phantom, size_voxel);
		kernel_interactions<<<grid, threads>>>(stackgamma2, dim_phantom, size_voxel);
		cudaThreadSynchronize();

		// Check simu
		get_nb_particles_simulated(stackgamma1, stackgamma2, phasespace1, phasespace2, &gamma_sim);		
		cudaThreadSynchronize();
	} // while

    // Rewind particules by mapping them to the phantom faces
    back_raytrace_phasespace(phasespace1, positron, dim_phantom, size_voxel);
    back_raytrace_phasespace(phasespace2, positron, dim_phantom, size_voxel);

	int i=0;
	while (i<positron) {
		if (phasespace1.live[i]) {
			GateSourceGPUVoxellizedOutputParticle particle;
			particle.E = phasespace1.E[i];
			particle.dx = phasespace1.dx[i];
			particle.dy = phasespace1.dy[i];
			particle.dz = phasespace1.dz[i];
			particle.px = phasespace1.px[i];
			particle.py = phasespace1.py[i];
			particle.pz = phasespace1.pz[i];
			particle.t = phasespace1.t[i];
            particle.initialID = firstInitialID + i;
			output.particles.push_back(particle);
            //printf("dum dum x %e y %e z %e\n", 
            //        phasespace1.px[i], phasespace1.py[i], phasespace1.pz[i]);
		}
		if (phasespace2.live[i]) {
			GateSourceGPUVoxellizedOutputParticle particle;
			particle.E = phasespace2.E[i];
			particle.dx = phasespace2.dx[i];
			particle.dy = phasespace2.dy[i];
			particle.dz = phasespace2.dz[i];
			particle.px = phasespace2.px[i];
			particle.py = phasespace2.py[i];
			particle.pz = phasespace2.pz[i];
			particle.t = phasespace2.t[i];
            particle.initialID = firstInitialID + i;
			output.particles.push_back(particle);
            //printf("dum dum x %e y %e z %e\n", 
            //        phasespace1.px[i], phasespace1.py[i], phasespace1.pz[i]);
		}
		++i;
	}
	
	cudaThreadExit();
	free_device_stackgamma(stackgamma1);
	free_device_stackgamma(stackgamma2);
	free_host_stackgamma(phasespace1);
	free_host_stackgamma(phasespace2);
}

