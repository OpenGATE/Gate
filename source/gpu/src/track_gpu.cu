#include "fun_gpu.cu"
#include "GateSourceGPUVoxellizedIO.hh"

void GateGPUGeneratePrimaries(const GateSourceGPUVoxellizedInput * input, 
                              GateSourceGPUVoxellizedOutput & output) {
	
}

/*

	int positron = input->nb_events / 2.0f; // positron generated (nb gamma = 2*ptot) 
	unsigned short int most_att_mat = input->phantom_most_attenuate_material; // 1 Water  -  7 RibBone

	// Energy
	float E = input->E; // 511 keV
	long seed = input->seed;

	// Define phantom
	int3 dim_phantom;
	dim_phantom.z = input->phantom_size_z; // vox
	dim_phantom.y = input->phantom_size_y; // vox
	dim_phantom.x = input->phantom_size_x; // vox
	float size_voxel = input->phantom_spacing; // mm
	//const char* phantom_name = "ncat_12mat_128x63x46.bin";
	//const char* activities_val_name = "ncat_act.bin";
	//const char* activities_ind_name = "ncat_ind.bin";

	// Select a GPU
	cudaSetDevice(0);

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
	cudaMemcpy(dphantom, input->phantom_phantom_data, mem_phantom, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_phantom, dphantom, mem_phantom);

	// load activities in texture mem
	unsigned int mem_act_f = nb * sizeof(float);
	float* activities = &(input->phantom_activity_data[0]); // &phan[0]   !!! check std:vector
	float* dactivities;
	cudaMalloc((void**) &dactivities, mem_act_f);
	cudaMemcpy(dactivities, activities, mem_act_f, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_act_val, dactivities, mem_act_f);

	// load activities indices in the texture mem
	unsigned int* index = &(input->phantom_activity_index[0]); // &ind[0]  !!! check std:vector
	unsigned int* dindex; // FIXME long
	cudaMalloc((void**) &dindex, mem_act_i);
	cudaMemcpy(dindex, , mem_act_i, cudaMemcpyHostToDevice);
	cudaBindTexture(NULL, tex_act_ind, dindex, mem_act_i);


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


	int i=0;
	while (i<gamma_max_sim) {
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
			output.push_back(particle);
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
			output.push_back(particle);
		}
		++i;
	}
	
	cudaThreadExit();
	free_device_stackgamma(stackgamma1);
	free_device_stackgamma(stackgamma2);
	free_host_stackgamma(phasespace1);
	free_host_stackgamma(phasespace2);
}

*/
