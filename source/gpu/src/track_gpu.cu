#include "fun_gpu.cu"
#include "GateSourceGPUVoxellizedIO.hh"

#include <vector>

void GateGPUGeneratePrimaries(const GateSourceGPUVoxellizedInput * input, 
                              GateSourceGPUVoxellizedOutput & output) {

	int positron = input->nb_events / 2.0f; // positron generated (nb gamma = 2*ptot) 
	unsigned short int most_att_mat = 7; // 1 Water  -  7 RibBone  FIXME add most att mat selector
	
	// Energy
	float E = input->E; // 511 keV
	long seed = input->seed;
	
	//printf("Energy = %f\n Seed = %d\n", E, seed);

	// Define phantom
	int3 dim_phantom;
	dim_phantom.z = input->phantom_size_z; // vox
	dim_phantom.y = input->phantom_size_y; // vox
	dim_phantom.x = input->phantom_size_x; // vox
	float size_voxel = input->phantom_spacing; // mm
	
	//printf("phantom_x = %d\n phantom_y = %d\n phantom_z = %d\n", dim_phantom.x, dim_phantom.y, dim_phantom.z);

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
			output.particles.push_back(particle);
			
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
			output.particles.push_back(particle);

		}
		++i;
	}
	
	printf("Simulated gamma = %d\n", gamma_max_sim);
	printf("Simulated gamma outpu = %d\n", output.particles.size());
	
	cudaThreadExit();
	free_device_stackgamma(stackgamma1);
	free_device_stackgamma(stackgamma2);
	free_host_stackgamma(phasespace1);
	free_host_stackgamma(phasespace2);

}

