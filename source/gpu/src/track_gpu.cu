#include "mc_fun_pet.cu"

int mainCUDA(int argc, char* argv[]) {

	/******************************************************************
	 * INIT
	 ******************************************************************/

	int positron = 5000000;   // positron generated (nb gamma = 2*ptot) 
	unsigned short int most_att_mat = 7; // 1 Water  -  7 RibBone
	// Energy
	float E = 0.511; // 511 keV
	int seed = 10;

	// Define phantom
	int3 dim_phantom;
	dim_phantom.z = 46;       // vox
	dim_phantom.y = 63;       // vox
	dim_phantom.x = 128;      // vox
	float size_voxel = 4.0f;  // mm
	const char* phantom_name = "ncat_12mat_128x63x46.bin";
	const char* activities_val_name = "ncat_act.bin";
	const char* activities_ind_name = "ncat_ind.bin";
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

	// Load phantom
	load_phantom_in_tex(phantom_name, dim_phantom);
	// load activities
	int nb_act = get_nb_active_voxel(activities_val_name);
	load_activities_in_tex(activities_val_name, activities_ind_name, nb_act);

	/************************************************************************
	 *  GPU RUN
	 ************************************************************************/

	// Generation
	kernel_voxelized_source_b2b<<<grid, threads>>>(stackgamma1, stackgamma2, nb_act, dim_phantom, E, size_voxel);
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
		
		printf("Tot particles simulated: %i\n", gamma_sim);
		
	} // while

	cudaThreadExit();
	free_device_stackgamma(stackgamma1);
	free_device_stackgamma(stackgamma2);
	free_host_stackgamma(phasespace1);
	free_host_stackgamma(phasespace2);
	
    return 0;
}

