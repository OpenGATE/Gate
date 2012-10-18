#include "voxsrc_fun_gpu.cu"
#include "GateSourceGPUVoxellizedIO.hh"

#include <vector>

void GateGPUGeneratePrimaries(const GateSourceGPUVoxellizedInput * input, 
                              GateSourceGPUVoxellizedOutput & output) {


    // TIMING
    double tg = voxsrc_time();
    double tinit = voxsrc_time();

	int positron = input->nb_events / 2.0f; // positron generated (nb gamma = 2*ptot) 
    unsigned short int most_att_mat = 7; // 1 Water  -  7 RibBone  FIXME add most att mat selector

	// T0 run
    int firstInitialID = input->firstInitialID;

	// Energy
	float E = input->E; // 511 keV
    
    // Seed managment
    srand(input->seed);
	
    // Define phantom
	int3 dim_phantom;
	dim_phantom.z = input->phantom_size_z; // vox
	dim_phantom.y = input->phantom_size_y; // vox
	dim_phantom.x = input->phantom_size_x; // vox
	float size_voxel = input->phantom_spacing; // mm

	// Select a GPU
	cudaSetDevice(0);

	// Vars
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
	int n=0; while (n<positron) {tmp[n] = rand(); ++n;};
	cudaMemcpy(stackgamma1.seed, tmp, positron * sizeof(int), cudaMemcpyHostToDevice);
	n=0; while (n<positron) {tmp[n] = rand(); ++n;};
	cudaMemcpy(stackgamma2.seed, tmp, positron * sizeof(int), cudaMemcpyHostToDevice);
	free(tmp);
	kernel_voxsrc_Brent_init<<<grid, threads>>>(stackgamma1);
	kernel_voxsrc_Brent_init<<<grid, threads>>>(stackgamma2);

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
	
    // Count simulated photons
    int* gamma_sim_d;
    int gamma_sim_h = 0;
    int gamma_sim = 0;
    cudaMalloc((void**) &gamma_sim_d, sizeof(int));
    cudaMemcpy(gamma_sim_d, &gamma_sim_h, sizeof(int), cudaMemcpyHostToDevice);

    // TIMING
    tinit = voxsrc_time() - tinit;
    double tsrc = voxsrc_time();
	
	// Generation
	kernel_voxelized_source_b2b<<<grid, threads>>>(stackgamma1, stackgamma2, dim_phantom, E, size_voxel);
	//cudaThreadSynchronize();
    
    // TIMING
    tsrc = voxsrc_time() - tsrc;
    double ttrack = voxsrc_time();    
	
	// Main loop
	while (gamma_sim_h < gamma_max_sim) {
		
        //kernel_voxsrc_regular_navigator<<<grid, threads>>>(dim_phantom, stackgamma1, 
        //                                                   size_voxel, gamma_sim_d);
        
        //kernel_voxsrc_regular_navigator<<<grid, threads>>>(dim_phantom, stackgamma2, 
        //                                                   size_voxel, gamma_sim_d);

        // Navigation Standard model
		kernel_voxsrc_woodcock_Standard<<<grid, threads>>>(dim_phantom, stackgamma1, size_voxel,
													most_att_mat, gamma_sim_d);
		kernel_voxsrc_woodcock_Standard<<<grid, threads>>>(dim_phantom, stackgamma2, size_voxel,
													most_att_mat, gamma_sim_d);
		//cudaThreadSynchronize();
		
		// Interaction
		kernel_voxsrc_interactions<<<grid, threads>>>(stackgamma1, dim_phantom, size_voxel,
                                                      gamma_sim_d);
		kernel_voxsrc_interactions<<<grid, threads>>>(stackgamma2, dim_phantom, size_voxel,
                                                      gamma_sim_d);
		//cudaThreadSynchronize();

        // get back the number of simulated photons
        cudaMemcpy(&gamma_sim_h, gamma_sim_d, sizeof(int), cudaMemcpyDeviceToHost);

		//cudaThreadSynchronize();
        //printf("gamma sim %i - %i / %i\n", gamma_sim, gamma_sim_h, gamma_max_sim);

	} // while

    // TIMING
    ttrack = voxsrc_time() - ttrack;
    double tray = voxsrc_time();

    // Rewind particules by mapping them to the phantom faces
	copy_device_to_host_stackgamma(stackgamma1, phasespace1);
	copy_device_to_host_stackgamma(stackgamma2, phasespace2);
    back_raytrace_phasespace(phasespace1, positron, dim_phantom, size_voxel);
    back_raytrace_phasespace(phasespace2, positron, dim_phantom, size_voxel);

    // TIMING
    tray = voxsrc_time() - tray;
    double texport = voxsrc_time();

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
    
    // TIMING
    texport = voxsrc_time()-texport;

	free_device_stackgamma(stackgamma1);
	free_device_stackgamma(stackgamma2);
	free_host_stackgamma(phasespace1);
	free_host_stackgamma(phasespace2);
    cudaFree(gamma_sim_d);

    // TIMING
    tg = voxsrc_time() - tg;
    printf(">> GPU: init %e src %e track %e ray %e exp %e tot %e\n", tinit, tsrc, ttrack, tray, texport, tg);

	cudaThreadExit();
}

