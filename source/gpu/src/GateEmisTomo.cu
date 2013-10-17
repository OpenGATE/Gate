#include "GateGPUIO.hh"
#include <vector>

void GPU_GateEmisTomo_init(const GateGPUIO_Input *input,
                           Materials &materials_d, Volume &phantom_d, Activities &activities_d,
                           StackParticle &gamma1_d, StackParticle &gamma2_d,
                           StackParticle &gamma1_h, StackParticle &gamma2_h,
                           unsigned int nb_of_particles, unsigned int seed) {

    // Select a GPU
    cudaSetDevice(input->cudaDeviceID); 
    
    // Seed managment
    srand(seed);

    // Photons Stacks
    stack_device_malloc(gamma1_d, nb_of_particles);
    stack_device_malloc(gamma2_d, nb_of_particles);
    stack_host_malloc(gamma1_h, nb_of_particles);
    stack_host_malloc(gamma2_h, nb_of_particles);
    printf(" :: Stacks init\n");

    // Materials def, alloc & loading
    Materials materials_h;
    materials_host_malloc(materials_h, input->nb_materials, input->nb_elements_total);

    materials_h.nb_elements = input->mat_nb_elements;
    materials_h.index = input->mat_index;
    materials_h.mixture = input->mat_mixture;
    materials_h.atom_num_dens = input->mat_atom_num_dens;
    materials_h.nb_atoms_per_vol = input->mat_nb_atoms_per_vol;
    materials_h.nb_electrons_per_vol = input->mat_nb_electrons_per_vol;
    materials_h.electron_cut_energy = input->electron_cut_energy;
    materials_h.electron_max_energy = input->electron_max_energy;
    materials_h.electron_mean_excitation_energy = input->electron_mean_excitation_energy;
    materials_h.fX0 = input->fX0;
    materials_h.fX1 = input->fX1;
    materials_h.fD0 = input->fD0;
    materials_h.fC = input->fC;
    materials_h.fA = input->fA;
    materials_h.fM = input->fM;

    materials_device_malloc(materials_d, input->nb_materials, input->nb_elements_total);
    materials_copy_host2device(materials_h, materials_d);
    printf(" :: Materials init\n");
    
    // Phantoms def, alloc & loading
    phantom_d.size_in_mm = make_float3(input->phantom_size_x*input->phantom_spacing_x,
                     input->phantom_size_y*input->phantom_spacing_y,
                     input->phantom_size_z*input->phantom_spacing_z);
    phantom_d.voxel_size = make_float3(input->phantom_spacing_x,
                     input->phantom_spacing_y,
                     input->phantom_spacing_z);
    phantom_d.size_in_vox = make_int3(input->phantom_size_x,
                    input->phantom_size_y,
                    input->phantom_size_z);
    phantom_d.nb_voxel_slice = phantom_d.size_in_vox.x * phantom_d.size_in_vox.y;
    phantom_d.nb_voxel_volume = phantom_d.nb_voxel_slice * phantom_d.size_in_vox.z;
    phantom_d.mem_data = phantom_d.nb_voxel_volume * sizeof(unsigned short int);
    volume_device_malloc(phantom_d, phantom_d.nb_voxel_volume); 
    cudaMemcpy(phantom_d.data, &(input->phantom_material_data[0]), 
               phantom_d.mem_data, cudaMemcpyHostToDevice);
    printf(" :: Phantom init\n");
    
    // Activities def, alloc & loading
    activities_d.nb_activities = input->activity_data.size();
    activities_d.tot_activity = input->tot_activity; 
    
    activities_device_malloc(activities_d, activities_d.nb_activities);

    cudaMemcpy(activities_d.act_index, &(input->activity_index[0]), 
               activities_d.nb_activities*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(activities_d.act_cdf, &(input->activity_data[0]), 
               activities_d.nb_activities*sizeof(float), cudaMemcpyHostToDevice);
    printf(" :: Activities init\n");

}

void GPU_GateEmisTomo_end(Materials &materials_d, Volume &phantom_d, Activities &activities_d,
                          StackParticle &gamma1_d, StackParticle &gamma2_d,
                          StackParticle &gamma1_h, StackParticle &gamma2_h) {
    // free memory
    stack_device_free(gamma1_d);
    stack_device_free(gamma2_d);
    stack_host_free(gamma1_h);
    stack_host_free(gamma2_h);

    materials_device_free(materials_d);
    volume_device_free(phantom_d);
    activities_device_free(activities_d);
    
    cudaThreadExit();
}


void GPU_GateEmisTomo(Materials &materials_d, Volume &phantom_d, Activities &activities_d,
                      StackParticle &gamma1_d, StackParticle &gamma2_d,
                      StackParticle &gamma1_h, StackParticle &gamma2_h,
                      unsigned int buffer_size) {

    printf(" :: Start GPU\n");

    // TIMING
    double tg = time();
    
    // Vars
    int gamma_max_sim = 2*buffer_size; // nb gammas = nb positrons * 2
    int* gamma_sim_d;
    int gamma_sim_h = 0;
    cudaMalloc((void**) &gamma_sim_d, sizeof(int));
    cudaMemcpy(gamma_sim_d, &gamma_sim_h, sizeof(int), cudaMemcpyHostToDevice);

    // Debug
    float* debug_d;
    cudaMalloc((void**) &debug_d, buffer_size*sizeof(float));
    float* debug_h = (float*)malloc(buffer_size*sizeof(float));
    int i=0; while(i<buffer_size) {debug_h[i] = 0.0f; ++i;}
    cudaMemcpy(debug_d, debug_h, buffer_size*sizeof(float), cudaMemcpyHostToDevice);

    // Energy
    float E = 0.511f; // MeV

    // Kernel vars
    dim3 threads, grid;
    int block_size = 128;
    int grid_size = (buffer_size + block_size - 1) / block_size;
    threads.x = block_size;
    grid.x = grid_size;

    // Init random
    int* tmp = (int*)malloc(buffer_size * sizeof(int));	
    int n=0; while (n<buffer_size) {tmp[n] = rand(); ++n;};
    cudaMemcpy(gamma1_d.seed, tmp, buffer_size * sizeof(int), cudaMemcpyHostToDevice);
    n=0; while (n<buffer_size) {tmp[n] = rand(); ++n;};
    cudaMemcpy(gamma2_d.seed, tmp, buffer_size * sizeof(int), cudaMemcpyHostToDevice);
    free(tmp);
    kernel_brent_init<<<grid, threads>>>(gamma1_d);
    kernel_brent_init<<<grid, threads>>>(gamma2_d);
    printf(" ::   Rnd ok\n");

    // Generation
    kernel_voxelized_source_b2b<<<grid, threads>>>(gamma1_d, gamma2_d, activities_d, E,
                                                   phantom_d.size_in_vox, phantom_d.voxel_size);
    cudaThreadSynchronize();
    printf(" ::   Generation ok\n");

    // Main loop
    int step=0;
    while (gamma_sim_h < gamma_max_sim) {
        ++step;
        // Regular navigator
        kernel_NavRegularPhan_Photon_NoSec<<<grid, threads>>>(gamma1_d, phantom_d,
                                                              materials_d, gamma_sim_d);
        
        kernel_NavRegularPhan_Photon_NoSec<<<grid, threads>>>(gamma2_d, phantom_d,
                                                              materials_d, gamma_sim_d);
        
        cudaThreadSynchronize();
          
        // get back the number of simulated photons
        cudaMemcpy(&gamma_sim_h, gamma_sim_d, sizeof(int), cudaMemcpyDeviceToHost);
        
        //printf("sim %i %i / %i tot\n", step, gamma_sim_h, gamma_max_sim);

        if (step >= 2000) {
            printf("WARNING - GPU reachs max step\n");
            break;
        }

    } // while

    // Get particles back to host
    stack_copy_device2host(gamma1_d, gamma1_h);
    stack_copy_device2host(gamma2_d, gamma2_h);
    printf(" ::   Tracking ok\n");

    cudaFree(gamma_sim_d);

    // TIMING
    tg = time() - tg;
    printf(" :: Stop GPU (time %e s)\n", tg);
}

