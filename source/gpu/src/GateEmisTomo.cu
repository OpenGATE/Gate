#include "GateGPUIO.hh"

#include <vector>

void GPU_GateEmisTomo(const GateGPUIO_Input * input, GateGPUIO_Output * output) {
    printf("====> GPU START\n");

    // TIMING
    double tg = time();
    double tinit = time();

    // Select a GPU
    cudaSetDevice(input->cudaDeviceID); 

    // Vars
    int gamma_max_sim = input->nb_events;
    int positron = gamma_max_sim / 2;     // positron generated (nb gamma = 2*ptot) 
    int* gamma_sim_d;
    int gamma_sim_h = 0;
    cudaMalloc((void**) &gamma_sim_d, sizeof(int));
    cudaMemcpy(gamma_sim_d, &gamma_sim_h, sizeof(int), cudaMemcpyHostToDevice);

    // T0 run
    int firstInitialID = input->firstInitialID;

    // Energy
    float E = input->E; // 511 keV

    // Seed managment
    srand(input->seed);

    // Photons Stacks
    StackParticle gamma1_d, gamma2_d;
    stack_device_malloc(gamma1_d, positron);
    stack_device_malloc(gamma2_d, positron);

    StackParticle gamma1_h, gamma2_h;
    stack_host_malloc(gamma1_h, positron);
    stack_host_malloc(gamma2_h, positron);
    printf(" :: Stacks ok\n");

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

    Materials materials_d;
    materials_device_malloc(materials_d, input->nb_materials, input->nb_elements_total);

    materials_copy_host2device(materials_h, materials_d);
    printf(" :: Materials ok\n");

    // Phantoms def, alloc & loading
    Volume phantom_d;
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
    printf(" :: Phantom ok\n");

    // Activities def, alloc & loading
    Activities activities_d;
    activities_d.nb_activities = input->activity_data.size();
    activities_d.tot_activity = 0.0f; // FIXME not used
    
    activities_device_malloc(activities_d, activities_d.nb_activities);

    cudaMemcpy(activities_d.act_index, &(input->activity_index[0]), 
               activities_d.nb_activities*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(activities_d.act_cdf, &(input->activity_data[0]), 
               activities_d.nb_activities*sizeof(float), cudaMemcpyHostToDevice);
    printf(" :: Activities ok\n");

    // Kernel vars
    dim3 threads, grid;
    int block_size = 512;
    int grid_size = (positron + block_size - 1) / block_size;
    threads.x = block_size;
    grid.x = grid_size;

    // Init random
    int* tmp = (int*)malloc(positron * sizeof(int));	
    int n=0; while (n<positron) {tmp[n] = rand(); ++n;};
    cudaMemcpy(gamma1_d.seed, tmp, positron * sizeof(int), cudaMemcpyHostToDevice);
    n=0; while (n<positron) {tmp[n] = rand(); ++n;};
    cudaMemcpy(gamma2_d.seed, tmp, positron * sizeof(int), cudaMemcpyHostToDevice);
    free(tmp);
    kernel_brent_init<<<grid, threads>>>(gamma1_d);
    kernel_brent_init<<<grid, threads>>>(gamma2_d);
    printf(" :: Rnd ok\n");

    // TIMING
    tinit = time() - tinit;
    double tsrc = time();

    // Generation
    kernel_voxelized_source_b2b<<<grid, threads>>>(gamma1_d, gamma2_d, activities_d, E,
                                                 phantom_d.size_in_vox, phantom_d.voxel_size);
    cudaThreadSynchronize();
    printf(" :: Generation ok\n");

    /*
    stack_copy_device2host(gamma1_d, gamma1_h);
    int i=0; while(i<gamma1_d.size) {
        printf("%e %.2f %.2f %.2f %.2f %.2f %.2f\n", gamma1_h.E[i],
                gamma1_h.px[i], gamma1_h.py[i], gamma1_h.pz[i],
                gamma1_h.dx[i], gamma1_h.dy[i], gamma1_h.dz[i]);
        ++i;
    }
    exit(0);
    */

    // TIMING
    tsrc = time() - tsrc;
    double ttrack = time();    

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
        
        printf("sim %i %i / %i tot\n", step, gamma_sim_h, gamma_max_sim);

        //if (step > 100) break;

    } // while

    // TIMING
    ttrack = time() - ttrack;
    double copy = time();

    // Get particles back to host
    stack_copy_device2host(gamma1_d, gamma1_h);
    stack_copy_device2host(gamma2_d, gamma2_h);

    // TIMING
    copy = time() - copy;
    double texport = time();
   
    /*
    // Debuging
    int i=0; while(i<gamma1_d.size) {
        if (gamma1_h.active[i]) {
            printf("%e %.2f %.2f %.2f %.2f %.2f %.2f\n", gamma1_h.E[i],
                    gamma1_h.px[i], gamma1_h.py[i], gamma1_h.pz[i],
                    gamma1_h.dx[i], gamma1_h.dy[i], gamma1_h.dz[i]);
        }
        ++i;
    }
    exit(0);
    */


    int i=0;
    while (i<positron) {
        if (gamma1_h.active[i]) {
            GateGPUIO_Particle particle;
            particle.E =  gamma1_h.E[i];
            particle.dx = gamma1_h.dx[i];
            particle.dy = gamma1_h.dy[i];
            particle.dz = gamma1_h.dz[i];
            particle.px = gamma1_h.px[i];
            particle.py = gamma1_h.py[i];
            particle.pz = gamma1_h.pz[i];
            particle.t =  gamma1_h.t[i];
            particle.initialID = firstInitialID + i;
            output->particles.push_back(particle);
        }
        if (gamma2_h.active[i]) {
            GateGPUIO_Particle particle;
            particle.E =  gamma2_h.E[i];
            particle.dx = gamma2_h.dx[i];
            particle.dy = gamma2_h.dy[i];
            particle.dz = gamma2_h.dz[i];
            particle.px = gamma2_h.px[i];
            particle.py = gamma2_h.py[i];
            particle.pz = gamma2_h.pz[i];
            particle.t =  gamma2_h.t[i];
            particle.initialID = firstInitialID + i;
            output->particles.push_back(particle);
        }
        ++i;
    }

    // TIMING
    texport = time()-texport;

    stack_device_free(gamma1_d);
    stack_device_free(gamma2_d);
    stack_host_free(gamma1_h);
    stack_host_free(gamma2_h);
    materials_device_free(materials_d);
    volume_device_free(phantom_d);
    activities_device_free(activities_d);
    cudaFree(gamma_sim_d);

    // TIMING
    tg = time() - tg;
    printf(">> GPU: init %e src %e track %e copy %e exp %e tot %e\n", 
            tinit, tsrc, ttrack, copy, texport, tg);

    DD(output->particles.size());

    cudaThreadExit();
    printf("====> GPU STOP\n");
}

