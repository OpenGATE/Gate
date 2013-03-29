#include "GateGPUIO.hh"
#include <vector>


void GPU_GatePhotRadThera_init(const GateGPUIO_Input *input, Dosimetry &dose_d,
                               Materials &materials_d, Volume &phantom_d,
                               StackParticle &photons_d, StackParticle &electrons_d,
                               StackParticle &photons_h, 
                               unsigned int nb_of_particles, unsigned int seed) {
    // Select a GPU
    cudaSetDevice(input->cudaDeviceID);

    // Init rand
    srand(seed);
    
    // Photons and electrons Stacks
    StackParticle electrons_h;
    stack_device_malloc(photons_d, nb_of_particles);
    stack_device_malloc(electrons_d, nb_of_particles);
    stack_host_malloc(photons_h, nb_of_particles);
    stack_host_malloc(electrons_h, nb_of_particles);
    
    // init electrons stack
    int i=0; while(i<nb_of_particles) {
        electrons_h.E[i]  = 0.0f;
        electrons_h.dx[i] = 0.0f;
        electrons_h.dy[i] = 0.0f;
        electrons_h.dz[i] = 0.0f;
        electrons_h.px[i] = 0.0f;
        electrons_h.py[i] = 0.0f;
        electrons_h.pz[i] = 0.0f;
        electrons_h.t[i]  = 0.0f;
        electrons_h.eventID[i] = 0;
        electrons_h.trackID[i] = 0;
        electrons_h.type[i] = 11; // G4_electron 
        electrons_h.seed[i] = rand();
        electrons_h.endsimu[i] = 0;
        electrons_h.active[i] = 0;
        ++i;
    }
    stack_copy_host2device(electrons_h, electrons_d);
    
    // init electrons PRNG
    dim3 threads, grid;
    int block_size = 512;
    int grid_size = (nb_of_particles + block_size - 1) / block_size;
    threads.x = block_size;
    grid.x = grid_size;
    kernel_brent_init<<<grid, threads>>>(electrons_d);
    printf(" :: Stack init\n");
   
    // Dosemap
    Dosimetry dose_h;

    dose_h.size_in_mm = make_float3(input->phantom_size_x*input->phantom_spacing_x,
                                  input->phantom_size_y*input->phantom_spacing_y,
                                  input->phantom_size_z*input->phantom_spacing_z);
    dose_h.voxel_size = make_float3(input->phantom_spacing_x,
                                  input->phantom_spacing_y,
                                  input->phantom_spacing_z);
    dose_h.size_in_vox = make_int3(input->phantom_size_x,
                                 input->phantom_size_y,
                                 input->phantom_size_z);
    dose_h.nb_voxel_slice = dose_h.size_in_vox.x * dose_h.size_in_vox.y;
    dose_h.nb_voxel_volume = dose_h.nb_voxel_slice * dose_h.size_in_vox.z;
    dose_h.mem_data = dose_h.nb_voxel_volume * sizeof(float);

    dosimetry_host_malloc(dose_h, dose_h.nb_voxel_volume);
    dosimetry_device_malloc(dose_d, dose_h.nb_voxel_volume);

    dosimetry_host_reset(dose_h);
    dosimetry_copy_host2device(dose_h, dose_d);
    printf(" :: GPU Dosemap init\n");
    
    // Materials
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
    materials_h.rad_length = input->rad_length;
    materials_h.fX0 = input->fX0;
    materials_h.fX1 = input->fX1;
    materials_h.fD0 = input->fD0;
    materials_h.fC = input->fC;
    materials_h.fA = input->fA;
    materials_h.fM = input->fM;

    materials_device_malloc(materials_d, input->nb_materials, input->nb_elements_total);
    materials_copy_host2device(materials_h, materials_d);
    printf(" :: Materials init\n");
    
    // Phantoms
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
    cudaMemcpy(phantom_d.data, &(input->phantom_material_data[0]), phantom_d.mem_data, cudaMemcpyHostToDevice);
    printf(" :: Phantoms init\n");

    // TODO free memory
}


void GPU_GatePhotRadThera_end(Dosimetry &dosemap_d, Materials &materials_d, Volume &phantom_d,
                              StackParticle &photons_d, StackParticle &electrons_d,
                              StackParticle &photons_h) {
    // Dosemap
    Dosimetry dosemap_h;

    dosemap_h.size_in_mm = dosemap_d.size_in_mm; 
    dosemap_h.voxel_size = dosemap_d.voxel_size;
    dosemap_h.size_in_vox = dosemap_d.size_in_vox;
    dosemap_h.nb_voxel_slice = dosemap_d.nb_voxel_slice;
    dosemap_h.nb_voxel_volume = dosemap_d.nb_voxel_volume;
    dosemap_h.mem_data = dosemap_d.nb_voxel_volume * sizeof(float);
    dosimetry_host_malloc(dosemap_h, dosemap_h.nb_voxel_volume);
    
    dosimetry_copy_device2host(dosemap_d, dosemap_h);
    dosimetry_dump(dosemap_h);
    printf(" :: Dosemap saved\n");
    
    // free memory
    materials_device_free(materials_d);
    volume_device_free(phantom_d);
    dosimetry_device_free(dosemap_d);
    
    stack_device_free(photons_d);
    stack_device_free(electrons_d);
    stack_host_free(photons_h);
    
    cudaThreadExit();
}

#define EPS 1.0e-03f
void GPU_GatePhotRadThera(Dosimetry &dosemap_d, Materials &materials_d,
                          Volume &phantom_d,
                          StackParticle &photons_d, StackParticle &electrons_d, 
                          StackParticle &photons_h, unsigned int nb_of_particles) {

    printf(" :: Start tracking\n");
    // FIXME
    float step_limiter = 1000.0f; // mm

    // TIMING
    double t_init = time();
    double t_g = time();

    // Copy paticles from host to device
    stack_copy_host2device(photons_h, photons_d);

    // Kernel vars
    dim3 threads, grid;
    int block_size = 512;
    int grid_size = (nb_of_particles + block_size - 1) / block_size;
    threads.x = block_size;
    grid.x = grid_size;

    // Init random
    kernel_brent_init<<<grid, threads>>>(photons_d);

    // Count simulated photons
    int *count_phot_d, *count_elec_d;
    int count_phot_h = 0, count_elec_h = 0;
    cudaMalloc((void**) &count_phot_d, sizeof(int));
    cudaMalloc((void**) &count_elec_d, sizeof(int));
    cudaMemcpy(count_phot_d, &count_phot_h, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(count_elec_d, &count_elec_h, sizeof(int), cudaMemcpyHostToDevice);

    // TIMING
    double t_track = time();

    //double ta, tb;

    // Simulation loop
    int step=0;
    while (count_phot_h < nb_of_particles) {
        ++step;

        //ta = time();
        // Regular photon navigator
        kernel_NavRegularPhan_Photon_WiSec<<<grid, threads>>>(photons_d, electrons_d, 
                                                              phantom_d, materials_d, 
                                                              dosemap_d,
                                                              count_phot_d, step_limiter);
        //cudaThreadSynchronize();
        //ta = time() - ta;

        //tb = time();
        // Regular electron navigator
        kernel_NavRegularPhan_Electron_BdPhoton<<<grid, threads>>>(electrons_d, photons_d,
                                                                   phantom_d, materials_d,
                                                                   dosemap_d,
                                                                   count_elec_d, step_limiter);
        //cudaThreadSynchronize();
        //tb = time() - tb;

        // get back the number of simulated photons
        cudaMemcpy(&count_phot_h, count_phot_d, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&count_elec_h, count_elec_d, sizeof(int), cudaMemcpyDeviceToHost);

        //printf("sim %i phot %i/%i e- %i\n", step, count_phot_h, 
        //        nb_of_particles, count_elec_h);

        if (step > 2000) {
            printf("WARNING - GPU reachs max step\n");
            break;
        }
    }

    // TIMING
    t_track = time() - t_track;

    cudaFree(count_phot_d);
    cudaFree(count_elec_d);

    t_g = time() - t_g;
    printf(">> GPU: track %e tot %e\n", t_track, t_g);

}
#undef EPS



