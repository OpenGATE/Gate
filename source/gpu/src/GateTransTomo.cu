#include "GateGPUIO.hh"
#include <vector>

void GPU_GateTransTomo_init(const GateGPUIO_Input *input,
                            Materials &materials_d, Volume &phantom_d,
                            StackParticle &photons_d, StackParticle &photons_h,
                            unsigned int nb_of_particles, unsigned int seed) {

    // Select a GPU
    cudaSetDevice(input->cudaDeviceID);

    // Seed management
    srand(seed);

    // Photons Stacks
    stack_device_malloc(photons_d, nb_of_particles);
    stack_host_malloc(photons_h, nb_of_particles);
    printf(" :: Stack init\n");

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

void GPU_GateTransTomo_end(Materials &materials_d, Volume &phantom_d,
                           StackParticle &photons_d, StackParticle &photons_h) {

    // free memory
    stack_device_free(photons_d);
    stack_host_free(photons_h);
    
    materials_device_free(materials_d);
    volume_device_free(phantom_d);
    
    cudaThreadExit();
}

#define EPS 1.0e-03f
void GPU_GateTransTomo(Materials &materials_d, Volume &phantom_d,
                       StackParticle &photons_d, StackParticle &photons_h,
                       unsigned int nb_of_particles) {

    printf(" :: Start tracking\n");

    // TIMING
    double t_g = time();

    // Copy particles from host to device
    stack_copy_host2device(photons_h, photons_d);
    printf(" :: Load particles from GATE\n");

    // Kernel vars
    dim3 threads, grid;
    int block_size = 512;
    int grid_size = (nb_of_particles + block_size - 1) / block_size;
    threads.x = block_size;
    grid.x = grid_size;

    // Init random
    kernel_brent_init<<<grid, threads>>>(photons_d);

    // Count simulated photons
    int* count_d;
    int count_h = 0;
    cudaMalloc((void**) &count_d, sizeof(int));
    cudaMemcpy(count_d, &count_h, sizeof(int), cudaMemcpyHostToDevice);

    // Simulation loop
    int step=0;
    while (count_h < nb_of_particles) {
        ++step;
        // Regular navigator
        kernel_NavRegularPhan_Photon_NoSec<<<grid, threads>>>(photons_d, phantom_d, 
                                                              materials_d, count_d);

        // get back the number of simulated photons
        cudaMemcpy(&count_h, count_d, sizeof(int), cudaMemcpyDeviceToHost);

        printf("sim %i %i / %i tot\n", step, count_h, nb_of_particles);
        
        if (step > 2000) {
            printf("WARNING - GPU reachs max step\n");
            break;
        }

    }

    // Copy photons from device to host
    stack_copy_device2host(photons_d, photons_h);

    cudaFree(count_d);

    t_g = time() - t_g;
    printf(">> GPU: tot time %e\n", t_g);

    printf("====> GPU STOP\n");
}
#undef EPS



