#include "GateGPUCollimIO.hh"
#include <vector>

void GPU_GateSPECT_init(const GateGPUCollimIO_Input *input, Colli &colli_d, 
						CoordHex2 &centerOfHexagons_h, CoordHex2 &centerOfHexagons_d, 
						StackParticle &photons_d, StackParticle &photons_h, Materials &materials_d,
						unsigned int nb_of_particles, unsigned int nb_of_hexagons, unsigned int seed) {

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
    
    // Center of Hexagons
    
    Hexa_host_malloc(centerOfHexagons_h, nb_of_hexagons);
    printf(" :: Center Hexagons init\n");
    
    // CubArrayRep
    for( int i = 0; i < input->CubRepNumZ; ++i )
    {
    	for( int j = 0; j < input->CubRepNumY; ++j )   
       	{	
       		int index = 2*i*input->CubRepNumY - i;
       		centerOfHexagons_h.y[ index + j ] =
            	( ( ( input->CubRepNumY - 1.0 ) / 2.0 ) - j ) * input->CubRepVecY;
            	
            centerOfHexagons_h.z[ index + j ] =
            	( ( ( input->CubRepNumZ - 1.0 ) / 2.0 ) - i ) * input->CubRepVecZ;
            
            //printf("index cub %d \n", (index + j));
            //printf("y %f z %f \n", centerOfHexagons_h.y[ index + j ], centerOfHexagons_h.z[ index + j ]);
        }
    }
    
    // LinearRep
    for( int i = 0; i < input->CubRepNumZ - 1; ++i )
    {
    	for( int j = 0; j < input->CubRepNumY - 1; ++j )   
       	{	
       		int index = (2*i*input->CubRepNumY - i) + j;
       		
       		centerOfHexagons_h.y[ index + input->CubRepNumY ] = 
       			centerOfHexagons_h.y[ index ] - input->LinRepVecY;
       			
            centerOfHexagons_h.z[ index + input->CubRepNumY ] = 
            	centerOfHexagons_h.z[ index ] - input->LinRepVecZ;
            	
             //printf("index lin %d \n", (index + input->CubRepNumY));
             //printf("y %f z %f \n", centerOfHexagons_h.y[ index + input->CubRepNumY ], 
             //				centerOfHexagons_h.z[ index + input->CubRepNumY ]);
        }
       
    }
    
    /*
    for (int i = 0; i < nb_of_hexagons; ++i)
    {
    	printf("center %f %f \n", centerOfHexagons_h.y[ i ], centerOfHexagons_h.z[ i ]); 
    }*/
    
    Hexa_device_malloc(centerOfHexagons_d, nb_of_hexagons);
    Hexa_copy_host2device(centerOfHexagons_h, centerOfHexagons_d);
    printf(" :: Center Hexagons load\n");
    
    
    colli_d.size_x = input->size_x;
    colli_d.size_y = input->size_y;
    colli_d.size_z = input->size_z;
    
    colli_d.HexaRadius = input->HexaRadius;
    colli_d.HexaHeight = input->HexaHeight;
    
    colli_d.CubRepNumY = input->CubRepNumY;
    colli_d.CubRepNumZ = input->CubRepNumZ;
    
    colli_d.CubRepVecX = input->CubRepVecX;
    colli_d.CubRepVecY = input->CubRepVecY;
    colli_d.CubRepVecZ = input->CubRepVecZ;
    
    colli_d.LinRepVecX = input->LinRepVecX;
    colli_d.LinRepVecY = input->LinRepVecY;
    colli_d.LinRepVecZ = input->LinRepVecZ;
    
    printf(" :: colli host init\n");

    Hexa_host_free(centerOfHexagons_h);
    printf(" :: Center Hexagons cpu delete\n"); 

}

void GPU_GateSPECT_end(CoordHex2 &centerOfHexagons_d, StackParticle &photons_d, StackParticle &photons_h,
						Materials &materials_d) {

    // free memory
    stack_device_free(photons_d);
    stack_host_free(photons_h);
    materials_device_free(materials_d);
    Hexa_device_free(centerOfHexagons_d);
        
    cudaThreadExit();
}

#define EPS 1.0e-03f
void GPU_GateSPECT(Colli &colli_d, CoordHex2 &centerOfHexagons_h, CoordHex2 &centerOfHexagons_d,
					StackParticle &photons_d, StackParticle &photons_h, 
					Materials &materials_d, unsigned int nb_of_particles) {

    printf(" :: Start tracking\n");

    // TIMING
    double t_g = time();

    // Copy particles from host to device
    stack_copy_host2device(photons_h, photons_d);
    printf(" :: Load particles from GATE\n");
    
    // Kernel vars
    dim3 threads, grid;
    //int block_size = 512;
    int block_size = 128;
    int grid_size = (nb_of_particles + block_size - 1) / block_size;
    threads.x = block_size;
    grid.x = grid_size;
    
    //printf("threads %d grid %d \n", threads.x, grid.x);

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
        //kernel_NavRegularPhan_Photon_NoSec<<<grid, threads>>>(photons_d, phantom_d, 
          //                                                    materials_d, count_d);

		kernel_NavHexaColli_Photon_NoSec<<<grid, threads>>>(photons_d, colli_d, centerOfHexagons_d, 
															materials_d, count_d);

        // get back the number of simulated photons
        cudaMemcpy(&count_h, count_d, sizeof(int), cudaMemcpyDeviceToHost);

        //printf("sim %i %i / %i tot\n", step, count_h, nb_of_particles);
        
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



