#include "actor_fun_gpu.cu"
#include "GateGPUIO.hh"

#include <vector>

void GateTrackingGPUActorTrack_RT(const GateTrackingGPUActorInput * input, 
                                  GateTrackingGPUActorOutput * output) {

  // FIXME
  // add ecut, emax, meanexcitation
  // add density correction coefficient
  // track event ID
  // init phantom
  printf("Gpu start\n");
    
  // Select a GPU
  cudaSetDevice(input->cudaDeviceID);
  // DD(input->cudaDeviceID);
    
  //int firstInitialID = input->firstInitialID;
  long seed = input->seed;
  DD(seed);
  srand(seed);

  // Vars
  int particle_simulated = 0;
  int nb_of_particles = input->particles.size();

  // Photons Stacks
  StackParticle photons_d;
  stack_device_malloc(photons_d, nb_of_particles);
  StackParticle photons_h;
  stack_host_malloc(photons_h, nb_of_particles);
	
  // Electrons Stacks
  StackParticle electrons_d;
  stack_device_malloc(electrons_d, nb_of_particles);
  StackParticle electrons_h;
  stack_host_malloc(electrons_h, nb_of_particles);

  // DD(input->phantom_size_x);
  // DD(input->phantom_size_y);
  // DD(input->phantom_size_z);

  // DD(input->phantom_spacing_x);
  // DD(input->phantom_spacing_y);
  // DD(input->phantom_spacing_z);

  // Fill photons stack with particles from GATE
  int i = 0;
  GateTrackingGPUActorInput::ParticlesList::const_iterator iter = input->particles.begin();
  while (iter != input->particles.end()) {
    // DD(i);
    GateTrackingGPUActorParticle p = *iter;
    photons_h.E[i] = p.E;
	DD(p.E);
    photons_h.dx[i] = p.dx;
    photons_h.dy[i] = p.dy;
    photons_h.dz[i] = p.dz;
    // FIXME need to change the world frame
    photons_h.px[i] = p.px + (input->phantom_size_x/2.0)*input->phantom_spacing_x;
    photons_h.py[i] = p.py + (input->phantom_size_y/2.0)*input->phantom_spacing_y;
    photons_h.pz[i] = p.pz + (input->phantom_size_z/2.0)*input->phantom_spacing_z;
    /* DD(photons_h.px[i]); */
    /* DD(photons_h.py[i]); */
    /* DD(photons_h.pz[i]); */

    /* DD(photons_h.dx[i]); */
    /* DD(photons_h.dy[i]); */
    /* DD(photons_h.dz[i]); */

    photons_h.t[i] = p.t;
    photons_h.eventID[i] = p.eventID;
    photons_h.trackID[i] = p.trackID;
    photons_h.type[i] = p.type; // FIXME
    photons_h.seed[i] = rand();

    photons_h.endsimu[i] = 0;
    photons_h.active[i] = 1;
    photons_h.interaction[i] = 1;

    electrons_h.seed[i] = rand();
    electrons_h.active[i] = 0;
    electrons_h.endsimu[i] = 0;
    electrons_h.interaction[i] = 0;
		
    ++iter;
    ++i;
  }
  // Copy particles from host to device
  stack_copy_host2device(photons_h, photons_d);
  stack_copy_host2device(electrons_h, electrons_d); // FIXME add seed copy
  cudaMemcpy(electrons_d.seed, electrons_h.seed, nb_of_particles*sizeof(int), cudaMemcpyHostToDevice);
	
  // Kernel vars
  dim3 threads, grid;
  int block_size = 512;
  int grid_size = (nb_of_particles + block_size - 1) / block_size;
  threads.x = block_size;
  grid.x = grid_size;
	
  // Init random
  kernel_brent_init<<<grid, threads>>>(photons_d);
  kernel_brent_init<<<grid, threads>>>(electrons_d);

  // Phantoms
  Volume<unsigned short int> phantom_d;
  // DD("here");
  phantom_d.most_att_data = 1;
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
  // DD(phantom_d.nb_voxel_slice);
  phantom_d.nb_voxel_volume = phantom_d.nb_voxel_slice * phantom_d.size_in_vox.z;
  // DD(phantom_d.nb_voxel_volume);
  phantom_d.mem_data = phantom_d.nb_voxel_volume * sizeof(unsigned short int);
  // DD(phantom_d.mem_data);
  volume_device_malloc<unsigned short int>(phantom_d, phantom_d.nb_voxel_volume); 
  // DD(input->phantom_material_data.size());
  // DD("before copy");
  cudaMemcpy(phantom_d.data, &(input->phantom_material_data[0]), phantom_d.mem_data, cudaMemcpyHostToDevice);
  // DD("after copy");

  // Count simulated photons
  int* count_d;
  int count_h = 0;
  cudaMalloc((void**) &count_d, sizeof(int));
  cudaMemcpy(count_d, &count_h, sizeof(int), cudaMemcpyHostToDevice);

  // Simualtion loop
  int step = 0;
  while (count_h < nb_of_particles) {
    // DD(step);
    ++step;

    /*
    // photons
    kernel_photons_woodcock<unsigned short int><<<grid, threads>>>(photons_d, phantom_d, count_d);
    kernel_photons_interactions<unsigned short int><<<grid, threads>>>(photons_d, electrons_d,phantom_d, count_d);

    // electrons
    kernel_electrons_woodcock<unsigned short int><<<grid, threads>>>(photons_d, electrons_d, phantom_d);
    kernel_electrons_interactions<unsigned short int><<<grid, threads>>>(photons_d, electrons_d, phantom_d);
    */

    kernel_photons_classical<unsigned short int><<<grid, threads>>>(photons_d, electrons_d, 
                                                            phantom_d, count_d);
        
    kernel_electrons_classical<unsigned short int><<<grid, threads>>>(photons_d, electrons_d, 
                                                            phantom_d);

    // get back the number of simulated photons
    cudaMemcpy(&count_h, count_d, sizeof(int), cudaMemcpyDeviceToHost);
    // DD(count_h);
  }

  // Copy photons from device to host
  stack_copy_device2host(photons_d, photons_h);
    

  /* 
     size_x = [ 61 ]
     [Core-0] input->phantom_size_y = [ 79 ]
     [Core-0] input->phantom_size_z = [ 16 ]
     [Core-0] input->phantom_spacing_x = [ 5 ]
     [Core-0] input->phantom_spacing_y = [ 5 ]
     [Core-0] input->phantom_spacing_z = [ 5 ]
  */


  i=0;
  /* DD("End kernel"); */
  while (i<nb_of_particles) {
    GateTrackingGPUActorParticle particle;
    printf("e = %e\n", photons_h.E[i]);
    particle.E =  photons_h.E[i];
    particle.dx = photons_h.dx[i];
    particle.dy = photons_h.dy[i];
    particle.dz = photons_h.dz[i];

    /* DD(photons_h.dx[i]); */
    /* DD(photons_h.dy[i]); */
    /* DD(photons_h.dz[i]); */

    /* DD(photons_h.px[i]); */
    /* DD(photons_h.py[i]); */
    /* DD(photons_h.pz[i]); */

    particle.px = photons_h.px[i] - (input->phantom_size_x/2.0)*input->phantom_spacing_x;
    particle.py = photons_h.py[i] - (input->phantom_size_y/2.0)*input->phantom_spacing_y;
    particle.pz = photons_h.pz[i] - (input->phantom_size_z/2.0)*input->phantom_spacing_z;

    /* DD(particle.px); */
    /* DD(particle.py); */
    /* DD(particle.pz); */


    particle.t =  photons_h.t[i];
    particle.type = photons_h.type[i];
    particle.eventID = photons_h.eventID[i];
    particle.trackID = photons_h.trackID[i];
    //particle.initialID = firstInitialID + i;

    // Test if the particle is still in the volume, it was absorbed -> no output.
    if (photons_h.px[i]<0 || photons_h.py[i] < 0 || photons_h.pz[i] < 0  ||
	photons_h.px[i]>input->phantom_size_x*input->phantom_spacing_x ||
	photons_h.py[i]>input->phantom_size_y*input->phantom_spacing_y ||
	photons_h.pz[i]>input->phantom_size_z*input->phantom_spacing_z) {
      output->particles.push_back(particle);
    }
    else {
      // DD("Particle is still in volume. Ignored.");
    }
    ++i;
  }
	
  stack_device_free(photons_d);
  stack_device_free(electrons_d);
  stack_host_free(photons_h);
  stack_host_free(electrons_h);
  volume_device_free(phantom_d);
  cudaThreadExit();
}

