#include "actor_ct_fun.cu"
#include "GateTrackingGPUActorIO.hh"

#include <vector>

void GateTrackingGPUActorTrack(const GateTrackingGPUActorInput * input, 
			       GateTrackingGPUActorOutput * output) {

  // FIXME
  // add ecut, emax, meanexcitation
  // add density correction coefficient
  // track event ID
  // init phantom
  printf("====> GPU START\n");

  // TIMING
  double t_init = time();
  double t_g = time();
    
  // Select a GPU
  //cudaSetDevice(input->cudaDeviceID);
  cudaSetDevice(1);
    
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
  
  /*
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
  materials_h.fC = input->fD;
  materials_h.fA = input->fA;
  materials_h.fM = input->fM;
  
  Materials materials_d;
  materials_device_malloc(materials_d, input->nb_materials, input->nb_elements_total);
  
  materials_copy_host2device(materials_h, materials_d);
  
  // To be continued... (JB)
   
  */
  

  // TIMING
  t_init = time() - t_init;
  double t_in = time();
	
  // Fill photons stack with particles from GATE
  int i = 0;
  GateTrackingGPUActorInput::ParticlesList::const_iterator iter = input->particles.begin();
  while (iter != input->particles.end()) {
    GateTrackingGPUActorParticle p = *iter;
    photons_h.E[i] = p.E;
	  //DD(p.E);
    photons_h.dx[i] = p.dx;
    photons_h.dy[i] = p.dy;
    photons_h.dz[i] = p.dz;
    // FIXME need to change the world frame
    photons_h.px[i] = p.px + (input->phantom_size_x/2.0)*input->phantom_spacing_x;
    photons_h.py[i] = p.py + (input->phantom_size_y/2.0)*input->phantom_spacing_y;
    photons_h.pz[i] = p.pz + (input->phantom_size_z/2.0)*input->phantom_spacing_z;
    photons_h.t[i] = p.t;
    photons_h.eventID[i] = p.eventID;
    photons_h.trackID[i] = p.trackID;
    photons_h.type[i] = p.type; // FIXME
    photons_h.seed[i] = rand();
    photons_h.endsimu[i] = 0;
    photons_h.active[i] = 1;
    photons_h.interaction[i] = 1;
        
    //printf("g %e %e %e %e %e %e %e\n", p.E, p.px, p.py, p.pz, p.dx, p.dy, p.dz);

    ++iter;
    ++i;
  }
  
  // Copy particles from host to device
  stack_copy_host2device(photons_h, photons_d);

  // TIMING
  t_in = time() - t_in;
  double t_init_2 = time();
	
  // Kernel vars
  dim3 threads, grid;
  int block_size = 512;
  int grid_size = (nb_of_particles + block_size - 1) / block_size;
  threads.x = block_size;
  grid.x = grid_size;
	
  // Init random
  kernel_brent_init<<<grid, threads>>>(photons_d);

  // Phantoms
  Volume<unsigned short int> phantom_d;
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
  phantom_d.nb_voxel_volume = phantom_d.nb_voxel_slice * phantom_d.size_in_vox.z;
  phantom_d.mem_data = phantom_d.nb_voxel_volume * sizeof(unsigned short int);
  volume_device_malloc<unsigned short int>(phantom_d, phantom_d.nb_voxel_volume); 
  cudaMemcpy(phantom_d.data, &(input->phantom_material_data[0]), phantom_d.mem_data, cudaMemcpyHostToDevice);

  // Count simulated photons
  int* count_d;
  int count_h = 0;
  cudaMalloc((void**) &count_d, sizeof(int));
  cudaMemcpy(count_d, &count_h, sizeof(int), cudaMemcpyHostToDevice);

  // TIMING
  t_init_2 = time() - t_init_2;
  double t_track = time();

  // Simualtion loop
  int step = 0;
  while (count_h < nb_of_particles) {
    ++step;

    kernel_ct_navigation_regular<unsigned short int><<<grid, threads>>>(photons_d, phantom_d, count_d);

    // get back the number of simulated photons
    cudaMemcpy(&count_h, count_d, sizeof(int), cudaMemcpyDeviceToHost);

  }

  // TIMING
  t_track = time() - t_track;
  double t_out = time();

  // Copy photons from device to host
  stack_copy_device2host(photons_d, photons_h);
 
  // DEBUG (not export particles)
  
  i=0;
  while (i<nb_of_particles) {
    
    // Test if the particle was absorbed -> no output.
    if (photons_h.active[i]) {
        GateTrackingGPUActorParticle particle;
        particle.E =  photons_h.E[i];
        particle.dx = photons_h.dx[i];
        particle.dy = photons_h.dy[i];
        particle.dz = photons_h.dz[i];
        particle.px = photons_h.px[i] - (input->phantom_size_x/2.0)*input->phantom_spacing_x;
        particle.py = photons_h.py[i] - (input->phantom_size_y/2.0)*input->phantom_spacing_y;
        particle.pz = photons_h.pz[i] - (input->phantom_size_z/2.0)*input->phantom_spacing_z;
        particle.t =  photons_h.t[i];
        particle.type = photons_h.type[i];
        particle.eventID = photons_h.eventID[i];
        particle.trackID = photons_h.trackID[i];
        
        output->particles.push_back(particle);
    
        //printf("g %e %e %e %e %e %e %e\n", photons_h.E[i], particle.px, particle.py,
        //                  particle.pz, photons_h.dx[i], photons_h.dy[i], photons_h.dz[i]);
    }
    //else {
      // DD("Particle is still in volume. Ignored.");
    //}
    ++i;
  }
  

  // TIMING
  t_out = time() - t_out;

  stack_device_free(photons_d);
  stack_host_free(photons_h);
  //volume_device_free(phantom_d);


  cudaDeviceSynchronize();
  t_g = time() - t_g;
  printf(">> GPU: init %e input %e track %e output %e tot %e\n", t_init+t_init_2, 
          t_in, t_track, t_out, t_g);

  cudaThreadExit();
  printf("====> GPU STOP\n");
}




