#include "GateGPUIO.hh"

#include <vector>

void GateOpticalBiolum_GPU(const GateGPUIO_Input * input, 
                           GateGPUIO_Output * output) {

  // Select a GPU
  cudaSetDevice(input->cudaDeviceID);

  // Vars
  int particle_simulated = 0;
  int nb_of_particles = input->nb_events;
  int nb_act = input->activity_data.size();
  float E = input->E;
  int firstInitialID = input->firstInitialID;
  long seed = input->seed;
  int i;
  srand(seed);

  DD(E);
  
  // Kernel vars
  dim3 threads, grid;
  int block_size = 512;
  int grid_size = (nb_of_particles + block_size - 1) / block_size;
  threads.x = block_size;
  grid.x = grid_size;

  // Photons Stacks
  StackParticle photons_d;
  stack_device_malloc(photons_d, nb_of_particles);
  StackParticle photons_h;
  stack_host_malloc(photons_h, nb_of_particles);
  
  // Init random
  i=0; while(i < nb_of_particles) {photons_h.seed[i] = rand(); ++i;};
  stack_copy_host2device(photons_h, photons_d);
  kernel_brent_init<<<grid, threads>>>(photons_d);
    
  // Phantoms Mat
  Volume phantom_mat_d;

  phantom_mat_d.size_in_mm = make_float3(input->phantom_size_x*input->phantom_spacing_x,
				     input->phantom_size_y*input->phantom_spacing_y,
				     input->phantom_size_z*input->phantom_spacing_z);
  phantom_mat_d.voxel_size = make_float3(input->phantom_spacing_x,
				     input->phantom_spacing_y,
				     input->phantom_spacing_z);
  phantom_mat_d.size_in_vox = make_int3(input->phantom_size_x,
				    input->phantom_size_y,
				    input->phantom_size_z);
  phantom_mat_d.nb_voxel_slice = phantom_mat_d.size_in_vox.x * phantom_mat_d.size_in_vox.y;
  phantom_mat_d.nb_voxel_volume = phantom_mat_d.nb_voxel_slice * phantom_mat_d.size_in_vox.z;
  
  phantom_mat_d.mem_data = phantom_mat_d.nb_voxel_volume * sizeof(unsigned short int);
  volume_device_malloc(phantom_mat_d, phantom_mat_d.nb_voxel_volume); 
  cudaMemcpy(phantom_mat_d.data, &(input->phantom_material_data[0]), phantom_mat_d.mem_data, cudaMemcpyHostToDevice);

  // Phantoms Activities + Indices
  float *phantom_act_d;
  cudaMalloc((void**) &phantom_act_d, sizeof(float)*nb_act);
  cudaMemcpy(phantom_act_d, &(input->activity_data[0]), sizeof(float)*nb_act, cudaMemcpyHostToDevice);
  unsigned int *phantom_ind_d;
  cudaMalloc((void**) &phantom_ind_d, sizeof(unsigned int)*nb_act);
  cudaMemcpy(phantom_ind_d, &(input->activity_index[0]), sizeof(float)*nb_act, cudaMemcpyHostToDevice);
 
  // Count simulated photons
  int* count_d;
  int count_h = 0;
  cudaMalloc((void**) &count_d, sizeof(int));
  cudaMemcpy(count_d, &count_h, sizeof(int), cudaMemcpyHostToDevice);

  
  // Source
  kernel_optical_voxelized_source<<<grid, threads>>>(photons_d, phantom_mat_d,
                                                     phantom_act_d, phantom_ind_d, E);
  /*i
  stack_copy_device2host(photons_d, photons_h);
    i=0; while(i<nb_of_particles) {
        // printf("%i %f %f %f %f %f %f\n", i, photons_h.px[i], photons_h.py[i], photons_h.pz[i],
                                            photons_h.dx[i], photons_h.dy[i], photons_h.dz[i]);

        ++i;

    }
  */

  // Simulation loop
  int step = 0;
  while (count_h < nb_of_particles) {
    ++step;
    //DD(step);
    //DD(count_h);
    kernel_optical_navigation_regular<<<grid, threads>>>(photons_d, phantom_mat_d, count_d);

    // get back the number of simulated photons
    cudaMemcpy(&count_h, count_d, sizeof(int), cudaMemcpyDeviceToHost);

    /*
    stack_copy_device2host(photons_d, photons_h);
    i=0; while(i<nb_of_particles) {
        // printf("%i %f %f %f %f %f %f\n", i, photons_h.px[i], photons_h.py[i], photons_h.pz[i],
                                            photons_h.dx[i], photons_h.dy[i], photons_h.dz[i]);

        ++i;
    }
    */

    //if (step > 50) {DD("WATCHDOG AT 50 STEP"); break;}
  }
  
  // Copy photons from device to host
  stack_copy_device2host(photons_d, photons_h);

  // ROOT export
  gROOT->Reset();
  gPluginMgr->AddHandler("TVirtualStreamerInfo", "*", "TStreamerInfo","RIO", "TStreamerInfo()");
  TFile* f = new TFile("gpu-test-Boundary-ON.root", "RECREATE", "ROOT file for phase space", 9);
  TTree* tree = new TTree("PhaseSpace", "Phase space tree");

  float px, py, pz, dx, dy, dz, energy;
  tree->Branch("Ekine", &energy, "Ekine/F");
  tree->Branch("X", &px, "X/F");
  tree->Branch("Y", &py, "Y/F");
  tree->Branch("Z", &pz, "Z/F");
  tree->Branch("dX", &dx, "dX/F");
  tree->Branch("dY", &dy, "dY/F");
  tree->Branch("dZ", &dz, "dZ/F");

  i=0; while (i<nb_of_particles) {
    energy = photons_h.E[i];
    dx = photons_h.dx[i];
    dy = photons_h.dy[i];
    dz = photons_h.dz[i];
    px = photons_h.px[i] - (input->phantom_size_x/2.0)*input->phantom_spacing_x;
    py = photons_h.py[i] - (input->phantom_size_y/2.0)*input->phantom_spacing_y;
    pz = photons_h.pz[i] - (input->phantom_size_z/2.0)*input->phantom_spacing_z;


  // printf("Result Main (BEFORE raytracing back): (x,y,z) =  %f %f %f  and (dx,dy,dz) = %f %f %f \n", px, py, pz, dx, dy, dz);


// Ray Tracing Back: start
   float3 backp = back_raytrace_particle(px, py, pz, dx, dy, dz);

	px = backp.x;
	py = backp.y;
	pz = backp.z;
// Ray Tracing Back: end

  // printf("Result Main (AFTER raytracing back): (x,y,z) =  %f %f %f  and (dx,dy,dz) = %f %f %f \n", px, py, pz, dx, dy, dz);


    tree->Fill();
    ++i;
  }
  f->Write();
  f->Close();

  i=0; while (i<nb_of_particles) {
    
    // Test if the particle was absorbed -> no output.
    //if (photons_h.active[i]) {
    GateGPUIO_Particle particle;
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
    particle.trackID = i; //photons_h.trackID[i];
    particle.initialID = firstInitialID + i;
    
    output->particles.push_back(particle);
    
 //        printf("g %e %e %e %e %e %e %e\n", photons_h.E[i], particle.px, particle.py,
   //                       particle.pz, photons_h.dx[i], photons_h.dy[i], photons_h.dz[i]);
    //}
    //else {
      // DD("Particle is still in volume. Ignored.");
    //}
    ++i;
  }
  

  stack_device_free(photons_d);
  stack_host_free(photons_h);
  volume_device_free(phantom_mat_d);

  cudaDeviceSynchronize();

  cudaThreadExit();
}




