/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateGPUEmisTomo.hh"
#include "GateClock.hh"
#include "Randomize.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "GateRandomEngine.hh"

#include "G4Gamma.hh"
#include "G4GenericIon.hh"
#include "G4Event.hh"
#include "G4UnitsTable.hh"

#include <vector>
#include <map>
#include "GateGPUEmisTomoMessenger.hh"
#include "GateVSourceVoxelReader.hh"
#include "GateSourceVoxelTestReader.hh"
#include "GateSourceVoxelImageReader.hh"
#include "GateSourceVoxelInterfileReader.hh"
#include "GateObjectStore.hh"
//#include "GateFictitiousVoxelMapParameterized.hh"
#include "GateRegularParameterized.hh"
#include "GateApplicationMgr.hh"

//----------------------------------------------------------
GateGPUEmisTomo::GateGPUEmisTomo(G4String name)
  : GateSourceVoxellized(name), m_gpu_input(NULL)
{
  // Build IO for gpu
  m_gpu_input = GateGPUIO_Input_new();
  m_gpu_output = GateGPUIO_Output_new();
  //m_gpu_input->E = 511*keV/MeV;
  attachedVolumeName = "no_attached_volume_given";

  // Create particle definition 
  G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
  gamma_particle_definition = particleTable->FindParticle("gamma");

  m_sourceGPUVoxellizedMessenger = new GateGPUEmisTomoMessenger(this); 
  //mNumberOfNextTime = 1;
  //mCurrentTimeID = 0;
  mCudaDeviceID = 0;
  mBeginRunFlag = 0;
  max_buffer_size = 0;
  mUserCount = 0;
     
  //printf("\n********************************************************\n");
  //double mytime = GateApplicationMgr::GetInstance()->GetTimeStop();
  //printf("Stop time %f\n", mytime);
  //GateApplicationMgr::GetInstance()->SetTimeStop(2*mytime);
  //printf("New Stop time %f\n\n", GateApplicationMgr::GetInstance()->GetTimeStop());
  
}
//----------------------------------------------------------


//----------------------------------------------------------
GateGPUEmisTomo::~GateGPUEmisTomo()
{
  GateGPUIO_Input_delete(m_gpu_input);
  GateGPUIO_Output_delete(m_gpu_output);
}
//----------------------------------------------------------

/*
//----------------------------------------------------------
G4double GateGPUEmisTomo::GetNextTime(G4double timeNow)
{
  // Loop on the mother's GetNextTime
  G4double t = 0.0;
  for(int i=0; i<mNumberOfNextTime; i++) {
    G4double a = timeNow + t;
    t += GateSourceVoxellized::GetNextTime(a);
  }

  GateMessage("Beam", 5, "Compute " << mNumberOfNextTime << " NextTime from " << G4BestUnit(timeNow, "Time") << " -> found = " << G4BestUnit(t, "Time") << "=" << G4BestUnit(timeNow+t, "Time") << std::endl);  

  return t;
}
//----------------------------------------------------------
*/

//----------------------------------------------------------
void GateGPUEmisTomo::SetGPUBufferSize(int n)
{
  max_buffer_size = n;
  printf("read by the messenger max buffer size %i\n", max_buffer_size);
}
//----------------------------------------------------------

void GateGPUEmisTomo::SetGPUDeviceID(int n)
{
  assert(m_gpu_input);
  m_gpu_input->cudaDeviceID = n;
}

//----------------------------------------------------------
void GateGPUEmisTomo::Dump(G4int level) 
{
  GateSourceVoxellized::Dump(level);
}
//----------------------------------------------------------

//----------------------------------------------------------
void GateGPUEmisTomo::AttachToVolume(const G4String& volume_name)
{
  attachedVolumeName = volume_name;
}
//----------------------------------------------------------

//----------------------------------------------------------
G4int GateGPUEmisTomo::GeneratePrimaries(G4Event* event) 
{

  // Initial checking
  if (!m_voxelReader) return 0;
  //  assert(GetType() == "backtoback");
  if (GetType() != "backtoback") {
    GateError("Error, the source GPUvoxel is only available for type 'backtoback' (PET application), but you used '" << GetType() << "'. Abort.");
  }
  assert(m_gpu_input);


  // STEP 1 -- INIT -----------------------------------------------
  if (!mBeginRunFlag) {
      // First time here -> phantom data are set
      if (m_gpu_input->phantom_material_data.empty())
      { // import phantom to gpu (fill input)
          SetPhantomVolumeData(); 
          //m_current_particle_index_in_buffer = 0;     
      }
      
      if (m_gpu_input->activity_index.empty())
      { // import activity to gpu
          //m_gpu_input->firstInitialID = mCurrentTimeID;
          ActivityMap activities = m_voxelReader->GetSourceActivityMap();
          GateGPUIO_Input_parse_activities(activities,m_gpu_input);
      }

      // Init phantom size
      half_phan_size_x = m_gpu_input->phantom_size_x * m_gpu_input->phantom_spacing_x * 0.5f;
      half_phan_size_y = m_gpu_input->phantom_size_y * m_gpu_input->phantom_spacing_y * 0.5f;
      half_phan_size_z = m_gpu_input->phantom_size_z * m_gpu_input->phantom_spacing_z * 0.5f;
    
      // Seed
      unsigned int seed = 
          static_cast<unsigned int>(*GateRandomEngine::GetInstance()->GetRandomEngine());
      srand(seed);

      id_event = 0;
      nb_event_in_buffer = 0;
      current_time = 0.0;
      tot_p = 0;

#ifdef GATE_USE_GPU
      // Init GPU' stuff
      GPU_GateEmisTomo_init(m_gpu_input,
                            gpu_materials, gpu_phantom, gpu_activities,
                            gpu_gamma1, gpu_gamma2, cpu_gamma1, cpu_gamma2,
                            max_buffer_size, seed);
#endif      

      mBeginRunFlag = 1;
  }

  // STEP 2 -- Fill buffer if need -------------------------------------

  if (nb_event_in_buffer <= 0) {
#ifdef GATE_USE_GPU
    GPU_GateEmisTomo(gpu_materials, gpu_phantom, gpu_activities,
                     gpu_gamma1, gpu_gamma2, cpu_gamma1, cpu_gamma2,
                     max_buffer_size);
    nb_event_in_buffer = max_buffer_size;
    id_event_in_buffer = 0;
#endif    
  }
  
  // STEP 3 -- Generate one event ------------------------------------

  // Create at least one particle
  int nb_p = 0;
  while (!nb_p) {
      
      // create one particle
      if (cpu_gamma1.active[id_event_in_buffer]) {
          // FIXME add transformation (R+T) to the new frame
          G4ThreeVector position1(cpu_gamma1.px[id_event_in_buffer] - half_phan_size_x,
                                  cpu_gamma1.py[id_event_in_buffer] - half_phan_size_y,
                                  cpu_gamma1.pz[id_event_in_buffer] - half_phan_size_z);
      
          G4ThreeVector direction1(cpu_gamma1.dx[id_event_in_buffer],
                                   cpu_gamma1.dy[id_event_in_buffer],
                                   cpu_gamma1.dz[id_event_in_buffer]);

          double p_time1 = current_time + cpu_gamma1.t[id_event_in_buffer]*ns;  // time + TOF
      
          G4ThreeVector p_momentum1 = cpu_gamma1.E[id_event_in_buffer] * MeV * direction1.unit();

          // Create the vertex
          G4PrimaryVertex* vertex1;
          vertex1 = new G4PrimaryVertex(position1, p_time1);
        
          // Create a G4PrimaryParticle
          G4PrimaryParticle* g4particle1 =  new G4PrimaryParticle(gamma_particle_definition, 
                                                                  p_momentum1.x(), 
                                                                  p_momentum1.y(), 
                                                                  p_momentum1.z());
          // Add the particle
          vertex1->SetPrimary( g4particle1 ); 
          event->AddPrimaryVertex( vertex1 );
          nb_p++;
      }

      if (cpu_gamma2.active[id_event_in_buffer]) {
          // FIXME add transformation (R+T) to the new frame
          G4ThreeVector position2(cpu_gamma2.px[id_event_in_buffer] - half_phan_size_x,
                                  cpu_gamma2.py[id_event_in_buffer] - half_phan_size_y,
                                  cpu_gamma2.pz[id_event_in_buffer] - half_phan_size_z);
      
          G4ThreeVector direction2(cpu_gamma2.dx[id_event_in_buffer],
                                   cpu_gamma2.dy[id_event_in_buffer],
                                   cpu_gamma2.dz[id_event_in_buffer]);

          double p_time2 = current_time + cpu_gamma2.t[id_event_in_buffer]*ns;  // time + TOF
      
          G4ThreeVector p_momentum2 = cpu_gamma2.E[id_event_in_buffer] * MeV * direction2.unit();

          // Create the vertex
          G4PrimaryVertex* vertex2;
          vertex2 = new G4PrimaryVertex(position2, p_time2);
        
          // Create a G4PrimaryParticle
          G4PrimaryParticle* g4particle2 =  new G4PrimaryParticle(gamma_particle_definition, 
                                                                  p_momentum2.x(), 
                                                                  p_momentum2.y(), 
                                                                  p_momentum2.z());
          // Add the particle
          vertex2->SetPrimary( g4particle2 ); 
          event->AddPrimaryVertex( vertex2 );
          nb_p++;
      }

      event->SetEventID(id_event);

      // update event
      id_event_in_buffer++;
      nb_event_in_buffer--;
      id_event++;

      // update current time
      double rnd = rand()/(double)(RAND_MAX);
      current_time += ((-log(rnd) / gpu_activities.tot_activity) * ns);

  }

  // FIXME EndOfRunAction
   
  return nb_p; 
}
//----------------------------------------------------------


/*
//----------------------------------------------------------
void GateGPUEmisTomo::GeneratePrimaryEventFromGPUOutput(const GateGPUIO_Particle & particle, 
                                                                G4Event * event)
{

  // Position
  G4ThreeVector particle_position;
  particle_position.setX(particle.px*mm-m_gpu_input->phantom_size_x*m_gpu_input->phantom_spacing_x/2.0*mm);
  particle_position.setY(particle.py*mm-m_gpu_input->phantom_size_y*m_gpu_input->phantom_spacing_y/2.0*mm);
  particle_position.setZ(particle.pz*mm-m_gpu_input->phantom_size_z*m_gpu_input->phantom_spacing_z/2.0*mm);

  
  // Create the vertex
  G4PrimaryVertex* vertex;
  double particle_time = particle.t*ns; // assume time is in ns

  
  std::cout << " gpu particle time (ns)             = " << particle.t << std::endl;
  std::cout << " gpu particle time check best unit  = " << G4BestUnit(particle_time, "Time") << std::endl;
  std::cout << " Gettime =" << G4BestUnit(GetTime() , "Time") << std::endl;
  std::cout << " a+b =" << G4BestUnit(GetTime()+particle_time, "Time") << std::endl;
  

  // Set the time of this particle to the current time plus the TOF.
  vertex = new G4PrimaryVertex(particle_position, GetTime() + particle_time);
  
  // Direction
  G4ThreeVector particle_direction;
  particle_direction.setX(particle.dx);
  particle_direction.setY(particle.dy);
  particle_direction.setZ(particle.dz);
  //FIXME -> change position according to source orientation/position

  // Compute momentum
  G4ThreeVector particle_momentum = (particle.E*MeV) * particle_direction.unit();
  
  mEnergy = particle.E*MeV;
  // Create a G4PrimaryParticle
  G4PrimaryParticle* g4particle =  new G4PrimaryParticle(gamma_particle_definition, 
                                                         particle_momentum.x(), 
                                                         particle_momentum.y(), 
                                                         particle_momentum.z());
  vertex->SetPrimary( g4particle ); 
  event->AddPrimaryVertex( vertex );
}
//----------------------------------------------------------
*/

//----------------------------------------------------------
void GateGPUEmisTomo::ReaderInsert(G4String readerType)
{
  G4cout << "GateGPUEmisTomoMessenger ReaderInsert" << G4endl;
  GateSourceVoxellized::ReaderInsert(readerType);
}
//----------------------------------------------------------


//----------------------------------------------------------
void GateGPUEmisTomo::ReaderRemove()
{
  G4cout << "GateGPUEmisTomoMessenger ReaderRemove" << G4endl;
  GateSourceVoxellized::ReaderRemove();
}
//----------------------------------------------------------


//----------------------------------------------------------
void GateGPUEmisTomo::Update(double time)
{
  G4cout << "GateGPUEmisTomoMessenger Update" << G4endl;
  return GateSourceVoxellized::Update(time);
}
//----------------------------------------------------------


//----------------------------------------------------------
void GateGPUEmisTomo::SetPhantomVolumeData() 
{
  GateVVolume* v = GateObjectStore::GetInstance()->FindVolumeCreator(attachedVolumeName);
  // FindVolumeCreator raise an error if not found
  // FIXME -> change the error message
  
  //GateFictitiousVoxelMapParameterized * m = dynamic_cast<GateFictitiousVoxelMapParameterized*>(v);
  GateRegularParameterized *m = dynamic_cast<GateRegularParameterized*>(v);
  if (m == NULL) {
    //GateError(attachedVolumeName << " is not a GateFictitiousVoxelMapParameterized.");
    GateError(attachedVolumeName << " is not a GateRegularParameterized.");
  }
  else {

    // Size of the image
    GateVGeometryVoxelReader* reader = m->GetReader();
    m_gpu_input->phantom_size_x = reader->GetVoxelNx();
    m_gpu_input->phantom_size_y = reader->GetVoxelNy();
    m_gpu_input->phantom_size_z = reader->GetVoxelNz();
    m_gpu_input->phantom_spacing_x = reader->GetVoxelSize().x();
    m_gpu_input->phantom_spacing_y = reader->GetVoxelSize().y();
    m_gpu_input->phantom_spacing_z = reader->GetVoxelSize().z();
    
    /* THIS IS ELEGANT, BUT IT DOESN'T WORK PROPERLY.... (JB)
    // Find the list of material in the image and set the pixel
    std::vector<G4Material*> materials;
    for(int k=0; k<m_gpu_input->phantom_size_z; k++)
      for(int j=0; j<m_gpu_input->phantom_size_y; j++)
        for(int i=0; i<m_gpu_input->phantom_size_x; i++) {
          // Get the material
          G4Material * m = reader->GetVoxelMaterial(i,j,k); 
          std::vector<G4Material*>::iterator iter;
          iter = std::find(materials.begin(), materials.end(), m);
          // Store it if this is the first time
          if (iter == materials.end()) {
            DD(m->GetName());
            materials.push_back(m);
          }
          // Store the pixel value with the correct index
          // DD(i);
          // DD(j);
          // DD(k);
          unsigned short int index = iter-materials.begin();
          //DD(materials.begin());
          //DD((*iter).first);
          DD(index);
          m_gpu_input->phantom_material_data.push_back(index);
        }
    */

    ///// This code is less fancy but is working properly - JB ////////////////////////////

    // list of materials
    std::vector<G4Material*> materials;

    // set voxel accoring materials index
    int i, j, k; unsigned int iter=0; int find=0;
    k=0; while (k < m_gpu_input->phantom_size_z) {
        j=0; while (j < m_gpu_input->phantom_size_y) {
            i=0; while (i < m_gpu_input->phantom_size_x) {
                // get materials
                G4Material* mat = reader->GetVoxelMaterial(i, j, k);

                // find the materials in the list
                iter=0; find=0;
                while (iter < materials.size()) {
                    if (mat->GetName() == materials[iter]->GetName()) {find = 1; break;}
                    ++iter;
                }
                
                // set voxel index
                m_gpu_input->phantom_material_data.push_back(iter);

                // if this material is not in the list add in it?
                if (!find) {materials.push_back(mat);}

                ++i;
            } // i
            ++j;
        } // j
        ++k;
    } // k

    DD(materials.size());

    // Init the materials
    G4String name = v->GetObjectName();
    GateGPUIO_Input_Init_Materials(m_gpu_input, materials, name);
    DD("mat done");
  }
}
//----------------------------------------------------------


//----------------------------------------------------------
/* 
 * GeneratePrimaries Version 2 - JB
 *
//----------------------------------------------------------
G4int GateGPUEmisTomo::GeneratePrimaries(G4Event* event) 
{

  // Initial checking
  if (!m_voxelReader) return 0;
  //  assert(GetType() == "backtoback");
  if (GetType() != "backtoback") {
    GateError("Error, the source GPUvoxel is only available for type 'backtoback' (PET application), but you used '" << GetType() << "'. Abort.");
  }
  assert(m_gpu_input);


  // STEP 1 -- INIT -----------------------------------------------
  if (!mBeginRunFlag) {
      // First time here -> phantom data are set
      if (m_gpu_input->phantom_material_data.empty())
      { // import phantom to gpu (fill input)
          SetPhantomVolumeData(); 
          //m_current_particle_index_in_buffer = 0;     
      }
      
      if (m_gpu_input->activity_index.empty())
      { // import activity to gpu
          //m_gpu_input->firstInitialID = mCurrentTimeID;
          ActivityMap activities = m_voxelReader->GetSourceActivityMap();
          GateGPUIO_Input_parse_activities(activities,m_gpu_input);
      }

      // Init phantom size
      half_phan_size_x = m_gpu_input->phantom_size_x * m_gpu_input->phantom_spacing_x * 0.5f;
      half_phan_size_y = m_gpu_input->phantom_size_y * m_gpu_input->phantom_spacing_y * 0.5f;
      half_phan_size_z = m_gpu_input->phantom_size_z * m_gpu_input->phantom_spacing_z * 0.5f;
    
      // Seed
      unsigned int seed = 
          static_cast<unsigned int>(*GateRandomEngine::GetInstance()->GetRandomEngine());
      srand(seed);

      id_event = 0;
      nb_event_in_buffer = 0;
      current_time = 0.0;
      tot_p = 0;

#ifdef GATE_USE_GPU
      // Init GPU' stuff
      GPU_GateEmisTomo_init(m_gpu_input,
                            gpu_materials, gpu_phantom, gpu_activities,
                            gpu_gamma1, gpu_gamma2, cpu_gamma1, cpu_gamma2,
                            max_buffer_size, seed);
#endif      

      mBeginRunFlag = 1;
  }

  // STEP 2 -- Fill buffer if need -------------------------------------

  if (nb_event_in_buffer <= 0) {
#ifdef GATE_USE_GPU
    GPU_GateEmisTomo(gpu_materials, gpu_phantom, gpu_activities,
                     gpu_gamma1, gpu_gamma2, cpu_gamma1, cpu_gamma2,
                     max_buffer_size);
    nb_event_in_buffer = max_buffer_size;
    id_event_in_buffer = 0;
#endif    
  }
  
  // STEP 3 -- Generate one event ------------------------------------

  // Create at least one particle
  int nb_p = 0;
  while (!nb_p) {

      // Check alternatively each cpu buffer in order to produce one event at a time
      
      // cpu_gamma1
      if (mUserCount == 0) {
      
          // create one particle
          if (cpu_gamma1.active[id_event_in_buffer]) {
              // FIXME add transformation (R+T) to the new frame
              G4ThreeVector position(cpu_gamma1.px[id_event_in_buffer] - half_phan_size_x,
                                     cpu_gamma1.py[id_event_in_buffer] - half_phan_size_y,
                                     cpu_gamma1.pz[id_event_in_buffer] - half_phan_size_z);
          
              G4ThreeVector direction(cpu_gamma1.dx[id_event_in_buffer],
                                      cpu_gamma1.dy[id_event_in_buffer],
                                      cpu_gamma1.dz[id_event_in_buffer]);

              double p_time = current_time + cpu_gamma1.t[id_event_in_buffer]*ns;  // time + TOF
          
              G4ThreeVector p_momentum = cpu_gamma1.E[id_event_in_buffer] * MeV * direction.unit();

              // Create the vertex
              G4PrimaryVertex* vertex;
              vertex = new G4PrimaryVertex(position, p_time);
            
              // Create a G4PrimaryParticle
              G4PrimaryParticle* g4particle =  new G4PrimaryParticle(gamma_particle_definition, 
                                                                     p_momentum.x(), 
                                                                     p_momentum.y(), 
                                                                     p_momentum.z());
              // Add the particle
              vertex->SetPrimary( g4particle ); 
              event->AddPrimaryVertex( vertex );
              event->SetEventID(id_event);
              nb_p++;
              tot_p++;
          }

          // update
          mUserCount = 1;
          id_event++;

      // cpu_gamma2
      } else {

          // create one particle
          if (cpu_gamma2.active[id_event_in_buffer]) {
              // FIXME add transformation (R+T) to the new frame
              G4ThreeVector position(cpu_gamma2.px[id_event_in_buffer] - half_phan_size_x,
                                     cpu_gamma2.py[id_event_in_buffer] - half_phan_size_y,
                                     cpu_gamma2.pz[id_event_in_buffer] - half_phan_size_z);
          
              G4ThreeVector direction(cpu_gamma2.dx[id_event_in_buffer],
                                      cpu_gamma2.dy[id_event_in_buffer],
                                      cpu_gamma2.dz[id_event_in_buffer]);

              double p_time = current_time + cpu_gamma2.t[id_event_in_buffer]*ns;  // time + TOF
          
              G4ThreeVector p_momentum = cpu_gamma2.E[id_event_in_buffer] * MeV * direction.unit();

              // Create the vertex
              G4PrimaryVertex* vertex;
              vertex = new G4PrimaryVertex(position, p_time);
            
              // Create a G4PrimaryParticle
              G4PrimaryParticle* g4particle =  new G4PrimaryParticle(gamma_particle_definition, 
                                                                     p_momentum.x(), 
                                                                     p_momentum.y(), 
                                                                     p_momentum.z());
              // Add the particle
              vertex->SetPrimary( g4particle ); 
              event->AddPrimaryVertex( vertex );
              event->SetEventID(id_event);
              nb_p++;
              tot_p++;
          }

          // update
          id_event_in_buffer++;
          nb_event_in_buffer--;
          id_event++;


          // update current time
          double rnd = rand()/(double)(RAND_MAX);
          current_time += ((-log(rnd) / gpu_activities.tot_activity) * ns);
          
          mUserCount = 0;
      }
      
  } // while nb_p

  // FIXME EndOfRunAction
  // Get simulation time required to define the last run
   
  return nb_p; // Return zero or one particle at a time
}
//----------------------------------------------------------
*/



//----------------------------------------------------------
/* 
 * GeneratePrimaries Version 1 - JB
 *
//----------------------------------------------------------
G4int GateGPUEmisTomo::GeneratePrimaries(G4Event* event) 
{

  // Initial checking
  if (!m_voxelReader) return 0;
  //  assert(GetType() == "backtoback");
  if (GetType() != "backtoback") {
    GateError("Error, the source GPUvoxel is only available for type 'backtoback' (PET application), but you used '" << GetType() << "'. Abort.");
  }
  assert(m_gpu_input);


  // STEP 1 -- INIT -----------------------------------------------
  if (!mBeginRunFlag) {
      // First time here -> phantom data are set
      if (m_gpu_input->phantom_material_data.empty())
      { // import phantom to gpu (fill input)
          SetPhantomVolumeData(); 
          //m_current_particle_index_in_buffer = 0;     
      }
      
      if (m_gpu_input->activity_index.empty())
      { // import activity to gpu
          //m_gpu_input->firstInitialID = mCurrentTimeID;
          ActivityMap activities = m_voxelReader->GetSourceActivityMap();
          GateGPUIO_Input_parse_activities(activities,m_gpu_input);
      }

      // Init phantom size
      half_phan_size_x = m_gpu_input->phantom_size_x * m_gpu_input->phantom_spacing_x * 0.5f;
      half_phan_size_y = m_gpu_input->phantom_size_y * m_gpu_input->phantom_spacing_y * 0.5f;
      half_phan_size_z = m_gpu_input->phantom_size_z * m_gpu_input->phantom_spacing_z * 0.5f;
    
      // Seed
      unsigned int seed = 
          static_cast<unsigned int>(*GateRandomEngine::GetInstance()->GetRandomEngine());
      srand(seed);

      id_event = 0;
      nb_event_in_buffer = 0;
      current_time = 0.0;

#ifdef GATE_USE_GPU
      // Init GPU' stuff
      GPU_GateEmisTomo_init(m_gpu_input,
                            gpu_materials, gpu_phantom, gpu_activities,
                            gpu_gamma1, gpu_gamma2, cpu_gamma1, cpu_gamma2,
                            max_buffer_size, seed);
#endif      

      mBeginRunFlag = 1;
  }

  // STEP 2 -- Fill buffer if need -------------------------------------
  if (nb_event_in_buffer <= 0) {
#ifdef GATE_USE_GPU
    GPU_GateEmisTomo(gpu_materials, gpu_phantom, gpu_activities,
                     gpu_gamma1, gpu_gamma2, cpu_gamma1, cpu_gamma2,
                     max_buffer_size);
    nb_event_in_buffer = max_buffer_size;
    id_event_in_buffer = 0;
#endif    
  }
  
  // STEP 3 -- Generate one event ------------------------------------

  // nb of particles return
  int nb_p = 0;

  // create one particle
  if (cpu_gamma1.active[id_event_in_buffer]) {
      // FIXME add transformation (R+T) to the new frame
      G4ThreeVector position(cpu_gamma1.px[id_event_in_buffer] - half_phan_size_x,
                             cpu_gamma1.py[id_event_in_buffer] - half_phan_size_y,
                             cpu_gamma1.pz[id_event_in_buffer] - half_phan_size_z);
  
      G4ThreeVector direction(cpu_gamma1.dx[id_event_in_buffer],
                              cpu_gamma1.dy[id_event_in_buffer],
                              cpu_gamma1.dz[id_event_in_buffer]);

      double p_time = current_time + cpu_gamma1.t[id_event_in_buffer]*ns;  // time + TOF
  
      G4ThreeVector p_momentum = cpu_gamma1.E[id_event_in_buffer] * MeV * direction.unit();

      // Create the vertex
      G4PrimaryVertex* vertex;
      vertex = new G4PrimaryVertex(position, p_time);
    
      // Create a G4PrimaryParticle
      G4PrimaryParticle* g4particle =  new G4PrimaryParticle(gamma_particle_definition, 
                                                             p_momentum.x(), 
                                                             p_momentum.y(), 
                                                             p_momentum.z());
      // Add the particle
      vertex->SetPrimary( g4particle ); 
      event->AddPrimaryVertex( vertex );
      event->SetEventID(id_event);
      nb_p++;
      printf("     New particle cpu1\n");
  }

  // create one particle
  if (cpu_gamma2.active[id_event_in_buffer]) {
      // FIXME add transformation (R+T) to the new frame
      G4ThreeVector position(cpu_gamma2.px[id_event_in_buffer] - half_phan_size_x,
                             cpu_gamma2.py[id_event_in_buffer] - half_phan_size_y,
                             cpu_gamma2.pz[id_event_in_buffer] - half_phan_size_z);
  
      G4ThreeVector direction(cpu_gamma2.dx[id_event_in_buffer],
                              cpu_gamma2.dy[id_event_in_buffer],
                              cpu_gamma2.dz[id_event_in_buffer]);

      double p_time = current_time + cpu_gamma2.t[id_event_in_buffer]*ns;  // time + TOF
  
      G4ThreeVector p_momentum = cpu_gamma2.E[id_event_in_buffer] * MeV * direction.unit();

      // Create the vertex
      G4PrimaryVertex* vertex;
      vertex = new G4PrimaryVertex(position, p_time);
    
      // Create a G4PrimaryParticle
      G4PrimaryParticle* g4particle =  new G4PrimaryParticle(gamma_particle_definition, 
                                                             p_momentum.x(), 
                                                             p_momentum.y(), 
                                                             p_momentum.z());
      // Add the particle
      //vertex->SetPrimary( g4particle ); 
      //event->AddPrimaryVertex( vertex );
      //event->SetEventID(id_event);
      //nb_p++;
      printf("     New particle cpu2\n");
  }

  // update particle ct
  id_event_in_buffer++;
  nb_event_in_buffer--;
  id_event++;

  // update time
  double rnd = rand()/(double)(RAND_MAX);
  current_time += (-log(rnd) / gpu_activities.tot_activity);
  printf(":: Id event %i time %e\n", id_event, current_time);
  
  // FIXME EndOfRunAction
  return nb_p; // Return one or two particles at a time
}
//----------------------------------------------------------
*/

