/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateSourceGPUVoxellized.hh"
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
#include "GateSourceGPUVoxellizedMessenger.hh"
#include "GateVSourceVoxelReader.hh"
#include "GateSourceVoxelTestReader.hh"
#include "GateSourceVoxelImageReader.hh"
#include "GateSourceVoxelInterfileReader.hh"
#include "GateObjectStore.hh"
#include "GateFictitiousVoxelMapParameterized.hh"
#include "GateApplicationMgr.hh"

//----------------------------------------------------------
GateSourceGPUVoxellized::GateSourceGPUVoxellized(G4String name)
  : GateSourceVoxellized(name), m_gpu_input(NULL)
{
  // Build IO for gpu
  m_gpu_input = GateGPUIO_Input_new();
  m_gpu_output = GateGPUIO_Output_new();
  m_gpu_input->E = 511*keV/MeV;
  attachedVolumeName = "no_attached_volume_given";

  // Create particle definition 
  G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
  gamma_particle_definition = particleTable->FindParticle("gamma");

  m_sourceGPUVoxellizedMessenger = new GateSourceGPUVoxellizedMessenger(this); 
  mNumberOfNextTime = 1;
  mCurrentTimeID = 0;
  mCudaDeviceID = 0;
}
//----------------------------------------------------------


//----------------------------------------------------------
GateSourceGPUVoxellized::~GateSourceGPUVoxellized()
{
  GateGPUIO_Input_delete(m_gpu_input);
  GateGPUIO_Output_delete(m_gpu_output);
}
//----------------------------------------------------------


//----------------------------------------------------------
G4double GateSourceGPUVoxellized::GetNextTime(G4double timeNow)
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


//----------------------------------------------------------
void GateSourceGPUVoxellized::SetGPUBufferSize(int n)
{
  assert(m_gpu_input);
  m_gpu_input->nb_events = n;
}
//----------------------------------------------------------

void GateSourceGPUVoxellized::SetGPUDeviceID(int n)
{
  assert(m_gpu_input);
  m_gpu_input->cudaDeviceID = n;
}

//----------------------------------------------------------
void GateSourceGPUVoxellized::Dump(G4int level) 
{
  GateSourceVoxellized::Dump(level);
}
//----------------------------------------------------------

//----------------------------------------------------------
void GateSourceGPUVoxellized::AttachToVolume(const G4String& volume_name)
{
  attachedVolumeName = volume_name;
}
//----------------------------------------------------------

//----------------------------------------------------------
G4int GateSourceGPUVoxellized::GeneratePrimaries(G4Event* event) 
{

  // Initial checking
  if (!m_voxelReader) return 0;
  //  assert(GetType() == "backtoback");
  if (GetType() != "backtoback") {
    GateError("Error, the source GPUvoxel is only available for type 'backtoback' (PET application), but you used '" << GetType() << "'. Abort.");
  }
  assert(m_gpu_input);

  // First time here -> phantom data are set
  if (m_gpu_input->phantom_material_data.empty())
    { // import phantom to gpu (fill input)
      SetPhantomVolumeData(); 
      m_current_particle_index_in_buffer = 0;     
    }
  
  if (m_gpu_input->activity_index.empty())
    { // import activity to gpu
      m_gpu_input->firstInitialID = mCurrentTimeID;
      ActivityMap activities = m_voxelReader->GetSourceActivityMap();
      GateGPUIO_Input_parse_activities(activities,m_gpu_input);
    }

  // Main loop : if particles buffer is empty, we ask the gpu 
  // FIXME  if (m_gpu_output->particles.empty()) {
  if (m_current_particle_index_in_buffer >= m_gpu_output->particles.size()) {
    GateMessage("Beam", 5, "No particles in the buffer, we ask the gpu for " 
                << m_gpu_input->nb_events << " events" << std::endl);

    // Go GPU
    m_gpu_input->firstInitialID = mCurrentTimeID; // fix a bug - JB
    m_gpu_input->seed = 
      static_cast<unsigned int>(*GateRandomEngine::GetInstance()->GetRandomEngine());
    printf("seed from input %ld\n", m_gpu_input->seed);

#ifdef GATE_USE_GPU
    GateGPU_VoxelSource_GeneratePrimaries(m_gpu_input, m_gpu_output);
#endif    


    GateMessage("Beam", 5, "Done : GPU send " << m_gpu_output->particles.size() 
                << " events" << std::endl);
    m_current_particle_index_in_buffer = 0;
  }

  // Generate one particle
  //FIXME  if (!m_gpu_output->particles.empty()) {
  if (m_current_particle_index_in_buffer < m_gpu_output->particles.size())  {
    
    // Create a new particle
    //FIXME    GeneratePrimaryEventFromGPUOutput(m_gpu_output->particles.front(), event);
    const GateGPUIO_Particle & part = m_gpu_output->particles[m_current_particle_index_in_buffer];
    GeneratePrimaryEventFromGPUOutput(part, event);
    //FIXME double tof = m_gpu_output->particles.front().t;
    double tof = part.t;
    //printf("t %e\n", tof);

    // Set the current timeID
    //FIXME mCurrentTimeID = m_gpu_output->particles.front().initialID;
    mCurrentTimeID = part.initialID;
    
    // Remove the particle from the list
    //m_gpu_output->particles.pop_front();
    m_current_particle_index_in_buffer++;

    // Display information
    G4PrimaryParticle  * p = event->GetPrimaryVertex(0)->GetPrimary(0);
    event->SetEventID(mCurrentTimeID);
    GateMessage("Beam", 3, "(" << event->GetEventID() << ") " << p->GetG4code()->GetParticleName() 
                << " pos=" << event->GetPrimaryVertex(0)->GetPosition()
                << " weight=" << p->GetWeight()                                
                << " energy=" <<  G4BestUnit(mEnergy, "Energy")
                << " mom=" << p->GetMomentum()
                << " prop_time=" <<  G4BestUnit(p->GetProperTime(), "Time")
                << " Gate_time=" <<  G4BestUnit(GetTime(), "Time")
                << " TOF_time=" <<  G4BestUnit(tof, "Time")
                << " current_timeID=" <<  mCurrentTimeID
                << G4endl);  

    // Prepare for next particle
    if (m_current_particle_index_in_buffer < m_gpu_output->particles.size())  {
      // FIXME if (!m_gpu_output->particles.empty()) {
      GateMessage("Beam", 5, "The next particle will be time ID = " << part.initialID << std::endl);
      mNumberOfNextTime = m_gpu_output->particles.front().initialID - mCurrentTimeID;
    }  
    else {
      GateMessage("Beam", 5, "No more particules in gpu buffer, time stay the same." << std::endl);
      mNumberOfNextTime = 1;
    }
  }

  return 1; // Return a single particle at a time
}
//----------------------------------------------------------



//----------------------------------------------------------
void GateSourceGPUVoxellized::GeneratePrimaryEventFromGPUOutput(const GateGPUIO_Particle & particle, 
                                                                G4Event * event)
{
  /*
  std::cout << "From gpu pos = " << particle.px << " " << particle.py << " " << particle.pz << std::endl
            << "         dir = " << particle.dx << " " << particle.dy << " " << particle.dz << std::endl
            << "         E   = " << G4BestUnit(particle.E*MeV, "Energy") << std::endl
            << "         t   = " << G4BestUnit(particle.t*ns, "Time") << std::endl;
  */

  // Position
  G4ThreeVector particle_position;
  particle_position.setX(particle.px*mm-m_gpu_input->phantom_size_x*m_gpu_input->phantom_spacing_x/2.0*mm);
  particle_position.setY(particle.py*mm-m_gpu_input->phantom_size_y*m_gpu_input->phantom_spacing_y/2.0*mm);
  particle_position.setZ(particle.pz*mm-m_gpu_input->phantom_size_z*m_gpu_input->phantom_spacing_z/2.0*mm);

  
/*
G4ThreeVector particle_position;
particle_position.setX(particle.px*mm-256*mm); // FIXME HECTOR to replace by m_gpu_input.phantom_size ...
particle_position.setY(particle.py*mm-126*mm);
particle_position.setZ(particle.pz*mm-92*mm);
*/
  // Create the vertex
  G4PrimaryVertex* vertex;
  double particle_time = particle.t*ns; // assume time is in ns

  /*
  std::cout << " gpu particle time (ns)             = " << particle.t << std::endl;
  std::cout << " gpu particle time check best unit  = " << G4BestUnit(particle_time, "Time") << std::endl;
  std::cout << " Gettime =" << G4BestUnit(GetTime() , "Time") << std::endl;
  std::cout << " a+b =" << G4BestUnit(GetTime()+particle_time, "Time") << std::endl;
  */

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
  
  /*
    std::cout << "Momentum = " << particle_momentum << std::endl;
    std::cout << "Energy = " << particle.E << std::endl;
    std::cout << "Energy = " << G4BestUnit(particle.E, "Energy")  << std::endl;
  */
  
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


//----------------------------------------------------------
void GateSourceGPUVoxellized::ReaderInsert(G4String readerType)
{
  G4cout << "GateSourceGPUVoxellizedMessenger ReaderInsert" << G4endl;
  GateSourceVoxellized::ReaderInsert(readerType);
}
//----------------------------------------------------------


//----------------------------------------------------------
void GateSourceGPUVoxellized::ReaderRemove()
{
  G4cout << "GateSourceGPUVoxellizedMessenger ReaderRemove" << G4endl;
  GateSourceVoxellized::ReaderRemove();
}
//----------------------------------------------------------


//----------------------------------------------------------
void GateSourceGPUVoxellized::Update(double time)
{
  G4cout << "GateSourceGPUVoxellizedMessenger Update" << G4endl;
  return GateSourceVoxellized::Update(time);
}
//----------------------------------------------------------


//----------------------------------------------------------
void GateSourceGPUVoxellized::SetPhantomVolumeData() 
{
  GateVVolume* v = GateObjectStore::GetInstance()->FindVolumeCreator(attachedVolumeName);
  // FindVolumeCreator raise an error if not found
  // FIXME -> change the error message
  
  GateFictitiousVoxelMapParameterized * m = dynamic_cast<GateFictitiousVoxelMapParameterized*>(v);
  if (m == NULL) {
    GateError(attachedVolumeName << " is not a GateFictitiousVoxelMapParameterized.");
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
          // DD(index);
          m_gpu_input->phantom_material_data.push_back(index);
        }
    DD(materials.size());

    // Init the materials
    G4String name = v->GetObjectName();
    GateGPUIO_Input_Init_Materials(m_gpu_input, materials, name);
    DD("mat done");
  }
}
//----------------------------------------------------------
