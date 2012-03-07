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

//-------------------------------------------------------------------------------------------------
GateSourceGPUVoxellized::GateSourceGPUVoxellized(G4String name)
  : GateSourceVoxellized(name), m_gpu_input(NULL)
{
  G4cout << "GateSourceGPUVoxellizedMessenger constructor" << G4endl;

  //  Build input and init (allocate) FIXME to be in Init
  // to init in contructor, fill in messenger
  m_gpu_input = GateSourceGPUVoxellizedInput_new();



  // Create particle definition 
  G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
  gamma_particle_definition = particleTable->FindParticle("gamma");

  m_sourceGPUVoxellizedMessenger = new GateSourceGPUVoxellizedMessenger(this);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
GateSourceGPUVoxellized::~GateSourceGPUVoxellized()
{
  G4cout << "GateSourceGPUVoxellizedMessenger destructor" << G4endl;
  GateSourceGPUVoxellizedInput_delete(m_gpu_input);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
G4double GateSourceGPUVoxellized::GetNextTime(G4double timeNow)
{
  G4cout << "GateSourceGPUVoxellizedMessenger GetNextTime" << G4endl;
  return GateSourceVoxellized::GetNextTime(timeNow);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateSourceGPUVoxellized::Dump(G4int level) 
{
  G4cout << "GateSourceGPUVoxellizedMessenger Dump" << G4endl;
  GateSourceVoxellized::Dump(level);
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
void GateSourceGPUVoxellized::AttachToVolume(const G4String& volume_name)
{
  G4cout << "GateSourceGPUVoxellizedMessenger Attach to " << volume_name << G4endl;
}
//-------------------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------------------
G4int GateSourceGPUVoxellized::GeneratePrimaries(G4Event* event) 
{
  G4cout << "GateSourceGPUVoxellizedMessenger GeneratePrimaries" << G4endl;
  G4cout << m_voxelReader << G4endl;
  if (!m_voxelReader) return 0;

  assert(m_gpu_input);
  if (m_gpu_input->phantom_activity_index == NULL)
  { // import activity to gpu (fill input)
	  ActivityMap buffer = m_voxelReader->GetSourceActivityMap();
	  GateSourceGPUVoxellizedInput_parse_activities(buffer,m_gpu_input);
	  G4cout << "PARSING ACTIVITIES HERE " << buffer.size() << G4endl;
  }


  if (m_gpu_output.particles.empty()) {
    std::cout << "output is empty" << std::endl;

    // Go GPU
    GateGPUGeneratePrimaries(m_gpu_input, m_gpu_output);
    
    std::cout << "End gpu with particles = " << m_gpu_output.particles.size() << std::endl;
  }

  // Generate one particle
  if (!m_gpu_output.particles.empty()) {
    GeneratePrimaryEventFromGPUOutput(m_gpu_output.particles.front(), event);
    m_gpu_output.particles.pop_front();
    std::cout << "even id = " << event->GetEventID() << std::endl;
  }

  return 0; // Number of vertex
}
//-------------------------------------------------------------------------------------------------



//-------------------------------------------------------------------------------------------------
void GateSourceGPUVoxellized::GeneratePrimaryEventFromGPUOutput(GateSourceGPUVoxellizedOutputParticle & particle, G4Event * event)
{
  // Position
  G4ThreeVector particle_position;
  particle_position.setX(particle.px);
  particle_position.setY(particle.py);
  particle_position.setZ(particle.pz);
  std::cout << "Position = " << particle_position << std::endl;
  //FIXME -> change position according to source orientation/position

  // Create the vertex
  G4PrimaryVertex* vertex;
  double particle_time = particle.t;
  vertex = new G4PrimaryVertex(particle_position, particle_time);

  // Direction
  G4ThreeVector particle_direction;
  particle_direction.setX(particle.dx);
  particle_direction.setY(particle.dy);
  particle_direction.setZ(particle.dz);
  std::cout << "Direction = " << particle_direction << std::endl;
  //FIXME -> change position according to source orientation/position

  // Compute momentum
  G4ThreeVector particle_momentum = particle.E * particle_direction.unit();
  std::cout << "Momentum = " << particle_momentum << std::endl;

  // Create a G4PrimaryParticle
  G4PrimaryParticle* g4particle =  new G4PrimaryParticle(gamma_particle_definition, 
                                                         particle_momentum.x(), 
                                                         particle_momentum.y(), 
                                                         particle_momentum.z());
  vertex->SetPrimary( g4particle ); 
  event->AddPrimaryVertex( vertex );
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateSourceGPUVoxellized::ReaderInsert(G4String readerType)
{
  G4cout << "GateSourceGPUVoxellizedMessenger ReaderInsert" << G4endl;
  GateSourceVoxellized::ReaderInsert(readerType);
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateSourceGPUVoxellized::ReaderRemove()
{
  G4cout << "GateSourceGPUVoxellizedMessenger ReaderRemove" << G4endl;
  GateSourceVoxellized::ReaderRemove();
}
//-------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------
void GateSourceGPUVoxellized::Update(double time)
{
  G4cout << "GateSourceGPUVoxellizedMessenger Update" << G4endl;
  return GateSourceVoxellized::Update(time);
}
//-------------------------------------------------------------------------------------------------

