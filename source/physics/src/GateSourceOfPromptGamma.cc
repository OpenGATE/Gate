/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GateSourceOfPromptGamma.hh"
#include "GateRandomEngine.hh"
#include "G4ParticleTable.hh"
#include "G4Event.hh"

//------------------------------------------------------------------------
GateSourceOfPromptGamma::GateSourceOfPromptGamma(G4String name)
  :GateVSource( name )
{
  DD("GSPGE::constructor");
  pMessenger = new GateSourceOfPromptGammaMessenger(this);
  // Create distribution object (will be initialized later)
  mDistrib = new GatePromptGammaSpatialEmissionDistribution;
  mIsInitializedFlag = false;
  mFilename = "no filename given";
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
GateSourceOfPromptGamma::~GateSourceOfPromptGamma()
{
  delete pMessenger;
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GateSourceOfPromptGamma::SetFilename(G4String filename)
{
  mFilename = filename;
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GateSourceOfPromptGamma::Initialize()
{
  // Get pointer to the random engine
  CLHEP::HepRandomEngine * engine = GateRandomEngine::GetInstance()->GetRandomEngine();
  DD(engine);
  engine->showStatus();

  CLHEP::HepRandomEngine * ee = CLHEP::HepRandom::getTheEngine();
  DD(ee);
  ee->showStatus();
  // mDistrib->SetRandomEngine(engine); FIXME ? Necessary or not ?

  // Get filename, load data
  //  mDistrib->LoadData(mFilename);

  // Compute cmulative marginals information
  mDistrib->Initialize();

  // Particle type (photon)
  DD("Particle definition");
  /*
  G4ParticleTable * particleTable = G4ParticleTable::GetParticleTable();
  particle_definition = particleTable->FindParticle("Gamma");
  if (particle_definition==0) {
    GateError("Could not find Gamma definition" << G4endl);
  }
  particle_weight = 1.0;
  */

  // It is initialized
  mIsInitializedFlag = true;
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GateSourceOfPromptGamma::GenerateVertex(G4Event* aEvent)
{
  DD("GSPGE::GenerateVertex");

  // Initialisation of the distribution information
  if (!mIsInitializedFlag) Initialize();

  // Position
  DD("Particle position");
  G4ThreeVector particle_position;
  mDistrib->SampleRandomPosition(particle_position);
  DD(particle_position);

  // Energy
  DD("Particle Energy");
  double particle_energy;
  mDistrib->SampleRandomEnergy(particle_energy);
  DD(particle_energy);
  //  std::cout << G4BestUnit("Energy", particle_energy) << std::endl;

  // Direction
  DD("Particle momentum direction");
  G4ParticleMomentum particle_direction;
  mDistrib->SampleRandomDirection(particle_direction);
  DD(particle_direction);

  // Momentum
  DD("Particle momentum");
  /*
  double mass = particle_definition->GetPDGMass();
  DD(mass);
  double pmom = std::sqrt(particle_energy*particle_energy-mass*mass);
  double d = std::sqrt(pow(particle_direction[0],2) +
                       pow(particle_direction[1],2) +
                       pow(particle_direction[2],2));
  double px = pmom * particle_direction[0]/d;
  double py = pmom * particle_direction[1]/d;
  double pz = pmom * particle_direction[2]/d;

  // Create vertex
  G4PrimaryParticle* particle =
    new G4PrimaryParticle(particle_definition, px, py, pz);
  G4PrimaryVertex* vertex;
  vertex = new G4PrimaryVertex(particle_position, particle_time);
  vertex->SetWeight(particle_weight);
  vertex->SetPrimary(particle);
  aEvent->AddPrimaryVertex(vertex);
  */

}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
G4int GateSourceOfPromptGamma::GeneratePrimaries(G4Event* event)
{
  GateMessage("Beam", 4, "GeneratePrimaries " << event->GetEventID() << G4endl);
  DD("GSPGE::GeneratePrimaries");
  GenerateVertex(event);
  return 1; // a single vertex
}
//------------------------------------------------------------------------
