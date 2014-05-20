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
#include "G4Gamma.hh"

//------------------------------------------------------------------------
GateSourceOfPromptGamma::GateSourceOfPromptGamma(G4String name)
  :GateVSource( name )
{
  DD("GSPGE::constructor");
  pMessenger = new GateSourceOfPromptGammaMessenger(this);
  // Create distribution object (will be initialized later)
  mData = new GateSourceOfPromptGammaData;
  mIsInitializedFlag = false;
  mFilename = "no filename given";
  gamma = 0;
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
  DD("GateSourceOfPromptGamma::Initialize");

  // Get pointer to the random engine
  CLHEP::HepRandomEngine * engine = GateRandomEngine::GetInstance()->GetRandomEngine();
  DD(engine);
  // engine->showStatus();

  CLHEP::HepRandomEngine * ee = CLHEP::HepRandom::getTheEngine();
  DD(ee);
  // ee->showStatus();
  // mData->SetRandomEngine(engine); FIXME ? Necessary or not ?

  // Get filename, load data
  mData->LoadData(mFilename);

  // Compute cmulative marginals information
  mData->Initialize();

  // // Particle type (photon)
  // DD("Particle definition");
  // G4ParticleTable * particleTable = G4ParticleTable::GetParticleTable();
  // SetParticleDefinition(particleTable->FindParticle("Gamma"));
  // DD("here");
  // if (GetParticleDefinition() == 0) {
  //   GateError("Could not find Gamma definition" << G4endl);
  // }


  //particle_weight = 1.0; // FIXME ???

  // It is initialized
  mIsInitializedFlag = true;
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GateSourceOfPromptGamma::GenerateVertex(G4Event* aEvent)
{
  // DD("GSPGE::GenerateVertex");

  // Initialisation of the distribution information
  if (!mIsInitializedFlag) Initialize();

  // Position
  // DD("Particle position");
  G4ThreeVector particle_position;
  mData->SampleRandomPosition(particle_position);
  // DD(particle_position);

  // The position coordinate is expressed in image coordinate
  // system which was in world CS. FIXME documentation
  // DD(mCentreCoords);
  // particle_position.setX(particle_position.x()-mCentreCoords.x());
  // particle_position.setY(particle_position.y()-mCentreCoords.y());
  // particle_position.setZ(particle_position.z()-mCentreCoords.z());
  // DD(particle_position);

  // Energy
  // DD("Particle Energy");
  double particle_energy;
  mData->SampleRandomEnergy(particle_energy);
  // DD(particle_energy);
  //  std::cout << G4BestUnit("Energy", particle_energy) << std::endl;

  // Direction
  // DD("Particle momentum direction");
  G4ParticleMomentum particle_direction;
  mData->SampleRandomDirection(particle_direction);
  // DD(particle_direction);

  // Momentum
  // DD("Particle momentum");
  double mass = GetParticleDefinition()->GetPDGMass();
  // DD(mass);
  double pmom = std::sqrt(particle_energy*particle_energy-mass*mass);
  // DD(pmom);
  double d = std::sqrt(pow(particle_direction[0],2) +
                       pow(particle_direction[1],2) +
                       pow(particle_direction[2],2));
  // DD(d);
  double px = pmom * particle_direction[0]/d;
  double py = pmom * particle_direction[1]/d;
  double pz = pmom * particle_direction[2]/d;
  // DD(px); DD(py); DD(pz);

  // Create vertex
  G4PrimaryParticle* particle =
    new G4PrimaryParticle(G4Gamma::Gamma(), px, py, pz);
  G4PrimaryVertex* vertex;
  vertex = new G4PrimaryVertex(particle_position, GetParticleTime());
  //  vertex->SetWeight(particle_weight); // FIXME
  vertex->SetPrimary(particle);
  aEvent->AddPrimaryVertex(vertex);
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
G4int GateSourceOfPromptGamma::GeneratePrimaries(G4Event* event)
{
  GateMessage("Beam", 4, "GeneratePrimaries " << event->GetEventID() << G4endl);
  // DD("GSPGE::GeneratePrimaries");
  GenerateVertex(event);
  return 1; // a single vertex
}
//------------------------------------------------------------------------
