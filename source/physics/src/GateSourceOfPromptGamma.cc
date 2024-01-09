/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateConfiguration.h"
#include "GateSourceOfPromptGamma.hh"
#include "GateRandomEngine.hh"
#include "GateApplicationMgr.hh"
#include "GateSourceMgr.hh"

#include "GateSourceOfPromptGammaData.hh"
#include "GateImageOfHistograms.hh"
#include <iostream>
#include <fstream>
#include "G4ParticleTable.hh"
#include "G4Event.hh"
#include "G4Gamma.hh"

//------------------------------------------------------------------------
GateSourceOfPromptGamma::GateSourceOfPromptGamma(G4String name)
  :GateVSource( name )
{
  pMessenger = new GateSourceOfPromptGammaMessenger(this);
  // Create data object (will be initialized later)
  mData = new GateSourceOfPromptGammaData;
  mIsInitializedFlag = false;
  mIsInitializedNumberOfPrimariesFlag = false;
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
void GateSourceOfPromptGamma::SetTof(G4bool newflag)
{
  mData->SetTof(newflag);
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GateSourceOfPromptGamma::Initialize()
{

  if (mIsInitializedFlag) return;
  // Get filename, load data
  mData->LoadData(mFilename);

  // Compute cmulative marginals information
  mData->Initialize();

  // Particle type is photon. Could not be initialize here.

  // Weight is fixed for the moment (could change in the future)
  //particle_weight = 1.0; // could not be initialized here

  // Weight is abused to scale to convert number of proton primaries
  // into number of gamma primaries. See GateApplicationMgrMessenger.cc
  // WILL NOT WORK WITH SEVERAL SOURCES !
  SetSourceWeight(mData->ComputeSum());

  // It is initialized
  mIsInitializedFlag = true;
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GateSourceOfPromptGamma::InitializeNumberOfPrimaries()
{
  // NOT USED, because has no effect (anymore?)

  // The user set the number of primaries in number of proton. To
  // generate the corresponding number of gamma we scale according to
  // the sum in the source. We then cheat by changing the timeStop of
  // the application. WILL NOT WORK WITH SEVERAL SOURCES !

  GateApplicationMgr* appMgr = GateApplicationMgr::GetInstance();
  GateSourceMgr * sourceMgr = GateSourceMgr::GetInstance();
  if (sourceMgr->GetNumberOfSources() != 1) {
    GateError("When using SourceOfPromptGamma, only use a single source. Abort.");
  }

  double np = appMgr->GetTotalNumberOfPrimaries();
  double sum = mData->ComputeSum();
  double ng = np*sum;
  double timestep = appMgr->GetTimeStepInTotalAmountOfPrimariesMode();

  double newTimeStop = timestep*ng;
  appMgr->SetTimeStop(newTimeStop);//set new endtime to timestep*newnumberofprims

  GateMessage("Run", 0, "Requested number of proton is " << np
              << ". According to the data source, it corresponds to "
              << ng << " gammas." << std::endl);

  // It is initialized
  mIsInitializedNumberOfPrimariesFlag = true;
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
void GateSourceOfPromptGamma::GenerateVertex(G4Event* aEvent)
{
  // Initialisation of the distribution information (only once)
  if (!mIsInitializedFlag) Initialize();
  //if (!mIsInitializedNumberOfPrimariesFlag) InitializeNumberOfPrimaries();

  // Position
  G4ThreeVector particle_position;
  mData->SampleRandomPosition(particle_position);

  // The position coordinate is expressed in the coordinate system
  // (CS) of the volume it was attached to during the TLEActor
  // simulation. Now we convert the coordinates into world
  // coordinates.
  ChangeParticlePositionRelativeToAttachedVolume(particle_position);

  // Energy
  mData->SampleRandomEnergy(mEnergy);

  // Time
  mTime = GetParticleTime();
  mData->SampleRandomPgtime(mTime);

  // Direction
  G4ParticleMomentum particle_direction;
  mData->SampleRandomDirection(particle_direction);
  ChangeParticleMomentumRelativeToAttachedVolume(particle_direction);

  // Momentum
  double mass = GetParticleDefinition()->GetPDGMass();
  double pmom = std::sqrt(mEnergy*mEnergy-mass*mass);
  double d = std::sqrt(pow(particle_direction[0],2) +
                       pow(particle_direction[1],2) +
                       pow(particle_direction[2],2));
  double px = pmom * particle_direction[0]/d;
  double py = pmom * particle_direction[1]/d;
  double pz = pmom * particle_direction[2]/d;

  // Create vertex
  G4PrimaryParticle* particle =
    new G4PrimaryParticle(G4Gamma::Gamma(), px, py, pz);
  G4PrimaryVertex* vertex;
  //vertex = new G4PrimaryVertex(particle_position, GetParticleTime()); 
  vertex = new G4PrimaryVertex(particle_position, mTime); 
  vertex->SetWeight(1.0); // FIXME
  vertex->SetPrimary(particle);
  vertex->SetT0(mTime); /** Modif Oreste **/
  aEvent->AddPrimaryVertex(vertex);
}
//------------------------------------------------------------------------


//------------------------------------------------------------------------
G4int GateSourceOfPromptGamma::GeneratePrimaries(G4Event* event)
{
  //  This does not work unfortunately. Events are not running, AbortingRun will just create newruns.
  //  TerminateEvenloop seems to have no effect.
  //  if(event->GetEventID()>nrGammaPrim){
  //      GateRunManager::GetRunManager()->AbortRun();
  //      GateRunManager::GetRunManager()->AbortEvent();
  //      GateRunManager::GetRunManager()->TerminateEventLoop();
  //      return 0;
  //  }
  GenerateVertex(event);
  G4PrimaryParticle  * p = event->GetPrimaryVertex(0)->GetPrimary(0);
  GateMessage("Beam", 3, "(" << event->GetEventID() << ") " << p->GetG4code()->GetParticleName()
              << " pos=" << event->GetPrimaryVertex(0)->GetPosition()
              << " weight=" << p->GetWeight()
              << " energy=" << G4BestUnit(mEnergy, "Energy")
              << " mom=" << p->GetMomentum()
              << " ptime=" <<  G4BestUnit(p->GetProperTime(), "Time")
              << " atime=" <<  G4BestUnit(GetTime(), "Time")
              << ")" << G4endl);
  return 1; // a single vertex
}
//------------------------------------------------------------------------
