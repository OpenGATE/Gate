/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/
#include "GateConfiguration.h"
#include "GateDoseSpectrumActor.hh"
#include "GateMiscFunctions.hh"
#include "G4VProcess.hh"
//#ifdef G4ANALYSIS_USE_ROOT
#include "GateScatterOrderTrackInformationActor.hh"
#include "GateDoseSpectrumActorMessenger.hh"

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateDoseSpectrumActor::GateDoseSpectrumActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateDoseSpectrumActor() -- begin\n");
  mDosePrimaryOnly = false;
  mCurrentEvent= 0;
  mEventEnergy = -1;
  mTotalEventEnergyDep = 0.;
  pMessenger = new GateDoseSpectrumActorMessenger(this);

  GateDebugMessageDec("Actor",4,"GateDoseSpectrumActor() -- end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor 
GateDoseSpectrumActor::~GateDoseSpectrumActor()
{
  GateDebugMessageInc("Actor",4,"~GateDoseSpectrumActor() -- begin\n");
  GateDebugMessageDec("Actor",4,"~GateDoseSpectrumActor() -- end\n");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateDoseSpectrumActor::Construct()
{
  GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnableUserSteppingAction(true);
  EnableEndOfEventAction(true);
  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateDoseSpectrumActor::SaveData()
{
  GateVActor::SaveData();
  std::ofstream DoseResponseFile;
  OpenFileOutput(mSaveFilename, DoseResponseFile);
  DoseResponseFile << "Incoming energy (keV)" << " "
                   << "Average energy deposit (keV)" << " "
                   << "Energy uncertainty (keV)"  << Gateendl;
  std::map< G4double, G4double>::iterator itermEnergy;
  for( itermEnergy = mEnergy.begin(); itermEnergy != mEnergy.end(); itermEnergy++)
  {
    G4double energyIn = itermEnergy->first;
    G4double totEnergyOut = mEnergy[energyIn];
    G4double numPart = mNumParticPerEnergy[energyIn];
    G4double avgEnergyOut = totEnergyOut / numPart;
    G4double avgEnergySq  = mEnergySquare[energyIn] / numPart;
    G4double error = sqrt( (1.0 / ( numPart - 1)) * ( avgEnergySq - pow( avgEnergyOut, 2)));
    DoseResponseFile << energyIn/keV << " "
                     << avgEnergyOut/keV << " "
                     << error/keV  << Gateendl;
  }

  if (!DoseResponseFile)
  {
    GateMessage("Output",1,"Error Writing file: " << mSaveFilename << Gateendl);
  }
  DoseResponseFile.close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDoseSpectrumActor::ResetData()
{
  mNumParticPerEnergy.clear();
  mEnergy.clear();
  mEnergySquare.clear();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseSpectrumActor::BeginOfRunAction(const G4Run *)
{
  GateDebugMessage("Actor", 3, "GateDoseSpectrumActor -- Begin of Run\n");
  ResetData();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseSpectrumActor::BeginOfEventAction(const G4Event* event)
{
  GateDebugMessage("Actor", 3, "GateDoseSpectrumActor -- Begin of Event\n");

  // Process the previous event (if there was one)
  if(mCurrentEvent) {
    mEnergy[mEventEnergy]       += mTotalEventEnergyDep;
    mEnergySquare[mEventEnergy] += mTotalEventEnergyDep*mTotalEventEnergyDep;
  }

  // Prepare the new event
  mTotalEventEnergyDep = 0.;
  mEventEnergy = event->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy();
  mCurrentEvent++;
  if(mNumParticPerEnergy.find(mEventEnergy)==mNumParticPerEnergy.end()) {
    mNumParticPerEnergy[mEventEnergy] = 1;
    mEnergy[mEventEnergy] = 0.;
    mEnergySquare[mEventEnergy] = 0.;
  }
  else
    mNumParticPerEnergy[mEventEnergy]++;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseSpectrumActor::UserSteppingAction(const GateVVolume *, const G4Step* step)
{
  GateScatterOrderTrackInformation * info = dynamic_cast<GateScatterOrderTrackInformation *>(step->GetTrack()->GetUserInformation());
  if( mDosePrimaryOnly &&
      step->GetTrack()->GetParticleDefinition()->GetParticleName() == "gamma" &&
      info->GetScatterOrder() ) {
    step->GetTrack()->SetTrackStatus(fKillTrackAndSecondaries);
  }
  else {
    mTotalEventEnergyDep += step->GetTotalEnergyDeposit()*step->GetTrack()->GetWeight();
  }
}
//-----------------------------------------------------------------------------



//#endif
