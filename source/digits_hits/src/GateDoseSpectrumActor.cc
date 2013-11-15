/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
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
  GateDebugMessageInc("Actor",4,"GateDoseSpectrumActor() -- begin"<<G4endl);
  mDosePrimaryOnly = false;
  mCurrentEvent= 0;
  mEventEnergy = -1;
  mLastHitEventImage = 1;
  pMessenger = new GateDoseSpectrumActorMessenger(this);

  GateDebugMessageDec("Actor",4,"GateDoseSpectrumActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor 
GateDoseSpectrumActor::~GateDoseSpectrumActor()
{
  GateDebugMessageInc("Actor",4,"~GateDoseSpectrumActor() -- begin"<<G4endl);
  GateDebugMessageDec("Actor",4,"~GateDoseSpectrumActor() -- end"<<G4endl);
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
  EnableEndOfEventAction(true); // for save every n
  mVolumeMass = GetVolume()->GetPhysicalVolume()->GetLogicalVolume()->GetMass();
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
  //G4int suma = 0;
  G4double numPart;
  G4double energyDose, doseEnergy, averageDoseEnergy, averageDoseEnergySquare, errorStandardDoseEnergy, averageEnergyTotal;
  std::map< G4double, G4double>::iterator itermDoseEnergy;
  for( itermDoseEnergy = mDoseEnergy.begin(); itermDoseEnergy != mDoseEnergy.end(); itermDoseEnergy++)
  {
    energyDose = itermDoseEnergy->first;
    //suma = suma + mNumParticPerEnergy[energyDose];
    //G4cout << "itermDoseEnergy: " << energyDose << " doseEnergy: " <<  mDoseEnergy[energyDose] << " mNumParticPerEnergy: " << mNumParticPerEnergy[energyDose] << " suma: " << suma << G4endl;
    doseEnergy = mDoseEnergy[energyDose];
    numPart = mNumParticPerEnergy[energyDose];
    averageDoseEnergy = doseEnergy / numPart;
    averageEnergyTotal = averageDoseEnergy*mVolumeMass;
    averageDoseEnergySquare = mDoseEnergySquare[energyDose] / numPart;
    errorStandardDoseEnergy = sqrt( (1.0 / ( numPart - 1)) * ( averageDoseEnergySquare - pow( averageDoseEnergy/gray, 2)));
    DoseResponseFile << "# energydose: " << energyDose/keV << " " << averageEnergyTotal/keV << " " << doseEnergy/gray << " " << averageDoseEnergy/gray << " " << errorStandardDoseEnergy  << std::endl;
  }

  if (!DoseResponseFile)
  {
    GateMessage("Output",1,"Error Writing file: " << mSaveFilename << G4endl);
  }
  DoseResponseFile.flush();
  DoseResponseFile.close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDoseSpectrumActor::ResetData()
{
  mDoseEnergy.clear();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseSpectrumActor::BeginOfRunAction(const G4Run *)
{
  GateDebugMessage("Actor", 3, "GateDoseSpectrumActor -- Begin of Run" << G4endl);
  ResetData();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseSpectrumActor::BeginOfEventAction(const G4Event* event)
{
  GateDebugMessage("Actor", 3, "GateDoseSpectrumActor -- Begin of Event" << G4endl);

  mOldEventEnergy = mEventEnergy;
  mEventEnergy = event->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy();

  mCurrentEvent = mCurrentEvent + 1;
  G4int numEventEnergy = 1;
  mNumParticPerEnergy[mEventEnergy] += numEventEnergy;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------


void GateDoseSpectrumActor::UserSteppingAction(const GateVVolume *, const G4Step* step)
{
  GateScatterOrderTrackInformation * info = dynamic_cast<GateScatterOrderTrackInformation *>(step->GetTrack()->GetUserInformation());
  if( mDosePrimaryOnly && step->GetTrack()->GetParticleDefinition()->GetParticleName() == "gamma" && info->GetScatterOrder())
  {
    step->GetTrack()->SetTrackStatus(fKillTrackAndSecondaries);
  }
  else
  {
    bool sameEvent = true;
    if(mCurrentEvent != mLastHitEventImage)
    {
      sameEvent = false;
      mLastHitEventImage = mCurrentEvent;
    }
    if(!sameEvent)
    {
      mDoseEnergySquare[mOldEventEnergy] += (mDoseEnergyTemp/gray)*(mDoseEnergyTemp/gray);
      mDoseEnergy[mOldEventEnergy] += mDoseEnergyTemp;
      mDoseEnergyTemp = 0;
    }

    G4double energyDepot = step->GetTotalEnergyDeposit();
    G4double doseEnergy = (energyDepot/mVolumeMass);
    mDoseEnergyTemp += doseEnergy;
  }
}

//-----------------------------------------------------------------------------



//#endif
