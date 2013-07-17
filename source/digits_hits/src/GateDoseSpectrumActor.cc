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
#ifdef G4ANALYSIS_USE_ROOT
#include "GateScatterOrderTrackInformationActor.hh"
#include "GateDoseSpectrumActorMessenger.hh"

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateDoseSpectrumActor::GateDoseSpectrumActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateDoseSpectrumActor() -- begin"<<G4endl);
  mDosePrimaryOnly = false;
  mEnergyDepot = 0.;

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

  std::map< G4double, G4double>::iterator itermDoseSpectrum;
  for( itermDoseSpectrum = mDoseSpectrum.begin(); itermDoseSpectrum != mDoseSpectrum.end(); itermDoseSpectrum++)
  {
    DoseResponseFile << "# energydose: " << itermDoseSpectrum->first << " " << itermDoseSpectrum->second << std::endl;
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
  mDoseSpectrum.clear();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseSpectrumActor::BeginOfRunAction(const G4Run *)
{
  GateDebugMessage("Actor", 3, "GateDoseSpectrumActor -- Begin of Run" << G4endl);
  ResetData();
  DOSIS = 0.;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseSpectrumActor::BeginOfEventAction(const G4Event* event)
{
  GateDebugMessage("Actor", 3, "GateDoseSpectrumActor -- Begin of Event" << G4endl);
  mEnergyDepot = 0.;
  mEventEnergy = event->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseSpectrumActor::EndOfEventAction(const G4Event* event)
{
  GateDebugMessage("Actor", 3, "GateDoseSpectrumActor -- End of Event" << G4endl);
  if (mEnergyDepot > 0)
  {
    //DOSIS += mEnergyDepot/mVolumeMass;
    //G4cout << "DOSIS[Gy]: " << DOSIS/gray << " DOSIS[G4]: " << DOSIS << G4endl;
    mDoseSpectrum[event->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy()] += (mEnergyDepot/mVolumeMass)/gray;
  }
}
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
    mEnergyDepot += step->GetTotalEnergyDeposit();
  }
}

//-----------------------------------------------------------------------------



#endif
