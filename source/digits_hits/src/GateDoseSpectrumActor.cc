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
#ifdef G4ANALYSIS_USE_ROOT

//#include "GateDoseSpectrumActorMessenger.hh"

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateDoseSpectrumActor::GateDoseSpectrumActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateDoseSpectrumActor() -- begin"<<G4endl);

  mEnergyDepot = 0.;

  pMessenger = new GateActorMessenger(this);

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

  //GateVVolume * v = GetVolume();
  //G4VPhysicalVolume * phys = v->GetPhysicalVolume();
  //G4double mVoxelVolume = GetVolume()->GetPhysicalVolume()->GetLogicalVolume()->GetSolid()->GetCubicVolume();
  mVolumeMass = GetVolume()->GetPhysicalVolume()->GetMotherLogical()->GetMass();
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
  G4cout << " size: " << mDoseSpectrum.size() << G4endl;
  for( itermDoseSpectrum = mDoseSpectrum.begin(); itermDoseSpectrum != mDoseSpectrum.end(); itermDoseSpectrum++)
  {
    DoseResponseFile << itermDoseSpectrum->first << " " << itermDoseSpectrum->second << std::endl;
  }

  if (!DoseResponseFile) {
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
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseSpectrumActor::BeginOfEventAction(const G4Event* event)
{
  GateDebugMessage("Actor", 3, "GateDoseSpectrumActor -- Begin of Event" << G4endl);
  mEnergyDepot = 0.;
//  if( mEnergyFlag[event->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy()] == true)
//  {
//    mEnergyFlag[event->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy()] = false;
//  }
//  mEnergyFlag[event->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy()] = true;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateDoseSpectrumActor::EndOfEventAction(const G4Event* event)
{
  GateDebugMessage("Actor", 3, "GateDoseSpectrumActor -- End of Event" << G4endl);
  if (mEnergyDepot > 0)
  {
    G4double a = mDoseSpectrum[event->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy()];
//    G4cout << "a: " << a << " ener: " << event->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy() << G4endl;
//    if ( mEnergyFlag[event->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy()])
//    {
//      G4cout << "energy: " << event->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy() << G4endl;
//      mDoseSpectrum[event->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy()] = 0;
//    }
    mDoseSpectrum[event->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy()] += mEnergyDepot/mVolumeMass;
  }
}
//-----------------------------------------------------------------------------


void GateDoseSpectrumActor::UserSteppingAction(const GateVVolume *, const G4Step* step)
{
  mEnergyDepot += step->GetTotalEnergyDeposit();
}
//-----------------------------------------------------------------------------



#endif
