/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/
#ifdef G4ANALYSIS_USE_ROOT

/*
  \brief Class GateFragmentationAndProductionActor : 
  \brief 
 */

#ifndef GATEFRAGMENTATIONANDPRODUCTIONACTOR_CC
#define GATEFRAGMENTATIONANDPRODUCTIONACTOR_CC

#include "GateFragmentationAndProductionActor.hh"

#include "GateMiscFunctions.hh"
#include "G4VProcess.hh"

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateFragmentationAndProductionActor::GateFragmentationAndProductionActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateFragmentationAndProductionActor() -- begin"<<G4endl);

  pMessenger = new GateFragmentationAndProductionActorMessenger(this);

  GateDebugMessageDec("Actor",4,"GateFragmentationAndProductionActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor 
GateFragmentationAndProductionActor::~GateFragmentationAndProductionActor() 
{
  GateDebugMessageInc("Actor",4,"~GateFragmentationAndProductionActor() -- begin"<<G4endl);
 


  GateDebugMessageDec("Actor",4,"~GateFragmentationAndProductionActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateFragmentationAndProductionActor::Construct()
{
  GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnablePostUserTrackingAction(true);
  EnableUserSteppingAction(true);
  EnableEndOfEventAction(true); // for save every n

  //mHistName = "Precise/output/EnergySpectrum.root";
  pTfile = new TFile(mSaveFilename,"RECREATE");

  double halfLength = mVolume->GetHalfDimension(2);
  GateMessage("Actor", 0, "GateFragmentationAndProductionActor -- Construct -- halfLength=" << halfLength << G4endl);
  pGammaProduction = new TH1D("gammaProduction","Gamma production",100,-halfLength,halfLength);
  pGammaProduction->SetXTitle("z [mm]");
  pGammaProduction->SetYTitle("count");

  GateMessage("Actor", 0, "GateFragmentationAndProductionActor -- Construct -- halfLength=" << halfLength << G4endl);
  pNeutronProduction = new TH1D("neutronProduction","Neutron production",100,-halfLength,halfLength);
  pNeutronProduction->SetXTitle("z [mm]");
  pNeutronProduction->SetYTitle("count");

  //pEnergySpectrum = new TH1D("energySpectrum","Energy Spectrum",GetENBins(),GetEmin() ,GetEmax() );
  //pEnergySpectrum->SetXTitle("Energy (MeV)");

  //pEdep  = new TH1D("edepHisto","Energy deposited",GetEdepNBins(),GetEdepmin() ,GetEdepmax() );
  //pEdep->SetXTitle("E_{dep} (MeV)");

  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateFragmentationAndProductionActor::SaveData()
{
  GateMessage("Actor", 0, "GateFragmentationAndProductionActor -- Saving data to " << mSaveFilename << G4endl);
  pTfile->Write();
  pTfile->Close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateFragmentationAndProductionActor::ResetData() 
{
  pGammaProduction->Reset();
  pNeutronProduction->Reset();
  //GateWarning("GateFragmentationAndProductionActor -- ResetData not implemented" << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateFragmentationAndProductionActor::BeginOfRunAction(const G4Run *)
{
  GateDebugMessage("Actor", 3, "GateFragmentationAndProductionActor -- Begin of Run" << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateFragmentationAndProductionActor::BeginOfEventAction(const G4Event*)
{
  GateDebugMessage("Actor", 3, "GateFragmentationAndProductionActor -- Begin of Event" << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateFragmentationAndProductionActor::EndOfEventAction(const G4Event*)
{
  GateDebugMessage("Actor", 3, "GateFragmentationAndProductionActor -- End of Event" << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateFragmentationAndProductionActor::PreUserTrackingAction(const GateVVolume *, const G4Track* t) 
{
  GateDebugMessage("Actor", 3, "GateFragmentationAndProductionActor -- Begin of Track" << G4endl);
  const G4String &name = t->GetDefinition()->GetParticleName();
  if (name=="gamma")   { pGammaProduction->Fill(t->GetPosition()[2],t->GetWeight()); }
  if (name=="neutron") { pNeutronProduction->Fill(t->GetPosition()[2],t->GetWeight()); }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateFragmentationAndProductionActor::PostUserTrackingAction(const GateVVolume *, const G4Track* t) 
{
  //GateDebugMessage("Actor", 3, "GateFragmentationAndProductionActor -- End of Track" << G4endl);
  //const G4String &name = t->GetDefinition()->GetParticleName();
  //G4cout << name << " " << (t->GetCreatorProcess()? t->GetCreatorProcess()->GetProcessName():"no process") << G4endl;
  //if (name=="C12[0.0]") {
  //  G4cout << "*** c12 fragment" << G4endl;
  //  const G4Step *step = t->GetStep();
  //  assert(step);
  //  const G4StepPoint *point = step->GetPostStepPoint();
  //  assert(point);
  //  const G4ThreeVector &position = point->GetPosition(); 
  //  G4cout << "*** end track name=" << name << " pos=" << position << position[2] << G4endl;
  //}
}
//-----------------------------------------------------------------------------

//G4bool GateFragmentationAndProductionActor::ProcessHits(G4Step * step , G4TouchableHistory* /*th*/)
void GateFragmentationAndProductionActor::UserSteppingAction(const GateVVolume *, const G4Step* step)
{
}
//-----------------------------------------------------------------------------



#endif /* end #define GATEFRAGMENTATIONANDPRODUCTIONACTOR_CC */
#endif
