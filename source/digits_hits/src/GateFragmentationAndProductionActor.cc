/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \brief Class GateFragmentationAndProductionActor :
  \brief
 */

#include "GateFragmentationAndProductionActor.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateMiscFunctions.hh"
#include "G4VProcess.hh"

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateFragmentationAndProductionActor::GateFragmentationAndProductionActor(G4String name, G4int depth):
  GateVActor(name,depth), pNBins(100)
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

  delete pMessenger;


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
  pTFile = new TFile(mSaveFilename,"RECREATE");

  double halfLength = mVolume->GetHalfDimension(2);
  GateMessage("Actor", 0, "GateFragmentationAndProductionActor -- Construct -- halfLength=" << halfLength << " nBins=" << pNBins << G4endl);

  pGammaProduction = new TH1D("gammaProduction","Gamma production",pNBins,-halfLength,halfLength);
  pGammaProduction->SetXTitle("z [mm]");
  pGammaProduction->SetYTitle("count");

  pNeutronProduction = new TH1D("neutronProduction","Neutron production",pNBins,-halfLength,halfLength);
  pNeutronProduction->SetXTitle("z [mm]");
  pNeutronProduction->SetYTitle("count");

  pFragmentation = new TH1D("fragmentation","Fragmentation",pNBins,-halfLength,halfLength);
  pFragmentation->SetXTitle("z [mm]");
  pFragmentation->SetYTitle("count");

  pNEvent = new TVector2(0,0);

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
  GateVActor::SaveData();
  GateMessage("Actor", 0, "GateFragmentationAndProductionActor -- Saving data to " << mSaveFilename << G4endl);
  pTFile->cd();
  pNEvent->Write("nevents");
  pTFile->Write();
  //pTFile->Close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateFragmentationAndProductionActor::ResetData()
{
  pNEvent->Set(0.f,0.f);
  pGammaProduction->Reset();
  pNeutronProduction->Reset();
  pFragmentation->Reset();
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
  pNEvent->Set(pNEvent->X()+1,0);
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
void GateFragmentationAndProductionActor::PostUserTrackingAction(const GateVVolume *, const G4Track* /*t*/)
{
  GateDebugMessage("Actor", 3, "GateFragmentationAndProductionActor -- End of Track" << G4endl);
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

//G4String getProcessName(const G4StepPoint *point) {
//  if (!point) return "nopoint";
//  const G4VProcess *process = point->GetProcessDefinedStep();
//  if (!process) return "noprocess";
//  return process->GetProcessName();
//}

//G4bool GateFragmentationAndProductionActor::ProcessHits(G4Step * step , G4TouchableHistory* /*th*/)
void GateFragmentationAndProductionActor::UserSteppingAction(const GateVVolume *, const G4Step* step)
{
  const G4StepPoint *point = step->GetPostStepPoint();
  assert(point);
  const G4String &processName = point->GetProcessDefinedStep()->GetProcessName();
  if (processName =="IonInelastic" ||
      processName =="NeutronInelastic" ||
      processName =="AlphaInelastic" ||
      processName =="ProtonInelastic" ||
      processName =="DeuteronInelastic" ||
      processName =="TritonInelastic") {
    double zfrag = (step->GetPostStepPoint()->GetPosition() + step->GetPreStepPoint()->GetPosition())[2]/2.;
    pFragmentation->Fill(zfrag,point->GetWeight());
  }
  //const G4String &name = step->GetTrack()->GetDefinition()->GetParticleName();
  //if (name=="e-") return;
  //G4cout << "name=" << name << G4endl;
  //G4cout << "trackid=" << step->GetTrack()->GetTrackID() << " stepnumber=" << step->GetTrack()->GetCurrentStepNumber() << G4endl;
  //G4cout << "prepoint=" << getProcessName(step->GetPreStepPoint()) << " postpoint=" << getProcessName(step->GetPostStepPoint()) << G4endl;
}
//-----------------------------------------------------------------------------



#endif
