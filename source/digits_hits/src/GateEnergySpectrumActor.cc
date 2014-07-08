/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateEnergySpectrumActor.hh"
#ifdef G4ANALYSIS_USE_ROOT

#include "GateEnergySpectrumActorMessenger.hh"

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateEnergySpectrumActor::GateEnergySpectrumActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateEnergySpectrumActor() -- begin"<<G4endl);

  mEmin = 0.;
  mEmax = 50.;
  mENBins = 100;

  mEdepmin = 0.;
  mEdepmax = 50.;
  mEdepNBins = 100;

  Ei = 0.;
  Ef = 0.;
  newEvt = true;
  newTrack = true;
  sumNi=0.;
  nTrack=0;
  sumM1=0.;
  sumM2=0.;
  sumM3=0.;
  edep = 0.;

  pMessenger = new GateEnergySpectrumActorMessenger(this);

  GateDebugMessageDec("Actor",4,"GateEnergySpectrumActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateEnergySpectrumActor::~GateEnergySpectrumActor()
{
  GateDebugMessageInc("Actor",4,"~GateEnergySpectrumActor() -- begin"<<G4endl);



  GateDebugMessageDec("Actor",4,"~GateEnergySpectrumActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateEnergySpectrumActor::Construct()
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

  pEnergySpectrum = new TH1D("energySpectrum","Energy Spectrum",GetENBins(),GetEmin() ,GetEmax() );
  pEnergySpectrum->SetXTitle("Energy (MeV)");

  pEdep  = new TH1D("edepHisto","Energy deposited per event",GetEdepNBins(),GetEdepmin() ,GetEdepmax() );
  pEdep->SetXTitle("E_{dep} (MeV)");

  pEdepTime  = new TH2D("edepHistoTime","Energy deposited with time per event",GetEdepNBins(),0,20,GetEdepNBins(),GetEdepmin(),GetEdepmax());
  pEdepTime->SetXTitle("t (ns)");
  pEdepTime->SetYTitle("E_{dep} (MeV)");

  pEdepTrack  = new TH1D("edepTrackHisto","Energy deposited per track",GetEdepNBins(),GetEdepmin() ,GetEdepmax() );
  pEdepTrack->SetXTitle("E_{dep} (MeV)");

  pDeltaEc = new TH1D("eLossHisto","Energy loss",GetEdepNBins(),GetEdepmin() ,GetEdepmax() );
  pDeltaEc ->SetXTitle("E_{loss} (MeV)");

  ResetData();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateEnergySpectrumActor::SaveData()
{
  GateVActor::SaveData();
  pTfile->Write();
  //pTfile->Close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEnergySpectrumActor::ResetData()
{
  pEnergySpectrum->Reset();
  pEdep->Reset();
  pEdepTime->Reset();
  pEdepTrack->Reset();
  pDeltaEc->Reset();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateEnergySpectrumActor::BeginOfRunAction(const G4Run *)
{
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- Begin of Run" << G4endl);
  ResetData();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateEnergySpectrumActor::BeginOfEventAction(const G4Event*)
{
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- Begin of Event" << G4endl);
  newEvt = true;
  edep = 0.;
  tof  = 0;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateEnergySpectrumActor::EndOfEventAction(const G4Event*)
{
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- End of Event" << G4endl);
  if (edep > 0)
  {
	  //G4cout << "hitted " << edep/MeV << "MeV " << tof/MeV << "ns" << G4endl;
	  pEdep->Fill(edep/MeV);
	  pEdepTime->Fill(tof/ns,edep/MeV);
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateEnergySpectrumActor::PreUserTrackingAction(const GateVVolume *, const G4Track* t)
{
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- Begin of Track" << G4endl);
  newTrack = true; //nTrack++;
  if(t->GetParentID()==1) nTrack++;
  edepTrack = 0.;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateEnergySpectrumActor::PostUserTrackingAction(const GateVVolume *, const G4Track* t)
{
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- End of Track" << G4endl);

  double eloss = Ei-Ef;
  if (eloss > 0) pDeltaEc->Fill(eloss/MeV,t->GetWeight() );
  if (edepTrack > 0)  pEdepTrack->Fill(edepTrack/MeV,t->GetWeight() );
}
//-----------------------------------------------------------------------------

//G4bool GateEnergySpectrumActor::ProcessHits(G4Step * step , G4TouchableHistory* /*th*/)
void GateEnergySpectrumActor::UserSteppingAction(const GateVVolume *, const G4Step* step)
{
  //sumNi+=step->fNonIonizingEnergyDeposit;//GetNonIonizingEnergyDeposit();
  /*if(step->GetPreStepPoint()->GetProcessDefinedStep() )
if(step->GetPreStepPoint()->GetProcessDefinedStep()->GetProcessName()!="eIonisation" )
  { G4cout<<nTrack<<"  "<<step->GetPreStepPoint()->GetProcessDefinedStep()->GetProcessName()<<"  "<< step->GetTotalEnergyDeposit()/ (step->GetPreStepPoint()->GetKineticEnergy()-step->GetPostStepPoint()->GetKineticEnergy())<<G4endl;
sumNi+=step->GetTotalEnergyDeposit();}
  if(step->GetPostStepPoint()->GetProcessDefinedStep())*/
//if(step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName()!="ElectronIonisation" )
 //   G4cout<<"post "<<step->GetPostStepPoint()->GetProcessDefinedStep()->GetProcessName()<<G4endl;

  assert(step->GetTrack()->GetWeight() == 1.); // edep doesnt handle weight

  if(step->GetTotalEnergyDeposit()>0.01) sumM1+=step->GetTotalEnergyDeposit();
  else if(step->GetTotalEnergyDeposit()>0.00001) sumM2+=step->GetTotalEnergyDeposit();
  else sumM3+=step->GetTotalEnergyDeposit();

  edep += step->GetTotalEnergyDeposit();
  edepTrack += step->GetTotalEnergyDeposit();

  //cout << "--- " << step->GetTrack()->GetTrackID() << " " << step->GetTrack()->GetParentID() << endl;
  if (newEvt) {
    double pretof = step->GetPreStepPoint()->GetGlobalTime();
    double posttof = step->GetPostStepPoint()->GetGlobalTime();
    tof = pretof + posttof;
    tof /= 2;
    //cout << "****************** new event tof=" << pretof << "/" << posttof << "/" << tof << " edep=" << edep << endl;
    newEvt = false;
  } else {
    double pretof = step->GetPreStepPoint()->GetGlobalTime();
    double posttof = step->GetPostStepPoint()->GetGlobalTime();
    double ltof = pretof + posttof;
    ltof /= 2;
    //cout << "****************** diff tof=" << ltof << " edep=" << edep << endl;
  }


  Ef=step->GetPostStepPoint()->GetKineticEnergy();
  if(newTrack){
     Ei=step->GetPreStepPoint()->GetKineticEnergy();
     pEnergySpectrum->Fill(Ei/MeV,step->GetTrack()->GetWeight());
     newTrack=false;
  }


  //return true;
}
//-----------------------------------------------------------------------------

#endif
