/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifdef G4ANALYSIS_USE_ROOT

/*
  \brief Class GateEnergySpectrumActor : 
  \brief 
 */

#ifndef GATEENERGYSPECTRUMACTOR_CC
#define GATEENERGYSPECTRUMACTOR_CC

#include "GateEnergySpectrumActor.hh"
#include "GateMiscFunctions.hh"

//-----------------------------------------------------------------------------
/// Constructors (Prototype)
GateEnergySpectrumActor::GateEnergySpectrumActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  GateDebugMessageInc("Actor",4,"GateEnergySpectrumActor() -- begin"<<G4endl);

   mEmin = 0.;
   mEmax = 50.;
   mNBins = 10;
   mEdepmin = 0.;
   mEdepmax = 50.;
   mEdepNBins = 10;
   Ei = 0.;
   Ef = 0.;
   newEvt = true;
   newTrack = true;
   sum =0.;sumNi=0.;nTrack=0;
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

   pEnergySpectrum = new TH1D("energySpectrum","Energy Spectrum",GetNBins(),GetEmin() ,GetEmax() );
   pEnergySpectrum->SetXTitle("Energy (MeV)");

   pEdep  = new TH1D("edepHisto","Energy deposited",GetEdepNBins(),GetEdepmin() ,GetEdepmax() );
   pEdep->SetXTitle("E_{dep} (MeV)");

   pEdepTrack  = new TH1D("edepTrackHisto","Energy deposited by Tracks",GetEdepNBins(),GetEdepmin() ,GetEdepmax() );
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
 pTfile->Write();
    pTfile->Close();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateEnergySpectrumActor::ResetData() 
{

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
  newEvt = true;  edep = 0.;

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateEnergySpectrumActor::EndOfEventAction(const G4Event*)
{
  GateDebugMessage("Actor", 3, "GateEnergySpectrumActor -- End of Event" << G4endl);
  pEdep->Fill(edep);
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
  pDeltaEc->Fill( eloss,t->GetWeight() );
  pEdepTrack->Fill(edepTrack,t->GetWeight() );
  sum+=eloss;
 // G4cout<<sum<<"  "<<sumNi<<"  "<<sumM1+sumM2+sumM3 <<G4endl;

}
//-----------------------------------------------------------------------------
//#include "G4VProcess.hh"
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





  if(step->GetTotalEnergyDeposit()>0.01) sumM1+=step->GetTotalEnergyDeposit();
  else if(step->GetTotalEnergyDeposit()>0.00001) sumM2+=step->GetTotalEnergyDeposit();
  else sumM3+=step->GetTotalEnergyDeposit();

  edep+=step->GetTotalEnergyDeposit()*step->GetTrack()->GetWeight();
  edepTrack+=step->GetTotalEnergyDeposit();

  Ef=step->GetPostStepPoint()->GetKineticEnergy();
  if(newTrack){
     Ei=step->GetPreStepPoint()->GetKineticEnergy();
     pEnergySpectrum->Fill( Ei,step->GetTrack()->GetWeight() ); 
     newTrack=false;  
  }


  //return true;
}
//-----------------------------------------------------------------------------



#endif /* end #define GATEENERGYSPECTRUMACTOR_CC */
#endif
