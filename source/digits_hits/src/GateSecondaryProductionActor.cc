/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*
  \brief Class GateSecondaryProductionActor :
  \brief
*/



#include "GateSecondaryProductionActor.hh"
#include "GateMiscFunctions.hh"
#include "G4VProcess.hh"
//-----------------------------------------------------------------------------
GateSecondaryProductionActor::GateSecondaryProductionActor(G4String name, G4int depth):
  GateVActor(name,depth) {

  mCurrentEvent=-1;

  pMessenger = new GateActorMessenger(this);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateSecondaryProductionActor::~GateSecondaryProductionActor()  {

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GateSecondaryProductionActor::Construct() {
  GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(true);
  EnableBeginOfEventAction(true);
  EnableEndOfEventAction(false);
  EnablePreUserTrackingAction(true);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(false);

  // Output Filename
  mProdFilename = G4String(removeExtension(mSaveFilename))+"-Secondaries.root";

   pTfile = new TFile(mProdFilename,"RECREATE");



   pFrag = new TH1F("fragments","Fragment production",3,0,1);
   pFrag->SetXTitle("Fragment name");
   pFrag->SetBit(TH1::kCanRebin);


   pFrag->Fill("e- Ioni",0.);

   pFrag->Fill("e-",0.);

   pFrag->Fill("e+",0.);

   pFrag->Fill("e+ Decay",0.);

   pFrag->Fill("gamma (EM)",0.);

   pFrag->Fill("gamma (HAD)",0.);

   pFrag->Fill("gamma Decay",0.);

   pFrag->Fill("gamma (others)",0.);

   pFrag->Fill("proton",0.);

   pFrag->Fill("neutron",0.);

   pFrag->Fill("alpha",0.);

   pFrag->Fill("deuteron",0.);

   pFrag->Fill("triton",0.);

   pFrag->Fill("He3",0.);

   pFrag->Fill("He3[0.0]",0.);

   pFrag->Fill("He4[0.0]",0.);

   pFrag->Fill("He5[0.0]",0.);

   pFrag->Fill("Li5[0.0]",0.);

   pFrag->Fill("Li6[0.0]",0.);

   pFrag->Fill("Li7[0.0]",0.);

   pFrag->Fill("Be7[0.0]",0.);

   pFrag->Fill("Be8[0.0]",0.);

   pFrag->Fill("B8[0.0]",0.);

   pFrag->Fill("B9[0.0]",0.);

   pFrag->Fill("B10[0.0]",0.);

   pFrag->Fill("B11[0.0]",0.);

   pFrag->Fill("C11[0.0]",0.);

   pFrag->Fill("C12[0.0]",0.);

   pFrag->Fill("C13[0.0]",0.);

   pFrag->Fill("C14[0.0]",0.);

   pFrag->Fill("N13[0.0]",0.);

   pFrag->Fill("N14[0.0]",0.);

   pFrag->Fill("N15[0.0]",0.);

   pFrag->Fill("N16[0.0]",0.);

   pFrag->Fill("O14[0.0]",0.);

   pFrag->Fill("O15[0.0]",0.);

   pFrag->Fill("O16[0.0]",0.);

   pFrag->Fill("O17[0.0]",0.);

   pFrag->Fill("O18[0.0]",0.);

   pFrag->Fill("F15[0.0]",0.);

   pFrag->Fill("F16[0.0]",0.);

   pFrag->Fill("F17[0.0]",0.);

   pFrag->Fill("F18[0.0]",0.);


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Save data
void GateSecondaryProductionActor::SaveData() {

  pTfile->Write();
  //pTfile->Close();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateSecondaryProductionActor::ResetData() {
  pFrag->Reset();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateSecondaryProductionActor::BeginOfRunAction(const G4Run * ) {
  GateDebugMessage("Actor", 3, "GateSecondaryProductionActor -- Begin of Run" << G4endl);
  ResetData();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback at each event
void GateSecondaryProductionActor::BeginOfEventAction(const G4Event * ) {
  mCurrentEvent++;
  GateDebugMessage("Actor", 3, "GateSecondaryProductionActor -- Begin of Event: "<<mCurrentEvent << G4endl);
}
//-----------------------------------------------------------------------------



//-----------------------------------------------------------------------------
void GateSecondaryProductionActor::PreUserTrackingAction(const GateVVolume *, const G4Track* t)
{
  if(t->GetTrackID() != 1 ){
    G4String name = t->GetDefinition()->GetParticleName();

    const G4VProcess* process = t->GetCreatorProcess();
    if(process){
      //G4cout<<process->GetProcessName()<<"  "<<process->GetProcessName().find("Decay")<<G4endl;
      if(name=="e-" && process->GetProcessName().find("Ionisation")!=std::string::npos) {name += " Ioni";}
      if(name=="e+" && process->GetProcessName().find("Decay")!=std::string::npos) {name += " Decay";}
      if(name=="gamma"){
	if(process->GetProcessName().find("Decay")!=std::string::npos) {name += " Decay";}
        else if(process->GetProcessName().find("Inelastic")!=std::string::npos) {name += " (HAD)";}
        else if(process->GetProcessTypeName(process->GetProcessType()).find("Electromagnetic")!=std::string::npos) {name += " (EM)";}
	else {name += " (others)";}
      }
    }
   // pFragPos->Fill(name,t->GetVertexPosition().z(),1);
    pFrag->Fill(name,1);
  }
}
//-----------------------------------------------------------------------------
