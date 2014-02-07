/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*
  \brief Class GatePhaseSpaceBrentActor :
  \brief
*/

#ifndef GatePhaseSpaceBrentACTOR_CC
#define GatePhaseSpaceBrentACTOR_CC

#include "GatePhaseSpaceBrentActor.hh"
#include "GatePhaseSpaceActor.hh"
#include <G4NistManager.hh>
//#include "GeantTrackInformation.hh"

//-----------------------------------------------------------------------------
GatePhaseSpaceBrentActor::GatePhaseSpaceBrentActor(G4String name, G4int depth):
  GatePhaseSpaceActor(name,depth) {
  GateDebugMessageInc("Actor",4,"GatePhaseSpaceBrentActor() -- begin"<<G4endl);
  pMessenger = new GatePhaseSpaceBrentActorMessenger(this);
  GateDebugMessageDec("Actor",4,"GatePhaseSpaceBrentActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GatePhaseSpaceBrentActor::~GatePhaseSpaceBrentActor()  {
  delete pMessenger;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Construct
void GatePhaseSpaceBrentActor::Construct() {
  GateDebugMessageInc("Actor", 4, "GatePhaseSpaceBrentActor -- Construct - begin" << G4endl);
  GatePhaseSpaceActor::Construct();
  // Enable callbacks
  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(true);
  EnableUserSteppingAction(true);

  if(mFileType == "rootFile"){
    pListeVar->Branch("primaryEnergy", &primaryEnergy,"primaryEnergy/F");
  }

  GateMessageDec("Actor", 4, "GatePhaseSpaceBrentActor -- Construct - end" << G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// Save data
void GatePhaseSpaceBrentActor::SaveData() {
  GatePhaseSpaceActor::SaveData();
  //G4cout << "BrentActorSaved!!!!!" << G4endl;

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------

void GatePhaseSpaceBrentActor::ResetData() {
  GatePhaseSpaceActor::ResetData();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------

void GatePhaseSpaceBrentActor::UserSteppingAction(const GateVVolume * vol, const G4Step* step){
  // Add code here that builds a histo over the length of the half-cylinder with particle count
  // Wish: find way to build these histos depending on primary proton energy.
  GatePhaseSpaceActor::UserSteppingAction(vol,step);
  
  //G4cout << "Allo Brent - UserSteppingAction" << G4endl;
  //GeantTrackInformation* info = (GeantTrackInformation*)(step->GetTrack()->GetUserInformation());
  //G4cout << info->GetOriginalEnergy() << G4endl;
  //G4cout << " Original Track ID " << info->GetOriginalTrackID() << G4endl;
  
  //DD() can print most/any Geant/Gate Actor with some usefull details.
  
  //DD(GetVolume()->GetSolidName());
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Callback at each event
void GatePhaseSpaceBrentActor::BeginOfEventAction(const G4Event * e) {
  GatePhaseSpaceActor::BeginOfEventAction(e);
  primaryEnergy = e->GetPrimaryVertex()->GetPrimary()->GetKineticEnergy();
  //G4cout << primaryEnergy << G4endl;
  GateDebugMessage("Actor", 3, "GatePhaseSpaceBrentActor -- Begin of Event: " << mCurrentEvent << G4endl);
}
//-----------------------------------------------------------------------------


#endif /* end #define GatePhaseSpaceBrentACTOR_CC */
