/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateComptonCameraActorMessenger
*/

#include <G4SystemOfUnits.hh>
#include "GateComptonCameraActorMessenger.hh"
#include "GateComptonCameraActor.hh"

//-----------------------------------------------------------------------------
GateComptonCameraActorMessenger::GateComptonCameraActorMessenger(GateComptonCameraActor * v)
  : GateActorMessenger(v),
    pActor(v)
{
    G4cout<<"buildCommand ComptonCamera messenger"<<G4endl;
  BuildCommands(baseName+pActor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateComptonCameraActorMessenger::~GateComptonCameraActorMessenger()
{
  delete pSaveHitsTree;
  delete pSaveSinglesText;
  delete pSaveCoincidenceText;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActorMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;

  //  bb = base+"/saveAsText";
  //  pSaveAsText = new G4UIcmdWithABool(bb, this);
  //  guidance = G4String("In addition to root output files, also write .txt files (that can be open as a source, 'UserSpectrum')");
  //  pSaveAsText->SetGuidance(guidance);

  bb = base+"/saveHitRootTree";
  pSaveHitsTree = new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition  save a root tree wit the hit info inside the attachedVolume");
  pSaveHitsTree->SetGuidance(guidance);

  bb = base+"/saveSinglesText";
  pSaveSinglesText = new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition  save a text file with singles info");
  pSaveSinglesText->SetGuidance(guidance);

  bb = base+"/saveCoincidencesText";
  pSaveCoincidenceText= new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition  save a text file with coincidence info");
  pSaveCoincidenceText->SetGuidance(guidance);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if(cmd == pSaveHitsTree) pActor->SetSaveHitsTreeFlag(  pSaveHitsTree->GetNewBoolValue(newValue)  ) ;
  if(cmd == pSaveSinglesText){
       G4cout<<"entra en comando singles"<<G4endl;
      pActor->SetSaveSinglesTextFlag(  pSaveSinglesText->GetNewBoolValue(newValue)  ) ;
  }
  if(cmd == pSaveCoincidenceText) pActor->SetSaveCoincidenceTextFlag(  pSaveCoincidenceText->GetNewBoolValue(newValue)  ) ;
  GateActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------
