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
    delete pSaveSinglesTree;
    delete pSaveCoincidencesTree;
    delete pSaveCoincidenceChainsTree;

    delete pSaveHitsText;
    delete pSaveSinglesText;
    delete pSaveCoincidencesText;
    delete pSaveCoincidenceChainsText;


    delete pNameOfAbsorberSDVol;
    delete pNameOfScattererSDVol;
    delete pNumberofDiffScattererLayers;
    delete pNumberofTotScattererLayers;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActorMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;


  bb = base+"/saveHitsRootTree";
  pSaveHitsTree = new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition  save a root tree wit the hit info inside the attachedVolume");
  pSaveHitsTree->SetGuidance(guidance);

  bb = base+"/saveSinglesRootTree";
  pSaveSinglesTree = new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition  save a root tree wit the singles info inside the attachedVolume");
  pSaveSinglesTree->SetGuidance(guidance);

  bb = base+"/saveCoincidencesRootTree";
  pSaveCoincidencesTree = new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition  save a root tree with the coincidences info inside the attachedVolume");
  pSaveCoincidencesTree->SetGuidance(guidance);

  bb = base+"/saveCoincidenceChainsRootTree";
  pSaveCoincidenceChainsTree = new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition  save a root tree wit the coincidence chain info inside the attachedVolume");
  pSaveCoincidenceChainsTree->SetGuidance(guidance);



  bb = base+"/saveHitsText";
 pSaveHitsText = new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition  save a text file with hits info");
  pSaveHitsText->SetGuidance(guidance);

  bb = base+"/saveSinglesText";
  pSaveSinglesText = new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition  save a text file with singles info");
  pSaveSinglesText->SetGuidance(guidance);

  bb = base+"/saveCoincidencesText";
  pSaveCoincidencesText= new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition  save a text file with coincidence info");
  pSaveCoincidencesText->SetGuidance(guidance);

  bb = base+"/saveCoincidenceChainsText";
  pSaveCoincidenceChainsText= new G4UIcmdWithABool(bb, this);
  guidance = G4String("In addition  save a text file with coincidence chains info");
  pSaveCoincidenceChainsText->SetGuidance(guidance);




  bb = base+"/absorberSDVolume";
  pNameOfAbsorberSDVol = new G4UIcmdWithAString(bb,this);
  guidance = "Specifies the absorber volume to track particles";
  pNameOfAbsorberSDVol->SetGuidance(guidance);
  //pNameOfAbsorberSDVol->SetParameterName(" absorber SD Volume name",false);




  bb = base+"/scattererSDVolume";
  pNameOfScattererSDVol = new G4UIcmdWithAString(bb,this);
  guidance = "Specifies the scatterer  volume to track particles";
  pNameOfScattererSDVol->SetGuidance(guidance);
  //pNameOfScattererSDVol->SetParameterName(" scatterer SD Volume name",false);


  bb = base+"/numberOfDiffScatterers";
  pNumberofDiffScattererLayers = new G4UIcmdWithAnInteger(bb,this);
  guidance = "Specifies the number of different  scatterer layers non repeaters.";
  pNumberofDiffScattererLayers->SetGuidance(guidance);
  //The name of the layers must me the same but for a number. When the repeaters are used that number is set by the copyNumber,
  //if the user generated name, nameNumb Study!!)

  bb = base+"/numberOfTotScatterers";
  pNumberofTotScattererLayers = new G4UIcmdWithAnInteger(bb,this);
  guidance = "Specifies the number of different  scatterer layers non repeaters.";
  pNumberofTotScattererLayers->SetGuidance(guidance);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if(cmd == pSaveHitsTree) pActor->SetSaveHitsTreeFlag(  pSaveHitsTree->GetNewBoolValue(newValue)  ) ;
  if(cmd == pSaveSinglesTree) pActor->SetSaveSinglesTreeFlag(  pSaveSinglesTree->GetNewBoolValue(newValue)  ) ;
  if(cmd == pSaveCoincidencesTree) pActor->SetSaveCoincidencesTreeFlag(  pSaveCoincidencesTree->GetNewBoolValue(newValue)  ) ;
  if(cmd == pSaveCoincidenceChainsTree) pActor->SetSaveCoincidenceChainsTreeFlag(  pSaveCoincidenceChainsTree->GetNewBoolValue(newValue)  ) ;

  if(cmd == pSaveHitsText) pActor->SetSaveHitsTextFlag(  pSaveHitsText->GetNewBoolValue(newValue)  ) ;
  if(cmd == pSaveSinglesText) pActor->SetSaveSinglesTextFlag(  pSaveSinglesText->GetNewBoolValue(newValue)  ) ;
  if(cmd == pSaveCoincidencesText) pActor->SetSaveCoincidenceTextFlag(  pSaveCoincidencesText->GetNewBoolValue(newValue)  ) ;
  if(cmd == pSaveCoincidenceChainsText) pActor->SetSaveCoincidenceChainsTextFlag(  pSaveCoincidenceChainsText->GetNewBoolValue(newValue)  ) ;

  if(cmd == pNumberofDiffScattererLayers) pActor->SetNumberOfDiffScattererLayers( pNumberofDiffScattererLayers->GetNewIntValue(newValue)  ) ;
   if(cmd == pNumberofTotScattererLayers) pActor->SetNumberOfTotScattererLayers( pNumberofTotScattererLayers->GetNewIntValue(newValue)  ) ;
  if(cmd == pNameOfScattererSDVol) pActor->SetNameOfScattererSDVol(newValue) ;
  if(cmd == pNameOfAbsorberSDVol) pActor->SetNameOfAbsorberSDVol( newValue ) ;
  GateActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------
