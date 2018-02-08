

#include "GateComptonCameraActorMessenger.hh"

//#ifdef G4ANALYSIS_USE_ROOT

#include "GateComptonCameraActor.hh"
#include "G4SystemOfUnits.hh"

//-----------------------------------------------------------------------------
GateComptonCameraActorMessenger::GateComptonCameraActorMessenger(GateComptonCameraActor * v)
  : GateActorMessenger(v),
    pActor(v)
{
  BuildCommands(baseName+pActor->GetObjectName());
  
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateComptonCameraActorMessenger::~GateComptonCameraActorMessenger()
{


  delete pSaveHitsTree;
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


  

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateComptonCameraActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{


   if(cmd == pSaveHitsTree) pActor->SetSaveHitsTreeFlag(  pSaveHitsTree->GetNewBoolValue(newValue)  ) ;

  GateActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------

//#endif
