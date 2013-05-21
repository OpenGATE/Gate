/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateScintillatorResponseActorMessenger.hh"
#include "GateScintillatorResponseActor.hh"

//-----------------------------------------------------------------------------
GateScintillatorResponseActorMessenger::GateScintillatorResponseActorMessenger(GateScintillatorResponseActor* sensor)
  :GateImageActorMessenger(sensor),
  pScintillatorResponseActor(sensor)
{
  pEnableScatterCmd = 0;
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateScintillatorResponseActorMessenger::~GateScintillatorResponseActorMessenger()
{
  if(pEnableScatterCmd) delete pEnableScatterCmd;
  delete pReadMuAbsortionListCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateScintillatorResponseActorMessenger::BuildCommands(G4String base)
{
  G4String  n = base+"/enableScatter";
  pEnableScatterCmd = new G4UIcmdWithABool(n, this); 
  G4String guid = G4String("Enable computation of scattered particles ScintillatorResponse");
  pEnableScatterCmd->SetGuidance(guid);

  n = base+"/readMuAbsortionList";
  pReadMuAbsortionListCmd = new G4UIcmdWithAString(n, this);
  guid = G4String( "List of Mu absortion coeficients");
  pReadMuAbsortionListCmd->SetGuidance( guid);

  n = base+"/scatterOrderFilename";
  pSetScatterOrderFilenameCmd = new G4UIcmdWithAString(n,this);
  guid = "Set the file name for the scatter x-rays that hit the detector (printf format with runId as a single parameter).";
  pSetScatterOrderFilenameCmd->SetGuidance(guid);


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateScintillatorResponseActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if( cmd == pEnableScatterCmd)
    pScintillatorResponseActor->EnableScatterImage(pEnableScatterCmd->GetNewBoolValue(newValue));
  if( cmd == pReadMuAbsortionListCmd)
      pScintillatorResponseActor->ReadMuAbsortionList(newValue);
  if( cmd == pSetScatterOrderFilenameCmd)
    pScintillatorResponseActor->SetScatterOrderFilename(newValue);
  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------
