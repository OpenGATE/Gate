/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateFluenceActorMessenger.hh"
#include "GateFluenceActor.hh"

//-----------------------------------------------------------------------------
GateFluenceActorMessenger::GateFluenceActorMessenger(GateFluenceActor* sensor)
  :GateImageActorMessenger(sensor),
  pFluenceActor(sensor)
{
  pEnableScatterCmd = 0;
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateFluenceActorMessenger::~GateFluenceActorMessenger()
{
  if(pEnableScatterCmd) delete pEnableScatterCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateFluenceActorMessenger::BuildCommands(G4String base)
{
  G4String  n = base+"/enableScatter";
  pEnableScatterCmd = new G4UIcmdWithABool(n, this); 
  G4String guid = G4String("Enable computation of scattered particles fluence");
  pEnableScatterCmd->SetGuidance(guid);

  n = base+"/responseDetectorFilename";
  pSetResponseDetectorFileCmd = new G4UIcmdWithAString(n, this);
  guid = G4String( "Response detector curve (weight to each energy)");
  pSetResponseDetectorFileCmd->SetGuidance( guid);

  n = base+"/scatterOrderFilename";
  pSetScatterOrderFilenameCmd = new G4UIcmdWithAString(n,this);
  guid = "Set the file name for the scatter x-rays that hit the detector (printf format with runId as a single parameter).";
  pSetScatterOrderFilenameCmd->SetGuidance(guid);

  n = base+"/comptonFilename";
  pSetComptonFilenameCmd = new G4UIcmdWithAString(n,this);
  guid = "Set the file name for compton scatter that hit the detector (printf format with runId as a single parameter).";
  pSetComptonFilenameCmd->SetGuidance(guid);

  n = base+"/rayleighFilename";
  pSetRayleighFilenameCmd = new G4UIcmdWithAString(n,this);
  guid = "Set the file name for rayleigh scatter that hit the detector (printf format with runId as a single parameter).";
  pSetRayleighFilenameCmd->SetGuidance(guid);

  n = base+"/fluorescenceFilename";
  pSetFluorescenceFilenameCmd = new G4UIcmdWithAString(n,this);
  guid = "Set the file name for fluorescence scatter that hit the detector (printf format with runId as a single parameter).";
  pSetFluorescenceFilenameCmd->SetGuidance(guid);


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateFluenceActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pEnableScatterCmd) 
    pFluenceActor->EnableScatterImage(pEnableScatterCmd->GetNewBoolValue(newValue));

  if( cmd == pSetResponseDetectorFileCmd)
      pFluenceActor->SetResponseDetectorFile(newValue);

  if(cmd == pSetScatterOrderFilenameCmd)
    pFluenceActor->SetScatterOrderFilename(newValue);

  if(cmd == pSetComptonFilenameCmd)
    pFluenceActor->SetComptonFilename(newValue);

  if(cmd == pSetRayleighFilenameCmd)
    pFluenceActor->SetRayleighFilename(newValue);

  if(cmd == pSetFluorescenceFilenameCmd)
    pFluenceActor->SetFluorescenceFilename(newValue);

  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------
