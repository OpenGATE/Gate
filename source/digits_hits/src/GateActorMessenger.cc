/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#ifndef GATEACTORMESSENGER_CC
#define GATEACTORMESSENGER_CC

#include "GateActorMessenger.hh"

#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWithAnInteger.hh"


#include "GateVActor.hh"

//-----------------------------------------------------------------------------
GateActorMessenger::GateActorMessenger(GateVActor* sensor)
  :pActor(sensor)
{
  baseName = "/gate/actor/";
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateActorMessenger::~GateActorMessenger()
{
  delete pSetFileNameCmd;
  delete pSetVolumeNameCmd;
  delete pSaveEveryNEventsCmd;
  delete pSaveEveryNSecondsCmd;
  delete pAddFilterCmd;
  delete pSetOverWriteFilesFlagCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateActorMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;

  bb = base+"/save";
  pSetFileNameCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Set the name of the save file.";
  pSetFileNameCmd->SetGuidance(guidance);
  pSetFileNameCmd->SetParameterName("File name",false);

  bb = base+"/attachTo";
  pSetVolumeNameCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Attaches the sensor to the given volume";
  pSetVolumeNameCmd->SetGuidance(guidance);
  pSetVolumeNameCmd->SetParameterName("Volume name",false);

  bb = base+"/saveEveryNEvents";
  pSaveEveryNEventsCmd = new G4UIcmdWithAnInteger(bb,this);
  guidance = "Save sensor every n Events.";
  pSaveEveryNEventsCmd->SetGuidance(guidance);
  pSaveEveryNEventsCmd->SetParameterName("Event number",false);

  bb = base+"/setOverWriteFilesFlag";
  pSetOverWriteFilesFlagCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("True by default, if false change the filename every time the outputs are written (append the current run & event ID to the filename). Do not work for root output.");
  pSetOverWriteFilesFlagCmd->SetGuidance(guidance);

  bb = base+"/setResetDataAtEachRunFlag";
  pSetResetDataAtEachRunFlagCmd = new G4UIcmdWithABool(bb, this);
  guidance = G4String("Reset data values at the end of each run.");
  pSetResetDataAtEachRunFlagCmd->SetGuidance(guidance);

  bb = base+"/saveEveryNSeconds";
  pSaveEveryNSecondsCmd = new G4UIcmdWithAnInteger(bb,this);
  guidance = "Save sensor every n seconds.";
  pSaveEveryNSecondsCmd->SetGuidance(guidance);
  pSaveEveryNSecondsCmd->SetParameterName("Number of seconds between next save",false);

  bb = base+"/addFilter";
  pAddFilterCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Add a new filter";
  pAddFilterCmd->SetGuidance(guidance);
  pAddFilterCmd->SetParameterName("Type",false);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateActorMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if(command == pSetVolumeNameCmd) {
    pActor->SetVolumeName(param);
    pActor->AttachToVolume(param);
  }

  if(command == pSetFileNameCmd) pActor->SetSaveFilename(param);

  if(command == pSaveEveryNEventsCmd)
    pActor->EnableSaveEveryNEvents(pSaveEveryNEventsCmd->GetNewIntValue(param));

  if(command == pSaveEveryNSecondsCmd)
    pActor->EnableSaveEveryNSeconds(pSaveEveryNSecondsCmd->GetNewIntValue(param));

  if (command == pSetOverWriteFilesFlagCmd)
    pActor->SetOverWriteFilesFlag(pSetOverWriteFilesFlagCmd->GetNewBoolValue(param));

  if (command == pSetResetDataAtEachRunFlagCmd)
    pActor->EnableResetDataAtEachRun(pSetResetDataAtEachRunFlagCmd->GetNewBoolValue(param));

  if(command == pAddFilterCmd)
    GateActorManager::GetInstance()->AddFilter(param, pActor->GetObjectName() );
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEACTORMESSENGER_CC */
