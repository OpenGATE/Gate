/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateMergedVolumeActor.hh"
#include "GateMergedVolumeActorMessenger.hh"

GateMergedVolumeActorMessenger::GateMergedVolumeActorMessenger( GateMergedVolumeActor* sensor )
: GateActorMessenger( sensor ),
  pMergedVolumeActor( sensor )
{
  BuildCommands( baseName + sensor->GetObjectName() );
}

GateMergedVolumeActorMessenger::~GateMergedVolumeActorMessenger()
{
  delete ListVolumeToMergeCmd;
}

void GateMergedVolumeActorMessenger::BuildCommands(G4String base)
{
  G4String cmdName;

  cmdName = base+"/volumeToMerge";
  ListVolumeToMergeCmd = new G4UIcmdWithAString(cmdName,this);
  ListVolumeToMergeCmd->SetGuidance("List of volume(s) to merge within a voxelized phantom");
}

void GateMergedVolumeActorMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if( command == ListVolumeToMergeCmd )
  {
    pMergedVolumeActor->ListOfVolumesToMerge(newValue);
  }
}
