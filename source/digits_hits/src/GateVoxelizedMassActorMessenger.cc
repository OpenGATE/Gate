/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateVoxelizedMassActorMessenger
  \author Thomas DESCHLER (thomas.deschler@iphc.cnrs.fr)
  \date	October 2015
*/

#ifndef GATEVOXELIZEDMASSACTORMESSENGER_CC
#define GATEVOXELIZEDMASSACTORMESSENGER_CC

#include "GateVoxelizedMassActorMessenger.hh"
#include "GateVoxelizedMassActor.hh"

//-----------------------------------------------------------------------------
GateVoxelizedMassActorMessenger::GateVoxelizedMassActorMessenger(GateVoxelizedMassActor* sensor)
  :GateImageActorMessenger(sensor),pVoxelizedMassActor(sensor)
{
  pEnableMassCmd= 0;

  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateVoxelizedMassActorMessenger::~GateVoxelizedMassActorMessenger()
{
  if(pEnableMassCmd) delete pEnableMassCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVoxelizedMassActorMessenger::BuildCommands(G4String base)
{
  G4String  n = base+"/enableMass";
  pEnableMassCmd = new G4UIcmdWithABool(n, this);
  G4String guid = G4String("Enable mass computation");
  pEnableMassCmd->SetGuidance(guid);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVoxelizedMassActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pEnableMassCmd) pVoxelizedMassActor->EnableMassImage(pEnableMassCmd->GetNewBoolValue(newValue));

  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif
