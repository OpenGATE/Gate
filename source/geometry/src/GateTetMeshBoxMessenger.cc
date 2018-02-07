/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
----------------------*/
#include <G4String.hh>
#include <G4UIcommand.hh>
#include <G4UIcmdWithAString.hh>
#include <G4UIcmdWithADoubleAndUnit.hh>
#include <G4UIcmdWith3VectorAndUnit.hh>

#include "GateVVolume.hh"
#include "GateTetMeshBox.hh"
#include "GateVolumeMessenger.hh"

#include "GateTetMeshBoxMessenger.hh"


GateTetMeshBoxMessenger::GateTetMeshBoxMessenger(GateTetMeshBox* itsCreator)
  : GateVolumeMessenger(itsCreator)
{
  const G4String& dir = GateMessenger::GetDirectoryName();
  G4String pathCmdName = dir + "reader/setPathToELEFile";
  G4String regionAttributeMapCmdName = dir + "setPathToAttributeMap";
  G4String unitOfLengthCmdName = dir + "reader/setUnitOfLength";

  pSetPathToELEFileCmd = new G4UIcmdWithAString(pathCmdName, this);
  pSetPathToELEFileCmd->SetGuidance("Set path to ELE file.");
  pSetPathToAttributeMapCmd = new G4UIcmdWithAString(regionAttributeMapCmdName, this);
  pSetPathToAttributeMapCmd->SetGuidance("Set path to material map (ASCII file).");
  pSetUnitOfLengthCmd = new G4UIcmdWithADoubleAndUnit(unitOfLengthCmdName, this);
  pSetUnitOfLengthCmd->SetGuidance("Unit of length to interpret the coordinates.");
}


GateTetMeshBoxMessenger::~GateTetMeshBoxMessenger()
{
  delete pSetPathToELEFileCmd;
  delete pSetPathToAttributeMapCmd;
  delete pSetUnitOfLengthCmd;
}


void GateTetMeshBoxMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  GateVVolume* creatorBase = GateVolumeMessenger::GetVolumeCreator();
  GateTetMeshBox* creator = dynamic_cast<GateTetMeshBox*>(creatorBase);
  
  if(command == pSetPathToELEFileCmd)
  {
    creator->SetPathToELEFile(newValue);
  }
  else if (command == pSetPathToAttributeMapCmd)
  {
    creator->SetPathToAttributeMap(newValue);
  }
  else if (command == pSetUnitOfLengthCmd)
  {
    creator->SetUnitOfLength(pSetUnitOfLengthCmd->GetNewDoubleValue(newValue));
  }
  else
  {
    GateVolumeMessenger::SetNewValue(command, newValue);
  }
}
