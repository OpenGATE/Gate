/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateVGeometryVoxelStoreMessenger.hh"
#include "GateVGeometryVoxelStore.hh"

//e#include "GateVVolume.hh"

#include "GateClock.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateVGeometryVoxelStoreMessenger::GateVGeometryVoxelStoreMessenger(GateVGeometryVoxelStore* voxelStore)
  : GateMessenger(voxelStore->GetCreator()->GetObjectName() + 
		  G4String("/") +
		  voxelStore->GetName(),true),
  m_voxelStore(voxelStore)
{ 

  G4String cmdName;

  cmdName = GetDirectoryName()+"setDefaultMaterial";
  DefaultMaterialCmd = new G4UIcmdWithAString(cmdName,this);
  DefaultMaterialCmd->SetGuidance("Sets the default material for voxels not set in the voxel store");
  DefaultMaterialCmd->SetGuidance("1. material name");

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

GateVGeometryVoxelStoreMessenger::~GateVGeometryVoxelStoreMessenger()
{
   delete DefaultMaterialCmd;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....

void GateVGeometryVoxelStoreMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  if ( command == DefaultMaterialCmd ) {
    m_voxelStore->SetDefaultMaterial(newValue);
  }

}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo....



