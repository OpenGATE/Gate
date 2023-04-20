/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

#include "GateVolumeMessenger.hh"
#include "GateVVolume.hh"
#include "GateVisAttributesMessenger.hh"

//-------------------------------------------------------------------------------------
GateVolumeMessenger::GateVolumeMessenger(GateVVolume* itsCreator, const G4String& itsDirectoryName)
  //: GateClockDependentMessenger(itsCreator, itsCreator->GetObjectName()+"/geometry"),
  : GateClockDependentMessenger(itsCreator, itsDirectoryName)
{
  G4String guidance = G4String("Controls the geometry for '") + GetVolumeCreator()->GetObjectName() +"'";

  GetDirectory()->SetGuidance(guidance.c_str());

  //  G4cout << " before new GateVisAttributesMessenger creating GetVolumeCreator() ="  << GetVolumeCreator()->GetObjectName() << Gateendl;
  pVisAttributesMessenger  =
    new GateVisAttributesMessenger(GetVolumeCreator()->GetVisAttributes(), GetVolumeCreator()->GetObjectName()+"/vis");

  G4String cmdName;
  cmdName = GetDirectoryName()+"setMaterial";

  pSetMaterCmd = new G4UIcmdWithAString(cmdName.c_str(),this);
  pSetMaterCmd->SetGuidance("Select Material of the detector");
  pSetMaterCmd->SetParameterName("choice",false);
  //  pSetMaterCmd->AvailableForStates(G4State_Idle);

  cmdName = GetDirectoryName()+"attachCrystalSD";
  pAttachCrystalSDCmd = new G4UIcmdWithoutParameter(cmdName,this);
  pAttachCrystalSDCmd->SetGuidance("Attach the crystal-SD to the object.");

  cmdName = GetDirectoryName()+"attachCrystalSDnoSystem";
  pAttachCrystalSDnoSystemCmd = new G4UIcmdWithoutParameter(cmdName,this);
  pAttachCrystalSDnoSystemCmd->SetGuidance("Attach the crystal-SD to the object.");

  cmdName = GetDirectoryName()+"attachPhantomSD";
  pAttachPhantomSDCmd = new G4UIcmdWithoutParameter(cmdName,this);
  pAttachPhantomSDCmd->SetGuidance("Attach the phantom-SD to the object.");

  cmdName = GetDirectoryName()+"saveGeometryAsVoxelizedImage";
  pDumpVoxelizedVolumeCmd = new G4UIcmdWith3VectorAndUnit(cmdName,this);
  pDumpVoxelizedVolumeCmd->SetGuidance("Select the voxel size and save the voxelized image of the geometry.");
  pDumpVoxelizedVolumeCmd->SetParameterName("Spacingx", "Spacingy", "Spacingz", false);
  pDumpVoxelizedVolumeCmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName()+"setSaveImageDirectory";
  pSetDumpPathCmd = new G4UIcmdWithAString(cmdName,this);
  pSetDumpPathCmd->SetGuidance("Select voxelized image saving directory.");
}
//-------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------
GateVolumeMessenger::~GateVolumeMessenger()
{
   delete pVisAttributesMessenger;
   delete pSetMaterCmd;
   delete pAttachCrystalSDCmd;
   delete pAttachCrystalSDnoSystemCmd;

   delete pAttachPhantomSDCmd;
   delete pDumpVoxelizedVolumeCmd;
   delete pSetDumpPathCmd;
}
//-------------------------------------------------------------------------------------

//-------------------------------------------------------------------------------------
void GateVolumeMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if( command == pSetMaterCmd )
    GetVolumeCreator()->SetMaterialName(newValue);
  else if( command == pAttachCrystalSDnoSystemCmd )
    GetVolumeCreator()->AttachCrystalSDnoSystem();
  else if( command == pAttachCrystalSDCmd )
    GetVolumeCreator()->AttachCrystalSD();
  else if( command == pAttachPhantomSDCmd )
    GetVolumeCreator()->AttachPhantomSD();
  else if( command == pDumpVoxelizedVolumeCmd )
    GetVolumeCreator()->DumpVoxelizedVolume(pDumpVoxelizedVolumeCmd->GetNew3VectorValue(newValue));
  else if( command == pSetDumpPathCmd )
    GetVolumeCreator()->SetDumpPath(newValue);
  else
    GateClockDependentMessenger::SetNewValue(command,newValue);
}
//-------------------------------------------------------------------------------------
