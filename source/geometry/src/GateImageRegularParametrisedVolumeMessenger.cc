/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


/*! \file
  \brief Implementation of GateImageRegularParametrisedVolumeMessenger
 */
#include "GateImageRegularParametrisedVolumeMessenger.hh"
#include "GateImageRegularParametrisedVolume.hh"

#include "G4UIcommand.hh"
#include "G4UIcmdWithADouble.hh"

//-----------------------------------------------------------------------------
GateImageRegularParametrisedVolumeMessenger::GateImageRegularParametrisedVolumeMessenger(GateImageRegularParametrisedVolume* volume):GateVImageVolumeMessenger(volume), pVolume(volume)
{
  GateMessageInc("Volume",6,"Begin GateImageRegularParametrisedVolumeMessenger()\n");
  G4String cmdName = GetDirectoryName()+"setSkipEqualMaterials";
  SkipEqualMaterialsCmd = new G4UIcmdWithABool(cmdName,this);
  SkipEqualMaterialsCmd->SetGuidance("Skip or not boundaries when neighbour voxels are made of same material (default: yes)");
  GateMessageDec("Volume",6,"End GateImageRegularParametrisedVolumeMessenger()\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateImageRegularParametrisedVolumeMessenger::~GateImageRegularParametrisedVolumeMessenger()
{
  GateMessageInc("Volume",6,"Begin ~GateImageRegularParametrisedVolumeMessenger()\n");
  delete SkipEqualMaterialsCmd;
  GateMessageDec("Volume",6,"End ~GateImageRegularParametrisedVolumeMessenger()\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageRegularParametrisedVolumeMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  GateMessage("Volume",6,"GateImageRegularParametrisedVolumeMessenger::SetNewValue " << command->GetCommandPath()
	      << " newValue=" << newValue << Gateendl);
  if (command == SkipEqualMaterialsCmd) {
    pVolume->SetSkipEqualMaterialsFlag(SkipEqualMaterialsCmd->GetNewBoolValue(newValue));
  }
  else {
    GateVImageVolumeMessenger::SetNewValue(command,newValue);
  }
}
//-----------------------------------------------------------------------------
