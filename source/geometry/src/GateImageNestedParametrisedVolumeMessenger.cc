/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


/*! \file
  \brief Implementation of GateImageNestedParametrisedVolumeMessenger
*/
#include "GateImageNestedParametrisedVolumeMessenger.hh"
#include "GateImageNestedParametrisedVolume.hh"

#include "G4UIcommand.hh"
#include "G4UIcmdWithADouble.hh"

//-----------------------------------------------------------------------------
GateImageNestedParametrisedVolumeMessenger::GateImageNestedParametrisedVolumeMessenger(GateImageNestedParametrisedVolume* volume)
  :GateVImageVolumeMessenger(volume)
{
  GateMessageInc("Volume",6,"Begin GateImageNestedParametrisedVolumeMessenger()\n");
  GateMessageDec("Volume",6,"End GateImageNestedParametrisedVolumeMessenger()\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateImageNestedParametrisedVolumeMessenger::~GateImageNestedParametrisedVolumeMessenger()
{
  GateMessageInc("Volume",6,"Begin ~GateImageNestedParametrisedVolumeMessenger()\n");
  GateMessageDec("Volume",6,"End ~GateImageNestedParametrisedVolumeMessenger()\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageNestedParametrisedVolumeMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  GateMessage("Volume",6,"GateImageNestedParametrisedVolumeMessenger::SetNewValue "
              << command->GetCommandPath()
	      << " newValue=" << newValue << Gateendl);
  GateVImageVolumeMessenger::SetNewValue(command,newValue);
}
//-----------------------------------------------------------------------------
