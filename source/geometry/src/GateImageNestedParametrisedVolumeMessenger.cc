/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
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
  : 
  GateVImageVolumeMessenger(volume),
  pVolume(volume)
{  
  GateMessageInc("Volume",6,"Begin GateImageNestedParametrisedVolumeMessenger()"<<G4endl);
  G4String dir = GetDirectoryName() + "geometry/";

 /* G4String n = dir +"/SetSmartLess";
  pSmartLessCmd = new G4UIcmdWithADouble(n,this);  
  pSmartLessCmd->SetGuidance("Sets the value of SmartLess : Quality for optimisation, average number of voxels to be spent per content.");*/

  GateMessageDec("Volume",6,"End GateImageNestedParametrisedVolumeMessenger()"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateImageNestedParametrisedVolumeMessenger::~GateImageNestedParametrisedVolumeMessenger()
{
  GateMessageInc("Volume",6,"Begin ~GateImageNestedParametrisedVolumeMessenger()"<<G4endl);
  //delete pSmartLessCmd;
  GateMessageDec("Volume",6,"End ~GateImageNestedParametrisedVolumeMessenger()"<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------    
void GateImageNestedParametrisedVolumeMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  GateMessage("Volume",6,"GateImageNestedParametrisedVolumeMessenger::SetNewValue " << command->GetCommandPath()
	      << " newValue=" << newValue << G4endl);

 /// if (command == pSmartLessCmd) {
 //   pVolume->SetSmartless(pSmartLessCmd->GetNewDoubleValue(newValue));
 // }

  GateVImageVolumeMessenger::SetNewValue(command,newValue);
  
}

//-----------------------------------------------------------------------------



