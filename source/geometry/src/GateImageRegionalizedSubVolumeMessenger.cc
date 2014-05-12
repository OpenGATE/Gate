/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*! \file 
  \brief Implementation of GateImageRegionalizedSubVolumeMessenger
 */

#include "GateImageRegionalizedSubVolumeMessenger.hh"
#include "GateImageRegionalizedSubVolume.hh"

#include "G4UIcommand.hh"

//====================================================================
GateImageRegionalizedSubVolumeMessenger::GateImageRegionalizedSubVolumeMessenger(GateImageRegionalizedSubVolume* volume)
  : 
  GateVolumeMessenger(volume),
  pVolume(volume)
{
  
  //GateMessage("Volume",5,"GateImageRegionalizedSubVolumeMessenger()"<<G4endl);
}
//====================================================================


//====================================================================
GateImageRegionalizedSubVolumeMessenger::~GateImageRegionalizedSubVolumeMessenger()
{
  //GateMessage("Volume",5,"~GateImageRegionalizedSubVolumeMessenger()"<<G4endl);
}
//====================================================================


//====================================================================    
void GateImageRegionalizedSubVolumeMessenger::SetNewValue(G4UIcommand* command, 
						 G4String newValue)
{
  ////GateMessage("Volume",5,"GateImageRegionalizedSubVolumeMessenger::SetNewValue " 
	    //  << command->GetCommandPath()
	   //   << " newValue=" << newValue << G4endl);
  
  GateVolumeMessenger::SetNewValue(command,newValue);
  
}

//====================================================================



