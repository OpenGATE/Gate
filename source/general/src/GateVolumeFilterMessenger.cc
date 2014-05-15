/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEVOLUMEFILTERMESSENGER_CC
#define GATEVOLUMEFILTERMESSENGER_CC

#include "GateVolumeFilterMessenger.hh"

#include "GateVolumeFilter.hh"



//-----------------------------------------------------------------------------
GateVolumeFilterMessenger::GateVolumeFilterMessenger(GateVolumeFilter* idFilter)

  : pVolumeFilter(idFilter)
{
  BuildCommands(pVolumeFilter->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateVolumeFilterMessenger::~GateVolumeFilterMessenger()
{
  delete pAddVolumeCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVolumeFilterMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;
  
  bb = base+"/addVolume";
  pAddVolumeCmd = new G4UIcmdWithAString(bb,this);
  guidance = "Add volume";
  pAddVolumeCmd->SetGuidance(guidance);
  pAddVolumeCmd->SetParameterName("Volume name",false);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVolumeFilterMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if(command==pAddVolumeCmd) pVolumeFilter->addVolume(param);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEVOLUMEFILTERMESSENGER_CC */
