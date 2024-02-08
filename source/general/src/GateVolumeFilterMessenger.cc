/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
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
  delete pInvertCmd;
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

  bb = base+"/invert";
  pInvertCmd = new G4UIcmdWithoutParameter(bb,this);
  guidance = "Invert the filter";
  pInvertCmd->SetGuidance(guidance);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVolumeFilterMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if(command==pAddVolumeCmd) pVolumeFilter->addVolume(param);
  if(command==pInvertCmd) pVolumeFilter->setInvert();
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEVOLUMEFILTERMESSENGER_CC */
