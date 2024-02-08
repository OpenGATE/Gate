/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateCreatorProcessFilterMessenger.hh"

#include "GateCreatorProcessFilter.hh"

//-----------------------------------------------------------------------------
GateCreatorProcessFilterMessenger::GateCreatorProcessFilterMessenger(GateCreatorProcessFilter* filter) : G4UImessenger(), pFilter(filter)
{
  G4String base = pFilter->GetObjectName();

  {
    G4String bb = base+"/addCreatorProcess";
    pAddCreatorProcessCmd = new G4UIcmdWithAString(bb,this);
    G4String guidance = "Add creator process";
    pAddCreatorProcessCmd->SetGuidance(guidance);
    pAddCreatorProcessCmd->SetParameterName("Creator process name",false);

    bb = base+"/invert";
    pInvertCmd = new G4UIcmdWithoutParameter(bb,this);
    guidance = "Invert the filter";
    pInvertCmd->SetGuidance(guidance);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateCreatorProcessFilterMessenger::~GateCreatorProcessFilterMessenger()
{
  delete pAddCreatorProcessCmd;
  delete pInvertCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateCreatorProcessFilterMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if(command==pAddCreatorProcessCmd) pFilter->AddCreatorProcess(param);
  if(command==pInvertCmd) pFilter->setInvert();
}
//-----------------------------------------------------------------------------

