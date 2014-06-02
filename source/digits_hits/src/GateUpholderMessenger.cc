/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateUpholderMessenger.hh"

#include "GateUpholder.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"

GateUpholderMessenger::GateUpholderMessenger(GateUpholder* itsUpholder)
    : GatePulseProcessorMessenger(itsUpholder)
{
  G4String guidance;
  G4String cmdName;

  cmdName = GetDirectoryName() + "setUphold";
  upholdCmd = new G4UIcmdWithADoubleAndUnit(cmdName,this);
  upholdCmd->SetGuidance("Set uphold (in keV) for pulse-limitation");
  upholdCmd->SetUnitCategory("Energy");
}


GateUpholderMessenger::~GateUpholderMessenger()
{
  delete upholdCmd;
}


void GateUpholderMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==upholdCmd )
    { GetUpholder()->SetUphold(upholdCmd->GetNewDoubleValue(newValue)); }
  else
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
