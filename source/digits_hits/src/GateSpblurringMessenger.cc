/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSpblurringMessenger.hh"

#include "GateSpblurring.hh"

#include "G4UIcmdWithADouble.hh"

GateSpblurringMessenger::GateSpblurringMessenger(GateSpblurring* itsSpresolution)
    : GatePulseProcessorMessenger(itsSpresolution)
{
  G4String guidance;
  G4String cmdName;

  cmdName = GetDirectoryName() + "setSpresolution";
  spresolutionCmd = new G4UIcmdWithADouble(cmdName,this);
  spresolutionCmd->SetGuidance("Set the resolution in position for gaussian spblurring");
}


GateSpblurringMessenger::~GateSpblurringMessenger()
{
  delete spresolutionCmd;
}


void GateSpblurringMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==spresolutionCmd )
    { GetSpblurring()->SetSpresolution(spresolutionCmd->GetNewDoubleValue(newValue)); }
  else
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
