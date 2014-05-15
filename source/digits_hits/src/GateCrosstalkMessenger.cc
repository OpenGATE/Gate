/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCrosstalkMessenger.hh"

#include "GateCrosstalk.hh"

#include "G4UIcmdWithAString.hh"
#include  "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithADouble.hh"

GateCrosstalkMessenger::GateCrosstalkMessenger(GateCrosstalk* itsCrosstalk)
    : GatePulseProcessorMessenger(itsCrosstalk)
{
  G4String guidance;
  G4String cmdName;
  G4String cmdName2;
  G4String cmdName3;

  cmdName = GetDirectoryName() + "chooseCrosstalkVolume";
  newVolCmd = new G4UIcmdWithAString(cmdName,this);
  newVolCmd->SetGuidance("Choose a volume for crosstalk (e.g. crystal)");

  cmdName2 = GetDirectoryName() + "setEdgesFraction";
  edgesFractionCmd = new G4UIcmdWithADouble(cmdName2,this);
  edgesFractionCmd->SetGuidance("Set the fraction of energy which leaves on each edge crystal");

  cmdName3 = GetDirectoryName() + "setCornersFraction";
  cornersFractionCmd = new G4UIcmdWithADouble(cmdName3,this);
  cornersFractionCmd->SetGuidance("Set the fraction of the energy which leaves on each corner crystal");
}



GateCrosstalkMessenger::~GateCrosstalkMessenger()
{
  delete newVolCmd;
  delete edgesFractionCmd;
  delete cornersFractionCmd;
}


void GateCrosstalkMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==newVolCmd ) {
    GetCrosstalk()->CheckVolumeName(newValue);
  }
  else if ( command==edgesFractionCmd ) {
    GetCrosstalk()->SetEdgesFraction (edgesFractionCmd->GetNewDoubleValue(newValue));
  }
  else if ( command==cornersFractionCmd ) {
    GetCrosstalk()->SetCornersFraction (cornersFractionCmd->GetNewDoubleValue(newValue));
  }
  else
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
