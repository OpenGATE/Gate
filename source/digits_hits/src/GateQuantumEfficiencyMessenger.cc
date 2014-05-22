/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateQuantumEfficiencyMessenger.hh"

#include "GateQuantumEfficiency.hh"

#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADouble.hh"

GateQuantumEfficiencyMessenger::GateQuantumEfficiencyMessenger(GateQuantumEfficiency* itsQE)
    : GatePulseProcessorMessenger(itsQE)
{
  G4String guidance;
  G4String cmdName;
  G4String cmdName2;
  G4String cmdName3;

  cmdName = GetDirectoryName() + "chooseQEVolume";
  newVolCmd = new G4UIcmdWithAString(cmdName,this);
  newVolCmd->SetGuidance("Choose a volume for quantum efficiency (e.g. crystal)");

  cmdName2 = GetDirectoryName() + "useFileDataForQE";
  newFileCmd = new G4UIcmdWithAString(cmdName2,this);
  newFileCmd->SetGuidance("Use data from a file to set your quantum efficiency inhomogeneity");

  cmdName3 = GetDirectoryName() + "setUniqueQE";
  uniqueQECmd = new G4UIcmdWithADouble(cmdName3,this);
  uniqueQECmd->SetGuidance("Set an unique quantum efficiency");
}



GateQuantumEfficiencyMessenger::~GateQuantumEfficiencyMessenger()
{
  delete newVolCmd;
  delete newFileCmd;
  delete uniqueQECmd;
}


void GateQuantumEfficiencyMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if ( command==newVolCmd )
    GetQE()->CheckVolumeName(newValue);
  else if ( command==newFileCmd )
    GetQE()->UseFile(newValue);
  else if ( command==uniqueQECmd )
    GetQE()->SetUniqueQE(uniqueQECmd->GetNewDoubleValue(newValue));
  else
    GatePulseProcessorMessenger::SetNewValue(command,newValue);
}
