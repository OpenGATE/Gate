/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCoincidenceGeometrySelectorMessenger.hh"

#include "GateCoincidenceGeometrySelector.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"

GateCoincidenceGeometrySelectorMessenger::GateCoincidenceGeometrySelectorMessenger(GateCoincidenceGeometrySelector* itsGeometrySelector)
    : GateClockDependentMessenger(itsGeometrySelector)
{
  G4String guidance;
  G4String cmdName;

  cmdName = GetDirectoryName() + "setSMax";
  maxSCmd= new G4UIcmdWithADoubleAndUnit(cmdName,this);
  maxSCmd->SetGuidance("Set max S value accepted (<0 --> all is accepted)");
  maxSCmd->SetUnitCategory("Length");

  cmdName = GetDirectoryName() + "setDeltaZMax";
  maxDeltaZCmd= new G4UIcmdWithADoubleAndUnit(cmdName,this);
  maxDeltaZCmd->SetGuidance("Set max delta Z value accepted (<0 --> all is accepted)");
  maxDeltaZCmd->SetUnitCategory("Length");
}


GateCoincidenceGeometrySelectorMessenger::~GateCoincidenceGeometrySelectorMessenger()
{
  delete maxSCmd;
  delete maxDeltaZCmd;
}


void GateCoincidenceGeometrySelectorMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command==maxSCmd)
    GetGeometrySelector()->SetMaxS(maxSCmd->GetNewDoubleValue(newValue));
  else if (command==maxDeltaZCmd)
    GetGeometrySelector()->SetMaxDeltaZ(maxDeltaZCmd->GetNewDoubleValue(newValue));
  else
    GateClockDependentMessenger::SetNewValue(command,newValue);
}
