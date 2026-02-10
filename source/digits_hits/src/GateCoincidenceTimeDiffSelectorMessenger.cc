/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateCoincidenceTimeDiffSelectorMessenger.hh"

#include "GateCoincidenceTimeDiffSelector.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "GateDigitizerMgr.hh"
#include "G4UImessenger.hh"
#include "globals.hh"
#include "GateClockDependentMessenger.hh"
#include "G4SystemOfUnits.hh"
#include "G4UIdirectory.hh"

GateCoincidenceTimeDiffSelectorMessenger::GateCoincidenceTimeDiffSelectorMessenger (GateCoincidenceTimeDiffSelector* CoincidenceTimeDiffSelector)
:GateClockDependentMessenger(CoincidenceTimeDiffSelector),
	  m_CoincidenceTimeDiffSelector(CoincidenceTimeDiffSelector)

{
  G4String guidance;
  G4String cmdName;

  cmdName = GetDirectoryName() + "setMin";
  minTimeCmd= new G4UIcmdWithADoubleAndUnit(cmdName,this);
  minTimeCmd->SetGuidance("Set min time diff to be accepted (negative values : any is time diff is ok)");
  minTimeCmd->SetUnitCategory("Time");

  cmdName = GetDirectoryName() + "setMax";
  maxTimeCmd= new G4UIcmdWithADoubleAndUnit(cmdName,this);
  maxTimeCmd->SetGuidance("Set max time diff to be accepted (negative values : any is time diff is ok)");
  maxTimeCmd->SetUnitCategory("Time");


}


GateCoincidenceTimeDiffSelectorMessenger::~GateCoincidenceTimeDiffSelectorMessenger()
{
  delete minTimeCmd;
  delete minTimeCmd;
}


void GateCoincidenceTimeDiffSelectorMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command==minTimeCmd)
	 m_CoincidenceTimeDiffSelector->SetMinTime(minTimeCmd->GetNewDoubleValue(newValue));
  else if (command == maxTimeCmd)
	  m_CoincidenceTimeDiffSelector->SetMaxTime(maxTimeCmd->GetNewDoubleValue(newValue));
  else
    GateClockDependentMessenger::SetNewValue(command,newValue);
}
