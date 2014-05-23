/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GateROOTBasicOutputMessenger.hh"

#ifdef G4ANALYSIS_USE_ROOT

#include "GateROOTBasicOutput.hh"
#include "GateActions.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithAString.hh"
#include "G4ios.hh"
#include "globals.hh"
#include "Randomize.hh"

class GateRunAction;
class GateEventAction;

//---------------------------------------------------------------------------------
GateROOTBasicOutputMessenger::GateROOTBasicOutputMessenger(GateROOTBasicOutput* LH)
  :xeHisto(LH)
{
  //  G4cout << " DEBUT Contrusteur GateROOTBasicOutputMessenger " << G4endl;
  plotDir = new G4UIdirectory("/gate/output/BasicROOT/");
  plotDir->SetGuidance("Basic ROOT output control.");
  setfileNameCmd = new G4UIcmdWithAString("/gate/output/BasicROOT/setFileName",this);
  setfileNameCmd->SetGuidance("Set name for the free output file root.");
}
//---------------------------------------------------------------------------------


//---------------------------------------------------------------------------------
GateROOTBasicOutputMessenger::~GateROOTBasicOutputMessenger()
{
  delete setfileNameCmd;
  delete plotDir;
}
//---------------------------------------------------------------------------------


//---------------------------------------------------------------------------------
void GateROOTBasicOutputMessenger::SetNewValue(G4UIcommand* command, G4String newValues)
{
  // G4cout << " GateROOTBasicOutputMessenger::SetNewValue = " << newValues << G4endl;
  if( command == setfileNameCmd ){
    xeHisto->SetfileName(newValues);
    GateRunAction::GetRunAction()->GateRunAction::SetFlagBasicOutput(true);
    GateEventAction::GetEventAction()->GateEventAction::SetFlagBasicOutput(true);
  }
}
//---------------------------------------------------------------------------------

#endif
