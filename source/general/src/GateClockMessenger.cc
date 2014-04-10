/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateClockMessenger.hh"
#include "GateClock.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"

//--------------------------------------------------------------------------
GateClockMessenger::GateClockMessenger()
{
  pGateTimingDir = new G4UIdirectory("/gate/timing/");
  pGateTimingDir->SetGuidance("GATE timing control.");

  pTimeCmd = new G4UIcmdWithADoubleAndUnit("/gate/timing/setTime",this);
  pTimeCmd->SetGuidance("Set time");
  pTimeCmd->SetParameterName("Time",false);
  pTimeCmd->SetUnitCategory("Time");
  //  TimeCmd->AvailableForStates(Idle);

  pVerboseCmd = new G4UIcmdWithAnInteger("/gate/timing/verbose",this);
  pVerboseCmd->SetGuidance("Set GATE event action verbose level");
  pVerboseCmd->SetParameterName("verbose",false);
  pVerboseCmd->SetRange("verbose>=0");
}
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
GateClockMessenger::~GateClockMessenger()
{
  delete pTimeCmd;
  delete pVerboseCmd;
}
//--------------------------------------------------------------------------


//--------------------------------------------------------------------------
void GateClockMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  GateClock* theClock = GateClock::GetInstance();
  if( command == pTimeCmd ) {
    theClock->SetTime(pTimeCmd->GetNewDoubleValue(newValue));
  }
  if( command == pVerboseCmd ) {
    theClock->SetVerboseLevel(pVerboseCmd->GetNewIntValue(newValue));
  }

}
//--------------------------------------------------------------------------
