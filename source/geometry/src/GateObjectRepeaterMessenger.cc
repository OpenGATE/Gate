/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateObjectRepeaterMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

#include "GateVGlobalPlacement.hh"

//--------------------------------------------------------------------------------------------------
GateObjectRepeaterMessenger::GateObjectRepeaterMessenger(GateVGlobalPlacement* itsObjectRepeater)
:GateClockDependentMessenger(itsObjectRepeater)
{ 
  G4String guidance;
  G4String cmdName;

  guidance = G4String("Control for the object repeater '") + GetObjectRepeater()->GetObjectName() + "'";
  GetDirectory()->SetGuidance(guidance.c_str());

}
//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
GateObjectRepeaterMessenger::~GateObjectRepeaterMessenger()
{
}
//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
void GateObjectRepeaterMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  GateClockDependentMessenger::SetNewValue(command,newValue);
}
//--------------------------------------------------------------------------------------------------
