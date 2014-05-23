/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateObjectMoveMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

//old #include "GateVObjectMove.hh"
#include "GateVGlobalPlacement.hh"

//old
//GateObjectMoveMessenger::GateObjectMoveMessenger(GateVObjectMove* itsMove)
//:GateObjectRepeaterMessenger(itsMove)
GateObjectMoveMessenger::GateObjectMoveMessenger(GateVGlobalPlacement* itsMove)
:GateObjectRepeaterMessenger(itsMove)
{ 
  G4String guidance = G4String("Control for the movement '") + GetMove()->GetObjectName() + "'";
  GetDirectory()->SetGuidance(guidance.c_str());

  G4String cmdName;

}

