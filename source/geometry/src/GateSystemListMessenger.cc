/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSystemListMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"

#include "G4UImanager.hh"

#include "GateSystemListManager.hh"

//------------------------------------------------------------------------------------------------------------------
// constructor
GateSystemListMessenger::GateSystemListMessenger(GateSystemListManager* itsListManager)
: GateListMessenger(itsListManager)
{ 
  G4String guidance;
  
  guidance = G4String("Control the GATE systems" );
  SetDirectoryGuidance(guidance);

  G4String cmdName;


}
//------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------
// destructor
GateSystemListMessenger::~GateSystemListMessenger()
{
}
//------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------
// UI command interpreter method
void GateSystemListMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
    GateListMessenger::SetNewValue(command,newValue);
}
//------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------
// Lists all the system-names into a string
const G4String& GateSystemListMessenger::DumpMap()
{
  return GetSystemListManager()->DumpChoices();
}
//------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------
// Pure virtual method: create and insert a new attachment
void GateSystemListMessenger::DoInsertion(const G4String& typeName)
{
  GetSystemListManager()->InsertNewSystem(typeName);
}
//------------------------------------------------------------------------------------------------------------------
