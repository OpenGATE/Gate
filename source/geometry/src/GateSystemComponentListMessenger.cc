/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateSystemComponentListMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

#include "GateSystemComponent.hh"
#include "GateBoxComponent.hh"
#include "GateCylinderComponent.hh"
#include "GateArrayComponent.hh"
#include "GateWedgeComponent.hh"
#include "GateVSystem.hh"

//------------------------------------------------------------------------------------------------------------------
// Constructor
GateSystemComponentListMessenger::GateSystemComponentListMessenger(GateSystemComponentList* itsSystemComponentList)
: GateListMessenger( itsSystemComponentList)
{ 
  pInsertCmd->SetCandidates(DumpMap());
}
//------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------
// Destructor
GateSystemComponentListMessenger::~GateSystemComponentListMessenger()
{
}
//------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------
// UI command interpreter method
void GateSystemComponentListMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
    GateListMessenger::SetNewValue(command,newValue);
}
//------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------
const G4String& GateSystemComponentListMessenger::DumpMap()
{
  static G4String theList = "boxComponent cylinderComponent arrayComponent";
  return theList;
}
//------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------
void GateSystemComponentListMessenger::DoInsertion(const G4String& childTypeName)
{
  if (GetNewInsertionBaseName().empty())
    SetNewInsertionBaseName(childTypeName);
    
  AvoidNameConflicts();

  //GateSystemComponent* newComponent=0;

  if (childTypeName=="boxComponent")
   new GateBoxComponent(GetNewInsertionBaseName(),GetMotherComponent(),GetMotherComponent()->GetSystem());
  else if (childTypeName=="cylinderComponent")
    new GateCylinderComponent(GetNewInsertionBaseName(),GetMotherComponent(),GetMotherComponent()->GetSystem());
  else if (childTypeName=="arrayComponent")
    new GateArrayComponent(GetNewInsertionBaseName(),GetMotherComponent(),GetMotherComponent()->GetSystem());
  else if (childTypeName=="wedgeComponent")
    new GateWedgeComponent(GetNewInsertionBaseName(),GetMotherComponent(),GetMotherComponent()->GetSystem());
  else {
    G4cout << "System-component type name '" << childTypeName << "' was not recognised --> insertion request must be ignored!\n";
    return;
  }
  
  SetNewInsertionBaseName("");
}
//------------------------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------------------------
/*  Check whether there may be a name conflict between a new
    system component and an already existing one

    name: the name of the attachment to create 
*/
G4bool GateSystemComponentListMessenger::CheckNameConflict(const G4String& name)
{
  // Check whether an object with the same name already exists in the list
  return ( GetMotherComponent()->GetSystem()->FindComponent( name , true ) != 0 ) ;
}
//------------------------------------------------------------------------------------------------------------------
