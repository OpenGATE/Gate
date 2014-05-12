/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateListMessenger.hh"

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

#include "GateListManager.hh"
#include "GateNamedObject.hh"

G4String GateListMessenger::mSystemType = "";
//----------------------------------------------------------------------------
// constructor
GateListMessenger::GateListMessenger(GateListManager* itsListManager)
: GateClockDependentMessenger(itsListManager)
{ 
  
  const G4String& elementTypeName = GetListManager()->GetElementTypeName();

  G4String guidance = G4String("Manages a list of ") + elementTypeName + "s.";
  GetDirectory()->SetGuidance(guidance.c_str());

  G4String cmdName;
  
  if (GetListManager()->AcceptNewElements()) {

    cmdName = GetDirectoryName()+"systemType";
    guidance = "Sets the system type.";
    pSystemTypeCmd = new G4UIcmdWithAString(cmdName,this);
    pSystemTypeCmd->SetGuidance(guidance);
    pSystemTypeCmd->SetParameterName("Name",false);

    cmdName = GetDirectoryName()+"name";
    guidance = "Sets the name given to the next " + elementTypeName + " to insert.";
    pDefineNameCmd = new G4UIcmdWithAString(cmdName,this);
    pDefineNameCmd->SetGuidance(guidance);
    pDefineNameCmd->SetParameterName("Name",false);

    cmdName = GetDirectoryName()+"insert";
    guidance = "Inserts a new " + elementTypeName + ".";
    pInsertCmd = new G4UIcmdWithAString(cmdName,this);
    pInsertCmd->SetGuidance(guidance);
    pInsertCmd->SetParameterName("choice",false);

    cmdName = GetDirectoryName()+"info";
    guidance = "List the " + elementTypeName + " choices available.";
    pListChoicesCmd = new G4UIcmdWithoutParameter(cmdName,this);
    pListChoicesCmd->SetGuidance(guidance);
  }
      
  cmdName = GetDirectoryName()+"list";
  guidance = "List the " + elementTypeName + "'s within '" + GetListManager()->GetObjectName() + "'";
  pListCmd = new G4UIcmdWithoutParameter(cmdName,this);
  pListCmd->SetGuidance(guidance);

}
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
// destructor
GateListMessenger::~GateListMessenger()
{
  delete pSystemTypeCmd;
  delete pListCmd;
  if (GetListManager()->AcceptNewElements()) {
    delete pListChoicesCmd;
    delete pInsertCmd;
    delete pDefineNameCmd;
  }
}
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
// UI command interpreter method
void GateListMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
   if( command==pSystemTypeCmd )
   { mSystemType = newValue; }
  else if( command==pDefineNameCmd )
    {    
     mNewInsertionBaseName = newValue; }   
  else if( command==pInsertCmd )
    { DoInsertion(newValue); }   
  else if( command==pListChoicesCmd )
    { ListChoices(); }   
  else if( command==pListCmd )
    //{ GetListManager()->ListElements(); 
    { GetListManager()->TheListElements(); }   
  else
    GateClockDependentMessenger::SetNewValue(command,newValue);
}
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
//  Check whether there may be a name conflict between a new
//  attachment and an already existing one
G4bool GateListMessenger::CheckNameConflict(const G4String& newBaseName)
{
  // Check whether an object with the same name already exists in the list
  return ( GetListManager()->FindElementByBaseName(newBaseName) != 0 ) ;
}
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
/*  Check for a potential name conflict between a new attachment and an already existing one.
    If such a conflict is found, a new, conflict-free, name is generated
*/
void GateListMessenger::AvoidNameConflicts()
{ 
  // Look for a potential name-conflict
  if (!CheckNameConflict( GetNewInsertionBaseName() )) {
    // No name conflict, it's OK
    return;
  }
  

  G4int copyNumber = 2;
  char buffer[256];
  
  // Try with 'name2', 'name-3',... until the conflict is solved
  do {
    sprintf(buffer, "%s%i", GetNewInsertionBaseName().c_str(), copyNumber);
    copyNumber++;
  } while (CheckNameConflict(buffer));  
    
  G4cout << "Warning: the selected insertion name ('" << GetNewInsertionBaseName() << "') was already in use.\n"
      	    "Name '" << buffer << "' will be used instead.\n";

  // A conflict-free name was found: store it into the insertion-name variable
  SetNewInsertionBaseName(buffer);
}
//----------------------------------------------------------------------------

