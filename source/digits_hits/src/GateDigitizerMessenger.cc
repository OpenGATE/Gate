/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/
//GND 2022 Class to Remove

#include "GateDigitizerMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

#include "GatePulseProcessorChain.hh"
#include "GateCoincidenceSorterOld.hh"
#include "GateCoincidencePulseProcessorChain.hh"

// Constructor
GateDigitizerMessenger::GateDigitizerMessenger(GateDigitizer* itsDigitizer)
: GateClockDependentMessenger( itsDigitizer)
{
//  G4cout << " DEBUT Constructor GateDigitizerMessenger \n";

  const G4String& elementTypeName = itsDigitizer->GetElementTypeName();

  G4String guidance = G4String("Manages a list of ") + elementTypeName + "s.";
  GetDirectory()->SetGuidance(guidance.c_str());

  G4String cmdName;

  cmdName = GetDirectoryName()+"name";
  guidance = "Sets the name given to the next " + elementTypeName + " to insert.";
  DefineNameCmd = new G4UIcmdWithAString(cmdName,this);
  DefineNameCmd->SetGuidance(guidance);
  DefineNameCmd->SetParameterName("Name",false);

  cmdName = GetDirectoryName()+"insert";
//  G4cout << " cmdName GateDigitizerMessenger : " << cmdName << Gateendl;
  guidance = "Inserts a new " + elementTypeName + ".";
  pInsertCmd = new G4UIcmdWithAString(cmdName,this);
  pInsertCmd->SetGuidance(guidance);
  pInsertCmd->SetParameterName("choice",false);

  cmdName = GetDirectoryName()+"info";
  guidance = "List the " + elementTypeName + " choices available.";
  ListChoicesCmd = new G4UIcmdWithoutParameter(cmdName,this);
  ListChoicesCmd->SetGuidance(guidance);

  cmdName = GetDirectoryName()+"list";
  guidance = "List the " + elementTypeName + "'s within '" + GetDigitizer()->GetObjectName() + "'";
  ListCmd = new G4UIcmdWithoutParameter(cmdName,this);
  ListCmd->SetGuidance(guidance);

  pInsertCmd->SetCandidates(DumpMap());

//  G4cout << " FIN Constructor GateDigitizerMessenger \n";
}



// Destructor
GateDigitizerMessenger::~GateDigitizerMessenger()
{
  delete ListCmd;
  delete ListChoicesCmd;
  delete pInsertCmd;
  delete DefineNameCmd;
}



// UI command interpreter method
void GateDigitizerMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
  if( command==DefineNameCmd )
    { m_newInsertionBaseName = newValue; }
  else if( command==pInsertCmd )
    { DoInsertion(newValue); }
  else if( command==ListChoicesCmd )
    { ListChoices(); }
  else if( command==ListCmd )
    { GetDigitizer()->ListElements(); }
  else
    GateClockDependentMessenger::SetNewValue(command,newValue);
}




const G4String& GateDigitizerMessenger::DumpMap()
{
  static G4String theList = "singleChain coincidenceSorter coincidenceChain";
  return theList;
}




void GateDigitizerMessenger::DoInsertion(const G4String& childTypeName)
{
  if (GetNewInsertionBaseName().empty())
    SetNewInsertionBaseName(childTypeName);

  AvoidNameConflicts();

  if (childTypeName=="singleChain") {
    GetDigitizer()->StoreNewPulseProcessorChain( new GatePulseProcessorChain(GetDigitizer(),GetNewInsertionBaseName()) );
  } else if (childTypeName=="coincidenceSorter") {
    GetDigitizer()->StoreNewCoincidenceSorter( new GateCoincidenceSorterOld(GetDigitizer(),GetNewInsertionBaseName(),10.*ns) );
  } else if (childTypeName=="coincidenceChain") {
    GetDigitizer()->StoreNewCoincidenceProcessorChain( new GateCoincidencePulseProcessorChain(GetDigitizer(),GetNewInsertionBaseName()) );
  } else {
    G4cout << "Digitizer module type name '" << childTypeName << "' was not recognised --> insertion request must be ignored!\n";
    return;
  }

  SetNewInsertionBaseName("");
}




//  Check whether there may be a name conflict between a new
//  attachment and an already existing one
G4bool GateDigitizerMessenger::CheckNameConflict(const G4String& newBaseName)
{
  // Check whether an object with the same name already exists in the list
  return ( GetDigitizer()->FindElementByBaseName(newBaseName) != 0 ) ;
}



/*  Check for a potential name conflict between a new attachment and an already existing one.
    If such a conflict is found, a new, conflict-free, name is generated
*/
void GateDigitizerMessenger::AvoidNameConflicts()
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
