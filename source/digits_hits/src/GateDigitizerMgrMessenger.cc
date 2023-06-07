/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateDigitizerMgrMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"


// Constructor
GateDigitizerMgrMessenger::GateDigitizerMgrMessenger(GateDigitizerMgr* itsDigitizerMgr)
: GateClockDependentMessenger(itsDigitizerMgr)
{
  //G4cout << " DEBUT Constructor GateDigitizerMgrMessenger \n";

  const G4String& elementTypeName = itsDigitizerMgr->GetElementTypeName();

  G4String guidance = G4String("Manages a list of ") + elementTypeName + "s.";
  GetDirectory()->SetGuidance(guidance.c_str());

  G4String cmdName;

 // G4cout<<GetDirectoryName()<<G4endl;

  cmdName = GetDirectoryName()+"name";
  guidance = "Sets the name given to the next " + elementTypeName + " to insert.";
  DefineNameCmd = new G4UIcmdWithAString(cmdName,this);
  DefineNameCmd->SetGuidance(guidance);
  DefineNameCmd->SetParameterName("Name",false);

  cmdName = GetDirectoryName()+"insert"; // /gate/digitizer/insert
  guidance = "Inserts a new " + elementTypeName + ".";
  pInsertCmd = new G4UIcmdWithAString(cmdName,this);
  pInsertCmd->SetGuidance(guidance);
  pInsertCmd->SetParameterName("choice",false);


  cmdName = GetDirectoryName()+"chooseSD";
  SetChooseSDCmd = new G4UIcmdWithAString(cmdName,this);
  SetChooseSDCmd->SetGuidance("Set the name of the input pulse channel");
  SetChooseSDCmd->SetParameterName("Name",false);


  cmdName = GetDirectoryName()+"info";
  guidance = "List the " + elementTypeName + " choices available.";
  ListChoicesCmd = new G4UIcmdWithoutParameter(cmdName,this);
  ListChoicesCmd->SetGuidance(guidance);

  cmdName = GetDirectoryName()+"list";
  guidance = "List the " + elementTypeName + "'s within '" + GetDigitizerMgr()->GetObjectName() + "'";
  ListCmd = new G4UIcmdWithoutParameter(cmdName,this);
  ListCmd->SetGuidance(guidance);

  pInsertCmd->SetCandidates(DumpMap());

//  G4cout << " FIN Constructor GateDigitizerMgrMessenger \n";
}



// Destructor
GateDigitizerMgrMessenger::~GateDigitizerMgrMessenger()
{
  delete ListCmd;
  delete ListChoicesCmd;
  delete pInsertCmd;
  delete SetChooseSDCmd;
  delete DefineNameCmd;
}



// UI command interpreter method
void GateDigitizerMgrMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{
	if( command==DefineNameCmd )
    { m_newCollectionName = newValue; }
  else if (command == SetChooseSDCmd)
        {  m_SDname=newValue; }
  else if( command==pInsertCmd )
    	{ DoInsertion(newValue); }
  else if( command==ListChoicesCmd )
    { ListChoices(); }
  else if( command==ListCmd )
    { GetDigitizerMgr()->ShowSummary(); }
  else
    GateClockDependentMessenger::SetNewValue(command,newValue);

}




const G4String& GateDigitizerMgrMessenger::DumpMap()
{
  static G4String theList = "SinglesDigitizer CoincidenceSorter CoincidenceDigitizer";
  return theList;
}




void GateDigitizerMgrMessenger::DoInsertion(const G4String& childTypeName)
{
	//G4cout<<"GateDigitizerMgrMessenger::DoInsertion "<<childTypeName<<G4endl;

	if (GetNewCollectionName().empty())
    SetNewCollectionName(childTypeName);

  AvoidNameConflicts();


  if (childTypeName=="SinglesDigitizer") {
	  if(m_SDname.empty())
	  	  GateError("***ERROR*** Please, choose the sensitive detector name for which you want to insert new digitizer. "
	  			  "Command: /gate/digitizerMgr/chooseSD\n"
	  			   "ATTENTION: this command can be called only before /insert command\n");

	  G4SDManager* SDman = G4SDManager::GetSDMpointer();
	  GateCrystalSD* SD = (GateCrystalSD*) SDman->FindSensitiveDetector(m_SDname, true);
	  GetDigitizerMgr()->AddNewSinglesDigitizer( new GateSinglesDigitizer(GetDigitizerMgr(),GetNewCollectionName(),SD) );

  } else if (childTypeName=="CoincidenceSorter") {
	  // One CoinSorter per System! Defined in the constructor of the system ! Only its parameters should be defiend with CS messenger
	 GetDigitizerMgr()->AddNewCoincidenceSorter( new GateCoincidenceSorter(GetDigitizerMgr(),GetNewCollectionName()) );
  }  else if (childTypeName=="CoincidenceDigitizer") {
    GetDigitizerMgr()->AddNewCoincidenceDigitizer( new GateCoincidenceDigitizer(GetDigitizerMgr(),GetNewCollectionName()) );
  }
else {
    G4cout << "Digitizer module type name '" << childTypeName << "' was not recognised --> insertion request must be ignored!\n";
    return;
  }

  SetNewCollectionName("");
}




//  Check whether there may be a name conflict between a new
//  attachment and an already existing one
G4bool GateDigitizerMgrMessenger::CheckNameConflict(const G4String& newName)
{
  // Check whether an object with the same name already exists in the list
  return (GetDigitizerMgr()->FindElement(newName) != 0 ) ;
}



/*  Check for a potential name conflict between a new attachment and an already existing one.
    If such a conflict is found, a new, conflict-free, name is generated
*/
void GateDigitizerMgrMessenger::AvoidNameConflicts()
{
	//	G4cout<<" AvoidNameConflicts "<<G4endl;
  // Look for a potential name-conflict
  if (!CheckNameConflict( GetNewCollectionName() )) {
    // No name conflict, it's OK
    return;
  }


  G4int copyNumber = 2;
  char buffer[256];

  // Try with 'name2', 'name-3',... until the conflict is solved
  do {
    sprintf(buffer, "%s%i", GetNewCollectionName().c_str(), copyNumber);
    copyNumber++;
  } while (CheckNameConflict(buffer));

  G4cout << "Warning: the selected insertion name ('" << GetNewCollectionName() << "') was already in use.\n"
      	    "Name '" << buffer << "' will be used instead.\n";

  // A conflict-free name was found: store it into the insertion-name variable
  SetNewCollectionName(buffer);
}
