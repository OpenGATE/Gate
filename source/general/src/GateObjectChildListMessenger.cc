/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateObjectChildListMessenger.hh"
#include "GateObjectChildList.hh"

#include "GateVVolume.hh"
#include "GateVolumeManager.hh"
#include "GateSystemListManager.hh"
#include "GateObjectStore.hh"

#include "GateMessageManager.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"

#include <string>
 
//-----------------------------------------------------------------------------------------
GateObjectChildListMessenger::GateObjectChildListMessenger(GateObjectChildList* itsChildList)
  :GateListMessenger(itsChildList)
{ 

  pInsertCmd->SetCandidates(DumpMap());

  /*
    pInsertCmd->AvailableForStates(G4State_PreInit);
  */
}
//-----------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------
GateObjectChildListMessenger::~GateObjectChildListMessenger(){}
//-----------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------
const G4String& GateObjectChildListMessenger::DumpMap()
{
  static G4String theList;
  typedef GateVVolume *(*maker_volume)(const G4String& itsName, G4bool acceptsChildren, G4int depth);
  
  std::map<G4String,maker_volume> Child;
  Child = GateVolumeManager::GetInstance()->theListOfVolumePrototypes;
  std::map<G4String,maker_volume>::iterator iter; 
  
  GateMessage("Geometry", 10, "The available types of child-object are: \n"); 
  for (iter = Child.begin(); iter!=Child.end(); iter++) {
    theList+=iter->first;
    theList+=" ";     
    GateMessage("Geometry", 10, " " << iter->first << G4endl;);
  }
    
  return theList;  
}
//-----------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------
void GateObjectChildListMessenger::ListChoices()
{

  static G4String theList;
  typedef GateVVolume *(*maker_volume)(const G4String& itsName, G4bool acceptsChildren, G4int depth);
  
  std::map<G4String,maker_volume> Child;
  Child = GateVolumeManager::GetInstance()->theListOfVolumePrototypes;
  std::map<G4String,maker_volume>::iterator iter; 
   
  GateMessage("Geometry", 10, "The available types of child-object are: \n");
  
  for (iter = Child.begin(); iter!=Child.end(); iter++) {
    GateMessage("Geometry", 10, " " << iter->first << G4endl;);
  }
}
//-----------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------
void GateObjectChildListMessenger::DoInsertion(const G4String& childTypeName)
{
  InsertIntoCreator(childTypeName);
}
//-----------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------
void GateObjectChildListMessenger::InsertIntoCreator(const G4String& childTypeName)
{
  // GetNewInsertionBaseName() is defined in GateListMessenger
  if (GetNewInsertionBaseName().empty())
    SetNewInsertionBaseName(childTypeName);

  AvoidNameConflicts();  

  GateVVolume* newChild=0;
  G4bool acceptsNewChildren = true;
  G4int depth = 0;

  if (GateVolumeManager::GetInstance()->theListOfVolumePrototypes[childTypeName]){
    newChild = (GateVolumeManager::GetInstance()->theListOfVolumePrototypes[childTypeName](GetNewInsertionBaseName(), acceptsNewChildren, depth));

    // GateMessage("Core", 0, "Create child '" << newChild->GetObjectName() 
    //                 << "' with parent '" << GetCreator()->GetObjectName() << G4endl);
    newChild->SetParentVolume(GetCreator());

  }
  else {
    GateError("Child type name '" << childTypeName << "' was not recognised --> insertion request must be ignored!\n");
    //G4cout << "Child type name '" << childTypeName << "' was not recognised --> insertion request must be ignored!\n";
    //return;
  }    
  GetChildList()->AddChild(newChild);
  SetNewInsertionBaseName("");   
  GateSystemListManager::GetInstance()->CheckScannerAutoCreation(newChild);
}
//-----------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------------
G4bool GateObjectChildListMessenger::CheckNameConflict(const G4String& name)
{ 
  return ( GateObjectStore::GetInstance()->FindCreator(name) != 0 ) ;
}
//-----------------------------------------------------------------------------------------

