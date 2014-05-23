/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateObjectMoveListMessenger.hh"
#include "GateMessageManager.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"

#include "GateObjectRepeaterList.hh"
#include "GateDetectorConstruction.hh"

#include "GateVolumePlacement.hh"
#include "GateTranslationMove.hh"
#include "GateRotationMove.hh"
#include "GateOrbitingMove.hh"
#include "GateEccentRotMove.hh"
#include "GateOscTranslationMove.hh"
#include "GateGenericRepeaterMove.hh"
#include "GateGenericMove.hh"

//------------------------------------------------------------------------------------------------
GateObjectMoveListMessenger::GateObjectMoveListMessenger(GateObjectRepeaterList* itsRepeaterList)
  :GateListMessenger(itsRepeaterList)
{ 
  //  SetObjectMoveListMess(this);
  //  SetFlagMove(false);
  
  // G4cout << " *** Constructeur GateObjectMoveListMessenger" << " SetFlagMove(0) =  " << GetFlagMove() << G4endl;
  pInsertCmd->SetCandidates(DumpMap());
}
//------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------
GateObjectMoveListMessenger::~GateObjectMoveListMessenger()
{
}
//------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------
void GateObjectMoveListMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  GateListMessenger::SetNewValue(command,newValue);
}
//------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------
const G4String& GateObjectMoveListMessenger::DumpMap() {
  static G4String theList = "placement translation rotation orbiting osc-trans eccent-rot genericRepeaterMove genericMove";
  return theList;
}
//------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------
void GateObjectMoveListMessenger::ListChoices()
{
  G4cout << "The available types of movement are: \n"; 
  G4cout << "  placement:\n"
    "    Sets the object's position and orientation \n";
  G4cout << "  translation:\n"
    "    Sets the object in uniform linear translation M(t) = t * V \n";
  G4cout << "  rotation:\n"
    "    Sets the object in uniform rotation around an axis passing \n"
    "    through its center: angle(t) = t * v \n";
  G4cout << "  orbiting:\n"
    "    Sets the object in uniform orbit around an arbitrary axis\n";
  G4cout << "  eccent-rot:\n"
    "    Sets the object in eccentric position (X-Y-Z) and rotate it round OZ lab frame axis\n";
  G4cout << "  genericRepeaterMove:\n"
    "    TODO\n";
  G4cout << "  genericMove:\n"
    "    TODO\n";
}
//------------------------------------------------------------------------------------------------


//------------------------------------------------------------------------------------------------
void GateObjectMoveListMessenger::DoInsertion(const G4String& moveTypeName)
{

  //   G4cout << " *** GateObjectMoveListMessenger::DoInsertion " << G4endl;
  //   G4cout << " *** moveTypeName = " << moveTypeName << G4endl;
      
  if (GetNewInsertionBaseName().empty())
    SetNewInsertionBaseName(moveTypeName);

  AvoidNameConflicts();

  GateVGlobalPlacement* newMove=0;

  G4String insertionName = GetRepeaterList()->MakeElementName(GetNewInsertionBaseName());

  if (moveTypeName=="placement")
    newMove = new GateVolumePlacement(GetCreator(), insertionName);
  else if (moveTypeName=="translation")
    newMove = new GateTranslationMove(GetCreator(), insertionName);
  else if (moveTypeName=="rotation")
    newMove = new GateRotationMove(GetCreator(), insertionName);
  else if (moveTypeName=="orbiting"){
    GateMessage("Geometry", 5, " Insert orbiting move\n");
    newMove = new GateOrbitingMove(GetCreator(), insertionName);}
  else if (moveTypeName=="eccent-rot")
    newMove = new GateEccentRotMove(GetCreator(), insertionName);
  else if (moveTypeName=="osc-trans")
    newMove = new GateOscTranslationMove(GetCreator(),insertionName);
  else if (moveTypeName=="genericRepeaterMove")
    newMove = new GateGenericRepeaterMove(GetCreator(),insertionName);
  else if (moveTypeName=="genericMove")
    newMove = new GateGenericMove(GetCreator(),insertionName);
  else {
    GateError("Move type name '" << moveTypeName << "' was not recognised --> insertion request must be ignored!\n");
  }
  
  GetRepeaterList()->AppendObjectRepeater(newMove);      
  SetNewInsertionBaseName("");
  
  GateDetectorConstruction::GetGateDetectorConstruction()->SetFlagMove(true);
}
//------------------------------------------------------------------------------------------------

