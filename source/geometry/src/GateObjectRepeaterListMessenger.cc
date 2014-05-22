/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateObjectRepeaterListMessenger.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithADouble.hh"

#include "GateObjectRepeaterList.hh"

#include "GateLinearRepeater.hh"
#include "GateArrayRepeater.hh"
#include "GateAngularRepeater.hh"
#include "GateSphereRepeater.hh"
#include "GateQuadrantRepeater.hh"
#include "GateGenericRepeater.hh"


//-------------------------------------------------------------------------------------------------------
GateObjectRepeaterListMessenger::GateObjectRepeaterListMessenger(GateObjectRepeaterList* itsRepeaterList)
:GateListMessenger(itsRepeaterList)
{ 
  pInsertCmd->SetCandidates(DumpMap());
  //InsertCmd->AvailableForStates(PreInit);
}
//-------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------
GateObjectRepeaterListMessenger::~GateObjectRepeaterListMessenger()
{
}
//-------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------
void GateObjectRepeaterListMessenger::SetNewValue(G4UIcommand* command,G4String newValue)
{ 
  GateListMessenger::SetNewValue(command,newValue);
}
//-------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------
const G4String& GateObjectRepeaterListMessenger::DumpMap() {
  static G4String theList = "linear cubicArray ring quadrant sphere genericRepeater";
  return theList;
}
//-------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------
void GateObjectRepeaterListMessenger::ListChoices()
{
  G4cout << "The available types of object repeater are: \n"; 
  G4cout << "  linear:\n"
            "    Repeats n times the object at the positions P[0], P[1], P[2], ...\n"
	    "    with P[i] = P0 + i* dP\n";
  G4cout << "  cubicArray:\n"
            "    Repeats Nx x Ny x nZ times the object at the positions P\n"
	    "    with P[i,j,k] = P0 + i * dx I + j * dy J + z*dk K\n";
  G4cout << "  ring:\n"
            "    Repeats n times the object on a ring at a distance R from the centre\n";
  G4cout << "  quadrant:\n"
            "    Repeats the object in a triangle-like pattern similar to that of a Derenzo resolution phantom\n";
  G4cout << "  sphere:\n"
  	    "    Repeats the object in a spherical pattern\n";
  G4cout << "  genericRepeater:\n"
  	    "    Repeats the object according to a given list of translation/rotation\n";
}
//-------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------
void GateObjectRepeaterListMessenger::DoInsertion(const G4String& objectRepeaterTypeName)
{
 
  if (GetNewInsertionBaseName().empty())
    SetNewInsertionBaseName(objectRepeaterTypeName);

  AvoidNameConflicts();

  GateVGlobalPlacement* newObjectRepeater=0;
  
  G4String insertionName = GetRepeaterList()->MakeElementName(GetNewInsertionBaseName());

  if (objectRepeaterTypeName=="linear")
    newObjectRepeater = new GateLinearRepeater(GetCreator(),insertionName);
  else if (objectRepeaterTypeName=="cubicArray")
    newObjectRepeater = new GateArrayRepeater(GetCreator(),insertionName);
  else if (objectRepeaterTypeName=="ring")
    newObjectRepeater = new GateAngularRepeater(GetCreator(),insertionName);
  else if (objectRepeaterTypeName=="quadrant")
    newObjectRepeater = new GateQuadrantRepeater(GetCreator(),insertionName);
  else if (objectRepeaterTypeName=="sphere")
    newObjectRepeater = new GateSphereRepeater(GetCreator(),insertionName);
  else if (objectRepeaterTypeName=="genericRepeater")
    newObjectRepeater = new GateGenericRepeater(GetCreator(),insertionName);
  else {
    GateError("Repeater type name '" << objectRepeaterTypeName << "' was not recognised --> insertion request is ignored!\n");
  }
  
  GetRepeaterList()->AppendObjectRepeater(newObjectRepeater);
  SetNewInsertionBaseName("");
}
//-------------------------------------------------------------------------------------------------------

