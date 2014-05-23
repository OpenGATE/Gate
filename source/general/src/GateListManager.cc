/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateListManager.hh"
#include "GateNamedObject.hh"
#include "GateDetectorConstruction.hh"
#include "GateTools.hh"

//--------------------------------------------------------------------------------------
GateListManager::GateListManager(const G4String& itsName,
				 const G4String& itsElementTypeName,
    		    	  	 G4bool canBeDisabled,
				 G4bool acceptNewElements)
  : GateClockDependent(itsName,canBeDisabled),
    mElementTypeName(itsElementTypeName),
    bAcceptNewElements(acceptNewElements)
{
}
//--------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------
GateListManager::~GateListManager()
{  
  for (std::vector<GateNamedObject*>::iterator it = theListOfNamedObject.begin(); it != theListOfNamedObject.end(); )
    {
      //delete (*it);
      it = theListOfNamedObject.erase(it);
    }
}
//--------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------
GateNamedObject* GateListManager::FindElement(const G4String& name)
{
  for (size_t i=0; i<theListOfNamedObject.size() ; i++)
    if (theListOfNamedObject[i]->GetObjectName() == name)
      return theListOfNamedObject[i];
  return 0;
}
//--------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------
//void GateListManager::ListElements(size_t indent) const
void GateListManager::TheListElements(size_t indent) const
{
  G4cout << GateTools::Indent(indent) << "Nb of elements:     " << size() << G4endl;
  for (size_t i=0; i<size() ; i++)
    G4cout << GateTools::Indent(indent+2) << '\'' << theListOfNamedObject[i]->GetObjectName() << '\'' << G4endl;
}
//--------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------
// Method overloading GateClockDependent::Describe()
// Print-out a description of the object
void GateListManager::Describe(size_t indent)
{
  GateClockDependent::Describe(indent);
  TheListElements(indent);
}
//--------------------------------------------------------------------------------------
