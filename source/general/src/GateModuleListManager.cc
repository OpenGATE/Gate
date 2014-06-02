/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateModuleListManager.hh"
#include "GateTools.hh"

//--------------------------------------------------------------------------------------------
GateModuleListManager::GateModuleListManager(GateNamedObject* itsMotherObject,
    			  		     const G4String& itsName,
				 	     const G4String& itsElementTypeName,
    		    	  		     G4bool canBeDisabled,
				 	     G4bool acceptNewElements)
  : GateListManager(itsName, itsElementTypeName, canBeDisabled, acceptNewElements),
    pMotherObject(itsMotherObject)
{
}
//--------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------
GateModuleListManager::~GateModuleListManager()
{  
}
//--------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------
// Method overloading GateClockDependent::Describe()
// Print-out a description of the object
void GateModuleListManager::Describe(size_t indent)
{
  GateListManager::Describe(indent);
  G4cout << GateTools::Indent(indent) << "Attached to:        '" << pMotherObject->GetObjectName() << "'" << G4endl;
}
//--------------------------------------------------------------------------------------------
