/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateModuleListManager_h
#define GateModuleListManager_h 1

#include "globals.hh"

#include "GateListManager.hh"

class GateVObjectCreator;

class GateModuleListManager : public GateListManager
{
  public:
    GateModuleListManager(GateNamedObject* itsMotherObject,
    			  const G4String& itsName,
    		    	  const G4String& itsElementTypeName,
    		    	  G4bool canBeDisabled=true,
    		    	  G4bool acceptNewElements=true);
    virtual ~GateModuleListManager();

  public:
    //! Method overloading GateNamedObject::Describe()
    //! Print-out a description of the object
    virtual void Describe(size_t indent=0);
     
    GateNamedObject* GetMotherObject() const
      { return pMotherObject; }

  protected:
    GateNamedObject*    pMotherObject;
};

#endif

