/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateListManager_h
#define GateListManager_h 1

#include "globals.hh"
#include <vector>
#include <algorithm>

#include "GateClockDependent.hh"

class GateVObjectCreator;

class GateListManager : public GateClockDependent
{
  public:
    GateListManager(const G4String& itsName,
    		    const G4String& itsElementTypeName,
    		    G4bool canBeDisabled=true,
    		    G4bool acceptNewElements=true);
    virtual ~GateListManager();

  public:
     virtual  void TheListElements(size_t indent=0) const;
     //virtual  void ListElements(size_t indent=0) const;
     
     virtual GateNamedObject* FindElement(const G4String& name);
     virtual inline GateNamedObject* GetElement(size_t i)
      	  {return (i<theListOfNamedObject.size()) ? theListOfNamedObject[i] : 0; }
     virtual inline size_t size() const
      	  {return theListOfNamedObject.size(); }
     virtual inline GateNamedObject* FindElementByBaseName(const G4String& baseName)
       { return FindElement( MakeElementName(baseName) ) ; }

     inline G4bool AcceptNewElements() const 
      	  { return bAcceptNewElements;}

     inline const G4String&  GetElementTypeName()
      	  { return mElementTypeName;}

    virtual inline G4String MakeElementName(const G4String& newBaseName)
      { return GetObjectName() + "/" + newBaseName; }

    //! Method overloading GateNamedObject::Describe()
    //! Print-out a description of the object
    virtual void Describe(size_t indent=0);
     

  protected:
    G4String mElementTypeName;
    G4bool   bAcceptNewElements;
    
    std::vector<GateNamedObject*>    theListOfNamedObject;
};

#endif

