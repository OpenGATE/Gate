/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateObjectChildListMessenger_hh
#define GateObjectChildListMessenger_hh 1

#include "GateObjectChildList.hh"
#include "GateListMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;
class G4UIcmdWithoutParameter;
class G4UIcmdWithABool;
class G4UIcmdWithADouble;


#include "globals.hh"

/*! \class GateObjectChildListMessenger
  
  \brief the messenger belonging to a GateSurfaceList

  Does not define any new methods and commands. All of them are inherited 
  from GateListMessenger.
  
  - GateSurfaceListMessenger - by d.j.vanderlaan@tnw.tudelft.nl
  
  */
class GateVVolume;

class GateObjectChildListMessenger : public GateListMessenger
{
  public:
   GateObjectChildListMessenger(GateObjectChildList* itsChildList);
   virtual ~GateObjectChildListMessenger();
    
  protected:
    //! returns ""
    virtual const G4String& DumpMap();
    //! creates a GateSurface and adds it to the GateSurfaceList
    virtual void DoInsertion(const G4String& volumeName);
    //! lists the possible choices for the insert command (inherited)
    virtual void InsertIntoCreator(const G4String& childTypeName);
    
    virtual void ListChoices();
    
    //! checks if name is ok
    virtual G4bool CheckNameConflict(const G4String& name);
  
    virtual GateObjectChildList* GetChildList() { return (GateObjectChildList*) GetListManager();}
    //! returns the GateVObjectCreator where the list this messenger belongs to belongs to
    virtual GateVVolume* GetCreator() { return GetChildList()->GetCreator();}

    GateObjectChildListMessenger* pMessenger;
};

#endif

