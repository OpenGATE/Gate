/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSystemListManagerMessenger_h
#define GateSystemListManagerMessenger_h 1

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

class GateSystemListManager;
class GateVSystem;

/*! \class GateSystemListManagerMessenger
    \brief A messenger for a GateSystemListManager
    
    - GateSystemListManagerMessenger - by Daniel.Strul@iphe.unil.ch 
    
    - The GateSystemListManagerMessenger inherits from the abilities/responsabilities
      of the GateClockDependentMessenger base-class, i.e. the creation and management
      of a Gate UI directory for a Gate object
      
    - In addition, the main responsability of this messenger is to handle
      the system store, and to allow the insertion of new systems.

    - It proposes and manages commands specific to the system-stores
      (these commands are fairly similar to those of other insertion messengers):
      definition of the name of a new system, listing of system choices,
      list of already created systems, and insertion of a new system

*/      
class GateSystemListMessenger: public GateListMessenger
{
  public:
    //! Constructor
    GateSystemListMessenger(GateSystemListManager* itsListManager);

    virtual ~GateSystemListMessenger();  //!< destructor
    
    //! UI command interpreter method
    void SetNewValue(G4UIcommand*, G4String);

    //! Lists all the system-names into a string
    virtual const G4String& DumpMap();

    //! Pure virtual method: create and insert a new attachment
    virtual void DoInsertion(const G4String& typeName);

    //! Get the store pointer
    inline GateSystemListManager* GetSystemListManager() 
      { return (GateSystemListManager*) GetListManager(); }

  private:
};

#endif

