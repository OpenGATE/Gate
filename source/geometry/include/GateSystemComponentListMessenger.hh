/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSystemComponentListMessenger_h
#define GateSystemComponentListMessenger_h 1

#include "GateListMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

#include "GateSystemComponentList.hh"



/*! \class GateSystemComponentListMessenger
    \brief This messenger manages a list of system-component
    
    - GateSystemComponentListMessenger - by Daniel.Strul@iphe.unil.ch 
*/      
class GateSystemComponentListMessenger: public GateListMessenger
{
  public:
    GateSystemComponentListMessenger(GateSystemComponentList* itsSystemComponentList);  //!< constructor
    ~GateSystemComponentListMessenger();  	      	      	      	      	        //!< destructor
    
     //! UI command interpreter method
    void SetNewValue(G4UIcommand*, G4String);

  private:
    //! Dumps the list of modules that the user can insert into the list
    virtual const G4String& DumpMap();
    
    //! Inserts a new module into the pulse-processor list
    virtual void DoInsertion(const G4String& typeName);
    
    //! Get the system componentList
    inline GateSystemComponentList* GetSystemComponentList() 
      { return (GateSystemComponentList*) GetListManager(); }
      
    //! Get the system componentList
    inline GateSystemComponent* GetMotherComponent() 
      { return GetSystemComponentList()->GetMotherComponent(); }
      
    /*  \brief Check whether there may be a name conflict between a new
        \brief system component and an already existing one

        \param name: the name of the attachment to create 
    */
    G4bool CheckNameConflict(const G4String& name);
 
  private:
    
};

#endif

