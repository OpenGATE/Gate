/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateSystemComponentMessenger_h
#define GateSystemComponentMessenger_h 1

#include "GateClockDependentMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

class GateSystemComponent;



/*! \class GateSystemComponentMessenger
    \brief Base class for GateSystemComponent messengers
    
    - GateSystemComponentMessenger - by Daniel.Strul@iphe.unil.ch 
    
    - The GateSystemComponentMessenger inherits from the abilities/responsabilities
      of the GateClockDependentMessenger base-class, i.e. the creation and management
      of a Gate UI directory for a Gate object
      
    - In addition, it proposes and manages UI commands that are specific
      to system components: 'attach', and 'describe'

    - Note: from July to Oct. 2002, a system was a vector of system-levels, each level managing a
      	    vector of components. On Oct. 2002, the whole system (which was too complex) was redesigned, 
      	    and the current mechanism (tree of system-components) replaced the previous one (vector of
      	    system-levels). The system-component messengers are now connected to these new system-components
	    rather than to the old ones.

*/      
//    Last modification in 12/2011 by Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr, for the multi-system approach.

class GateSystemComponentMessenger: public GateClockDependentMessenger
{
  public:
    GateSystemComponentMessenger(GateSystemComponent* itsSystemComponent);  //!< constructor
    ~GateSystemComponentMessenger();  	      	      	      	      	    //!< destructor
    
    //! UI command interpreter method
    void SetNewValue(G4UIcommand*, G4String);

    //! Get the system component
    inline GateSystemComponent* GetSystemComponent() 
      { return (GateSystemComponent*) GetClockDependent(); }
      
  protected:
    //! Method to apply the UI command 'attach'
    //! Finds an creator from its name and attaches this creator to the system component
    void AddCreator(const G4String& creatorName);
    G4String FabricateDirName(const GateSystemComponent* component);// Fabricate the directory name (multi-system approach)

  private:
    
    G4UIcmdWithAString	      	*AttachCmd;         //!< The UI command "attach"

		G4UIcmdWithAnInteger*          minSectorDiffCmd ;
    G4UIcmdWithAString*            setInCoincidenceWithCmd ;
    G4UIcmdWithAnInteger*          setRingIDCmd;
    G4String                       m_dirName;
};

#endif

