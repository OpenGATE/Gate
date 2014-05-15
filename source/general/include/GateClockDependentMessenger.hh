/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateClockDependentMessenger_h
#define GateClockDependentMessenger_h 1

#include "GateNamedObjectMessenger.hh"

class GateClockDependent;



/*! \class GateClockDependentMessenger
    \brief Base class for GateClockDependent messengers
    
    - GateClockDependentMessenger - by Daniel.Strul@iphe.unil.ch 
    
    - The GateClockDependentMessenger inherits from the abilities/responsabilities
      of the GateNamedObjectMessenger base-class: creation and management
      of a Gate UI directory for a Gate object; creation of the UI command "describe"
      
    - In addition, it proposes and manages the UI commands 'enable' and 'disable'.

*/      
class GateClockDependentMessenger: public GateNamedObjectMessenger
{
  public:
    //! Constructor
    //! The flags are passed to the base-class GateNamedObjectMessenger
    GateClockDependentMessenger(GateClockDependent* itsClockDependent,
    			        const G4String& itsDirectoryName="");

   ~GateClockDependentMessenger();  //!< destructor
    
    //! UI command interpreter method
    void SetNewValue(G4UIcommand*, G4String);
    void SetARFCommands();/* PY Descourt 08/09/2009 */  
    //! Get the clock-dependent volume
    inline GateClockDependent* GetClockDependent() 
      { return (GateClockDependent*) GetNamedObject(); }

  private:
    G4UIcmdWithABool* 	      pEnableCmd;    //!< The UI command "enable"
    G4UIcmdWithABool* 	      pDisableCmd;   //!< The UI command "disable"
	/* PY Descourt 08/09/2009 */  
    G4UIcmdWithAString*       ARFcmd;
    G4UIcmdWithoutParameter*  AttachARFSDcmd;
	/* PY Descourt 08/09/2009 */  
};

#endif
