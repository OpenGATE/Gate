/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateNamedObjectMessenger_h
#define GateNamedObjectMessenger_h 1

#include "GateMessenger.hh"

class GateNamedObject;


/*! \class GateNamedObjectMessenger
    \brief Base class for GateNamedVolume messengers
    
    - GateNamedObjectMessenger - by Daniel.Strul@iphe.unil.ch 
    
    - The main function of a GateNamedObjectMessenger is to create/manage 
      a (new) UI directory for a GateNamedObjectMessenger

    - It also proposes/handles the UI command 'describe'.
    
    - Note that, since the GateNamedObjectMessenger is based on the GateMessenger
      base-class, it inherits the methods GateMessenger::TellGeometryToUpdate()
      and GateMessenger::TellGeometryToRebuild() that allow messengers to command the
      geometry updating.
*/      
class GateNamedObjectMessenger: public GateMessenger
{
  public:
    //! Constructor
    //! The flag 'itsFlagDescribe' tells whether this messenger should propose a command "Describe"
    //! The flag 'flagCreateDirectory' is passed to the base-class GateMessenger
    GateNamedObjectMessenger(GateNamedObject* itsNamedVolume,
    			     const G4String& itsDirectoryName="");

   ~GateNamedObjectMessenger(); //!< destructor
    
    //! UI command interpreter method
    void SetNewValue(G4UIcommand*, G4String);

    //! Get the object
    inline GateNamedObject* GetNamedObject() const 
      { return pNamedVolume; }

  protected:
    GateNamedObject*  	      pNamedVolume;  	//!< The named object for this messenger

    G4UIcmdWithoutParameter*  pDescribeCmd;    	//!< The UI command "describe"
    G4UIcmdWithAString*       pDefineNameCmd;    //!< The command "name"
    G4UIcmdWithAnInteger*     pVerbosityCmd;  	    //!< UI command 'verbose'
    
};

#endif

