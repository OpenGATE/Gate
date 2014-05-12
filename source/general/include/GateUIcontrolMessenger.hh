/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*----------------------

   GATE - Geant4 Application for Tomographic Emission 
   OpenGATE Collaboration 
     
   Daniel Strul <daniel.strul@iphe.unil.ch> 
     
   Copyright (C) 2002,2003 UNIL/IPHE, CH-1015 Lausanne 

This software is distributed under the terms 
of the GNU Lesser General  Public Licence (LGPL) 
See GATE/LICENSE.txt for further details 
----------------------*/

#ifndef GateUIcontrolMessenger_h
#define GateUIcontrolMessenger_h 1

#include "globals.hh"
#include "GateMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithoutParameter;
class G4UIcmdWithAString;
class G4UIcmdWithABool;
class G4UIcmdWithAnInteger;
class G4UIcmdWithADouble;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWith3Vector;
class G4UIcmdWith3VectorAndUnit;

/*! \class GateUIcontrolMessenger
    \brief Provides new control commands
    
    - GateUIcontrolMessenger - by Daniel.Strul@iphe.unil.ch 
    
    - The GateUIcontrolMessenger inherits from the abilities/responsabilities
      of the GateMessenger base-class, i.e. the creation and management
      of a Gate UI directory
      
    - In addition, it provides the new command "/gate/control/execute", 
      which can find and execute a macro file either in the current 
      directory or in $GATEHOME
*/      
class GateUIcontrolMessenger: public GateMessenger
{
  public:
    //! Constructor
    GateUIcontrolMessenger();

    //! Destructor
    virtual ~GateUIcontrolMessenger();
    
    //! UI command interpreter method
    void SetNewValue(G4UIcommand*, G4String);

    //! Execute a macrofile located either in the current directory or in $GATEHOME
    void LaunchMacroFile(G4String fileName);

  private:

    G4UIcmdWithAString*         ExecuteCmd;	//!< the UI command 'execute'
};

#endif

