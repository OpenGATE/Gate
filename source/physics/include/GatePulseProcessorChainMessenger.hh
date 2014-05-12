/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GatePulseProcessorChainMessenger_h
#define GatePulseProcessorChainMessenger_h 1

#include "globals.hh"
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

#include "GatePulseProcessorChain.hh"


class GateVPulseProcessor;

/*! \class  GatePulseProcessorChainMessenger
    \brief  This messenger manages a chain of pulse-processor modules 

    - GatePulseProcessorChainMessenger - by Daniel.Strul@iphe.unil.ch
*/      
//    Last modification in 12/2011 by Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr, for the multi-system approach.

class GatePulseProcessorChainMessenger: public GateListMessenger
{
  public:
    //! Constructor: the argument is the chain of pulse-processor modules
    GatePulseProcessorChainMessenger(GatePulseProcessorChain* itsProcessorChain);

    ~GatePulseProcessorChainMessenger();
        
    //! Standard messenger command interpreter
    void SetNewValue(G4UIcommand*, G4String);

    G4bool CheckNameConflict(const G4String& name);

  private:
    //! Dumps the list of modules that the user can insert into the chain
    virtual const G4String& DumpMap();
    
    //! Inserts a new module into the pulse-processor chain
    virtual void DoInsertion(const G4String& typeName);
    
    //! Returns the pulse-processor chain managed by this messenger 
    virtual GatePulseProcessorChain* GetProcessorChain()
      { return (GatePulseProcessorChain*) GetListManager(); }
    
  private:
  
    G4UIcmdWithAString*         SetInputNameCmd;        //!< The UI command "set input name"
};

#endif

