/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateCoincidencePulseProcessorChainMessenger_h
#define GateCoincidencePulseProcessorChainMessenger_h 1

#include "globals.hh"
#include "GateListMessenger.hh"

class G4UIdirectory;
class G4UIcmdWithAString;
class G4UIcmdWithABool;

#include "GateCoincidencePulseProcessorChain.hh"


class GateVPulseProcessor;

/*! \class  GateCoincidencePulseProcessorChainMessenger
    \brief  This messenger manages a chain of pulse-processor modules 

    - GateCoincidencePulseProcessorChainMessenger - by davguez@yahoo.fr
*/      
//    Last modification in 12/2011 by Abdul-Fattah.Mohamad-Hadi@subatech.in2p3.fr, for the multi-system approach.

class GateCoincidencePulseProcessorChainMessenger: public GateListMessenger
{
  public:
    //! Constructor: the argument is the chain of pulse-processor modules
    GateCoincidencePulseProcessorChainMessenger(GateCoincidencePulseProcessorChain* itsProcessorChain);

    ~GateCoincidencePulseProcessorChainMessenger();
        
    //! Standard messenger command interpreter
    void SetNewValue(G4UIcommand*, G4String);

    G4bool CheckNameConflict(const G4String& name);

  private:
    //! Dumps the list of modules that the user can insert into the chain
    virtual const G4String& DumpMap();
    
    //! Inserts a new module into the pulse-processor chain
    virtual void DoInsertion(const G4String& typeName);
    
    //! Returns the pulse-processor chain managed by this messenger 
    virtual GateCoincidencePulseProcessorChain* GetProcessorChain()
      { return (GateCoincidencePulseProcessorChain*) GetListManager(); }
    
  private:
  
    G4UIcmdWithAString*         AddInputNameCmd;        //!< The UI command "add input name"
    G4UIcmdWithABool*           usePriorityCmd;    //!< The UI command "usePriority"
};

#endif

