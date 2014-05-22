/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateObjectRepeaterMessenger_h
#define GateObjectRepeaterMessenger_h 1

#include "GateClockDependentMessenger.hh"

class GateVGlobalPlacement;


/*! \class GateObjectRepeaterMessenger
    \brief Base class for GateVGlobalPlacement messengers
    
    - GateObjectRepeaterMessenger - by Daniel.Strul@iphe.unil.ch 
    
    - The GateObjectRepeaterMessenger inherits from the abilities/responsabilities
      of the GateClockDependentMessenger base-class: creation and management
      of a Gate UI directory for a Gate object; UI commands "describe",
      'enable' and 'disable'.
      
    - It currently does not propose any new command, but may have 
      some in the future

*/      
class GateObjectRepeaterMessenger: public GateClockDependentMessenger
{
  public:
    GateObjectRepeaterMessenger(GateVGlobalPlacement* itsObjectRepeater);
   ~GateObjectRepeaterMessenger();
    
    void SetNewValue(G4UIcommand*, G4String);

    virtual inline GateVGlobalPlacement* GetObjectRepeater() 
      { return (GateVGlobalPlacement*) GetClockDependent(); }
    
};

#endif

