/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#ifndef GateOpticalAdderMessenger_h
#define GateOpticalAdderMessenger_h 1

#include "GatePulseProcessorMessenger.hh"

class G4UIdirectory;
class GateOpticalAdder;

/*! \class  GateOpticalAdderMessenger
    \brief  Messenger for the GateOpticalAdder

    - GateOpticalAdderMessenger - by d.j.vanderlaan@iri.tudelft.nl
*/
class GateOpticalAdderMessenger: public GatePulseProcessorMessenger
{
  public:
    GateOpticalAdderMessenger(GateOpticalAdder* itsPulseAdder);
    inline ~GateOpticalAdderMessenger() {}

    void SetNewValue(G4UIcommand* aCommand, G4String aString);

    inline GateOpticalAdder* GetPulseAdder()
    { return (GateOpticalAdder*) GetPulseProcessor();}
};

#endif

#endif
