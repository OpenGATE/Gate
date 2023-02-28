/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*! \class  GateOpticalAdderMessenger
    \brief  Messenger for the GateOpticalAdder

    - GateOpticalAdderMessenger - by d.j.vanderlaan@iri.tudelft.nl

    Added to GND in Feb. 2023 by OK

*/

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#ifndef GateOpticalAdderMessenger_h
#define GateOpticalAdderMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateOpticalAdder;
class G4UIcmdWithAString;

class GateOpticalAdderMessenger : public GateClockDependentMessenger
{
public:
  
  GateOpticalAdderMessenger(GateOpticalAdder*);
  ~GateOpticalAdderMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateOpticalAdder* m_OpticalAdder;


};

#endif
#endif







