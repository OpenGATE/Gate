/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*! \class  GatePulseAdderMessenger
    \brief  Messenger for the GatePulseAdder

    - GatePulseAdderMessenger - by Daniel.Strul@iphe.unil.ch

    \sa GatePulseAdder, GatePulseProcessorMessenger
*/


#ifndef GateAdderMessenger_h
#define GateAdderMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateAdder;
class G4UIcmdWithAString;

class GateAdderMessenger : public GateClockDependentMessenger
{
public:
  
  GateAdderMessenger(GateAdder*);
  ~GateAdderMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

private:
  GateAdder* m_Adder;
  //G4UIdirectory           *Dir;
  G4UIcmdWithAString          *positionPolicyCmd;


};

#endif








