/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

/*! \class  GateNoiseMessenger
    \brief  Messenger for the GateNoise

    - GateNoise - by name.surname@email.com

    \sa GateNoise, GateNoiseMessenger
*/


#ifndef GateNoiseMessenger_h
#define GateNoiseMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateNoise;
class G4UIcmdWithAString;

class GateNoiseMessenger : public GateClockDependentMessenger
{
public:
  
  GateNoiseMessenger(GateNoise*);
  ~GateNoiseMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateNoise* m_Noise;
  G4UIcmdWithAString        *m_deltaTDistribCmd;
  G4UIcmdWithAString        *m_energyDistribCmd;


};

#endif








