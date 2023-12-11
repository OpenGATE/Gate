/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022
/*This class is not used by GATE !
  The purpose of this class is to help to create new users digitizer module(DM).
  Please, check GateCrosstalk.cc for more detals
  */


/*! \class  GateCrosstalkMessenger
    \brief  Messenger for the GateCrosstalk

    - GateCrosstalk - by name.surname@email.com

    \sa GateCrosstalk, GateCrosstalkMessenger
*/


#ifndef GateCrosstalkMessenger_h
#define GateCrosstalkMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateCrosstalk;

class G4UIcmdWithAString;
class G4UIcmdWithADouble;

class GateCrosstalkMessenger : public GateClockDependentMessenger
{
public:
  
  GateCrosstalkMessenger(GateCrosstalk*);
  ~GateCrosstalkMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateCrosstalk* m_Crosstalk;

  G4UIcmdWithADouble   *edgesFractionCmd;
  G4UIcmdWithADouble   *cornersFractionCmd;


};

#endif








