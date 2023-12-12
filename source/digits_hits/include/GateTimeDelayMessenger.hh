/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*! 
  \class  GateTimeDelayMessenger
  \brief  Messenger for the GateTimeDelay
  \sa GateTimeDelay, GateTimeDelayMessenger
    
  Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/


#ifndef GateTimeDelayMessenger_h
#define GateTimeDelayMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateTimeDelay;
class G4UIcmdWithAString;

class GateTimeDelayMessenger : public GateClockDependentMessenger
{
public:
  
  GateTimeDelayMessenger(GateTimeDelay*);
  ~GateTimeDelayMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateTimeDelay* m_TimeDelay;
  G4UIcmdWithADoubleAndUnit*          TimeDCmd;


};

#endif






