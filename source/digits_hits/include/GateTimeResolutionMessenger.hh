/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

/*! \class  GateTimeResolutionMessenger
    \brief  Messenger for the GateTimeResolution

    - GateTimeResolution - by Martin.Rey@epfl.ch (July 2003)

    \sa GateTimeResolution, GateTimeResolutionMessenger
*/


#ifndef GateTimeResolutionMessenger_h
#define GateTimeResolutionMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateTimeResolution;
class G4UIcmdWithAString;

class GateTimeResolutionMessenger : public GateClockDependentMessenger
{
public:
  
  GateTimeResolutionMessenger(GateTimeResolution*);
  ~GateTimeResolutionMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateTimeResolution* m_TimeResolution;
  G4UIcmdWithADoubleAndUnit   *fwhmCmd;
  G4UIcmdWithADoubleAndUnit   *ctrCmd;
  G4UIcmdWithADoubleAndUnit   *doiCmd;


};

#endif








