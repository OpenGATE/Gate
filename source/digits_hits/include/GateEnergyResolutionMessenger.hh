/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

/*! \class  GateEnergyResolutionMessenger
    \brief  Messenger for the GateEnergyResolution

    - GateEnergyResolution - by by Martin.Rey@epfl.ch (nov 2002)

    \sa GateEnergyResolution, GateEnergyResolutionMessenger
*/


#ifndef GateEnergyResolutionMessenger_h
#define GateEnergyResolutionMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateEnergyResolution;
class G4UIcmdWithAString;

class GateEnergyResolutionMessenger : public GateClockDependentMessenger
{
public:
  
  GateEnergyResolutionMessenger(GateEnergyResolution*);
  ~GateEnergyResolutionMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateEnergyResolution* m_EnergyResolution;
  G4UIcmdWithADouble   *resoCmd;
  G4UIcmdWithADouble   *resoMinCmd;
  G4UIcmdWithADouble   *resoMaxCmd;
  G4UIcmdWithADoubleAndUnit *erefCmd;
  G4UIcmdWithADoubleAndUnit *slopeCmd;


};

#endif








