/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022


/*! \class  GateEfficiencyMessenger
    \brief  Messenger for the GateEfficiency

    - GateEfficiency

    Last modification: olga.kochebina@cea.fr
	Previous authors are unknown

    \sa GateEfficiency, GateEfficiencyMessenger
*/


#ifndef GateEfficiencyMessenger_h
#define GateEfficiencyMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateEfficiency;

class GateEfficiencyMessenger : public GateClockDependentMessenger
{
public:
  
  GateEfficiencyMessenger(GateEfficiency*);
  ~GateEfficiencyMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateEfficiency* m_EnergyEfficiency;

  G4UIcmdWithADouble   *uniqueEfficiencyCmd;
  G4UIcmdWithAString   *efficiencyCmd;
  G4UIcmdWithAnInteger *enableCmd;
  G4UIcmdWithAnInteger *disableCmd;
  G4UIcmdWithAString   *modeCmd;



};

#endif








