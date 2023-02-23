/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

/*! \class  GateEnergyFramingMessenger
    \brief  Messenger for the GateEnergyFraming

    - GateEnergyFraming

	Previous authors: Daniel.Strul@iphe.unil.ch, Steven.Staelens@rug.ac.be
   Added to GND in November 2022 by olga.kochebina@cea.fr


    \sa GateEnergyFraming, GateEnergyFramingMessenger
*/


#ifndef GateEnergyFramingMessenger_h
#define GateEnergyFramingMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
class GateEnergyFraming;

class GateEnergyFramingMessenger : public GateClockDependentMessenger
{
public:
  
  GateEnergyFramingMessenger(GateEnergyFraming*);
  ~GateEnergyFramingMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateEnergyFraming* m_EnergyFraming;
  G4UIcmdWithADoubleAndUnit          *setMinCmd;
  G4UIcmdWithADoubleAndUnit          *setMaxCmd;

};

#endif








