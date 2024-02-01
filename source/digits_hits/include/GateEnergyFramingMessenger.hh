/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  \class  GateEnergyFramingMessenger
  \brief  Messenger for the GateEnergyFraming

  - GateEnergyFraming

  Previous authors: Daniel.Strul@iphe.unil.ch, Steven.Staelens@rug.ac.be
  Added to GND in November 2022 by olga.kochebina@cea.fr
  // OK GND 2022
	
  Last modification (Adaptation to GND): June 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/


#ifndef GateEnergyFramingMessenger_h
#define GateEnergyFramingMessenger_h 1

#include "G4UImessenger.hh"
#include "globals.hh"

#include "GateClockDependentMessenger.hh"
#include "GateVEffectiveEnergyLaw.hh"

class GateEnergyFraming;

class GateEnergyFramingMessenger : public GateClockDependentMessenger
{
public:
  
  GateEnergyFramingMessenger(GateEnergyFraming*);
  ~GateEnergyFramingMessenger();
  
  void SetNewValue(G4UIcommand*, G4String);

  
private:
  GateEnergyFraming* m_EnergyFraming;
  GateVEffectiveEnergyLaw* SetEnergyFLaw(const G4String& law);
  G4UIcmdWithADoubleAndUnit          *setMinCmd;
  G4UIcmdWithADoubleAndUnit          *setMaxCmd;
  G4UIcmdWithAString			 	 *setLawCmd;

};

#endif








