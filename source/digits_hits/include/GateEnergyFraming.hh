/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*!
  \class  GateEnergyFraming
  \brief  GateEnergyFraming applies an energy window selection
  ex-GateThresholder + ex-GateUpholder

  - GateEnergyFraming

  Previous authors: Daniel.Strul@iphe.unil.ch, Steven.Staelens@rug.ac.be

  Added to GND in November 2022 by olga.kochebina@cea.fr
  // OK GND 2022
  
  Last modification (Adaptation to GND): June 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/


#ifndef GateEnergyFraming_h
#define GateEnergyFraming_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateEnergyFramingMessenger.hh"
#include "GateSinglesDigitizer.hh"
#include "GateVEffectiveEnergyLaw.hh"


class GateEnergyFraming : public GateVDigitizerModule
{
public:
  
  GateEnergyFraming(GateSinglesDigitizer *digitizer, G4String name);
  ~GateEnergyFraming();
  
  void Digitize() override;

  void SetMin(G4double val)   { m_min = val;  }
  G4double GetMin()   	      { return m_min; }

  void SetMax(G4double val)   { m_max = val;  }
  G4double GetMax()   	      { return m_max; }
  
  inline void SetEnergyFLaw(GateVEffectiveEnergyLaw* law)   { m_EnergyFramingLaw = law; }
  void DescribeMyself(size_t );

protected:
  G4double   m_min;
  G4double   m_max;

private:
  GateDigi* m_outputDigi;

  GateEnergyFramingMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateVEffectiveEnergyLaw* m_EnergyFramingLaw;

  GateSinglesDigitizer *m_digitizer;

};

#endif








