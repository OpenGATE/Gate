/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

/*! \class  GateEnergyResolution
    \brief  GateEnergyResolution
    Digitizer Module for simulating a Gaussian resolution on the energy spectrum.
 	The user can choose a specific resolution for each type of crystal.
    Each time, (s)he must choose the resolution and the energy of reference.

    - GateEnergyResolution - by Martin.Rey@epfl.ch (nov 2002)

    \sa GateEnergyResolution, GateEnergyResolutionMessenger
*/

#ifndef GateEnergyResolution_h
#define GateEnergyResolution_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateEnergyResolutionMessenger.hh"
#include "GateSinglesDigitizer.hh"


class GateEnergyResolution : public GateVDigitizerModule
{
public:
  
  GateEnergyResolution(GateSinglesDigitizer *digitizer, G4String name);
  ~GateEnergyResolution();
  void SetEnergyResolutionParameters();

  void Digitize() override;

  void SetResolution(G4double val)   { m_reso = val;  }
  void SetResolutionMin(G4double val)   { m_resoMin = val;  }
  void SetResolutionMax(G4double val)   { m_resoMax = val;  }
  void SetEnergyRef(G4double val)   { m_eref = val;  }
  void SetSlope(G4double val)   { m_slope = val;  }


  void DescribeMyself(size_t );

protected:
  G4double m_reso;
  G4double m_resoMin;
  G4double m_resoMax;
  G4double m_eref;
  G4double m_slope;

private:
  GateDigi* m_outputDigi;
  GateEnergyResolutionMessenger *m_Messenger;
  GateDigiCollection*  m_OutputDigiCollection;
  GateSinglesDigitizer *m_digitizer;
  //G4bool m_IsFirstEntrance;


};

#endif








