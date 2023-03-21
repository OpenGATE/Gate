/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022


/*! \class  GateNoise
    \brief  GateNoise does some dummy things with input digi
    to create output digi

    - GateNoise - by name.surname@email.com

    \sa GateNoise, GateNoiseMessenger
*/

#ifndef GateNoise_h
#define GateNoise_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"

#include "GateNoiseMessenger.hh"
#include "GateSinglesDigitizer.hh"

class GateVDistribution;

class GateNoise : public GateVDigitizerModule
{
public:
  
  GateNoise(GateSinglesDigitizer *digitizer, G4String name);
  ~GateNoise();
  
  void Digitize() override;

  void SetEnergyDistribution(GateVDistribution* energyDistrib) {m_energyDistrib=energyDistrib;}
  GateVDistribution* GetEnergyDistribution() const {return m_energyDistrib;}
  void SetDeltaTDistribution(GateVDistribution* deltaTDistrib) {m_deltaTDistrib=deltaTDistrib;}
  GateVDistribution* GetDeltaTDistribution() const {return m_deltaTDistrib;}

  G4double ComputeStartTime(GateDigiCollection* IDC);
  G4double ComputeFinishTime(GateDigiCollection* IDC);


  void DescribeMyself(size_t );

protected:
  GateVDistribution* m_deltaTDistrib;  //! Delta arrival time distribution
  GateVDistribution* m_energyDistrib;  //! The energy distribution
  std::vector<GateDigi*> m_createdDigis;       //! Trans. pulse list
  GateNoiseMessenger *m_messenger;     //!< Messenger

  static const G4String& theTypeName;  //!< Default type-name for all efficiency
  G4double   	m_oldTime;                   //!< Time of last event

private:
  GateDigi* m_outputDigi;

  GateNoiseMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;


};

#endif








