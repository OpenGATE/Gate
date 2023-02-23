/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

// OK GND 2022

/*! \class  GateEfficiency
    \brief  GateEfficiency apples the efficiency as a function of energy.
 	 It uses GateVDistribution class to define either analytic
 	 function or list of values read from a file.

    - GateEfficiency


   	Last modification: olga.kochebina@cea.fr
	Previous authors are unknown

    \sa GateEfficiency, GateEfficiencyMessenger
*/

#ifndef GateEfficiency_h
#define GateEfficiency_h 1

#include "GateVDigitizerModule.hh"
#include "GateDigi.hh"
#include "GateClockDependent.hh"
#include "GateCrystalSD.hh"

#include "globals.hh"
#include "GateEfficiencyMessenger.hh"

#include "GateSinglesDigitizer.hh"

class GateVDistribution;

class GateEfficiency : public GateVDigitizerModule
{
public:
  
  GateEfficiency(GateSinglesDigitizer *digitizer, G4String name);
  ~GateEfficiency();
  
  void Digitize() override;

  inline void SetMode(G4String val) {m_mode=val; }
  inline G4String GetMode() {return m_mode; }

  void SetLevel(size_t i,G4bool val);

  inline void SetUniqueEfficiency(G4double val){m_uniqueEff = val; }
  inline G4double GetUniqueEfficiency(){return m_uniqueEff; }

  inline void SetEfficiency(GateVDistribution* dist) {m_efficiency_distr = dist;}
  inline GateVDistribution* GetEfficiency() const {return m_efficiency_distr;}

  void ComputeSizes();

  void DescribeMyself(size_t );

protected:
  G4String m_mode;
  G4double m_uniqueEff;
  std::vector<G4bool> m_enabled;    	  //!< is the level enabled
  GateVDistribution* m_efficiency_distr;    	   //!< efficiency table

private:
  GateDigi* m_outputDigi;

  GateEfficiencyMessenger *m_Messenger;

  GateDigiCollection*  m_OutputDigiCollection;

  GateSinglesDigitizer *m_digitizer;

  G4bool m_firstPass;


};

#endif








