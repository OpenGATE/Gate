/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCrystalBlurring.hh"

#include "GateCrystalBlurringMessenger.hh"
#include "GateTools.hh"

#include "Randomize.hh"

#include "G4UnitsTable.hh"


GateCrystalBlurring::GateCrystalBlurring(GatePulseProcessorChain* itsChain,
      	      	      	      	 const G4String& itsName,
				 G4double itsCrystalresolutionmin,
				 G4double itsCrystalresolutionmax,
			         G4double itsCrystalenergyRef,
				 G4double itsCrystalQE)
  : GateVPulseProcessor(itsChain,itsName),
    m_crystalresolutionmin(itsCrystalresolutionmin),
    m_crystalresolutionmax(itsCrystalresolutionmax),
    m_crystalQE(itsCrystalQE),
    m_crystaleref(itsCrystalenergyRef)
{
  m_messenger = new GateCrystalBlurringMessenger(this);
}




GateCrystalBlurring::~GateCrystalBlurring()
{
  delete m_messenger;
}



void GateCrystalBlurring::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{

         m_crystalresolution = m_crystalresolutionmin + (m_crystalresolutionmax - m_crystalresolutionmin)*G4UniformRand();
	 m_crystalcoeff = m_crystalresolution * sqrt(m_crystaleref);
	 GatePulse* outputPulse = new GatePulse(*inputPulse);
	 G4double m_QE = G4UniformRand();

	 if(m_QE <= m_crystalQE)
	 {outputPulse->SetEnergy(G4RandGauss::shoot(inputPulse->GetEnergy(),m_crystalcoeff*sqrt(inputPulse->GetEnergy())/2.35));
	 outputPulseList.push_back(outputPulse);}
	 else {outputPulse->SetEnergy(0);
	 outputPulseList.push_back(outputPulse);}

}

void GateCrystalBlurring::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Resolution of " << m_crystalresolution  << " for " <<  G4BestUnit(m_crystaleref,"Energy") << G4endl;
}
