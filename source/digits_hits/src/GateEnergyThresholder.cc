/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateEnergyThresholder.hh"

#include "G4UnitsTable.hh"

#include "GateEnergyThresholderMessenger.hh"
#include "GateTools.hh"

GateEnergyThresholder::GateEnergyThresholder(GatePulseProcessorChain* itsChain,
			       const G4String& itsName,
      	      	      	      	 G4double itsThreshold)
  : GateVPulseProcessor(itsChain,itsName),m_threshold(itsThreshold)
{
  m_messenger = new GateEnergyThresholderMessenger(this);
 //Asign the effective law to the default one. Just deposited energy
  //m_EffectiveEnergyLaw= new GateSolidAngleWeightedEnergyLaw(GetObjectName());
  m_effectiveEnergyLaw= new GateDepositedEnergyLaw(GetObjectName());

}




GateEnergyThresholder::~GateEnergyThresholder()
{
  delete m_messenger;
    delete m_effectiveEnergyLaw;
}



void GateEnergyThresholder::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  if (!inputPulse) {
    if (nVerboseLevel>1)
        G4cout << "[GateEnergyThresholder::ProcessOnePulse]: input pulse was null -> nothing to do\n\n";
    return;
  }
  if (inputPulse->GetEnergy()==0) {
    if (nVerboseLevel>1)
        G4cout << "[GateEnergyThresholder::ProcessOneHit]: energy is null for " << inputPulse << " -> pulse ignored\n\n";
    return;
  }

  if (  m_effectiveEnergyLaw->ComputeEffectiveEnergy(*inputPulse)>= m_threshold )
  {
    GatePulse* outputPulse = new GatePulse(*inputPulse);
    outputPulseList.push_back(outputPulse);
    if (nVerboseLevel>1)
      	G4cout << "Copied pulse to output:\n"
      	       << *outputPulse << Gateendl << Gateendl ;
  }
  else {
      if (nVerboseLevel>1)
      	G4cout << "Ignored pulse with energy below threshold:\n"
      	       << *inputPulse << Gateendl << Gateendl ;
  }
}



void GateEnergyThresholder::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Threshold: " << G4BestUnit(m_threshold,"Energy") << Gateendl;
}
