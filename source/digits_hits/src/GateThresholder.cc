/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateThresholder.hh"

#include "G4UnitsTable.hh"

#include "GateThresholderMessenger.hh"
#include "GateTools.hh"

GateThresholder::GateThresholder(GatePulseProcessorChain* itsChain,
			       const G4String& itsName,
      	      	      	      	 G4double itsThreshold)
  : GateVPulseProcessor(itsChain,itsName),
    m_threshold(itsThreshold)
{
  m_messenger = new GateThresholderMessenger(this);
}




GateThresholder::~GateThresholder()
{
  delete m_messenger;
}



void GateThresholder::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  if (!inputPulse) {
    if (nVerboseLevel>1)
      	G4cout << "[GateThresholder::ProcessOnePulse]: input pulse was null -> nothing to do\n\n";
    return;
  }
  if (inputPulse->GetEnergy()==0) {
    if (nVerboseLevel>1)
      	G4cout << "[GateThresholder::ProcessOneHit]: energy is null for " << inputPulse << " -> pulse ignored\n\n";
    return;
  }

  if ( inputPulse->GetEnergy() >= m_threshold )
  {
    GatePulse* outputPulse = new GatePulse(*inputPulse);
    outputPulseList.push_back(outputPulse);
    if (nVerboseLevel>1)
      	G4cout << "Copied pulse to output:" << G4endl
      	       << *outputPulse << G4endl << G4endl ;
  }
  else {
      if (nVerboseLevel>1)
      	G4cout << "Ignored pulse with energy below threshold:" << G4endl
      	       << *inputPulse << G4endl << G4endl ;
  }
}



void GateThresholder::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Threshold: " << G4BestUnit(m_threshold,"Energy") << G4endl;
}
