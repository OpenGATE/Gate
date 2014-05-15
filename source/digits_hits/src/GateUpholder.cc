/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateUpholder.hh"

#include "G4UnitsTable.hh"

#include "GateUpholderMessenger.hh"
#include "GateTools.hh"

GateUpholder::GateUpholder(GatePulseProcessorChain* itsChain,
			       const G4String& itsName,
      	      	      	      	 G4double itsUphold)
  : GateVPulseProcessor(itsChain,itsName),
    m_uphold(itsUphold)
{
  m_messenger = new GateUpholderMessenger(this);
}




GateUpholder::~GateUpholder()
{
  delete m_messenger;
}



void GateUpholder::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  if (!inputPulse) {
    if (nVerboseLevel>1)
      	G4cout << "[GateUpholder::ProcessOnePulse]: input pulse was null -> nothing to do\n\n";
    return;
  }
  if (inputPulse->GetEnergy()==0) {
    if (nVerboseLevel>1)
      	G4cout << "[GateUpholder::ProcessOneHit]: energy is null for " << inputPulse << " -> pulse ignored\n\n";
    return;
  }

  if ( inputPulse->GetEnergy() <= m_uphold )
  {
    GatePulse* outputPulse = new GatePulse(*inputPulse);
    outputPulseList.push_back(outputPulse);
    if (nVerboseLevel>1)
      	G4cout << "Copied pulse to output:" << G4endl
      	       << *outputPulse << G4endl << G4endl ;
  }
  else {
      if (nVerboseLevel>1)
      	G4cout << "Ignored pulse with energy above uphold:" << G4endl
      	       << *inputPulse << G4endl << G4endl ;
  }
}



void GateUpholder::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Uphold: " << G4BestUnit(m_uphold,"Energy") << G4endl;
}
