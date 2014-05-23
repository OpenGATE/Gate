/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateCoincidenceMultiplesKiller.hh"
#include "GateCoincidenceMultiplesKillerMessenger.hh"
#include "GateTools.hh"





GateCoincidenceMultiplesKiller::GateCoincidenceMultiplesKiller(GateCoincidencePulseProcessorChain* itsChain,
			   const G4String& itsName)
  : GateVCoincidencePulseProcessor(itsChain,itsName)
{

  m_messenger = new GateCoincidenceMultiplesKillerMessenger(this);
}




GateCoincidenceMultiplesKiller::~GateCoincidenceMultiplesKiller()
{
  delete m_messenger;
}




GateCoincidencePulse* GateCoincidenceMultiplesKiller::ProcessPulse(GateCoincidencePulse* inputPulse,G4int )
{
  if (!inputPulse) {
      if (nVerboseLevel>1)
      	G4cout << "[GateCoincidenceMultiplesKiller::ProcessOnePulse]: input pulse was null -> nothing to do\n\n";
      return 0;
  }
  return (inputPulse->size()==2) ? new GateCoincidencePulse(*inputPulse):0;
}


void GateCoincidenceMultiplesKiller::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "MultiplesKiller "<< G4endl;
}
