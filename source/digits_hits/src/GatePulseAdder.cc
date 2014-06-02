/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"
#include "GatePulseAdder.hh"

#include "G4UnitsTable.hh"

#include "GatePulseAdderMessenger.hh"


GatePulseAdder::GatePulseAdder(GatePulseProcessorChain* itsChain,
      	      	      	       const G4String& itsName)
  : GateVPulseProcessor(itsChain,itsName)
{
  m_messenger = new GatePulseAdderMessenger(this);
}




GatePulseAdder::~GatePulseAdder()
{
  delete m_messenger;
}



void GatePulseAdder::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
#ifdef GATE_USE_OPTICAL
  // ignore pulses based on optical photons. These can be added using the opticaladder
  if (!inputPulse->IsOptical())
#endif
  {
    GatePulseIterator iter;
    for (iter=outputPulseList.begin(); iter!= outputPulseList.end() ; ++iter)
      if ( (*iter)->GetVolumeID()   == inputPulse->GetVolumeID() )
      {
	(*iter)->CentroidMerge( inputPulse );
	if (nVerboseLevel>1)
	  G4cout << "Merged previous pulse for volume " << inputPulse->GetVolumeID()
		 << " with new pulse of energy " << G4BestUnit(inputPulse->GetEnergy(),"Energy") <<".\n"
		 << "Resulting pulse is: " << G4endl
		 << **iter << G4endl << G4endl ;
	break;
      }

    if ( iter == outputPulseList.end() )
    {
      GatePulse* outputPulse = new GatePulse(*inputPulse);
      if (nVerboseLevel>1)
	  G4cout << "Created new pulse for volume " << inputPulse->GetVolumeID() << ".\n"
		 << "Resulting pulse is: " << G4endl
		 << *outputPulse << G4endl << G4endl ;
      outputPulseList.push_back(outputPulse);
    }
  }
}


//void GatePulseAdder::DescribeMyself(size_t indent)
void GatePulseAdder::DescribeMyself(size_t )
{
  ;
}
