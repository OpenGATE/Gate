/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"
#include "GatePulseAdderGPUSpect.hh"

#include "G4UnitsTable.hh"
#include "GatePulseAdderGPUSpectMessenger.hh"


GatePulseAdderGPUSpect::GatePulseAdderGPUSpect(GatePulseProcessorChain* itsChain,
      	      	      	       const G4String& itsName)
  : GateVPulseProcessor(itsChain,itsName)
{
  m_messenger = new GatePulseAdderGPUSpectMessenger(this);
}




GatePulseAdderGPUSpect::~GatePulseAdderGPUSpect()
{
  delete m_messenger;
}



void GatePulseAdderGPUSpect::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
    GatePulseIterator iter;
    for (iter=outputPulseList.begin(); iter!= outputPulseList.end() ; ++iter)
		{

      if ( ( (*iter)->GetVolumeID() == inputPulse->GetVolumeID() ) &&
	( ::fabs( (*iter)->GetTime() - inputPulse->GetTime() ) < 0.001 ) )
      {
	(*iter)->CentroidMerge( inputPulse );
	if (nVerboseLevel>1)
	  G4cout << "Merged previous pulse for volume " << inputPulse->GetVolumeID()
		 << " with new pulse of energy " << G4BestUnit(inputPulse->GetEnergy(),"Energy") <<".\n"
		 << "Resulting pulse is: " << G4endl
		 << **iter << G4endl << G4endl ;
	break;
      }
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


void GatePulseAdderGPUSpect::DescribeMyself(size_t )
{
  ;
}
