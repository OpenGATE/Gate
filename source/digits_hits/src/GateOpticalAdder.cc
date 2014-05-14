/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"

#ifdef GATE_USE_OPTICAL

#include "GateOpticalAdder.hh"
#include "G4UnitsTable.hh"
#include "GateOpticalAdderMessenger.hh"


GateOpticalAdder::GateOpticalAdder(GatePulseProcessorChain* itsChain, const G4String& itsName)  :
  GateVPulseProcessor(itsChain,itsName)
{ m_messenger = new GateOpticalAdderMessenger(this);}

GateOpticalAdder::~GateOpticalAdder()
{ delete m_messenger;}

void GateOpticalAdder::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  if (inputPulse->IsOptical())
  {

    GatePulseIterator iter;
    for (iter=outputPulseList.begin(); iter!= outputPulseList.end() ; ++iter)
      if ( (*iter)->GetVolumeID()   == inputPulse->GetVolumeID() )
      {
//	G4double energy = (*iter)->GetEnergy();

	(*iter)->CentroidMerge( inputPulse );
//	energy += 1;
//	(*iter)->SetEnergy(energy);

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
//      outputPulse->SetEnergy(1);
      if (nVerboseLevel>1)
	  G4cout << "Created new pulse for volume " << inputPulse->GetVolumeID() << ".\n"
		 << "Resulting pulse is: " << G4endl
		 << *outputPulse << G4endl << G4endl ;
      outputPulseList.push_back(outputPulse);
    }
  }
}

void GateOpticalAdder::DescribeMyself(size_t )
{}

#endif
