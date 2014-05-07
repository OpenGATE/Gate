/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateReadout.hh"

#include "G4UnitsTable.hh"

#include "GateOutputVolumeID.hh"
#include "GateReadoutMessenger.hh"
#include "GateTools.hh"


GateReadout::GateReadout(GatePulseProcessorChain* itsChain,
      	      	      	 const G4String& itsName)
  : GateVPulseProcessor(itsChain,itsName),
    m_depth(1)
{
  m_messenger = new GateReadoutMessenger(this);
}




GateReadout::~GateReadout()
{
  delete m_messenger;
}



void GateReadout::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  const GateOutputVolumeID& blockID  = inputPulse->GetOutputVolumeID().Top(m_depth);

  if (blockID.IsInvalid()) {
    if (nVerboseLevel>1)
      	G4cout << "[GateReadout::ProcessOnePulse]: out-of-block hit for " << G4endl
	      <<  *inputPulse << G4endl
	      << " -> pulse ignored\n\n";
    return;
  }

  GatePulseIterator iter;
  for (iter = outputPulseList.begin() ; iter != outputPulseList.end() ; ++iter )
    if ( (*iter)->GetOutputVolumeID().Top(m_depth) == blockID )
      break;

  if ( iter != outputPulseList.end() )
  {
     G4double energySum = (*iter)->GetEnergy() + inputPulse->GetEnergy();
     if ( inputPulse->GetEnergy() > (*iter)->GetEnergy() )
      	**iter = *inputPulse;
     (*iter)->SetEnergy(energySum);
     if (nVerboseLevel>1)
      	  G4cout  << "Overwritten previous pulse for block " << blockID << " with new pulse with higer energy.\n"
      	          << "Resulting pulse is: " << G4endl
		  << **iter << G4endl << G4endl ;
  }
  else
  {
    GatePulse* outputPulse = new GatePulse(*inputPulse);
    if (nVerboseLevel>1)
      	G4cout << "Created new pulse for block " << blockID << ".\n"
      	       << "Resulting pulse is: " << G4endl
	       << *outputPulse << G4endl << G4endl ;
    outputPulseList.push_back(outputPulse);
  }
}





void GateReadout::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Readout at depth:      " << m_depth << G4endl;
}
