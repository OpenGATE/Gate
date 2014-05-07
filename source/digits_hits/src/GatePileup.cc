/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GatePileup.hh"

#include "G4UnitsTable.hh"

#include "GateOutputVolumeID.hh"
#include "GatePileupMessenger.hh"
#include "GateTools.hh"


GatePileup::GatePileup(GatePulseProcessorChain* itsChain,
      	      	      	 const G4String& itsName)
  : GateVPulseProcessor(itsChain,itsName),
    m_depth(1),
    m_pileup(0),
    m_waiting("")
{
  m_messenger = new GatePileupMessenger(this);
}




GatePileup::~GatePileup()
{
  delete m_messenger;
}


GatePulseList* GatePileup::ProcessPulseList(const GatePulseList* inputPulseList)
{
  G4double minTime = inputPulseList->ComputeStartTime();
  GatePulseList* ans = new GatePulseList(GetObjectName());
  std::vector<GatePulseIterator> toDel;
  GatePulseIterator iter;
  for (iter = m_waiting.begin() ; iter != m_waiting.end() ; ++iter ){
    if ( (*iter)->GetTime()+m_pileup<minTime) {
    	ans->push_back( (*iter) );
	toDel.push_back(iter);
    }
  }
  for (int i= (int)toDel.size()-1;i>=0;i--){
    m_waiting.erase( toDel[i] );
  }

  GatePulseConstIterator itr;
  for (itr = inputPulseList->begin() ; itr != inputPulseList->end() ; ++itr)
      	ProcessOnePulse( *itr, m_waiting);
  return ans;
}


void GatePileup::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  const GateOutputVolumeID& blockID  = inputPulse->GetOutputVolumeID().Top(m_depth);

  if (blockID.IsInvalid()) {
    if (nVerboseLevel>1)
      	G4cout << "[GatePileup::ProcessOnePulse]: out-of-block hit for " << G4endl
	      <<  *inputPulse << G4endl
	      << " -> pulse ignored\n\n";
    return;
  }

  GatePulseIterator iter;
  for (iter = outputPulseList.begin() ; iter != outputPulseList.end() ; ++iter )
    if ( ((*iter)->GetOutputVolumeID().Top(m_depth) == blockID )
         &&  (std::abs((*iter)->GetTime()-inputPulse->GetTime())<m_pileup) )
      break;

  if ( iter != outputPulseList.end() ){
     G4double energySum = (*iter)->GetEnergy() + inputPulse->GetEnergy();
     if ( inputPulse->GetEnergy() > (*iter)->GetEnergy() ){
     	G4double time = std::max( (*iter)->GetTime() ,inputPulse->GetTime());
      	**iter = *inputPulse;
	(*iter)->SetTime(time);
     }
     (*iter)->SetEnergy(energySum);
     if (nVerboseLevel>1)
      	  G4cout  << "Overwritten previous pulse for block " << blockID << " with new pulse with higer energy.\n"
      	          << "Resulting pulse is: " << G4endl
		  << **iter << G4endl << G4endl ;
  } else {
    GatePulse* outputPulse = new GatePulse(*inputPulse);
    if (nVerboseLevel>1)
      	G4cout << "Created new pulse for block " << blockID << ".\n"
      	       << "Resulting pulse is: " << G4endl
	       << *outputPulse << G4endl << G4endl ;
    outputPulseList.push_back(outputPulse);
  }
}





void GatePileup::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Pileup at depth:      " << m_depth << G4endl;
}
