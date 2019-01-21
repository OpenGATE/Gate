/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateConfiguration.h"
#include "GatePulseAdder.hh"

#include "G4UnitsTable.hh"

#include "GatePulseAdderMessenger.hh"



GatePulseAdder::GatePulseAdder(GatePulseProcessorChain* itsChain,
      	      	      	       const G4String& itsName)
  : GateVPulseProcessor(itsChain,itsName),
     m_positionPolicy(kenergyWeightedCentroid)
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
           if(m_positionPolicy==kTakeEnergyWin){
                (*iter)->MergePositionEnergyWin(inputPulse);




           }
           else{
               (*iter)->CentroidMerge( inputPulse );
           }


	if (nVerboseLevel>1)
	  G4cout << "Merged previous pulse for volume " << inputPulse->GetVolumeID()
		 << " with new pulse of energy " << G4BestUnit(inputPulse->GetEnergy(),"Energy") <<".\n"
		 << "Resulting pulse is: \n"
		 << **iter << Gateendl << Gateendl ;
	break;
      }

    if ( iter == outputPulseList.end() )
    {
      GatePulse* outputPulse = new GatePulse(*inputPulse);
      outputPulse->SetEnergyIniTrack(-1);
      outputPulse->SetEnergyFin(-1);
      if (nVerboseLevel>1)
	  G4cout << "Created new pulse for volume " << inputPulse->GetVolumeID() << ".\n"
		 << "Resulting pulse is: \n"
		 << *outputPulse << Gateendl << Gateendl ;
      outputPulseList.push_back(outputPulse);
    }
  }
}


//void GatePulseAdder::DescribeMyself(size_t indent)
void GatePulseAdder::DescribeMyself(size_t )
{
  ;
}

void GatePulseAdder::SetPositionPolicy(const G4String &policy){

    if (policy=="takeEnergyWinner")
        m_positionPolicy=kTakeEnergyWin;

    else {
        if (policy!="energyWeightedCentroid")
            G4cout<<"WARNING : policy not recognized, using default :energyWeightedCentroid\n";
       m_positionPolicy=kenergyWeightedCentroid;
    }
}
