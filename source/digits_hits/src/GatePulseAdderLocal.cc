/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateConfiguration.h"
#include "GatePulseAdderLocal.hh"

#include "G4UnitsTable.hh"

#include "GatePulseAdderLocalMessenger.hh"

#include "GateObjectStore.hh"

GatePulseAdderLocal::GatePulseAdderLocal(GatePulseProcessorChain* itsChain,
      	      	      	       const G4String& itsName)
  : GateVPulseProcessor(itsChain,itsName)
{
  m_messenger = new GatePulseAdderLocalMessenger(this);
}



GatePulseAdderLocal::~GatePulseAdderLocal()
{
  delete m_messenger;
}


G4int GatePulseAdderLocal::chooseVolume(G4String val){
    GateObjectStore* m_store = GateObjectStore::GetInstance();
    if (m_store->FindCreator(val)!=0) {

      m_param.m_positionPolicy=kenergyWeightedCentroid;//default value
      m_table[val] = m_param;
      G4cout << "value inserted in chosen Volume "<<val<<G4endl;
      return 1;
    }
    else {
      G4cout << "Wrong Volume Name\n";
      return 0;
    }
}


void GatePulseAdderLocal::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
#ifdef GATE_USE_OPTICAL
    // ignore pulses based on optical photons. These can be added using the opticaladder
    if (!inputPulse->IsOptical())
#endif
    {

        im=m_table.find(((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName());
        //GatePulse* outputPulse = new GatePulse(*inputPulse);
        if(im != m_table.end()){

            GatePulseIterator iter;
            for (iter=outputPulseList.begin(); iter!= outputPulseList.end() ; ++iter)
                if ( (*iter)->GetVolumeID()   == inputPulse->GetVolumeID() )
                {
                    if((*im).second.m_positionPolicy==kTakeEnergyWin){
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
                << *outputPulse << Gateendl << Gateendl;

                outputPulseList.push_back(outputPulse);
            }
        }
        //If the pulse is not in the volume where the adder is applied we just added to the outpl
        else{
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


//void GatePulseAdderLocal::DescribeMyself(size_t indent)
void GatePulseAdderLocal::DescribeMyself(size_t )
{
  ;
}

void GatePulseAdderLocal::SetPositionPolicy(G4String & name,const G4String &policy){

    if (policy=="takeEnergyWinner")
        m_table[name].m_positionPolicy=kTakeEnergyWin;

    else {
        if (policy!="energyWeightedCentroid")
            G4cout<<"WARNING : policy not recognized, using default :energyWeightedCentroid\n";
        m_table[name].m_positionPolicy=kenergyWeightedCentroid;
    }
}
