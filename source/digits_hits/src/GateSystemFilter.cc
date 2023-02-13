/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#include "GateSystemFilter.hh"
#include "GateDigitizer.hh"
#include "GateVSystem.hh"
#include "GateSystemFilterMessenger.hh"

GateSystemFilter::GateSystemFilter(GatePulseProcessorChain* itsChain,
      	      	      	       const G4String& itsName)
   : GateVPulseProcessor(itsChain,itsName), m_systemName("")
{
  m_messenger = new GateSystemFilterMessenger(this);
}


GateSystemFilter::~GateSystemFilter()
{
  delete m_messenger;
}

//=============================================================================
//=============================================================================
GatePulseList* GateSystemFilter::ProcessPulseList(const GatePulseList* inputPulseList)
{
   GatePulseList* outputPulseList = new GatePulseList(GetObjectName());

   GatePulseConstIterator iter;
   for (iter = inputPulseList->begin() ; iter != inputPulseList->end() ; ++iter){

      if (nVerboseLevel>1)
      {
      G4cout<<"[GateSystemFilter::ProcessPulseList]\n This pulse is from "<<m_systemName<<"' system', \n"<<"Its volumeID is : "
            <<(*iter)->GetVolumeID()<<",\n and its its outputVoulumeID is "<<(*iter)->GetOutputVolumeID()<< Gateendl;
      }

      G4String pulseSystemName = (*iter)->GetVolumeID().GetVolume(1)->GetName();
      size_t n = pulseSystemName.size();
      pulseSystemName.erase(n-5,5);

      if (pulseSystemName.compare(m_systemName) == 0)
      {
	 GatePulse* outputPulse = new GatePulse(*iter);
	 outputPulseList->push_back(outputPulse);
      }
   }

   return outputPulseList;
}

//=============================================================================
//=============================================================================
void GateSystemFilter::SetSystemToItsChain()
{
   GateDigitizer* digitizer = GateDigitizer::GetInstance();
   std::vector<GateCoincidenceSorterOld*> CoincidenceSorterList = digitizer->GetCoinSorterList();
   GateSystemList* systemList = digitizer->GetSystemList();

   GateSystemConstIterator iter;
   for(iter=systemList->begin(); iter!=systemList->end(); iter++)
   {
      if(m_systemName.compare((*iter)->GetOwnName()) == 0)
      {
         this->GetChain()->SetSystem(*iter);

         if(this->GetChain()->GetOutputName() == "Singles")
            for (std::vector<GateCoincidenceSorterOld*>::iterator itr=CoincidenceSorterList.begin(); itr!=CoincidenceSorterList.end(); ++itr)
         {
            if((*itr)->GetOutputName() == "Coincidences")
            {
               (*itr)->SetSystem(*iter);
               break;
            }
         }
         break;
      }
   }
}

//=============================================================================
//=============================================================================
void GateSystemFilter::ProcessOnePulse(const GatePulse* /*inputPulse*/,
                                       GatePulseList& /*outputPulseList*/)
{
  ;
}

//=============================================================================
//=============================================================================
void GateSystemFilter::DescribeMyself(size_t )
{
  ;
}
