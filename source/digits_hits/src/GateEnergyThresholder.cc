/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateEnergyThresholder.hh"

#include "G4UnitsTable.hh"

#include "GateEnergyThresholderMessenger.hh"
#include "GateTools.hh"

GateEnergyThresholder::GateEnergyThresholder(GatePulseProcessorChain* itsChain,
			       const G4String& itsName,
      	      	      	      	 G4double itsThreshold)
  : GateVPulseProcessor(itsChain,itsName),m_threshold(itsThreshold)
{
  m_messenger = new GateEnergyThresholderMessenger(this);
 //Asign the effective law to the default one. Just deposited energy
  //m_EffectiveEnergyLaw= new GateSolidAngleWeightedEnergyLaw(GetObjectName());
  m_effectiveEnergyLaw= new GateDepositedEnergyLaw(GetObjectName());
  flgTriggerAW=0;
}




GateEnergyThresholder::~GateEnergyThresholder()
{
  delete m_messenger;
  delete m_effectiveEnergyLaw;
}




GatePulseList* GateEnergyThresholder::ProcessPulseList(const GatePulseList* inputPulseList)
{

  if (!inputPulseList)
    return 0;

  size_t n_pulses = inputPulseList->size();
  if (nVerboseLevel==1)
        G4cout << "[" << GetObjectName() << "::ProcessPulseList]: processing input list with " << n_pulses << " entries\n";
  if (!n_pulses)
    return 0;

  GatePulseList* outputPulseList = new GatePulseList(GetObjectName());

   flgTriggerAW=0;
   while(vID.size()){
       vID.erase(vID.end()-1);
   }

  GatePulseConstIterator iter;
  for (iter = inputPulseList->begin() ; iter != inputPulseList->end() ; ++iter)
      ProcessOnePulse( *iter, *outputPulseList);


  if(m_effectiveEnergyLaw->GetObjectName()=="digitizer/layers/energyThresholder/solidAngleWeighted" ){
      if(flgTriggerAW==0){
          //REJECT THE EVENT
          if(!outputPulseList->empty()){

              while ( outputPulseList->size() ) {
                delete outputPulseList->back();
                outputPulseList->erase(outputPulseList->end()-1);
              }

          }
     }
     else if(flgTriggerAW==1){
          GatePulseIterator iter;
          std::vector<GateVolumeID>::iterator  it;
          for (iter=outputPulseList->begin(); iter!= outputPulseList->end() ; ++iter){
               it= find (vID.begin(), vID.end(), (*iter)->GetVolumeID());
               if (it == vID.end()){
                  // G4cout<<"tiro el de la position "<<std::distance(outputPulseList->begin(),iter)<<G4endl;
               //reject pulse
                  delete (*iter);
                 outputPulseList->erase(iter);
                 --iter;
               }
          }
     }
  }

  if (nVerboseLevel==1) {
      G4cout << "[" << GetObjectName() << "::ProcessPulseList]: returning output pulse-list with " << outputPulseList->size() << " entries\n";
      for (iter = outputPulseList->begin() ; iter != outputPulseList->end() ; ++iter)
        G4cout << **iter << Gateendl;
      G4cout << Gateendl;
  }

  return outputPulseList;
}


void GateEnergyThresholder::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  if (!inputPulse) {
    if (nVerboseLevel>1)
        G4cout << "[GateEnergyThresholder::ProcessOnePulse]: input pulse was null -> nothing to do\n\n";
    return;
  }
  if (inputPulse->GetEnergy()==0) {
    if (nVerboseLevel>1)
        G4cout << "[GateEnergyThresholder::ProcessOneHit]: energy is null for " << inputPulse << " -> pulse ignored\n\n";
    return;
  }




  GatePulse* outputPulse = new GatePulse(*inputPulse);
  //G4cout << "eventID"<<inputPulse->GetEventID()<<"  effectEnergy="<<m_effectiveEnergyLaw->ComputeEffectiveEnergy(*outputPulse)<<G4endl;
  if ( m_effectiveEnergyLaw->ComputeEffectiveEnergy(*outputPulse)>= m_threshold ) {

      outputPulseList.push_back(outputPulse);
      if (nVerboseLevel>1)
          G4cout << "Copied pulse to output:\n"
                 << *outputPulse << Gateendl << Gateendl ;
       if( m_effectiveEnergyLaw->GetObjectName()=="digitizer/layers/energyThresholder/solidAngleWeighted"){
            vID.push_back(outputPulse->GetVolumeID());
            flgTriggerAW=1;
       }


  }
  else {
      if (nVerboseLevel>1)
          G4cout << "Ignored pulse with energy below threshold except for solidAngleWeighted:\n"
                 << *outputPulse << Gateendl << Gateendl ;
      if( m_effectiveEnergyLaw->GetObjectName()=="digitizer/layers/energyThresholder/solidAngleWeighted"){

          outputPulseList.push_back(outputPulse);
      }
      else{
          delete outputPulse;
      }

  }
}



void GateEnergyThresholder::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "Threshold: " << G4BestUnit(m_threshold,"Energy") << Gateendl;
}
