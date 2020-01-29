

#include "GateLocalEnergyThresholder.hh"

#include "G4UnitsTable.hh"

#include "GateLocalEnergyThresholderMessenger.hh"
#include "GateTools.hh"

GateLocalEnergyThresholder::GateLocalEnergyThresholder(GatePulseProcessorChain* itsChain,
                   const G4String& itsName)
  : GateVPulseProcessor(itsChain,itsName)
{
  m_messenger = new GateLocalEnergyThresholderMessenger(this);

  flgTriggerAW=0;


  im=m_table.begin();

}




GateLocalEnergyThresholder::~GateLocalEnergyThresholder()
{
  delete m_messenger;

  for (im = m_table.begin() ; im != m_table.end() ; ++im){
       delete (*im).second.m_effectiveEnergyLaw;
   }

}


G4int GateLocalEnergyThresholder::ChooseVolume(G4String val)
{

  GateObjectStore* m_store = GateObjectStore::GetInstance();


  if (m_store->FindCreator(val)!=0) {
      m_param.m_effectiveEnergyLaw= new GateDepositedEnergyLaw(GetObjectName());
      m_param.m_threshold=-1;


      m_table[val] = m_param;

      return 1;
  }
  else {
    G4cout << "Wrong Volume Name\n";
    return 0;
  }

}






GatePulseList* GateLocalEnergyThresholder::ProcessPulseList(const GatePulseList* inputPulseList)
{
  if (!inputPulseList)
    return 0;

  size_t n_pulses = inputPulseList->size();
  if (nVerboseLevel==1)
        G4cout << "[" << GetObjectName() << "::ProcessPulseList]: processing input list with " << n_pulses << " entries\n";
  if (!n_pulses)
    return 0;

  GatePulseList* outputPulseList = new GatePulseList(GetObjectName());

 // PulsesIndexBelowSolidAngleTHR.erase(PulsesIndexBelowSolidAngleTHR.begin(), PulsesIndexBelowSolidAngleTHR.end());
   flgTriggerAW=0;
   //volumeID where the signal surprases the THR when applying solid Angle weightes energy
   while(vID.size()){
       vID.erase(vID.end()-1);
   }

  GatePulseConstIterator iter;
  for (iter = inputPulseList->begin() ; iter != inputPulseList->end() ; ++iter)
      ProcessOnePulse( *iter, *outputPulseList);


  if(!outputPulseList->empty()){

   /*  //checking if multiples in a volume  which
      double EffEnergyM=0;

      for (auto const& x : PulsesIndexBelowSolidAngleTHR ){

          if(x.second.size()>1){
               EffEnergyM=0;
              im=m_table.find(((outputPulseList->at(x.second[0])->GetVolumeID()).GetBottomCreator())->GetObjectName());
              double distX=0;
              double distY=0;
                if(im!=m_table.end()){
                    GateSolidAngleWeightedEnergyLaw* a=dynamic_cast<GateSolidAngleWeightedEnergyLaw*> ((*im).second.m_effectiveEnergyLaw);
                   distX=2*a->GetRectangleSzX();
                    distY=2*a->GetRectangleSzY();
                    G4cout<<"dist ="<< distY<<G4endl;
                }
                G4cout<<"size of vector="<<x.second.size() <<G4endl;
              for(unsigned int i1=0; i1<x.second.size(); i1++){
                  // outputPulseList.at(x.second[i1])
                  EffEnergyM=0;
                  for(unsigned int i2=i1+1; i2<x.second.size(); i2++){
                      //aprox I only consider two  singles that can be next to each other

                      if((outputPulseList->at(x.second[i1])->GetLocalPos().getX()-outputPulseList->at(x.second[i2])->GetLocalPos().getX())<distX){
                           if((outputPulseList->at(x.second[i1])->GetLocalPos().getY()-outputPulseList->at(x.second[i2])->GetLocalPos().getY())<distY){
                              EffEnergyM+=(*im).second.m_effectiveEnergyLaw->ComputeEffectiveEnergy(*outputPulseList->at(x.second[i1]))+(*im).second.m_effectiveEnergyLaw->ComputeEffectiveEnergy(*outputPulseList->at(x.second[i2]));
                              G4cout<<"NewEffectiveE="<<EffEnergyM<<G4endl;
                               G4cout<<"THR="<<(*im).second.m_threshold <<G4endl;
                              if(EffEnergyM>= (*im).second.m_threshold ){
                                  vID.push_back(outputPulseList->at(x.second[0])->GetVolumeID());
                                  flgTriggerAW=1;
                              }
                              break;
                           }
                      }


                  }
                  if(EffEnergyM>0)break;
              }
          }

      }*/




      GatePulseIterator iter;
      for (iter=outputPulseList->begin(); iter!= outputPulseList->end() ; ++iter){
            im=m_table.find((((*iter)->GetVolumeID()).GetBottomCreator())->GetObjectName());
          if(im!=m_table.end()){

              G4String lawP=(*im).second.m_effectiveEnergyLaw->GetObjectName();
              std::size_t pos = lawP.rfind("/");
              lawP=lawP.substr(pos+1);

              if(lawP=="solidAngleWeighted" ){
                  if(flgTriggerAW==0){
                      //Reject the pulse and delete
                      //I am deleting the pulses that I stored in the volume in case any of the other pulses in the same event was above the thr for solidAngleEnergyWeighted

                      delete (*iter);
                      outputPulseList->erase(iter);
                      --iter;
                  }
                  else if(flgTriggerAW==1){
                      // delete the pulse for with solida angle law was set if any of the opulses in that volume was above the thr
                      std::vector<GateVolumeID>::iterator  itV;
                      itV= find (vID.begin(), vID.end(), (*iter)->GetVolumeID());
                      if(itV==vID.end()){
                          delete (*iter);
                          outputPulseList->erase(iter);
                          --iter;
                      }

                  }
              }

          }
          //Option to rject the whole event
          //if(){}

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


void GateLocalEnergyThresholder::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{
  if (!inputPulse) {
    if (nVerboseLevel>1)
        G4cout << "[GateLocalEnergyThresholder::ProcessOnePulse]: input pulse was null -> nothing to do\n\n";
    return;
  }
  if (inputPulse->GetEnergy()==0) {
    if (nVerboseLevel>1)
        G4cout << "[GateLocalEnergyThresholder::ProcessOneHit]: energy is null for " << inputPulse << " -> pulse ignored\n\n";
    return;
  }

  //Only are processed those in the selected volume
   im=m_table.find(((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName());
  GatePulse* outputPulse = new GatePulse(*inputPulse);

  if(im != m_table.end()){

      G4String lawP=(*im).second.m_effectiveEnergyLaw->GetObjectName();
      std::size_t pos = lawP.rfind("/");
      lawP=lawP.substr(pos+1);
      //G4cout<<lawP<<G4endl;
      //G4cout << "eventID"<<inputPulse->GetEventID()<<"  effectEnergy="<<m_effectiveEnergyLaw->ComputeEffectiveEnergy(*outputPulse)<<G4endl;
      if ((*im).second.m_effectiveEnergyLaw->ComputeEffectiveEnergy(*outputPulse)>= (*im).second.m_threshold ) {

          outputPulseList.push_back(outputPulse);
          if (nVerboseLevel>1)
              G4cout << "Copied pulse to output:\n"
                     << *outputPulse << Gateendl << Gateendl ;



          if(lawP=="solidAngleWeighted"){
             vID.push_back(outputPulse->GetVolumeID());
             flgTriggerAW=1;
          }

      }
      else {
          if (nVerboseLevel>1)
              G4cout << "Ignored pulse with energy below threshold except for solidAngleWeighted:\n"
                     << *outputPulse << Gateendl << Gateendl ;
          if( lawP=="solidAngleWeighted"){


               /*G4String currentVolume=(outputPulse->GetVolumeID().GetBottomCreator())->GetObjectName()+std::to_string(outputPulse->GetVolumeID().GetBottomVolume()->GetCopyNo());

               if(PulsesIndexBelowSolidAngleTHR.find(currentVolume)!=PulsesIndexBelowSolidAngleTHR.end()){;
                    PulsesIndexBelowSolidAngleTHR[currentVolume].push_back(outputPulseList.size());
               }
               else{
                   PulsesIndexBelowSolidAngleTHR.insert(std::pair<G4String,std::vector<int>> (currentVolume,{(int)(outputPulseList.size())}));
               }*/


              outputPulseList.push_back(outputPulse);
          }
          else{
              delete outputPulse;
          }

      }
  }
  else{
      outputPulseList.push_back(outputPulse);
  }
}



void GateLocalEnergyThresholder::DescribeMyself(size_t indent)
{


    for (im=m_table.begin(); im!=m_table.end(); im++)
        G4cout << GateTools::Indent(indent) << "Threshold " << (*im).first << ":\n"
           << GateTools::Indent(indent+1) << (*im).second.m_threshold << "  effective threshold  for law "
           <<(*im).second.m_effectiveEnergyLaw->GetObjectName()<<  Gateendl;
}
