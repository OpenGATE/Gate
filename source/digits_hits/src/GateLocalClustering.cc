

#include "GateLocalClustering.hh"

#include "G4UnitsTable.hh"
#include "GateVolumeID.hh"
#include "GateLocalClusteringMessenger.hh"
#include "G4Electron.hh"
#include "GateConfiguration.h"
#include "GateObjectStore.hh"
#include "GateConstants.hh"
#include "GateTools.hh"

#include "GateVVolume.hh"
#include "G4VoxelLimits.hh"




GateLocalClustering::GateLocalClustering(GatePulseProcessorChain* itsChain,
                                                               const G4String& itsName)
    : GateVPulseProcessor(itsChain,itsName)
{


   m_messenger = new GateLocalClusteringMessenger(this);

    DescribeMyself(1);

   im=m_table.begin();


}

GateLocalClustering::~GateLocalClustering()
{
	delete m_messenger;
}


G4int GateLocalClustering::ChooseVolume(G4String val)
{

  GateObjectStore* m_store = GateObjectStore::GetInstance();


  if (m_store->FindCreator(val)!=0) {
      m_param.distance = -1;
     m_param.Rejectionflg= 0;

      m_table[val] = m_param;
      //there can be several that is why a map is used SetVolumeName(val);


      return 1;
  }
  else {
    G4cout << "Wrong Volume Name\n";
    return 0;
  }

}


void GateLocalClustering::ProcessOnePulse(const GatePulse* inputPulse,GatePulseList& outputPulseList)
{


    im=m_table.find(((inputPulse->GetVolumeID()).GetBottomCreator())->GetObjectName());
    GatePulse* outputPulse = new GatePulse(*inputPulse);

    if(im != m_table.end()){
        outputPulse->SetEnergyFin(-1);
        outputPulse->SetEnergyIniTrack(-1);
         //G4cout<<"pulso al que aplicar el cluster"<<G4endl;
                std::vector<double> dist;
                 std::vector<int> index4ClustSameVol;
                 if(index4Clusters.empty()){
                     outputPulseList.push_back(outputPulse);
                     index4Clusters.push_back(outputPulseList.size()-1);
                 }
                 else{
                     //for(unsigned int i=0; i<clustersGlobPositions.size();i++){
                     for(unsigned int i=0; i<index4Clusters.size();i++){
                         if(outputPulse->GetVolumeID() == (outputPulseList.at(index4Clusters.at(i)))->GetVolumeID()){
                         //dist.push_back(getDistance(outputPulse->GetGlobalPos(),clustersGlobPositions.at(i)));
                           dist.push_back(getDistance(outputPulse->GetGlobalPos(),outputPulseList.at(index4Clusters.at(i))->GetGlobalPos()));
                           index4ClustSameVol.push_back(index4Clusters.at(i));
                         }
                     }

                      if(dist.size()>0){


                          std::vector<double>::iterator itmin = std::min_element(std::begin(dist), std::end(dist));
                          unsigned int posMin=std::distance(std::begin(dist), itmin);
                          if(dist.at(posMin)< (*im).second.distance){
                              //Sum the hit to the cluster. sum of energies, position (global and local) weighted in energies, min time
                              outputPulseList.at( index4ClustSameVol.at(posMin))->CentroidMerge(outputPulse);
                              delete outputPulse;

                          }
                          else{
                              outputPulseList.push_back(outputPulse);
                              index4Clusters.push_back(outputPulseList.size()-1);
                          }
                      }
                      else{
                          outputPulseList.push_back(outputPulse);
                          index4Clusters.push_back(outputPulseList.size()-1);
                      }


                 }
    }
    else{
           //G4cout<<"pulso al que  no aplicar el cluster"<<G4endl;
         outputPulseList.push_back(outputPulse);
    }


}


GatePulseList* GateLocalClustering::ProcessPulseList(const GatePulseList* inputPulseList)
{
  if (!inputPulseList)
    return 0;

  size_t n_pulses = inputPulseList->size();
  if (nVerboseLevel==1)
        G4cout << "[" << GetObjectName() << "::ProcessPulseList]: processing input list with " << n_pulses << " entries\n";
  if (!n_pulses)
    return 0;

  GatePulseList* outputPulseList = new GatePulseList(GetObjectName());

   index4Clusters.clear();


  GatePulseConstIterator iter;
  for (iter = inputPulseList->begin() ; iter != inputPulseList->end() ; ++iter){
      ProcessOnePulse( *iter, *outputPulseList);
  }
 // G4cout<<" after processing all the pulses of an event "<<index4Clusters.size()<<G4endl;
  // Maybe check distance between centers
  //things bad written for rejection (repeated code)
   //check in the maps if there is any flagRejectionset to 1
  bool flagM=0;
   std::vector<int> pos2Erase;
  if(index4Clusters.size()>1){
       if(m_table.size()==1){
            im=m_table.begin();
           if((*im).second.Rejectionflg==1){
               // One volume name but maybe different volumeID if repeaters are used
               for(unsigned int i=0; i<index4Clusters.size()-1; i++){
                          for(unsigned int k=i+1;k<index4Clusters.size(); k++){
                                   if(outputPulseList->at(index4Clusters.at(i))->GetVolumeID() == outputPulseList->at(index4Clusters.at(k))->GetVolumeID()){
                                       flagM=1;
                                       break;

                                  }
                            }
                           if(flagM==1)break;
              }

           }

       }
       else if (m_table.size()>1){
           //it can be a volume with clusterising and rejection and another without rejection
           //Volumes with different names
           //names for volumes with rejections ah'i mismo check
           for(im=m_table.begin(); im!=m_table.end(); ++im){
               if((*im).second.Rejectionflg==0){
                   std::vector<int> pos2Erase;
                   for(unsigned int i=0;i<index4Clusters.size(); i++){
                       if((*im).first==((outputPulseList->at(index4Clusters.at(i))->GetVolumeID()).GetBottomCreator())->GetObjectName()){
                           pos2Erase.push_back(i);
                       }
                   }
                  // G4cout<<" position to erase saved"<<G4endl;
                   for(unsigned int i=0; i<pos2Erase.size(); i++){
                       index4Clusters.erase(index4Clusters.begin()+pos2Erase.at(i)-i);
                   }
                  // G4cout<<" position erased"<<G4endl;
               }
           }

           if(index4Clusters.size()>1){

              // G4cout<<" tamao el inex4Clusters"<<G4endl;
               // One volume name but maybe different volumeID if repeaters are used
               for(unsigned int i=0; i<index4Clusters.size()-1; i++){
                   for(unsigned int k=i+1;k<index4Clusters.size(); k++){
                       if(outputPulseList->at(index4Clusters.at(i))->GetVolumeID() == outputPulseList->at(index4Clusters.at(k))->GetVolumeID()){
                           flagM=1;
                           break;

                       }
                   }
                   if(flagM==1)break;
               }
           }

       }

       if(flagM==1){
           while (outputPulseList->size())
           {
               delete outputPulseList->back();
               outputPulseList->erase(outputPulseList->end()-1);
           }
       }
  }





  //G4cout<<" outputList cluster number="<<outputPulseList->size()<<G4endl;


  if (nVerboseLevel==1) {
      G4cout << "[" << GetObjectName() << "::ProcessPulseList]: returning output pulse-list with " << outputPulseList->size() << " entries\n";
      for (iter = outputPulseList->begin() ; iter != outputPulseList->end() ; ++iter)
        G4cout << **iter << Gateendl;
      G4cout << Gateendl;
  }

  return outputPulseList;
}






void GateLocalClustering::DescribeMyself(size_t  indent)
{
   for (im=m_table.begin(); im!=m_table.end(); im++)
        G4cout << GateTools::Indent(indent) << "Distance of  " << (*im).first << ":\n"
           << GateTools::Indent(indent+1) << G4BestUnit( (*im).second.distance,"Length")<<  Gateendl;
}



 double GateLocalClustering::getDistance(G4ThreeVector pos1, G4ThreeVector pos2 ){
    double dist=std::sqrt(std::pow(pos1.getX()-pos2.getX(),2)+std::pow(pos1.getY()-pos2.getY(),2)+std::pow(pos1.getZ()-pos2.getZ(),2));
    return dist;


 }


//this is standalone only because it repeats twice in processOnePulse()
inline void GateLocalClustering::PulsePushBack(const GatePulse* inputPulse, GatePulseList& outputPulseList)
{
	GatePulse* outputPulse = new GatePulse(*inputPulse);
    if (nVerboseLevel>1)
		G4cout << "Created new pulse for volume " << inputPulse->GetVolumeID() << ".\n"
		<< "Resulting pulse is: \n"
		<< *outputPulse << Gateendl << Gateendl ;
	outputPulseList.push_back(outputPulse);
}


