/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateClustering
  
  Last modification (Adaptation to GND): June 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/


#include "GateClustering.hh"
#include "GateClusteringMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4UnitsTable.hh"

#include "GateVolumeID.hh"
#include "GateConfiguration.h"
#include "GateObjectStore.hh"
#include "GateConstants.hh"
#include "GateTools.hh"

#include "GateVVolume.hh"
#include "G4VoxelLimits.hh"


GateClustering::GateClustering(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_GateClustering(0),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateClusteringMessenger(this);
}

double GateClustering::getDistance(G4ThreeVector pos1, G4ThreeVector pos2 ){
  double dist=std::sqrt(std::pow(pos1.getX()-pos2.getX(),2)+std::pow(pos1.getY()-pos2.getY(),2)+std::pow(pos1.getZ()-pos2.getZ(),2));
  return dist;
}

GateClustering::~GateClustering()
{
  delete m_Messenger;
}


void GateClustering::Digitize()
{
  G4String digitizerName = m_digitizer->m_digitizerName;
  G4String outputCollName = m_digitizer-> GetOutputName();

  m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

  G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();



  GateDigiCollection* IDC = 0;
  IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

  GateDigi* inputDigi;

  if (IDC)
     {
	  G4int n_digi = IDC->entries();

	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];
		  std::vector< GateDigi* >* OutputDigiCollectionVector = m_OutputDigiCollection->GetVector ();
		  std::vector<GateDigi*>::iterator iter;
		  if (nVerboseLevel==1) {
		      G4cout << "[" << GetObjectName() << "::ProcessPulseList]: processing input list with " << n_digi << " entries\n";
		      for (iter = OutputDigiCollectionVector->begin() ; iter != OutputDigiCollectionVector->end() ; ++iter)
		      	G4cout << **iter << Gateendl;
		      G4cout << Gateendl;
		  }

		      //Check if there are clusters whose centers are closer than the accepted distance to each other to merge them
		     // checkClusterCentersDistance(*OutputDigiCollectionVector);
		     
		      if(m_flgMRejection==1){
		      //If Multiple clusters in the same volumeID the whole event is rejected
		      
		          if(int(OutputDigiCollectionVector->size()>1)){
		              bool flagM=0;
		              for(unsigned int i=0; i<int(OutputDigiCollectionVector->size()-1); i++){
		                  for(unsigned int k=i+1;k<int(OutputDigiCollectionVector->size()); k++){
		                      if(OutputDigiCollectionVector->at(i)->GetVolumeID() == OutputDigiCollectionVector->at(k)->GetVolumeID()){
		                          flagM=1;
		                          break;
		                      }
		                  }
		                  if(flagM==1)break;
		              }
		              if(flagM==1){
		                  while (int(OutputDigiCollectionVector->size()))
		                  {
		                      delete OutputDigiCollectionVector->back();
		                      OutputDigiCollectionVector->erase(OutputDigiCollectionVector->end()-1);
		                  }
		              }
		          }

		      }
		      if (nVerboseLevel==1) {
		          G4cout << "[" << GetObjectName() << "::ProcessPulseList]: returning output pulse-list with " << int(OutputDigiCollectionVector->size()) << " entries\n";
		          for (iter = OutputDigiCollectionVector->begin() ; iter != OutputDigiCollectionVector->end() ; ++iter)
		              G4cout << **iter << Gateendl;
		          G4cout << Gateendl;
		      }

		      GateDigi* m_outputDigi = new GateDigi(*inputDigi);
		          m_outputDigi->SetEnergyFin(-1);
		          m_outputDigi->SetEnergyIniTrack(-1);
		          std::vector<double> dist;
		          std::vector<int> index4ClustSameVol;

		          if(OutputDigiCollectionVector->empty()){
		        	  m_OutputDigiCollection->insert(m_outputDigi);

		          }
		          else{
		              //store the distance to the different clusters in the m_outputDigi
		              for(unsigned int i=0; i<int(OutputDigiCollectionVector->size());i++){
		           
		                  if(m_outputDigi->GetVolumeID() == (OutputDigiCollectionVector->at(i))->GetVolumeID() && m_outputDigi->GetEventID()==(OutputDigiCollectionVector->at(i))->GetEventID() ){
		                      //Since I using only same volumeID pulses I could also use local position
		                      dist.push_back(getDistance(m_outputDigi->GetGlobalPos(),OutputDigiCollectionVector->at(i)->GetGlobalPos()));
		                      index4ClustSameVol.push_back(i);
		                  }
		              }

		              if(dist.size()>0){
		                  std::vector<double>::iterator itmin = std::min_element(std::begin(dist), std::end(dist));
		                  unsigned int posMin=std::distance(std::begin(dist), itmin);
		                  if(dist.at(posMin)< m_acceptedDistance){
		                      //Sum the pulse to the cluster. sum of energies, position (global and local) weighted in energies, min time
		               	      CentroidMerge(inputDigi,m_outputDigi);
		                      delete m_outputDigi;
		                  }
		                  else{
		                	  m_OutputDigiCollection->insert(m_outputDigi);
		                  }
		              }
		              else{
		            	  m_OutputDigiCollection->insert(m_outputDigi);
		              }


		          }


		if (nVerboseLevel==1) {
			G4cout << "[GateDummyDigitizerModule::Digitize]: returning output pulse-list with " << int(OutputDigiCollectionVector->size()) << " entries\n";
			for (iter=OutputDigiCollectionVector->begin(); iter!= OutputDigiCollectionVector->end() ; ++iter)
				G4cout << **iter << Gateendl;
			G4cout << Gateendl;
		}
	  } //loop  over input digits
    }
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateDummyDigitizerModule::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);

}

void GateClustering::DescribeMyself(size_t indent )
{
  ;
}
