/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateMultipleRejection

  Digitizer module for simulating a MultipleRejection

  Last modification (Adaptation to GND): August 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/

#include "GateMultipleRejection.hh"
#include "GateMultipleRejectionMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"

#include "GateTools.hh"
#include "GateVolumeID.hh"
#include "Randomize.hh"
#include "GateObjectStore.hh"
#include "GateConstants.hh"



GateMultipleRejection::GateMultipleRejection(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_MultipleRejection(0),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateMultipleRejectionMessenger(this);
}


GateMultipleRejection::~GateMultipleRejection()
{
  delete m_Messenger;

}


void GateMultipleRejection::Digitize()
{

  G4String digitizerName = m_digitizer->m_digitizerName;
  G4String outputCollName = m_digitizer-> GetOutputName();
  m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

  G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();



  GateDigiCollection* IDC = 0;
  IDC = (GateDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

  GateDigi* inputDigi;
  std::vector< GateDigi* >* OutputDigiCollectionVector = m_OutputDigiCollection->GetVector ();
  std::vector<GateDigi*>::iterator iter;

  if (IDC)
     {
	  G4int n_digi = IDC->entries();

	  //loop over input digits
	  for (G4int i=0;i<n_digi;i++)
	  {
		  inputDigi=(*IDC)[i];

		  if (nVerboseLevel==1) {
		 		  G4cout << "[" << GetObjectName() << "::OutputDigiCollectionVector]: processing input list with " << n_digi << " entries\n";
		 		  for (iter = OutputDigiCollectionVector->begin() ; iter != OutputDigiCollectionVector->end() ; ++iter)
		 		    G4cout << **iter << Gateendl;
		 		  G4cout << Gateendl;
		  }

		  multiplesIndex.erase(multiplesIndex.begin(), multiplesIndex.end());
		  if( OutputDigiCollectionVector->size()>1){
		      G4bool flagDeleteAll=false;
		      std::vector<int> posErase;
		      if(flagDeleteAll==true){
		          while (OutputDigiCollectionVector->size())
		         {
		             delete OutputDigiCollectionVector->back();
		             OutputDigiCollectionVector->erase(OutputDigiCollectionVector->end()-1);
		           }
		      }
		      else if (flagDeleteAll==false && posErase.size()>1){
		           std::sort (posErase.begin(), posErase.end());
		          for(unsigned int i=0; i<posErase.size(); i++){
		              // G4cout<<" position to delete="<<posErase.at(i)<<G4endl;
		              delete OutputDigiCollectionVector->at(posErase.at(i)-i);
		              OutputDigiCollectionVector->erase(OutputDigiCollectionVector->begin()+posErase.at(i)-i);
		          }
		      }
	     }


		  GateDigi* m_outputDigi = new GateDigi(*inputDigi);
		  if (((m_outputDigi->GetVolumeID()).GetBottomCreator())){
		      currentVolumeName=(m_outputDigi->GetVolumeID().GetBottomCreator())->GetObjectName();
		      if(m_param.multipleDef==kvolumeID){
		        currentNumber=m_outputDigi->GetVolumeID().GetBottomVolume()->GetCopyNo();
		        currentVolumeName=currentVolumeName+std::to_string(currentNumber);
		      }

		       if( multiplesRejPol.find(currentVolumeName)== multiplesRejPol.end()){
		           multiplesRejPol.insert(std::pair<G4String,G4bool> (currentVolumeName,m_param.rejectionAllPolicy));
		       }

		         //multiplesIndex.
		      if(multiplesIndex.find(currentVolumeName)!=multiplesIndex.end()){;
		           multiplesIndex[currentVolumeName].push_back(OutputDigiCollectionVector->size());
		      }
		      else{
		          multiplesIndex.insert(std::pair<G4String,std::vector<int>> (currentVolumeName,{(int)(OutputDigiCollectionVector->size())}));
		      }


		   }
		   m_OutputDigiCollection->insert(m_outputDigi);


        if (nVerboseLevel==1) {
			G4cout << "[GateMultipleRejection::Digitize]: returning output pulse-list with " << OutputDigiCollectionVector->size() << " entries\n";
			for (iter=OutputDigiCollectionVector->begin(); iter!= OutputDigiCollectionVector->end() ; ++iter)
				G4cout << **iter << Gateendl;
			G4cout << Gateendl;
		}
	  } //loop  over input digits
    } //IDC
  else
    {
  	  if (nVerboseLevel>1)
  	  	G4cout << "[GateMultipleRejection::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);


}

void GateMultipleRejection::DescribeMyself(size_t indent )
{
  ;
}
