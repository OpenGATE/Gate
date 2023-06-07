/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateCoinDigitizerInitializationModule
  This class is a specific Coincidence Digitizers class that are called before running all users
  Coincidence Digitizers and Coincidence Digitizers modules.
  It takes as input two Coincidence Digitizers collections and create a new coin digi collection
  Names for this input and output coin digi collections are defined with CoincidenceDigitizerMessenger
  Digitize() of this class is called by default by DigitizerMng at the beginning of RunCoincidenceDigitizers()


	05/2023 Olga.Kochebina@cea.fr
*/


#include "GateCoinDigitizerInitializationModule.hh"
#include "GateCoincidenceDigi.hh"
#include "GateCrystalSD.hh"

#include "GateHit.hh"


#include "GateOutputMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"




GateCoinDigitizerInitializationModule::GateCoinDigitizerInitializationModule(GateCoincidenceDigitizer *coinDigitizer)
  :GateVDigitizerModule("CoinDigiInit","digitizerMgr/CoincidenceDigitizer/"+coinDigitizer->m_digitizerName),
   m_FirstEvent(true),
   m_CDCIDs(),
   m_outputDigi(0),
   m_outputDigiCollection(0),
   m_inputNames(),
   m_coinDigitizer(coinDigitizer),
   test_times()
{

	G4String colName = coinDigitizer->GetOutputName();
	collectionName.push_back(colName);

	GateOutputMgr::GetInstance()->RegisterNewCoincidenceDigiCollection(coinDigitizer->GetOutputName()+"_CoinDigiInit",false);

}


GateCoinDigitizerInitializationModule::~GateCoinDigitizerInitializationModule()
{
	delete  m_coinDigitizer;
}


void GateCoinDigitizerInitializationModule::Digitize()
{

	m_outputDigiCollection = new GateCoincidenceDigiCollection (GetName(),  m_coinDigitizer->GetOutputName() ); // to create the Digi Collection
	m_inputDigiCollections.clear();
	test_times.clear();

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();


	std::vector< GateCoincidenceDigi* >* outputDigiCollectionVector = m_outputDigiCollection->GetVector ();
	std::vector<GateCoincidenceDigi*>::iterator iter;


	if (m_FirstEvent)
	{
		m_inputNames=m_coinDigitizer->GetInputNames();
		// loop over input collections to fill collections IDs
		for(G4int i=0; i<(G4int)m_inputNames.size();i++)
		{
			G4String CDCname=m_coinDigitizer->GetName()+"Collection" ;
			G4int ID=DigiMan->GetDigiCollectionID(m_inputNames[i]);

			if (ID==-1) //not found by G4DigiManager
			{
				GateError("Error in CoincidenceDigitizer: the argument of "
						"/gate/digitizerMgr/CoincidenceDigitizer/"<< m_coinDigitizer->GetName() <<"/addInputCollection : "<<  m_inputNames[i]<< " is UNKNOWN \n");

			}
			m_CDCIDs.push_back(ID);

		}

		m_FirstEvent=false;

	}

	GateCoincidenceDigi* inputDigi = new GateCoincidenceDigi();

	for(G4int i=0; i<(G4int)m_CDCIDs.size();i++)
	{
		GateCoincidenceDigiCollection* inCDC = (GateCoincidenceDigiCollection*) (DigiMan->GetDigiCollection(m_CDCIDs[i]));
		if (inCDC)
		    {
				G4int n_coinDigi = inCDC->entries();

				 for (G4int i=0;i< n_coinDigi;i++)
				 {
					 inputDigi=(*inCDC)[i];
					 if (inputDigi->empty()) continue;

					 m_outputDigi = new GateCoincidenceDigi(*inputDigi);

					 G4double time = m_outputDigi->GetEndTime();
					 bool last=true;
					 //ordering by time m_outputDigiCollection
					 for (std::vector<GateCoincidenceDigi*>::iterator iter = outputDigiCollectionVector->begin() ; iter != outputDigiCollectionVector->end() ; ++iter)
					 {
						 if ( (*iter)->GetEndTime()>time)
						 {
							 outputDigiCollectionVector->insert(iter,m_outputDigi) ;
							 last=false;
							 break;
						 }
						 /*if ( m_noPriority && ((*it2)->GetTime()==time)) {
							  	  G4cout<<"!!!!!! SAME TIME "<<G4endl;
							//S.Jan 15/02/2006
						    	//G4do	uble p = RandFlat::shoot();
						    	G4double p = G4UniformRand();
						    	if (p<0.5) {ans.insert(it2,pulse) ; last=false; break;}
						  	  	  }
						  	  	  */
					 }

					 if (last) outputDigiCollectionVector->push_back(m_outputDigi);

					 if ( iter == outputDigiCollectionVector->end() )
						 m_outputDigiCollection->insert(m_outputDigi);


				 }
		    }
	}





  StoreDigiCollection(m_outputDigiCollection);


}

void GateCoinDigitizerInitializationModule::DescribeMyself(size_t )
{
  ;
}







