
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*! \class  GateDigitizerMerger
    \brief  GateDigitizerMerger merges  digis from several sensitive detectors

    - GateDigitizerMerger - by olga.kochebina@cea.fr 03/03/23

    \sa GateDigitizerMerger, GateDigitizerMergerMessenger
*/
#include "GateDigitizerMerger.hh"
#include "GateDigitizerMergerMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"



GateDigitizerMerger::GateDigitizerMerger(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_names(),
   m_inputCollectionIDs(),
   isFirstEvent(true),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateDigitizerMergerMessenger(this);
	//m_namesList.push_back(" ");
}


GateDigitizerMerger::~GateDigitizerMerger()
{
  delete m_Messenger;

}


void GateDigitizerMerger::Digitize()
{

	//Input collection ID (m_DCID, defined in GateVDigitizerModule) is calculated automatically
	//Other collectionIDs are deduced from users name for addInput

	if (m_names.empty())
	{
		GateError("***ERROR*** Merger Digitizer Module: addInput option is not set! \n\n");

	}

	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* fDM = G4DigiManager::GetDMpointer();

	//Do only in the first event as it is time consuming procedure
	if(isFirstEvent)
	{
		for(G4int i=0; i<m_names.size();i++)
		{
			//G4cout<<m_names[i]<< " "<< fDM->GetDigiCollectionID(m_names[i])<<G4endl;
			m_inputCollectionIDs.push_back(fDM->GetDigiCollectionID(m_names[i]));
		}

		isFirstEvent=false;
	}

	GateDigiCollection* IDC = 0;
	IDC = (GateDigiCollection*) (fDM->GetDigiCollection(m_DCID));
	std::vector< GateDigi* >* IDCVector = IDC->GetVector ();
	GateDigi* inputDigi;

	if (IDC)
	 {
		  G4int n_digi = IDC->entries();

		  //loop over input digits
		  for (G4int i=0;i<n_digi;i++)
		  {
			  inputDigi=(*IDC)[i];
			  m_outputDigi = new GateDigi(*inputDigi);

			  m_OutputDigiCollection->insert(m_outputDigi);

		  }
	  }

	for(G4int i=0; i<m_inputCollectionIDs.size();i++)
		{
			GateDigi* inputDigi_tmp;
			GateDigiCollection* IDCtmp = 0;
			IDCtmp = (GateDigiCollection*) (fDM->GetDigiCollection(m_inputCollectionIDs[i]));
			if (m_inputCollectionIDs[i]==m_DCID)
			{
				//TODO add check also for all other inserted collection between themselves
				GateError("***ERROR*** Merger Digitizer Module: you try to merge the same things. Please, check used names carefully \n\n");

			}

			if(!IDCtmp)
			{
				//GateDigiCollection* IDCerror = 0;
				//G4String err = fDM->GetDigiCollection(m_DCID-1)->GetName();
				GateError("***ERROR*** Wrong usage of Merger Digitizer Module: the Digi collection that you want to use doesn't exist yet (not digitized yet?). The Merger must be inserted as a module of last called sensitive detector\n "
						"Please, read the description here: https://opengate.readthedocs.io/en/latest/digitizer_and_detector_modeling.html#id28 \n "
						"It is also possible that your input collection is empty at the first event. This bug will be addressed soon. \n\n");
				return;

			}
			std::vector< GateDigi* >* IDCVector_tmp = IDCtmp->GetVector ();

			if (IDCtmp)
				 {
					  G4int n_digi = IDCtmp->entries();

					  //loop over input digits
					  for (G4int i=0;i<n_digi;i++)
					  {
						  inputDigi_tmp=(*IDCtmp)[i];
						  m_outputDigi = new GateDigi(*inputDigi_tmp);

						  m_OutputDigiCollection->insert(m_outputDigi);

					  }
				  }
		}


  StoreDigiCollection(m_OutputDigiCollection);

}

void GateDigitizerMerger::AddInputCollection(const G4String& name)
{
	m_names.push_back(name);

}


void GateDigitizerMerger::DescribeMyself(size_t indent )
{
  ;
}
