
/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*!
  \class  GateTimeDelay

  Digitizer module for simulating a TimeDelay
  The user can choose a specific TimeDelay for each tracked volume.
  
  Last modification (Adaptation to GND): May 2023 by Mohamed-Jordan Soumano mjsoumano@yahoo.com
*/


#include "GateTimeDelay.hh"
#include "GateTimeDelayMessenger.hh"
#include "GateDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"

GateTimeDelay::GateTimeDelay(GateSinglesDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/"+digitizer->GetSD()->GetName()+"/SinglesDigitizer/"+digitizer->m_digitizerName+"/"+name,digitizer,digitizer->GetSD()),
   m_TimeDelay(0),
   m_outputDigi(0),
   m_OutputDigiCollection(0),
   m_digitizer(digitizer)
 {
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
	m_Messenger = new GateTimeDelayMessenger(this);
}

GateTimeDelay::~GateTimeDelay()
{
  delete m_Messenger;
}

void GateTimeDelay::Digitize()
{
        //G4cout<< "Time delay = "<<m_TimeDelay<<G4endl;

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

		  GateDigi* m_outputDigi = new GateDigi(*inputDigi);

		  m_outputDigi->SetTime(inputDigi->GetTime()+ m_TimeDelay);

		  m_OutputDigiCollection->insert(m_outputDigi);
	  }


	  if (nVerboseLevel==1)
	  {
		G4cout << "[GateTimeDelay::Digitize]: returning output pulse-list with " << OutputDigiCollectionVector->size() << " entries\n";
		for (iter=OutputDigiCollectionVector->begin(); iter!= OutputDigiCollectionVector->end() ; ++iter)
			G4cout << **iter << Gateendl;
		G4cout << Gateendl;
	  }


	  //loop  over input digits
      } //IDC
      else
      {
  	    if (nVerboseLevel>1)
  	    	G4cout << "[GateTimeDelay::Digitize]: input digi collection is null -> nothing to do\n\n";
  	    return;
    }
  StoreDigiCollection(m_OutputDigiCollection);

}



void GateTimeDelay::DescribeMyself(size_t indent)
{
  ;
}
