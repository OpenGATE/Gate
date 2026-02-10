/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "../include/GateCoincidenceTimeDiffSelector.hh"
#include "G4UnitsTable.hh"
#include "GateCoincidenceTimeDiffSelectorMessenger.hh"
#include "GateTools.hh"
#include "GateVolumeID.hh"
#include "GateOutputVolumeID.hh"
#include "GateDetectorConstruction.hh"
#include "GateCrystalSD.hh"
#include "GateVSystem.hh"
#include "GateObjectChildList.hh"
#include "GateVVolume.hh"
#include "GateMaps.hh"
#include "GateCoincidenceDigi.hh"

#include "GateDigitizerMgr.hh"

#include "G4SystemOfUnits.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"
#include "G4SDManager.hh"
#include "G4DigiManager.hh"
#include "G4ios.hh"
#include "G4UnitsTable.hh"
GateCoincidenceTimeDiffSelector::GateCoincidenceTimeDiffSelector(GateCoincidenceDigitizer *digitizer, G4String name)
  :GateVDigitizerModule(name,"digitizerMgr/CoincidenceDigitizer/"+digitizer->m_digitizerName+"/"+name, digitizer),
  m_outputDigi(0),
  m_OutputDigiCollection(0),
  m_digitizer(digitizer)

  {  m_minTime = -1;
     m_maxTime = -1;
	G4String colName = digitizer->GetOutputName() ;
	collectionName.push_back(colName);
  m_messenger = new GateCoincidenceTimeDiffSelectorMessenger(this);
}




GateCoincidenceTimeDiffSelector::~GateCoincidenceTimeDiffSelector()
{
  delete m_messenger;
}


void GateCoincidenceTimeDiffSelector::Digitize()
{
	G4String digitizerName = m_digitizer->m_digitizerName;
	G4String outputCollName = m_digitizer-> GetOutputName();

	m_OutputDigiCollection = new GateCoincidenceDigiCollection(GetName(),outputCollName); // to create the Digi Collection

	G4DigiManager* DigiMan = G4DigiManager::GetDMpointer();




	GateCoincidenceDigiCollection* IDC = 0;
	IDC = (GateCoincidenceDigiCollection*) (DigiMan->GetDigiCollection(m_DCID));

	GateCoincidenceDigi* inputDigi = new GateCoincidenceDigi();

	std::vector< GateCoincidenceDigi* >* OutputDigiCollectionVector = m_OutputDigiCollection->GetVector ();
	std::vector<GateCoincidenceDigi*>::iterator iter;

	 if (IDC) {
	        G4int n_digi = IDC->entries();
	        // Loop over input digits
	        for (G4int i = 0; i < n_digi; i++) {
	            GateCoincidenceDigi* inputDigi = (*IDC)[i]; // Retrieve input digi
	            G4double timeDiff = inputDigi->ComputeFinishTime() - inputDigi->GetStartTime();
	            if (((m_minTime > 0) && (timeDiff < m_minTime)) || ((m_maxTime > 0) && (timeDiff > m_maxTime))) {
	                continue; // Skip this digi
	            } else {
	                m_outputDigi = new GateCoincidenceDigi(*inputDigi);
	                m_OutputDigiCollection->insert(m_outputDigi);
	            }//loop  over input digits
	        }}//IDC
else
  {
	  if (nVerboseLevel>1)
	  	G4cout << "[GateCoincidenceDeadTime::Digitize]: input digi collection is null -> nothing to do\n\n";
	    return;
  }
StoreDigiCollection(m_OutputDigiCollection);

}

void GateCoincidenceTimeDiffSelector::DescribeMyself(size_t indent)
{
  G4cout << GateTools::Indent(indent) << "TimeDiffSelector: "
      	 << G4BestUnit(m_minTime,"Time") << "/"<< G4BestUnit(m_minTime,"Time")  << Gateendl;
}
