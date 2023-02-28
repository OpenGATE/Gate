/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateToSummary.hh"
#include "GateDigitizerMgr.hh"

#ifdef G4ANALYSIS_USE_FILE

#include "GateToSummaryMessenger.hh"
#include "GateOutputMgr.hh"
#include "GateMiscFunctions.hh"

#include "G4DigiManager.hh"

//--------------------------------------------------------------------------------
GateToSummary::GateToSummary(const G4String& name, GateOutputMgr* outputMgr, DigiMode digiMode)
  : GateVOutputModule(name, outputMgr, digiMode),
    m_fileName(" ")
{
  m_summaryMessenger = new GateToSummaryMessenger(this);
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
GateToSummary::~GateToSummary()
{
  delete m_summaryMessenger;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
const G4String& GateToSummary::GiveNameOfFile()
{
  return m_fileName;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void GateToSummary::RecordBeginOfAcquisition()
{
  //m_nb_of_hits = 0;
	//OK GND 2022
	GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();

	for (size_t i=0;i<digitizerMgr->m_SDlist.size();i++)
	{
		G4String SDname=digitizerMgr->m_SDlist[i]->GetName();
		m_hits_collection_names.push_back(SDname);
		m_hits_nb[SDname] = 0;
	}

	m_nb_of_events = 0;
	m_nb_of_runs = 0;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void GateToSummary::RecordEndOfAcquisition()
{
	GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();
  std::ofstream os;
  OpenFileOutput(m_fileName, os);
  if (digitizerMgr->m_SDlist.size() ==1)
 	  {
	  os << "# NumberOfHits = " << m_hits_nb[digitizerMgr->m_SDlist[0]->GetName()] << std::endl;
 	  }
  else
  	  {
	  for (size_t i=0;i<digitizerMgr->m_SDlist.size();i++)
	  	{
		  G4String SDname=digitizerMgr->m_SDlist[i]->GetName();
	  	  os << "# NumberOfHits in "<< SDname <<" = " << m_hits_nb[SDname] << std::endl;

	  	}

  	  }
  os << "# NumberOfRun = " << m_nb_of_runs << std::endl;
  os << "# NumberOfEvents = " << m_nb_of_events << std::endl;
  //OK GND 2022
  for(auto s:m_collection_names) {
	  if (digitizerMgr->m_SDlist.size() ==1)
	  {
		  std::string tmp_str = s.substr(0, s.find("_"));
	  	  os << "# " << tmp_str << " = " << m_collection_nb[s] << std::endl;
	  }
	  else
		  os << "# " << s << " = " << m_collection_nb[s] << std::endl;
  }
  os.flush();
  os.close();
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void GateToSummary::RecordEndOfRun(const G4Run * )
{
  m_nb_of_runs++;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void GateToSummary::RecordEndOfEvent(const G4Event* )
{
  m_nb_of_events++;

  // store number of hits during this events
  //OK GND 2022
    std::vector<GateHitsCollection*> CHC_vector = GetOutputMgr()->GetHitCollections();
    for (size_t i=0; i<CHC_vector.size();i++ )//HC_vector.size()
       {
    	   GateHitsCollection* CHC = CHC_vector[i];
    	if (CHC) {
    		G4int NbHits = CHC->entries();
    		G4int n = 0;
    		for (G4int iHit=0; iHit<NbHits; iHit++) {
    			GateHit* aHit = (*CHC)[iHit];
    			if (aHit->GoodForAnalysis()) n++;
    		}

    		m_hits_nb[CHC->GetSDname()] +=n;

    	}
      }
  // store all collections (Singles, Coincidences etc)
  G4DigiManager * fDM = G4DigiManager::GetDMpointer();
  if (!fDM) return;

  for(auto n:m_collection_names) {
	  //OK GND 2022
    auto m_collectionID = GetCollectionID(n); //fDM->GetDigiCollectionID(n);
    const GateDigiCollection * SDC =
      (GateDigiCollection*) (fDM->GetDigiCollection( m_collectionID ));
    if (!SDC) continue;
    G4int n_digi =  SDC->entries();
    m_collection_nb[n] += n_digi;
  }
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void GateToSummary::addCollection(const std::string & str)
{
  m_collection_names.push_back(str);
  m_collection_nb[str] = 0;
}
//--------------------------------------------------------------------------------


#endif
