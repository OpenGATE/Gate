/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateToSummary.hh"

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
  m_nb_of_hits = 0;
  m_nb_of_events = 0;
  m_nb_of_runs = 0;
}
//--------------------------------------------------------------------------------


//--------------------------------------------------------------------------------
void GateToSummary::RecordEndOfAcquisition()
{
  std::ofstream os;
  OpenFileOutput(m_fileName, os);
  os << "# NumberOfHits = " << m_nb_of_hits << std::endl;
  os << "# NumberOfRun = " << m_nb_of_runs << std::endl;
  os << "# NumberOfEvents = " << m_nb_of_events << std::endl;
  for(auto s:m_collection_names) {
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
  GateCrystalHitsCollection* CHC = GetOutputMgr()->GetCrystalHitCollection();
  if (CHC) {
    G4int NbHits = CHC->entries();
    G4int n = 0;
    for (G4int iHit=0; iHit<NbHits; iHit++) {
      GateCrystalHit* aHit = (*CHC)[iHit];
      if (aHit->GoodForAnalysis()) n++;
    }
    m_nb_of_hits += n;
  }

  // store all collections (Singles, Coincidences etc)
  G4DigiManager * fDM = G4DigiManager::GetDMpointer();
  if (!fDM) return;

  for(auto n:m_collection_names) {
    auto m_collectionID = fDM->GetDigiCollectionID(n);
    const GateSingleDigiCollection * SDC =
      (GateSingleDigiCollection*) (fDM->GetDigiCollection( m_collectionID ));
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
