/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "globals.hh"
#include "G4Run.hh"
#include "G4Step.hh"
#include "G4Event.hh"
#include "G4VHitsCollection.hh"
#include "G4HCofThisEvent.hh"
#include "G4TrajectoryContainer.hh"
#include "G4VProcess.hh"
#include "G4ios.hh"
#include "G4UImanager.hh"
#include "G4ParticleDefinition.hh"
#include "G4Positron.hh"
#include "G4GenericIon.hh"
#include "G4Navigator.hh"
#include "G4TransportationManager.hh"

#include "GatePrimaryGeneratorAction.hh"
#include "GateSourceMgr.hh"
#include "GateHit.hh"
#include "GatePhantomHit.hh"
#include "GateAnalysis.hh"
#include "GateAnalysisMessenger.hh"
#include "GateTrajectoryNavigator.hh"
#include "GateOutputMgr.hh"
#include "GateVVolume.hh"
#include "GateActions.hh"
#include "GateToRoot.hh"
#include "GateDigitizerMgr.hh"

//--------------------------------------------------------------------------------------------------
GateAnalysis::GateAnalysis(const G4String& name, GateOutputMgr* outputMgr,DigiMode digiMode)
  : GateVOutputModule(name,outputMgr,digiMode)
{
  m_isEnabled = true; // This module is essential for the hit processing, so let it enabled !!!
  m_analysisMessenger = new GateAnalysisMessenger(this);
  m_trajectoryNavigator = new GateTrajectoryNavigator();
  SetVerboseLevel(0);
}
//--------------------------------------------------------------------------------------------------

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo...

//--------------------------------------------------------------------------------------------------
GateAnalysis::~GateAnalysis()
{
  delete m_analysisMessenger;
  //modifs seb 3/3/2009
  delete m_trajectoryNavigator;
  if (nVerboseLevel > 0)
    G4cout << "GateAnalysis deleting...m_analysisMessenger - m_trajectoryNavigator\n";
}
//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
const G4String& GateAnalysis::GiveNameOfFile()
{
  m_noFileName = "  "; // 2 spaces for output module with no fileName
  return m_noFileName;
}
//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
void GateAnalysis::RecordBeginOfAcquisition()
{
  if (nVerboseLevel > 2)
    G4cout << "GateAnalysis::RecordBeginOfAcquisition\n";
}
//--------------------------------------------------------------------------------------------------



//--------------------------------------------------------------------------------------------------
void GateAnalysis::RecordEndOfAcquisition()
{
  if (nVerboseLevel > 2)
    G4cout << "GateAnalysis::RecordEndOfAcquisition\n";
}
//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
void GateAnalysis::RecordBeginOfRun(const G4Run * )
{
  if (nVerboseLevel > 2)
    G4cout << "GateAnalysis::RecordBeginOfRun\n";
}
//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
void GateAnalysis::RecordEndOfRun(const G4Run * )
{
  if (nVerboseLevel > 2)
    G4cout << "GateAnalysis::RecordEndOfRun\n";
}
//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
void GateAnalysis::RecordBeginOfEvent(const G4Event* )
{
  if (nVerboseLevel > 2)
    G4cout << "GateAnalysis::RecordBeginOfEvent\n";

}
//--------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------
void GateAnalysis::RecordEndOfEvent(const G4Event* event)
{
 if (nVerboseLevel > 2)
    G4cout << "GateAnalysis::RecordEndOfEvent "<< Gateendl;

  G4TrajectoryContainer* trajectoryContainer = event->GetTrajectoryContainer();

  if (trajectoryContainer)
    m_trajectoryNavigator->SetTrajectoryContainer(trajectoryContainer);

  G4int eventID = event->GetEventID();
  G4int runID   = GateRunManager::GetRunManager()->GetCurrentRun()->GetRunID();
  //G4cout << "GateAnalysis::EventID et RunID :  " <<eventID<<" - "<<runID<< Gateendl;

  //G4int i;

  if (!trajectoryContainer)
    {
      if (nVerboseLevel > 0)
        G4cout << "GateAnalysis::RecordEndOfEvent : WARNING : G4TrajectoryContainer not found\n";
    }
  else
    {
	  //OK GND 2022
      std::vector<GateHitsCollection*> CHC_vector = GetOutputMgr()->GetHitCollections();

      for (size_t i=0; i<CHC_vector.size();i++ )
      {
      GateHitsCollection* CHC = CHC_vector[i];
      G4int NbHits = 0;
      G4int NpHits = 0;

      if (CHC)
        {
			  NbHits = CHC->entries();
			  //G4cout << "     " << NbHits << " hits are stored in essaiHitsCollection.\n";

			  //G4int ionID      = 1; // the primary vertex particle
			  //G4int positronID = 0; // no more needed
			  G4int photon1ID  = 0;
			  G4int photon2ID  = 0;
			  G4int rootID     = 0;
			  G4int primaryID  = 0;

			  G4int photon1_phantom_compton = 0;
			  G4int photon2_phantom_compton = 0;

			  G4int photon1_crystal_compton = 0;
			  G4int photon2_crystal_compton = 0;

			  G4int photon1_phantom_Rayleigh = 0;
			  G4int photon2_phantom_Rayleigh = 0;

			  G4int photon1_crystal_Rayleigh = 0;
			  G4int photon2_crystal_Rayleigh = 0;

			  G4int septalNb = 0; // HDS : septal penetration

		  ////////////
			  // search the positron
			  //positronID = // No more needed
			  m_trajectoryNavigator->FindPositronTrackID();

			  /*if (positronID == 0)
				{
				if (nVerboseLevel > 0) G4cout << "GateAnalysis::RecordEndOfEvent : WARNING : positronID == 0\n";
				}

				if (nVerboseLevel > 1) G4cout << "GateAnalysis::RecordEndOfEvent : positronID : " << positronID << Gateendl;
			  */

		  ////////////
		  //search the two gammas

			  std::vector<G4int> photonIDVec = m_trajectoryNavigator->FindAnnihilationGammasTrackID();
			  if (photonIDVec.size() == 0)
				{
				  // no gamma coming from a positron or an ion, or shooted as primary
				  if (nVerboseLevel > 0) G4cout
										   << "GateAnalysis::RecordEndOfEvent : WARNING : photonIDs not found\n";
				}
			  else
				{
				  //  This warning is somewhat irrelevant with 124I
				  if (nVerboseLevel > 0 && photonIDVec.size() > 2)
					G4cout << "GateAnalysis::RecordEndOfEvent : WARNING : photonID vector size > 2\n";

				  photon1ID = photonIDVec[0];
				  photon2ID = (photonIDVec.size() >= 2) ? photonIDVec[1] : 0;
				}

			  if (photon1ID == 0)
				{
				  if (nVerboseLevel > 0) G4cout
										   << "GateAnalysis::RecordEndOfEvent : WARNING : photon1ID == 0\n";
				}
			  if (photon2ID == 0) {
				if (nVerboseLevel > 1) G4cout
										 << "GateAnalysis::RecordEndOfEvent : WARNING : photon2ID == 0\n";
			  }
			  if (nVerboseLevel > 1) G4cout
									   << "GateAnalysis::RecordEndOfEvent : photon1ID : " << photon1ID
									   << "     photon2ID : " << photon2ID << Gateendl;


			  // analysis of the phantom hits to count the comptons, etc.

			  GatePhantomHitsCollection* PHC = GetOutputMgr()->GetPhantomHitCollection();
			  NpHits = PHC->entries();

			  G4String theComptonVolumeName("NULL");
			  G4String theComptonVolumeName1("NULL");
			  G4String theComptonVolumeName2("NULL");

			  G4String theRayleighVolumeName("NULL");
			  G4String theRayleighVolumeName1("NULL");
			  G4String theRayleighVolumeName2("NULL");

			  for (G4int iPHit=0;iPHit<NpHits;iPHit++)
				{

				  // HDS : septal penetration record
				  if ( m_recordSeptalFlag ) {
					if ((*PHC)[iPHit]->GetPhysVolName() == m_septalPhysVolumeName) {
					  ++septalNb;
					}
				  }
				  //

				  G4int    phantomTrackID = (*PHC)[iPHit]->GetTrackID();
				  G4String processName    = (*PHC)[iPHit]->GetProcess();
				  G4int    PDGcode        = (*PHC)[iPHit]->GetPDGEncoding();
				  G4ThreeVector hitPos    = (*PHC)[iPHit]->GetPos();

				  if (nVerboseLevel > 2)
					G4cout << "GateAnalysis::RecordEndOfEvent : GatePhantomHitsCollection : trackID : " << std::setw(5) << phantomTrackID
						   << "    PDG code : " << std::setw(5) << PDGcode << "  processName : <" << processName << ">\n";
				  theComptonVolumeName = G4String("NULL");

				  if (nVerboseLevel > 2) G4cout
										   << "GateAnalysis::RecordEndOfEvent : GatePhantomHitsCollection : trackID : " << std::setw(5) << phantomTrackID
										   << "    PDG code : " << std::setw(5) << PDGcode << "  processName : <" << processName << ">\n";
				  theRayleighVolumeName = G4String("NULL");

				  // Modif by DS and LS on Oct 4, 2002: we need to be able to recognise both 'compt'
				  // and 'LowEnCompt", hence the find on 'ompt'
				  //	if (processName.find("ompt") != G4String::npos || processName.find("Rayleigh") != G4String::npos) {
				  // modif. by CJG to separate Compton and Rayleigh photons
				  if (processName.find("ompt") != G4String::npos)
					{
					  if ((phantomTrackID == photon1ID)||(phantomTrackID == photon2ID))
						{
						  G4Navigator *gNavigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
						  G4ThreeVector null(0.,0.,0.);
						  G4ThreeVector *ptr;
						  ptr = &null;
						  theComptonVolumeName = gNavigator->LocateGlobalPointAndSetup(hitPos,ptr,false)->GetName();
						  if (nVerboseLevel > 1)
							G4cout << "GateAnalysis::RecordEndOfEvent :  theComptonVolumeName: "
								   << theComptonVolumeName << Gateendl;
						}
					  if (phantomTrackID == photon1ID)
						{
						  photon1_phantom_compton++;
						  theComptonVolumeName1 = theComptonVolumeName;
						  if (nVerboseLevel > 0) G4cout
												   << "GateAnalysis::RecordEndOfEvent : photon1_phantom_compton : " << photon1_phantom_compton << Gateendl;
						}
					  if (phantomTrackID == photon2ID)
						{
						  photon2_phantom_compton++;
						  theComptonVolumeName2 = theComptonVolumeName;
						  if (nVerboseLevel > 0) G4cout
												   << "GateAnalysis::RecordEndOfEvent : photon2_phantom_compton : " << photon2_phantom_compton << Gateendl;
						}
					}

				  // Counting Rayleigh scatter in phantom
				  if (processName.find("Rayl") != G4String::npos)
					{
					  if ((phantomTrackID == photon1ID)||(phantomTrackID == photon2ID))
						{
						  G4Navigator *gNavigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
						  G4ThreeVector null(0.,0.,0.);
						  G4ThreeVector *ptr;
						  ptr = &null;
						  theRayleighVolumeName = gNavigator->LocateGlobalPointAndSetup(hitPos,ptr,false)->GetName();
						  if (nVerboseLevel > 1)
							G4cout << "GateAnalysis::RecordEndOfEvent :  theRayleighVolumeName: "
								   << theRayleighVolumeName << Gateendl;
						}
					  if (phantomTrackID == photon1ID)
						{
						  photon1_phantom_Rayleigh++;
						  theRayleighVolumeName1 = theRayleighVolumeName;
						  if (nVerboseLevel > 0) G4cout
												   << "GateAnalysis::RecordEndOfEvent : photon1_phantom_Rayleigh : " << photon1_phantom_Rayleigh << Gateendl;
						}
					  if (phantomTrackID == photon2ID)
						{
						  photon2_phantom_Rayleigh++;
						  theRayleighVolumeName2 = theRayleighVolumeName;
						  if (nVerboseLevel > 0) G4cout
												   << "GateAnalysis::RecordEndOfEvent : photon2_phantom_Rayleigh : " << photon2_phantom_Rayleigh << Gateendl;
						}
					}
				} // end loop NpHits

			  TrackingMode theMode =( (GateSteppingAction *)(GateRunManager::GetRunManager()->GetUserSteppingAction() ) )->GetMode();


			  if (  theMode == TrackingMode::kTracker ) // in tracker mode we store the infos about the number of compton and rayleigh
				{ // G4cout << " GateAnalysis eventID "<<eventID<< Gateendl;
				  GateToRoot* gateToRoot = (GateToRoot*) (GateOutputMgr::GetInstance()->GetModule("root"));
				  ComptonRayleighData aCRData;
				  aCRData.photon1_phantom_Rayleigh = photon1_phantom_Rayleigh;
				  aCRData.photon2_phantom_Rayleigh = photon2_phantom_Rayleigh;
				  aCRData.photon1_phantom_compton  = photon1_phantom_compton;
				  aCRData.photon2_phantom_compton  = photon2_phantom_compton;
				  strcpy(aCRData.theComptonVolumeName1 , theComptonVolumeName1.c_str() );
				  strcpy(aCRData.theComptonVolumeName2 , theComptonVolumeName2.c_str() );
				  strcpy(aCRData.theRayleighVolumeName1 , theRayleighVolumeName1.c_str() );
				  strcpy(aCRData.theRayleighVolumeName2 , theRayleighVolumeName2.c_str() );
				  gateToRoot->RecordPHData( aCRData );
				  // return;
				}

			  if (  theMode == TrackingMode::kDetector ) // in tracker mode we store the infos about the number of compton and rayleigh
				{
				  // we are in detector mode
				  GateToRoot* gateToRoot = (GateToRoot*) (GateOutputMgr::GetInstance()->GetModule("root"));
				  ComptonRayleighData aCRData;
				  gateToRoot->GetPHData( aCRData);

				  photon1_phantom_Rayleigh += aCRData.photon1_phantom_Rayleigh;
				  photon2_phantom_Rayleigh += aCRData.photon2_phantom_Rayleigh;
				  photon1_phantom_compton  += aCRData.photon1_phantom_compton;
				  photon2_phantom_compton  += aCRData.photon2_phantom_compton;
				  /*
					if( theComptonVolumeName1 == G4String("NULL") ) {theComptonVolumeName1    = aCRData.theComptonVolumeName1;}
					if( theComptonVolumeName2 == G4String("NULL") ) {theComptonVolumeName2    = aCRData.theComptonVolumeName2;}
					if( theRayleighVolumeName1 == G4String("NULL") ) {theRayleighVolumeName1   = aCRData.theRayleighVolumeName1;}
					if( theRayleighVolumeName2 == G4String("NULL") ){theRayleighVolumeName2   = aCRData.theRayleighVolumeName2;}
				  */
				  theComptonVolumeName1    = aCRData.theComptonVolumeName1;
				  theComptonVolumeName2    = aCRData.theComptonVolumeName2;
				  theRayleighVolumeName1   = aCRData.theRayleighVolumeName1;
				  theRayleighVolumeName2   = aCRData.theRayleighVolumeName2;

				}


			  /////////
			  // Source info
			  // DS : if gate source is not used (with /gate/EnableGeneralParticleSource)
			  // there are no GateSource, so skip
			  if ((GateSourceMgr::GetInstance())->GetSourcesForThisEvent().size() == 0) return;

			  G4int sourceID = (((GateSourceMgr::GetInstance())->GetSourcesForThisEvent())[0])->GetSourceID();
			  G4ThreeVector sourceVertex = m_trajectoryNavigator->FindSourcePosition();

			  // Hits loop
			  for (G4int iHit=0;iHit<NbHits;iHit++)
				{
				  G4int    crystalTrackID = (*CHC)[iHit]->GetTrackID();
				  G4String processName = (*CHC)[iHit]->GetProcess();
				  // Counting Compton in the Crystal
				  //      if (processName.find("ompt") != G4String::npos || processName.find("Rayleigh") != G4String::npos) {
				  if (processName.find("ompt") != G4String::npos)
					{

					  if (crystalTrackID == photon1ID) photon1_crystal_compton++;
					  if (crystalTrackID == photon2ID) photon2_crystal_compton++;
					}

				  // Counting Rayleigh scatter in crystal
				  if (processName.find("Rayl") != G4String::npos)
					{

					  if (crystalTrackID == photon1ID) photon1_crystal_Rayleigh++;
					  if (crystalTrackID == photon2ID) photon2_crystal_Rayleigh++;
					}

				  G4int PDGEncoding  = (*CHC)[iHit]->GetPDGEncoding();
				  if (nVerboseLevel > 2)
					G4cout << "GateAnalysis::RecordEndOfEvent : HitsCollection: processName : <" << processName
						   << ">    Particls PDG code : " << PDGEncoding << Gateendl;
				  if ((*CHC)[iHit]->GoodForAnalysis())
					{
					  // fill in values with the branch with C struct

					  G4int trackID  = (*CHC)[iHit]->GetTrackID();
					  //G4int parentID = (*CHC)[iHit]->GetParentID();

					  G4int photonID = 0;
					  G4int nPhantomCompton = 0;
					  G4int nCrystalCompton = 0;

					  G4int nPhantomRayleigh = 0;
					  G4int nCrystalRayleigh = 0;

					  //	  if ((photon1ID != 0) && (photon2ID != 0)) {
					  if (photon1ID != 0)
						{ // this means that at least 1 photon has been found, requiring 2 is wrong for SPECT
						  // search the gamma from which this hit comes --> photonID
						  photonID = m_trajectoryNavigator->FindPhotonID(trackID);
						  if (photonID == rootID)
							{
							  if (nVerboseLevel > 2) G4cout
													   << "GateAnalysis::RecordEndOfEvent : trackID: " << trackID << " photonID = " << rootID << Gateendl;
							}
						}

					  if (photonID == 1)
						{
						  nPhantomCompton = photon1_phantom_compton;
						  nPhantomRayleigh = photon1_phantom_Rayleigh;
						  theComptonVolumeName = theComptonVolumeName1;
						  theRayleighVolumeName = theRayleighVolumeName1;
						  nCrystalCompton = photon1_crystal_compton;
						  nCrystalRayleigh = photon1_crystal_Rayleigh;
						}
					  else if (photonID == 2)
						{
						  nPhantomCompton = photon2_phantom_compton;
						  nPhantomRayleigh = photon2_phantom_Rayleigh;
						  theComptonVolumeName = theComptonVolumeName2;
						  theRayleighVolumeName = theRayleighVolumeName2;
						  nCrystalCompton = photon2_crystal_compton;
						  nCrystalRayleigh = photon2_crystal_Rayleigh;
						}

					  // search the primary that originated the track
					  primaryID = m_trajectoryNavigator->FindPrimaryID(trackID);

					  (*CHC)[iHit]->SetSourceID          (sourceID);
					  (*CHC)[iHit]->SetSourcePosition    (sourceVertex);
					  (*CHC)[iHit]->SetNPhantomCompton   (nPhantomCompton);
					  (*CHC)[iHit]->SetNPhantomRayleigh   (nPhantomRayleigh);
					  (*CHC)[iHit]->SetComptonVolumeName (theComptonVolumeName);
					  (*CHC)[iHit]->SetRayleighVolumeName (theRayleighVolumeName);
					  (*CHC)[iHit]->SetPhotonID          (photonID);
					  (*CHC)[iHit]->SetPrimaryID         (primaryID);
					  (*CHC)[iHit]->SetEventID           (eventID);
					  (*CHC)[iHit]->SetRunID             (runID);
					  (*CHC)[iHit]->SetNCrystalCompton   (nCrystalCompton);
					  (*CHC)[iHit]->SetNCrystalRayleigh   (nCrystalRayleigh);
					  (*CHC)[iHit]->SetNSeptal   (septalNb); // HDS : septal penetration
					}
				}
			} // end if (CHC)
      }
    } // end if (!trajectoryContainer)
  //OK GND 2022
   //RunDigitizers is called here otherwise we don't have all attributes filled for aHit
  GateDigitizerMgr* digitizerMgr=GateDigitizerMgr::GetInstance();

	 if(!digitizerMgr->m_alreadyRun)
	 {

		 if (digitizerMgr->m_recordSingles|| digitizerMgr->m_recordCoincidences)
			 {

			 	 digitizerMgr->RunDigitizers();
			 	 digitizerMgr->RunCoincidenceSorters();
			 	 digitizerMgr->RunCoincidenceDigitizers();
			 }
	 }


} // end function
//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
void GateAnalysis::RecordStepWithVolume(const GateVVolume *, const G4Step* )
{
  if (nVerboseLevel > 2)
    G4cout << "GateAnalysis::RecordStep\n";
}
//--------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------
void GateAnalysis::SetVerboseLevel(G4int val) {
  nVerboseLevel = val;
  if (m_trajectoryNavigator) m_trajectoryNavigator->SetVerboseLevel(val);
}
//--------------------------------------------------------------------------------------------------
