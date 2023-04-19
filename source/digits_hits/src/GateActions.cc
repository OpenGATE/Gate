/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GATEACTION_CC
#define GATEACTION_CC

#include "G4Run.hh"
#include "G4UImanager.hh"
#include "G4VVisManager.hh"
#include "G4Polyline.hh"
#include "G4Colour.hh"
#include "G4VisAttributes.hh"
#include "G4NeutrinoE.hh"
#include "G4SteppingManager.hh"
#include "GateActions.hh"
#include "GateUserActions.hh"
#include "GateTrack.hh"

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_GENERAL
#include "GateOutputMgr.hh"
#endif
#include "GateARFDataToRoot.hh"
#include "GateVolumeID.hh"
#include "GateToRoot.hh"
#include "GateSPECTHeadSystem.hh"
#include "GateSystemListManager.hh"
#include "GateObjectStore.hh"
#include "G4VSensitiveDetector.hh"
#include "GateSourceMgr.hh"
#include <iostream>
#include <fstream>
#include<string>
#include <sstream>
#include "G4PhysicalVolumeStore.hh"
#include "G4ProcessTable.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4Trajectory.hh"
#include "GateDetectorConstruction.hh"
#ifdef G4_USE_G4BESTUNIT_FOR_VERBOSE
#include "G4UnitsTable.hh"
#else
#define G4BestUnit(a,b) a
#endif

#include "GateSteppingActionMessenger.hh"
#include "GateCrystalSD.hh"

#include "GateDigitizerMgr.hh"

GateRunAction* GateRunAction::prunAction=0;
GateEventAction* GateEventAction::peventAction=0;

//-----------------------------------------------------------------------------
GateRunAction::GateRunAction(GateUserActions * cbm)
  : pCallbackMan(cbm), flagBasicOutput(false)
{
	SetRunAction(this); runIDcounter = 0;

	//OK GND 2022. moved from Gate.cc
	//Very first initialization of GateDigitizerMgr
#ifdef G4ANALYSIS_USE_GENERAL
	GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();
	//digitizerMgr->Enable(false);
#endif

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateRunAction::BeginOfRunAction(const G4Run* aRun)
{
  GateMessage("Core", 1, "Begin Of Run " << aRun->GetRunID() << Gateendl);

  //#ifdef GATE_BasicROOT_Output
  //if(GateApplicationMgr::GetInstance()->GetOutputMode()){

  //--------------------------------------------------------------
  //}
  //#endif

#ifdef G4ANALYSIS_USE_GENERAL
  // Here we fill the histograms of the Analysis manager
  if(GateApplicationMgr::GetInstance()->GetOutputMode()){
    GateOutputMgr* outputMgr = GateOutputMgr::GetInstance();
    outputMgr->RecordBeginOfRun(aRun);
  }
#endif

  pCallbackMan->BeginOfRunAction(aRun);

  // OK GND 2022
  // Filling CHCollID for
  //In order to get unique CHCollID each time when we create a new SD
  	// one can take GetCollectionCapacity of G4SDManager
  	// by default there is always phantomSD attached
  	// thus: 1st CHCollID = 1
  	// This is done in order to replace a block from Intialize()
  	/*
  	 static G4int CHCollID=-1;
  	 if(CHCollID<0 ) // call only in the first event
  	 			{
  				CHCollID = G4SDManager::GetSDMpointer()->GetCollectionID(GetName()+"Collection");

  			}
  	 */

  G4SDManager* SDman = G4SDManager::GetSDMpointer();

  for (int i=0; i< SDman->GetCollectionCapacity(); i++)
  {
	   G4String HCname = SDman->GetHCtable()->GetHCname(i);
	  // G4cout<<SDman->GetHCtable()->GetSDname(i)<<" "<<SDman->GetHCtable()->GetHCname(i)  <<G4endl;
	  G4int CHCollID = G4SDManager::GetSDMpointer()->GetCollectionID(SDman->GetHCtable()->GetHCname(i));
	  //G4cout<<CHCollID<<" "<< SDman->GetHCtable()->GetHCname(i)<<G4endl;
	  m_CHCollIDs.push_back(CHCollID);

  }

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
inline void GateRunAction::EndOfRunAction(const G4Run* aRun)
{
  GateMessage("Core", 1, "End Of Run " << aRun->GetRunID() << Gateendl);



#ifdef G4ANALYSIS_USE_GENERAL
  // Here we fill the histograms of the Analysis manager
  if(GateApplicationMgr::GetInstance()->GetOutputMode()){
    GateOutputMgr* outputMgr = GateOutputMgr::GetInstance();
    outputMgr->RecordEndOfRun(aRun);
  }
#endif

  pCallbackMan->EndOfRunAction(aRun);
}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateEventAction::GateEventAction(GateUserActions * cbm)
  : pCallbackMan(cbm), flagBasicOutput(false)
{ SetEventAction(this); }
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
inline void GateEventAction::BeginOfEventAction(const G4Event* anEvent)
{
  GateMessage("Core", 2, "Begin Of Event " << anEvent->GetEventID() << "\n");
 // G4cout<<"Begin Of Event " << anEvent->GetEventID() << G4endl;

  TrackingMode theMode =( (GateSteppingAction *)(GateRunManager::GetRunManager()->GetUserSteppingAction() ) )->GetMode();
  if ( theMode != TrackingMode::kTracker )
    {


#ifdef G4ANALYSIS_USE_GENERAL
      // Here we fill the histograms of the OutputMgr manager
      if(GateApplicationMgr::GetInstance()->GetOutputMode()){
        GateOutputMgr* outputMgr = GateOutputMgr::GetInstance();
        outputMgr->RecordBeginOfEvent(anEvent);
      }
#endif
    }

  if(anEvent->GetNumberOfPrimaryVertex() > 0) pCallbackMan->BeginOfEventAction(anEvent);

  // OK GND 2022
  GateDigitizerMgr* digitizerMgr=GateDigitizerMgr::GetInstance();
  digitizerMgr->m_alreadyRun=false;
 // G4cout<<"m_alreadyRun "<< digitizerMgr->m_alreadyRun<<G4endl;

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
inline void GateEventAction::EndOfEventAction(const G4Event* anEvent)
{
  GateMessage("Core", 2, "End Of Event " << anEvent->GetEventID() << "\n");

  //OK GND 2022 TODO
   //I would like to RunDigitizers here but some aHit attributes are filled in OutputMng/GateAnalysis->RecordEndOfEvent
   //GateDigitizerMgr* digitizerMgr = GateDigitizerMgr::GetInstance();
   // 	  digitizerMgr->RunDigitizers();

#ifdef G4ANALYSIS_USE_GENERAL
  // Here we fill the histograms of the OutputMgr manager
  // Pre-digitalisation outputMgr (hits)
  if(GateApplicationMgr::GetInstance()->GetOutputMode()){
    GateOutputMgr* outputMgr = GateOutputMgr::GetInstance();
    outputMgr->RecordEndOfEvent(anEvent);
  }
#endif

  /* PY Descourt 08/09/2009 */

  GateSteppingAction* myAction = ( (GateSteppingAction *)(GateRunManager::GetRunManager()->GetUserSteppingAction() ) );
  TrackingMode theMode = myAction->GetMode();

  //OK GND 2022
  GateRunManager* RunMan = GateRunManager::GetRunManager();
  GateRunAction* RunAction = ( (GateRunAction*)(RunMan->GetUserRunAction()) );

    for (size_t i=0; i<RunAction->m_CHCollIDs.size(); i++)
     {
  	  // TODO: OK GND 2022, test that in tracking mode it still works
		  if ( theMode == TrackingMode::kTracker )
			{

			  //OK GND 2022
			  G4int CHCollID = RunAction->m_CHCollIDs[i];
			  GateHitsCollection * CHC = (GateHitsCollection *) ( anEvent->GetHCofThisEvent()->GetHC( CHCollID ) );



			  if (CHC != 0)
			{ if ( CHC->GetSize() > 0 )
				{                          G4int i = anEvent->GetEventID();
				  std::stringstream event_id; // convert event_id into string
				  event_id << i ;
				  std::stringstream size; // convert size into string
				  i = CHC->GetSize();
				  size << i ;
				  G4String message = " GateEventAction::EndOfEventAction : ERROR  Event "+ event_id.str() + " processed " + size.str() + " Crystal Hits.\n"+"Your Stepping policies may not be appropriately set. For instance You specified to stop after Phantom Boundaries and the distance between the phantom and the detectors is not sufficient so some particles reached the detectors.\n";
				  G4Exception( "GateEventAction::EndOfEventAction", "EndOfEventAction", FatalException, message );
				}
			}
			  if (  anEvent->GetNumberOfPrimaryVertex() > 0 )
			{
			  GateOutputMgr::GetInstance()->RecordTracks(myAction);
			  //   RECORD THE PHANTOM HITS COLLECTION OF THE CURRENT EVENT
			  GateToRoot* gateToRoot = (GateToRoot*) (GateOutputMgr::GetInstance()->GetModule("root"));
			  // STORE TO A ROOT FILE  THE DATA COLLECTED IN THE RECORDSTEP METHOD DURING STEPPING
			  gateToRoot->RecordRecStepData( anEvent );
			}
			  // se charge de remplir les histos      : steppingAction contient la colllection de tracks
		}//tracker mode
     }

  if(anEvent->GetNumberOfPrimaryVertex() > 0) pCallbackMan->EndOfEventAction(anEvent);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateTrackingAction::GateTrackingAction(GateUserActions * cbm)
  : pCallbackMan(cbm)
{
}
//-----------------------------------------------------------------------------

void GateTrackingAction::PreUserTrackingAction(const G4Track* a)
{
  // Create trajectory only for primaries
  if(a->GetParentID()==0) {
    fpTrackingManager->SetStoreTrajectory(true);
  } else {
    fpTrackingManager->SetStoreTrajectory(true);
    //fpTrackingManager->SetStoreTrajectory(false);
  }

  /* PY Descourt 08/09/2009 */

  GateSteppingAction*  myAction = (GateSteppingAction *) (GateRunManager::GetRunManager()->GetUserSteppingAction()) ;

  TrackingMode theMode = myAction->GetMode();

  if ( theMode == TrackingMode::kDetector )
    {
      std::vector<GateTrack*>* aTrackVector = myAction->GetPPTrackVector();
      G4int size =  aTrackVector->size() ;
      if ( size > 0 )
        {    G4bool test =false;
          G4Track* TmpTrack = const_cast<G4Track*>(a) ;
          std::vector<GateTrack*>::iterator iter;
          for ( iter = aTrackVector->begin(); iter != aTrackVector->end(); iter++)
            {   // G4cout << "  GateTrackingAction::PreUserTrackingAction()   Tracks Vector size  " << aTrackVector->size()<< Gateendl;
              G4int track_id = (*iter)->GetTrackID();
              G4int p_id   = (*iter)->GetParentID();
              test = (*iter)->Compare(TmpTrack);
              if (test == true )
                {
                  G4ThreeVector aVp = (*iter)->GetVertexPosition()  ;
                  if ( (*iter)->GetWasKilled() == 1 ) {  TmpTrack->SetTrackStatus( fStopAndKill ); }
                  G4DynamicParticle* aDP = const_cast<G4DynamicParticle*> ( TmpTrack->GetDynamicParticle() );
                  aDP->SetMomentum( (*iter)->GetMomentum() );
                  aDP->SetMomentumDirection( (*iter)->GetMomentumDirection().x() , (*iter)->GetMomentumDirection().y() ,(*iter)->GetMomentumDirection().z() );
                  aDP->SetKineticEnergy(  (*iter)->GetKineticEnergy()  );
                  aDP->SetPolarization( (*iter)->GetPolarization().x(),(*iter)->GetPolarization().y(),(*iter)->GetPolarization().z() );
                  aDP->SetProperTime( (*iter)->GetProperTime() );
                  const G4ThreeVector aVMD = (*iter)->GetVertexMomentumDirection();
                  const G4double aVKE =  (*iter)->GetVertexKineticEnergy() ;
                  // get the creator process pointer from its name
                  G4String processName = (*iter)->GetProcessName();
                  G4String parentparticleName = (*iter)->GetParentParticleName();
                  G4ProcessManager* pM = 0;
                  if ( parentparticleName != G4String( "None" ) )
                    {
                      pM = G4ParticleTable::GetParticleTable()->FindParticle( parentparticleName)->GetProcessManager();
                    }
                  G4VProcess* theProcess = G4ProcessTable::GetProcessTable()->FindProcess(   processName,  pM  );
                  TmpTrack->SetVertexPosition( aVp );
                  TmpTrack->SetTrackID( track_id );
                  TmpTrack->SetParentID( p_id );
                  TmpTrack->SetVertexMomentumDirection( aVMD   );
                  TmpTrack->SetVertexKineticEnergy( aVKE ) ;
                  TmpTrack->SetCreatorProcess( theProcess );
                  TmpTrack->SetLocalTime( (*iter)->GetLocalTime() );
                  TmpTrack->SetGlobalTime( (*iter)->GetGlobalTime() );
                  TmpTrack->SetProperTime( (*iter)->GetProperTime() );
                  G4LogicalVolumeStore* theLStore = G4LogicalVolumeStore::GetInstance();
                  G4LogicalVolume* theLogVol = 0;
                  std::vector<G4LogicalVolume*>::iterator itLog;
                  for ( itLog = theLStore->begin(); itLog != theLStore->end(); itLog++)
                    { G4String aLogName = (*itLog)->GetName();
                      if ( aLogName == (*iter)->GetVertexVolumeName() )
                        {
                          theLogVol = (*itLog);
                          break;
                        }
                    }
                  TmpTrack->SetLogicalVolumeAtVertex( theLogVol );
                  delete (*iter);
                  aTrackVector->erase( iter );
                  break;
                } // end if
            } // end for
        } // end if
    }

  /* PY Descourt 08/09/2009 */

  pCallbackMan->PreUserTrackingAction(a);
}


/* PY Descourt 08/09/2009 */
void GateTrackingAction::PostUserTrackingAction(const G4Track* aTrack)
{

  if ( !dummy_track_vector.empty() )
    { for ( size_t i = 0; i < dummy_track_vector.size();i++)
        delete dummy_track_vector[i];
      dummy_track_vector.clear();
    }
  if ( !dummy_step_vector.empty() )
    { for ( size_t i = 0; i < dummy_step_vector.size();i++)
        delete dummy_step_vector[i];
      dummy_step_vector.clear();
    }

  GateSteppingAction*  myAction = (GateSteppingAction *) (GateRunManager::GetRunManager()->GetUserSteppingAction()) ;
  TrackingMode theMode = myAction->GetMode();
  if ( theMode == TrackingMode::kDetector )
    {
      //G4int eventID = G4EventManager::GetEventManager()->GetNonconstCurrentEvent()->GetEventID();
      // In Detector Mode : look at trajectories
      if ( aTrack->GetParentID() < 2 )
        {
          G4TrackingManager* trackManager = G4EventManager::GetEventManager()->GetTrackingManager();

          G4Trajectory* trj = (G4Trajectory*) ( trackManager->GimmeTrajectory() );

          // for the GateTrajectoryNavigator::FindAnnihilationGammasTrackID() method
          // we need to store the vertex position of each track
          // so we append it at the end because it is not possible in GEANT4 to set a position for a G4TrajectoryPoint
          // for this we create a dummy G4Step
          // which position is the Vertex Position of the track
          G4Track* dummy_track = new G4Track();
          dummy_track_vector.push_back( dummy_track );
          G4Step* dummy_step   = new G4Step();
          dummy_step_vector.push_back( dummy_step );
          dummy_track->SetStep( dummy_step );
          // dummy_track_vector.push_back( dummy_track );

          // this dummy track must have as position the vertex position
          dummy_track->SetPosition( aTrack->GetVertexPosition() );
          dummy_step->GetPostStepPoint()->SetPosition( aTrack->GetVertexPosition() );
          // we append it to the trajectory
          // so when searching for the vertex position in detector mode
          // we must look at the end of the trajectory
          trj->AppendStep( dummy_step );
        }
    }
  pCallbackMan->PostUserTrackingAction(aTrack);
}

/* PY Descourt 08/09/2009 */

void GateTrackingAction::ShowG4TrackInfos( G4String outF, G4Track* aTrack )
{

  std::ofstream outFile;
  outFile.open (outF,std::ios::app);

  G4int eventID = G4EventManager::GetEventManager()->GetNonconstCurrentEvent()->GetEventID();

  outFile << " Event ID is "  << eventID << Gateendl;
  outFile << "GateTrackingAction::ShowG4TrackInfos :::: current  Track :::::::: \n";

  G4String particleName = aTrack->GetDefinition()->GetParticleName();

  outFile << " Particle " << particleName<< Gateendl;
  outFile  << " Parent ID " << aTrack->GetParentID()  <<"  track ID "<< aTrack->GetTrackID() << Gateendl;
  outFile << "      -----------------------------------------------\n";
  outFile << "        G4Track Information  " << std::setw(20) << Gateendl;
  outFile << "      -----------------------------------------------\n";
  outFile << "        Step number         : " << std::setw(20) << aTrack->GetCurrentStepNumber() << Gateendl;

#ifdef G4_USE_G4BESTUNIT_FOR_VERBOSE
  outFile << "        Position - x        : "
          << std::setw(20) << G4BestUnit(aTrack->GetPosition().x(), "Length")
          << Gateendl;
  outFile << "        Position - y        : "
          << std::setw(20) << G4BestUnit(aTrack->GetPosition().y(), "Length")
          << Gateendl;
  outFile << "        Position - z        : "
          << std::setw(20) << G4BestUnit(aTrack->GetPosition().z(), "Length")
          << Gateendl;
  outFile << "        Global Time         : "
          << std::setw(20) << G4BestUnit(aTrack->GetGlobalTime(), "Time")
          << Gateendl;
  outFile << "        Local Time          : "
          << std::setw(20) << G4BestUnit(aTrack->GetLocalTime(), "Time")
          << Gateendl;
#else
  outFile << "        Position - x (mm)   : "
          << std::setw(20) << aTrack->GetPosition().x() /mm
          << Gateendl;
  outFile << "        Position - y (mm)   : "
          << std::setw(20) << aTrack->GetPosition().y() /mm
          << Gateendl;
  outFile << "        Position - z (mm)   : "
          << std::setw(20) << aTrack->GetPosition().z() /mm
          << Gateendl;
  outFile << "        Global Time (ns)    : "
          << std::setw(20) << aTrack->GetGlobalTime() /ns
          << Gateendl;
  outFile << "        Local Time (ns)     : "
          << std::setw(20) << aTrack->GetLocalTime() /ns
          << Gateendl;
#endif
  outFile << "        Momentum Direct - x : "
          << std::setw(20) << aTrack->GetMomentumDirection().x()
          << Gateendl;
  outFile << "        Momentum Direct - y : "
          << std::setw(20) << aTrack->GetMomentumDirection().y()
          << Gateendl;
  outFile << "        Momentum Direct - z : "
          << std::setw(20) << aTrack->GetMomentumDirection().z()
          << Gateendl;
#ifdef G4_USE_G4BESTUNIT_FOR_VERBOSE
  outFile << "        Kinetic Energy      : "
#else
    outFile << "        Kinetic Energy (MeV): "
#endif
          << std::setw(20) << G4BestUnit(aTrack->GetKineticEnergy(), "Energy")
          << Gateendl;
  outFile << "        Polarization - x    : "
          << std::setw(20) << aTrack->GetPolarization().x()
          << Gateendl;
  outFile << "        Polarization - y    : "
          << std::setw(20) << aTrack->GetPolarization().y()
          << Gateendl;
  outFile << "        Polarization - z    : "
          << std::setw(20) << aTrack->GetPolarization().z()
          << Gateendl;
  outFile << "        Track Length        : "
          << std::setw(20) << G4BestUnit(aTrack->GetTrackLength(), "Length")
          << Gateendl;
  outFile << "        Track ID #          : "
          << std::setw(20) << aTrack->GetTrackID()
          << Gateendl;
  outFile << "        Parent Track ID #   : "
          << std::setw(20) << aTrack->GetParentID()
          << Gateendl;
  outFile << "        Next Volume         : "
          << std::setw(20);
  if( aTrack->GetNextVolume() != 0 ) {
    outFile << aTrack->GetNextVolume()->GetName() << " ";
  } else {
    outFile << "OutOfWorld" << " ";
  }
  outFile << Gateendl;
  outFile << "        Track Status        : "
          << std::setw(20);
  if( aTrack->GetTrackStatus() == fAlive ){
    outFile << " Alive";
  } else if( aTrack->GetTrackStatus() == fStopButAlive ){
    outFile << " StopButAlive";
  } else if( aTrack->GetTrackStatus() == fStopAndKill ){
    outFile << " StopAndKill";
  } else if( aTrack->GetTrackStatus() == fKillTrackAndSecondaries ){
    outFile << " KillTrackAndSecondaries";
  } else if( aTrack->GetTrackStatus() == fSuspend ){
    outFile << " Suspend";
  } else if( aTrack->GetTrackStatus() == fPostponeToNextEvent ){
    outFile << " PostponeToNextEvent";
  }
  outFile << Gateendl;
#ifdef G4_USE_G4BESTUNIT_FOR_VERBOSE
  outFile << "        Vertex - x          : "
          << std::setw(20) << G4BestUnit(aTrack->GetVertexPosition().x(),"Length")
          << Gateendl;
  outFile << "        Vertex - y          : "
          << std::setw(20) << G4BestUnit(aTrack->GetVertexPosition().y(),"Length")
          << Gateendl;
  outFile << "        Vertex - z          : "
          << std::setw(20) << G4BestUnit(aTrack->GetVertexPosition().z(),"Length")
          << Gateendl;
#else
  outFile << "        Vertex - x (mm)     : "
          << std::setw(20) << aTrack->GetVertexPosition().x()/mm
          << Gateendl;
  outFile << "        Vertex - y (mm)     : "
          << std::setw(20) << aTrack->GetVertexPosition().y()/mm
          << Gateendl;
  outFile << "        Vertex - z (mm)     : "
          << std::setw(20) << aTrack->GetVertexPosition().z()/mm
          << Gateendl;
#endif
  outFile << "        Vertex - Px (MomDir): "
          << std::setw(20) << aTrack->GetVertexMomentumDirection().x()
          << Gateendl;
  outFile << "        Vertex - Py (MomDir): "
          << std::setw(20) << aTrack->GetVertexMomentumDirection().y()
          << Gateendl;
  outFile << "        Vertex - Pz (MomDir): "
          << std::setw(20) << aTrack->GetVertexMomentumDirection().z()
          << Gateendl;
#ifdef G4_USE_G4BESTUNIT_FOR_VERBOSE
  outFile << "        Vertex - KineE      : "
#else
    outFile << "        Vertex - KineE (MeV): "
#endif
          << std::setw(20) << G4BestUnit(aTrack->GetVertexKineticEnergy(),"Energy")
          << Gateendl;

  outFile << "        Creator Process     : "
          << std::setw(20);
  if( aTrack->GetCreatorProcess() == NULL){
    outFile << " Event Generator\n";
  } else {
    outFile << aTrack->GetCreatorProcess()->GetProcessName() << Gateendl;
  }

  outFile << "      -----------------------------------------------"
          << Gateendl;

}


//-----------------------------------------------------------------------------
GateSteppingAction::GateSteppingAction(GateUserActions * cbm)
  : pCallbackMan(cbm)
{
  m_drawTrjLevel = 1;
  m_verboseLevel = 0;
  /* PY Descourt Tracker/Detector 18/12/2008 */
  m_steppingMessenger = new GateSteppingActionMessenger(this);
  m_trackingMode = TrackingMode::kBoth;
  Boundary = 1;
  fStpAKill = fStopAndKill;
  fKeepOnlyP = 0;
  fKeepOnlyPhotons = 0;
  fKeepOnlyElectrons = 0;
  TxtOn = 0;
  m_Nfiles = 1;
  m_currentN = 0;
  fKillNextIsSet = false;
  PPTrackVector = new std::vector<GateTrack*>;
  fStartVolumeIsPhantomSD = false;
  m_energyThreshold = 0.;
  /* PY Descourt 18/12/2008 */
}
//-----------------------------------------------------------------------------
void GateSteppingAction::SetEnergyThreshold(G4double aE){ m_energyThreshold = aE; }


G4int GateSteppingAction::SeekNewFile(G4bool increase)
{
  if ( m_verboseLevel > 0 ) G4cout << " GateSteppingAction::SeekNewFile  :::: m_currentN = " << m_currentN << Gateendl;

  if ( m_currentN == m_Nfiles - 1 ) {if ( m_verboseLevel > 0 ) G4cout << " GateSteppingAction::SeekNewFile : No more Root Tracks Data File to open.\n";
    return 0;}

  if ( m_verboseLevel > 0 ) G4cout << " GateSteppingAction::SeekNewFile : Found one more Root Tracks Data File to open.\n";
  GateToRoot* gateToRoot = (GateToRoot* ) ( GateOutputMgr::GetInstance()->GetModule("root") );
  if ( increase == true ) { m_currentN++;}
  if ( m_verboseLevel > 0 ) G4cout << " currrent file number in GateSteppingAction::SeekNewFile " << m_currentN << Gateendl;
  if ( gateToRoot != 0 ) {gateToRoot->OpenTracksFile();}
  return 1;
}

void GateSteppingAction::SetTxtOut(G4String aString)
{
  if ( aString == "On" ) { TxtOn = 1;return; }
  if ( aString == "Off" ) { TxtOn = 0;return; }
  G4cout << " GateSteppingAction::SetTxtOut : WARNING " << aString<< " in command SetTxtOuput is not correct. IGNORED!!!\n";
}


void GateSteppingAction::StopOnBoundary(G4int aI)
{ Boundary = aI;}

void GateSteppingAction::StopAndKill(G4String aString )
{
  if (aString == "KeepOnlyPhotons") {fKeepOnlyPhotons = 1;
    G4cout << " GateSteppingAction Module Message : Only Photons are stored.\n";
  }
  if (aString == "KeepOnlyElectrons") {fKeepOnlyElectrons = 1;
    G4cout << " GateSteppingAction Module Message : Only Electrons are stored.\n";
  }
  if (aString == "StopAndKill") {fStpAKill = fStopAndKill;
    G4cout << " GateSteppingAction Module Message : Once a Particle reaches Phantom Boundaries its Secondaries are Kept Alive\n";
  }

  if (aString == "KillTrackAndSecondaries") {fStpAKill = fKillTrackAndSecondaries;
    G4cout << " GateSteppingAction Module Message : Once a Particle reaches Phantom Boundaries its Secondaries are Killed at the same time\n";
  }

  if ( aString == "KeepAll" ) { fKeepOnlyP = 0;
    G4cout << " GateSteppingAction Module Message : All Particles are stored once they reached the Phantom Boundaries.\n";
  }

  if ( aString == "KeepOnlyPrimaries" ) { fKeepOnlyP = 1;
    G4cout << " GateSteppingAction Module Message : Only Primary Particles are stored once they reached the Phantom Boundaries.\n";
  }
  G4String msg("Off");
  if ( Boundary == 1 )msg = "On";
  G4cout << " GateSteppingAction Module Message : Stop-On-Boundary policy is " <<msg<< Gateendl;


}
void GateSteppingAction::SetMode( TrackingMode aMode)
{
  m_trackingMode = aMode;
}

TrackingMode  GateSteppingAction::GetMode()
{ return m_trackingMode;}

//-----------------------------------------------------------------------------
void GateSteppingAction::UserSteppingAction(const G4Step* theStep)
{
  // GateDebugMessage("Actor", 1, "GateSteppingAction::UserSteppingAction(a)\n");

  static G4int ARFStage = -3;

  if ( ARFStage == -3 )
    {
      GateSPECTHeadSystem* theS = dynamic_cast<GateSPECTHeadSystem*>( GateSystemListManager::GetInstance()->FindSystem("systems/SPECThead") );
      if ( theS !=0 ) ARFStage = theS->GetARFStage();
    }

  G4Track* theTrack = static_cast<G4Track*>( theStep->GetTrack() );

  if ( ARFStage == 0 )
    {
      //G4int eventID = G4EventManager::GetEventManager()->GetNonconstCurrentEvent()->GetEventID();

      static GateARFDataToRoot* ARFDataToRoot = dynamic_cast<GateARFDataToRoot*>( GateOutputMgr::GetInstance()->GetModule("arf") );
      if ( ARFDataToRoot == 0 )
        G4Exception( "GateSteppingAction::UserSteppingAction", "UserSteppingAction", FatalException, "ARF stage is 'generateARFTables' but the output module ARFDataToRoot is not enabled.just add '/gate/output/arf/enable' in your macro. Exiting");
      static G4int IsCountedInCrystal = 0;
      static G4int IsCountedOutCrystal = 0;
      static G4int IsCountedOutCamera = 0;
      static G4int head_number = 0;
      static G4int previous_inCamera = 0;
      static G4int counted_dead_incrystal = 0;
      static G4int counted_dead_inphantom = 0;


      G4bool IsGood = ( theTrack->GetDefinition()->GetPDGEncoding() == 22 ) && ( theTrack->GetParentID() == 0 );
      if ( IsGood  )
        {
          G4int step_number = theTrack->GetCurrentStepNumber();
          if ( step_number == 1  )
            {
              IsCountedInCrystal = 0;
              IsCountedOutCrystal = 0;
              IsCountedOutCamera = 0;
              previous_inCamera = 0;
              ARFDataToRoot->IncrementNbOfSourcePhotons();
              counted_dead_incrystal = 0;
              counted_dead_inphantom = 0;
            }

          G4bool isInsideCamera = false;
          G4bool isInsideCrystal = false;
          G4bool NextisInCrystal = false;
          G4bool isInColli = false;

          G4String theNextLVName("unknown");
          G4String theLVName("unknown");
          G4VPhysicalVolume* currentPV = theTrack->GetVolume();
          G4LogicalVolume* currentLV = 0;


          if ( currentPV != 0 ) currentLV = currentPV->GetLogicalVolume();

          G4VSensitiveDetector* currentSD = 0;
          if ( currentLV != 0 )
            { theLVName = currentLV->GetName();
              G4int pos = theLVName.length() - 2;
              theLVName.erase(pos,2);
              currentSD = currentLV->GetSensitiveDetector();
            }

          G4VPhysicalVolume* NextPV = theTrack->GetNextVolume();
          G4LogicalVolume* NextLV = 0;

          if ( NextPV != 0 ) NextLV = NextPV->GetLogicalVolume();

          G4VSensitiveDetector* NextSD = 0;
          if( NextLV != 0 )
            {
              theNextLVName = NextLV->GetName();
              G4int pos = theNextLVName.length() - 2;
              theNextLVName.erase(pos,2);
              NextSD = NextLV->GetSensitiveDetector();

            }
          // is the primary inside the camera ?
          // Get the step-points
          G4StepPoint  *oldStepPoint = theStep->GetPreStepPoint(),
            *newStepPoint = theStep->GetPostStepPoint();

          const G4VProcess*process = newStepPoint->GetProcessDefinedStep();

          //  For all processes except transportation, we select the PostStepPoint volume
          //  For the transportation, we select the PreStepPoint volume
          const G4TouchableHistory* touchable;
          if ( process->GetProcessType() == fTransportation )
            touchable = (const G4TouchableHistory*)(oldStepPoint->GetTouchable() );
          else
            touchable = (const G4TouchableHistory*)(newStepPoint->GetTouchable() );
          GateVolumeID volumeID(touchable);
          isInsideCamera = ( volumeID.GetCreatorDepth("SPECThead") != -1 );


          //if ( isInsideCamera ) {G4cout << " inserter found " << volumeID.GetCreator( volumeID.GetCreatorDepth("SPECThead") )->GetObjectName()<< Gateendl;
          //                      G4cout <<"event ID " << eventID<<"  step # "<<step_number<<"  "<<volumeID<< Gateendl;}

          if( ( isInsideCamera ) && ( previous_inCamera == 0 ) )
            {
              if ( head_number == 0 )
                {
                  GateVVolume* theInserter = GateObjectStore::GetInstance()->FindCreator("SPECThead");
                  head_number = theInserter->GetVolumeNumber();
                  ARFDataToRoot->SetNHeads( head_number );
                }

              ARFDataToRoot->IncrementInCamera();

              previous_inCamera = 1;

              //G4cout <<"event ID " << eventID<<"  step # "<<step_number<<"   going inside camera "<<volumeID<< Gateendl;

            }

          //if( isInsideCamera ) G4cout <<"event ID " << eventID<<"  step #  "<<step_number<<"   inside camera "<<volumeID<< Gateendl;

          G4bool isGoingOutCamera = (!isInsideCamera) && ( previous_inCamera == 1 );

          if ( isGoingOutCamera && (IsCountedOutCamera == 0) )
            { ARFDataToRoot->IncrementOutCamera();
              IsCountedOutCamera = 1;
              //G4cout <<"event ID " << eventID<<"  step # "<<step_number<<"   going outside camera "<<volumeID<< Gateendl;
              //theTrack->SetTrackStatus(fStopAndKill);
            }

          if ( currentSD != 0 )
            {
              isInsideCrystal = (currentSD->GetName() == "crystal");
              isInColli = (currentSD->GetName() == "phantom");
            }
          G4bool track_dead = (theTrack->GetTrackStatus() == fStopAndKill);

          if ( isInsideCrystal && track_dead && counted_dead_incrystal == 0)
            {
              ARFDataToRoot->IncrementKilledInsideCrystalPhotons();
              counted_dead_incrystal = 1;
            }

          if ( isInColli &&  track_dead && counted_dead_inphantom == 0)
            {
              ARFDataToRoot->IncrementKilledInsideColliPhotons();
              counted_dead_inphantom = 1;
            }

          if ( NextSD != 0 )  NextisInCrystal = (NextSD->GetName() == "crystal");

          G4bool isOutCrystal = !isInsideCrystal;
          G4bool NextisOutCrystal = !NextisInCrystal;

          if( isInsideCamera && track_dead) {ARFDataToRoot->IncrementKilledInsideCamera();
            //G4cout <<"  step # "<<step_number<<"   killed inside camera "<<volumeID<< Gateendl;
          }

          G4bool isGoingOutCrystal  = isInsideCrystal && NextisOutCrystal;
          G4bool isGoingInCrystal   = isOutCrystal && NextisInCrystal;
          if ( isGoingInCrystal && IsCountedInCrystal == 0)
            { ARFDataToRoot->IncrementGoingInPhotons();
              IsCountedInCrystal = 1;
            }
          if ( isGoingOutCrystal && IsCountedOutCrystal == 0)
            { ARFDataToRoot->IncrementGoingOutPhotons();
              IsCountedOutCrystal = 1;
            }
        }
    } // generate ARF - Data PY Descourt 08/09/2008

  if (m_trackingMode == TrackingMode::kTracker )
    {
      G4int EventID = G4EventManager::GetEventManager()->GetNonconstCurrentEvent()->GetEventID();
      G4int RunID   = GateRunManager::GetRunManager()->GetCurrentRun()->GetRunID();
      G4Track * fTrack = theStep->GetTrack();
      G4int ParentID  =  fTrack->GetParentID();
      G4int TrackID = fTrack->GetTrackID();
      G4StepPoint * thePPoint;
      G4String particleName = "unknown";
      if ( fTrack->GetDefinition() != 0 )
        particleName =  fTrack->GetDefinition()->GetParticleName();
      G4String NxtVol = G4String("OutOfWorld");
      if( fTrack->GetNextVolume() != 0 )
        {
          NxtVol = (G4String)  (fTrack->GetNextVolume()->GetName()) ;
        }


      //	G4cout<<"Event ID " << EventID<< Gateendl;
      //G4cout << " Track ID " << TrackID<<" Parent ID " << ParentID<< Gateendl;
      //G4cout<<" particle " << particleName << Gateendl;
      //G4cout<<" step number = " <<fTrack->GetCurrentStepNumber()<< Gateendl;


      if ( fTrack->GetCurrentStepNumber() == 1 ) // initialize for current Track
        {
          fStartVolumeIsPhantomSD = false;
          fKillNextIsSet = false;
          thePPoint = theStep->GetPreStepPoint();
          //
          //// check if the volume at vertex is a sensitive detector of phantom type
          //
          const    G4LogicalVolume* LVolAtVertex = fTrack->GetLogicalVolumeAtVertex();
          if ( LVolAtVertex != 0 )
            {
              G4VSensitiveDetector* SDetector = LVolAtVertex->GetSensitiveDetector();
              if ( SDetector != 0 )
                {
                  //
                  //// get the pointer to the Phantom SD instanciated in GateDetectorConstruction
                  //
                  G4VSensitiveDetector* GPhantomSD = (G4VSensitiveDetector*) ( GateDetectorConstruction::GetGateDetectorConstruction()->GetPhantomSD() );
                  if ( SDetector == GPhantomSD ){ fStartVolumeIsPhantomSD = true; }  // the volume where the particle was created is indeed a phantom !
                }
            }

          //
          ////// check if the mother volume is a phantom volume. this is the case for parameterized phantom
          //
          G4VPhysicalVolume* PVol = fTrack->GetVolume();
          if ( PVol != 0 )
            {
              // get the mother logical volume
              G4LogicalVolume* Mother = PVol->GetMotherLogical();
              if ( Mother != 0 )
                {
                  G4VSensitiveDetector* SDetector = Mother->GetSensitiveDetector();
                  if ( SDetector != 0 )
                    {
                      //
                      //// get the pointer to the Phantom SD instanciated in GateDetectorConstruction
                      //
                      G4VSensitiveDetector* GPhantomSD = (G4VSensitiveDetector*) ( GateDetectorConstruction::GetGateDetectorConstruction()->GetPhantomSD() );
                      if ( SDetector == GPhantomSD ){ fStartVolumeIsPhantomSD = true; }  // the volume where the particle was created is indeed a phantom !
                    }
                }
            }
          //////
          ////
          // store the volume name where the particle is created
          m_StartingVolName = G4String("OutOfWorld");
          if ( thePPoint->GetPhysicalVolume() != 0 ){ m_StartingVolName = (G4String) (thePPoint->GetPhysicalVolume()->GetName());}
        }
      //
      //// lets analyze only particles starting in a phantom type sensitive detector
      //
      if ( fStartVolumeIsPhantomSD == true ) // the particle was created in a Phantom type Sensitive Detector !
        {
          ////
          //
          // determine next volume and compare with the starting one to know if it is going out or not
          G4String NxtVol = G4String("OutOfWorld");
          if( fTrack->GetNextVolume() != 0 )
            {
              NxtVol = (G4String)  (fTrack->GetNextVolume()->GetName()) ;
            }
          G4bool test_ID = 1;
          G4String particleName = G4String("UnKnown");
          if( fTrack->GetDefinition() != 0 )
            {
              particleName =  fTrack->GetDefinition()->GetParticleName();
              if( fKeepOnlyPhotons == 1 )   test_ID = ( fTrack->GetDefinition()->GetPDGEncoding() == 22 );
              if( fKeepOnlyElectrons == 1 ) test_ID = ( fTrack->GetDefinition()->GetPDGEncoding() == 11 );
            }
          if( fKeepOnlyP == 1 ) test_ID = ( ParentID == 0 );

          test_ID = test_ID && ( fTrack->GetTotalEnergy() - m_energyThreshold >= 0. );


          if ( NxtVol  != m_StartingVolName ) // it means: At next step we go out of the phantom so we have to kill it now or next step depending on policy!
            {
              if ( test_ID == 1 ) // test if current particle has required ID datas
                {
                  if ( Boundary == 1) // here we kill particle at phantom boundary : so we kill it NOW !!!!!!
                    {
                      fTrack->SetTrackStatus(fStpAKill);
                      GateTrack* aTrack = new GateTrack();
                      aTrack->Fill_Track( fTrack  );
                      aTrack->SetTime( GateSourceMgr::GetInstance()->GetTime() );
                      aTrack->SetEventID(EventID);
                      aTrack->SetRunID(RunID);
                      aTrack->SetWasKilled(0);
                      aTrack->SetSourceID(  GateSourceMgr::GetInstance()->GetCurrentSourceID() );
                      PPTrackVector->push_back( aTrack );
                      if ( m_verboseLevel > 3  )
                        {
                          const G4Event* currentEvent = G4EventManager::GetEventManager()->GetConstCurrentEvent();
                          std::ofstream outFile;
                          G4String outF = "GoingOutParticles.txt";
                          outFile.open (outF ,std::ios::app);
                          outFile << " Event ID : " << currentEvent->GetEventID() << "         A " << particleName << " Primary Particle is detected";
                          outFile << " going out of the Phantom Parent ID " << ParentID <<"  track ID "<< TrackID << Gateendl;
                          ShowG4TrackInfos(outF, fTrack);
                        } // verbose end block
                    }

                  if ( Boundary == 0) // boundary == 0 so we kill particle after they passed phantom boundary
                    // so we WAIT one more step after phantom boundary !!!!!
                    {
                      if (   fKillNextIsSet == true  ) // the step number at which we just got out the phantom IS the current step number !!
                        {
                          if ( m_verboseLevel > 3 )
                            {
                              const G4Event* currentEvent = G4EventManager::GetEventManager()->GetConstCurrentEvent();
                              std::ofstream outFile;
                              G4String outF = "KilledAfterBoundaryParticles.txt";
                              outFile.open (outF ,std::ios::app);
                              outFile << " Event ID : " << currentEvent->GetEventID() << "         A " << particleName << " Primary Particle is detected";
                              outFile << " going out of the Phantom Parent ID " << ParentID <<"  track ID "<< TrackID << Gateendl;
                              ShowG4TrackInfos(outF, fTrack);
                            } // verbose end block
                          ///// We check that the next volume is not "OutOfWorld". If so NO NEED to save it !
                          G4String NxtVol = G4String("OutOfWorld");
                          if( fTrack->GetNextVolume() != 0 )
                            { NxtVol = (G4String)  (fTrack->GetNextVolume()->GetName()) ;}
                          if (   NxtVol != G4String( "OutOfWorld" )    )
                            {
                              fTrack->SetTrackStatus(fStpAKill);
                              GateTrack* aTrack = new GateTrack();
                              aTrack->Fill_Track( fTrack  );
                              aTrack->SetTime( GateSourceMgr::GetInstance()->GetTime() );
                              aTrack->SetEventID(EventID);
                              aTrack->SetRunID(RunID);
                              aTrack->SetWasKilled(0);
                              aTrack->SetSourceID(  GateSourceMgr::GetInstance()->GetCurrentSourceID() );
                              PPTrackVector->push_back( aTrack );
                            }
                        }
                      else {
                        fKillNextIsSet = true;  // we go for one more step and kill particle !
                      }
                    }  //  boundary == 0 end if  block
                  if (m_verboseLevel>0 )
                    {
                      std::ofstream outFile;
                      G4String outF = "GoingOutParticles.txt";
                      outFile.open ( outF ,std::ios::app);
                      const G4Event* currentEvent = G4EventManager::GetEventManager()->GetConstCurrentEvent();
                      outFile << " Event ID : " << currentEvent->GetEventID() << "         A " << particleName << " Particle is detected going out of the Phantom Parent ID " << ParentID <<"  track ID "<< TrackID << Gateendl;
                      if ( m_verboseLevel>3 )
                        {
                          ShowG4TrackInfos(outF, fTrack);
                        }
                    } // verbose end block
                } // test on particle IDs
              else
                { // particle goes out of the phantom but with wrong IDs so kill it to avoid it hit the detectors !
                  fTrack->SetTrackStatus( fStpAKill );
                }

            } // test next volume name is different end if block
          if ( NxtVol  == m_StartingVolName )   // here we are still in the same volume for next step
            {
              if ( test_ID == 1 ) // test if current particle has required ID datas
                {                 //// ok  check if it is killed in the phantom !!!!!
                  if ( fTrack->GetTrackStatus() == fStopAndKill )
                    {
                      //// in that case we also keep it because it is needed in GateTrajectoryNavigator methods
                      GateTrack* aTrack = new GateTrack();
                      aTrack->Fill_Track( fTrack  );
                      aTrack->SetTime( GateSourceMgr::GetInstance()->GetTime() );
                      aTrack->SetEventID(EventID);
                      aTrack->SetRunID(RunID);
                      aTrack->SetWasKilled(1);  /// the primary particle was killed during stepping inside the phantom !
                      aTrack->SetSourceID(  GateSourceMgr::GetInstance()->GetCurrentSourceID() );
                      PPTrackVector->push_back( aTrack );
                      if ( m_verboseLevel > 0 )
                        {
                          const G4Event* currentEvent = G4EventManager::GetEventManager()->GetConstCurrentEvent();
                          std::ofstream outFile;
                          G4String outF = "KilledInsideParticles.txt";
                          outFile.open ( outF ,std::ios::app);
                          outFile << " Event ID : " << currentEvent->GetEventID() << "         A " << particleName << " Primary Particle is killed inside the Phantom Parent ID "                                 << ParentID <<"  track ID "<< TrackID << Gateendl;
                          if ( m_verboseLevel > 3 )
                            {
                              ShowG4TrackInfos(outF, fTrack);
                            }
                        } // verbose end block
                    }
                } // test the particle IDs
            } //  case where we are still in the phantom for next step
        } // test on the Sensitive Detector Type
      else  // the particle was not created in a phantom so kill it to avoid it to hit detectors !
        {
          fTrack->SetTrackStatus( fStopAndKill );
        }
    } // Tracker Mode


  // GateDebugMessage("Actor", 1, "GateSteppingAction::UserSteppingAction à la fin\n");

#ifdef G4ANALYSIS_USE_GENERAL
  // Here we fill the histograms of the OutputMgr manager
  // Pre-digitalisation outputMgr (hits)
  GateOutputMgr* outputMgr = GateOutputMgr::GetInstance();
  outputMgr->RecordStepWithVolume(0,theStep);
#endif

  pCallbackMan->UserSteppingAction(theStep);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateSteppingAction::UserSteppingAction(const GateVVolume *v, const G4Step* theStep)
{
  // GateDebugMessage("Actor", 2 , "GateSteppingAction::UserSteppingAction(v,theStep)\n");



  G4SteppingManager* SM = fpSteppingManager;
  //! visualization

#ifdef G4ANALYSIS_USE_GENERAL
  // Here we fill the histograms of the Analysis manager
  if(GateApplicationMgr::GetInstance()->GetOutputMode()){
    GateOutputMgr* outputMgr = GateOutputMgr::GetInstance();
    outputMgr->RecordStepWithVolume(v, theStep);
  }
#endif

  G4bool drawTrj = false;
  if (m_drawTrjLevel == 0) {
  } else if (m_drawTrjLevel == 1) {
    G4int currentEvent = GateRunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
    if (currentEvent <= 10) {
      drawTrj = true;
    }
  } else if (m_drawTrjLevel >= 2) {
    drawTrj = true;
  }

  if(drawTrj) {
    G4VVisManager* pVVisManager = G4VVisManager::GetConcreteInstance();

    if (pVVisManager) {
      G4Polyline polyline;

      G4double  charge = SM->GetTrack()->GetDefinition()->GetPDGCharge();
      G4Colour  colour;
      //       G4Colour  white   (1.0, 1.0, 1.0);   // white
      //       G4Colour  grey    (0.5, 0.5, 0.5);   // grey
      //       G4Colour  black   (0.0, 0.0, 0.0);   // black
      G4Colour  red     (1.0, 0.0, 0.0);   // red
      G4Colour  green   (0.0, 1.0, 0.0);   // green
      G4Colour  blue    (0.0, 0.0, 1.0);   // blue
      //       G4Colour  cyan    (0.0, 1.0, 1.0);   // cyan
      //       G4Colour  magenta (1.0, 0.0, 1.0);   // magenta
      G4Colour  yellow  (1.0, 1.0, 0.0);   // yellow
      if      (charge < 0.) colour = red;
      else if (charge > 0.) colour = blue;
      else                  colour = green;
      G4VisAttributes attribs(colour);
      if (SM->GetTrack()->GetDefinition() == G4NeutrinoE::NeutrinoEDefinition()) {
	attribs.SetColour(yellow);
	attribs.SetLineStyle(G4VisAttributes::dashed);
      }
      polyline.SetVisAttributes(attribs);
      polyline.push_back(SM->GetStep()->GetPreStepPoint()->GetPosition());
      polyline.push_back(SM->GetStep()->GetPostStepPoint()->GetPosition());
      pVVisManager -> Draw(polyline);
    }
  }

  static G4int ARFStage = -3;

  if ( ARFStage == -3 )
    {
      GateSPECTHeadSystem* theS = dynamic_cast<GateSPECTHeadSystem*>( GateSystemListManager::GetInstance()->FindSystem("systems/SPECThead") );
      if ( theS !=0 ) ARFStage = theS->GetARFStage();
    }

  G4Track* theTrack = static_cast<G4Track*>( theStep->GetTrack() );


  G4cout << " ARFStage = " <<ARFStage << Gateendl;

  if ( ARFStage == 0 )
    {
      G4int eventID = G4EventManager::GetEventManager()->GetNonconstCurrentEvent()->GetEventID();

      static GateARFDataToRoot* ARFDataToRoot = dynamic_cast<GateARFDataToRoot*>( GateOutputMgr::GetInstance()->GetModule("arf") );
      if ( ARFDataToRoot == 0 )
        G4Exception( "GateSteppingAction::UserSteppingAction", "UserSteppingAction", FatalException, "ARF stage is 'generateARFTables' but the output module ARFDataToRoot is not enabled.just add '/gate/output/arf/enable' in your macro. Exiting");
      static G4int IsCountedInCrystal = 0;
      static G4int IsCountedOutCrystal = 0;
      static G4int IsCountedOutCamera = 0;
      static G4int head_number = 0;
      static G4int previous_inCamera = 0;
      static G4int counted_dead_incrystal = 0;
      static G4int counted_dead_inphantom = 0;


      G4bool IsGood = ( theTrack->GetDefinition()->GetPDGEncoding() == 22 ) && ( theTrack->GetParentID() == 0 );
      if ( IsGood  )
        {
          G4int step_number = theTrack->GetCurrentStepNumber();
          if ( step_number == 1  )
            {
              IsCountedInCrystal = 0;
              IsCountedOutCrystal = 0;
              IsCountedOutCamera = 0;
              previous_inCamera = 0;
              ARFDataToRoot->IncrementNbOfSourcePhotons();
              counted_dead_incrystal = 0;
              counted_dead_inphantom = 0;
            }

          G4bool isInsideCamera = false;
          G4bool isInsideCrystal = false;
          G4bool NextisInCrystal = false;
          G4bool isInColli = false;

          G4String theNextLVName("unknown");
          G4String theLVName("unknown");
          G4VPhysicalVolume* currentPV = theTrack->GetVolume();
          G4LogicalVolume* currentLV = 0;


          if ( currentPV != 0 ) currentLV = currentPV->GetLogicalVolume();

          G4VSensitiveDetector* currentSD = 0;
          if ( currentLV != 0 )
            { theLVName = currentLV->GetName();
              G4int pos = theLVName.length() - 2;
              theLVName.erase(pos,2);
              currentSD = currentLV->GetSensitiveDetector();
            }

          G4VPhysicalVolume* NextPV = theTrack->GetNextVolume();
          G4LogicalVolume* NextLV = 0;

          if ( NextPV != 0 ) NextLV = NextPV->GetLogicalVolume();

          G4VSensitiveDetector* NextSD = 0;
          if( NextLV != 0 )
            {
              theNextLVName = NextLV->GetName();
              G4int pos = theNextLVName.length() - 2;
              theNextLVName.erase(pos,2);
              NextSD = NextLV->GetSensitiveDetector();

            }
          // is the primary inside the camera ?
          // Get the step-points
          G4StepPoint  *oldStepPoint = theStep->GetPreStepPoint(),
            *newStepPoint = theStep->GetPostStepPoint();

          const G4VProcess*process = newStepPoint->GetProcessDefinedStep();

          //  For all processes except transportation, we select the PostStepPoint volume
          //  For the transportation, we select the PreStepPoint volume
          const G4TouchableHistory* touchable;
          if ( process->GetProcessType() == fTransportation )
            touchable = (const G4TouchableHistory*)(oldStepPoint->GetTouchable() );
          else
            touchable = (const G4TouchableHistory*)(newStepPoint->GetTouchable() );
          GateVolumeID volumeID(touchable);
          isInsideCamera = ( volumeID.GetCreatorDepth("SPECThead") != -1 );

          // if ( isInsideCamera ) G4cout << " inserter found " << volumeID.GetInserter( volumeID.GetInserterDepth("SPECThead") )->GetObjectName()<< Gateendl;

          G4cout <<"event ID " << eventID<<"  step # "<<step_number<<"  "<<volumeID<< Gateendl;

          if( ( isInsideCamera ) && ( previous_inCamera == 0 ) )
            {
              if ( head_number == 0 )
                {
                  GateVVolume* theInserter = GateObjectStore::GetInstance()->FindCreator("SPECThead");
                  head_number = theInserter->GetVolumeNumber();
                  ARFDataToRoot->SetNHeads( head_number );
                }

              ARFDataToRoot->IncrementInCamera();

              previous_inCamera = 1;

              //G4cout <<"event ID " << eventID<<"  step # "<<step_number<<"   going inside camera "<<volumeID<< Gateendl;

            }

          //if( isInsideCamera ) G4cout <<"event ID " << eventID<<"  step #  "<<step_number<<"   inside camera "<<volumeID<< Gateendl;

          G4bool isGoingOutCamera = (!isInsideCamera) && ( previous_inCamera == 1 );

          if ( isGoingOutCamera && (IsCountedOutCamera == 0) )
            { ARFDataToRoot->IncrementOutCamera();
              IsCountedOutCamera = 1;
              //G4cout <<"event ID " << eventID<<"  step # "<<step_number<<"   going outside camera "<<volumeID<< Gateendl;
              //theTrack->SetTrackStatus(fStopAndKill);
            }

          if ( currentSD != 0 )
            {
              isInsideCrystal = (currentSD->GetName() == "crystal");
              isInColli = (currentSD->GetName() == "phantom");
            }
          G4bool track_dead = (theTrack->GetTrackStatus() == fStopAndKill);

          if ( isInsideCrystal && track_dead && counted_dead_incrystal == 0)
            {
              ARFDataToRoot->IncrementKilledInsideCrystalPhotons();
              counted_dead_incrystal = 1;
            }

          if ( isInColli &&  track_dead && counted_dead_inphantom == 0)
            {
              ARFDataToRoot->IncrementKilledInsideColliPhotons();
              counted_dead_inphantom = 1;
            }

          if ( NextSD != 0 )  NextisInCrystal = (NextSD->GetName() == "crystal");

          G4bool isOutCrystal = !isInsideCrystal;
          G4bool NextisOutCrystal = !NextisInCrystal;

          if( isInsideCamera && track_dead) {ARFDataToRoot->IncrementKilledInsideCamera();
            //G4cout <<"  step # "<<step_number<<"   killed inside camera "<<volumeID<< Gateendl;
          }

          G4bool isGoingOutCrystal  = isInsideCrystal && NextisOutCrystal;
          G4bool isGoingInCrystal   = isOutCrystal && NextisInCrystal;
          if ( isGoingInCrystal && IsCountedInCrystal == 0)
            { ARFDataToRoot->IncrementGoingInPhotons();
              IsCountedInCrystal = 1;
            }
          if ( isGoingOutCrystal && IsCountedOutCrystal == 0)
            { ARFDataToRoot->IncrementGoingOutPhotons();
              IsCountedOutCrystal = 1;
            }
        }
    } // generate ARF - Data PY Descourt 11/12/2008

  if (m_trackingMode == TrackingMode::kTracker )
    {
      G4int EventID = G4EventManager::GetEventManager()->GetNonconstCurrentEvent()->GetEventID();
      G4int RunID   = GateRunManager::GetRunManager()->GetCurrentRun()->GetRunID();
      G4Track * fTrack = theStep->GetTrack();
      G4int ParentID  =  fTrack->GetParentID();
      G4int TrackID = fTrack->GetTrackID();
      G4StepPoint * thePPoint;
      G4String particleName = "unknown";
      if ( fTrack->GetDefinition() != 0 )
        particleName =  fTrack->GetDefinition()->GetParticleName();
      G4String NxtVol = G4String("OutOfWorld");
      if( fTrack->GetNextVolume() != 0 )
        {
          NxtVol = (G4String)  (fTrack->GetNextVolume()->GetName()) ;
        }


      //	G4cout<<"Event ID " << EventID<< Gateendl;
      //G4cout << " Track ID " << TrackID<<" Parent ID " << ParentID<< Gateendl;
      //G4cout<<" particle " << particleName << Gateendl;
      //G4cout<<" step number = " <<fTrack->GetCurrentStepNumber()<< Gateendl;


      if ( fTrack->GetCurrentStepNumber() == 1 ) // initialize for current Track
        {
          fStartVolumeIsPhantomSD = false;
          fKillNextIsSet = false;
          thePPoint = theStep->GetPreStepPoint();
          //
          //// check if the volume at vertex is a sensitive detector of phantom type
          //
          const    G4LogicalVolume* LVolAtVertex = fTrack->GetLogicalVolumeAtVertex();
          if ( LVolAtVertex != 0 )
            {
              G4VSensitiveDetector* SDetector = LVolAtVertex->GetSensitiveDetector();
              if ( SDetector != 0 )
                {
                  //
                  //// get the pointer to the Phantom SD instanciated in GateDetectorConstruction
                  //
                  G4VSensitiveDetector* GPhantomSD = (G4VSensitiveDetector*) ( GateDetectorConstruction::GetGateDetectorConstruction()->GetPhantomSD() );
                  if ( SDetector == GPhantomSD ){ fStartVolumeIsPhantomSD = true; }  // the volume where the particle was created is indeed a phantom !
                }
            }

          //
          ////// check if the mother volume is a phantom volume. this is the case for parameterized phantom
          //
          G4VPhysicalVolume* PVol = fTrack->GetVolume();
          if ( PVol != 0 )
            {
              // get the mother logical volume
              G4LogicalVolume* Mother = PVol->GetMotherLogical();
              if ( Mother != 0 )
                {
                  G4VSensitiveDetector* SDetector = Mother->GetSensitiveDetector();
                  if ( SDetector != 0 )
                    {
                      //
                      //// get the pointer to the Phantom SD instanciated in GateDetectorConstruction
                      //
                      G4VSensitiveDetector* GPhantomSD = (G4VSensitiveDetector*) ( GateDetectorConstruction::GetGateDetectorConstruction()->GetPhantomSD() );
                      if ( SDetector == GPhantomSD ){ fStartVolumeIsPhantomSD = true; }  // the volume where the particle was created is indeed a phantom !
                    }
                }
            }
          //////
          ////
          // store the volume name where the particle is created
          m_StartingVolName = G4String("OutOfWorld");
          if ( thePPoint->GetPhysicalVolume() != 0 ){ m_StartingVolName = (G4String) (thePPoint->GetPhysicalVolume()->GetName());}
        }
      //
      //// lets analyze only particles starting in a phantom type sensitive detector
      //
      if ( fStartVolumeIsPhantomSD == true ) // the particle was created in a Phantom type Sensitive Detector !
        {
          ////
          //
          // determine next volume and compare with the starting one to know if it is going out or not
          G4String NxtVol = G4String("OutOfWorld");
          if( fTrack->GetNextVolume() != 0 )
            {
              NxtVol = (G4String)  (fTrack->GetNextVolume()->GetName()) ;
            }
          G4bool test_ID = 1;
          G4String particleName = G4String("UnKnown");
          if( fTrack->GetDefinition() != 0 )
            {
              particleName =  fTrack->GetDefinition()->GetParticleName();
              if( fKeepOnlyPhotons == 1 )   test_ID = ( fTrack->GetDefinition()->GetPDGEncoding() == 22 );
              if( fKeepOnlyElectrons == 1 ) test_ID = ( fTrack->GetDefinition()->GetPDGEncoding() == 11 );
            }
          if( fKeepOnlyP == 1 ) test_ID = ( ParentID == 0 );

          test_ID = test_ID && ( fTrack->GetTotalEnergy() - m_energyThreshold >= 0. );


          if ( NxtVol  != m_StartingVolName ) // it means: At next step we go out of the phantom so we have to kill it now or next step depending on policy!
            {
              if ( test_ID == 1 ) // test if current particle has required ID datas
                {
                  if ( Boundary == 1) // here we kill particle at phantom boundary : so we kill it NOW !!!!!!
                    {
                      fTrack->SetTrackStatus(fStpAKill);
                      GateTrack* aTrack = new GateTrack();
                      aTrack->Fill_Track( fTrack  );
                      aTrack->SetTime( GateSourceMgr::GetInstance()->GetTime() );
                      aTrack->SetEventID(EventID);
                      aTrack->SetRunID(RunID);
                      aTrack->SetWasKilled(0);
                      aTrack->SetSourceID(  GateSourceMgr::GetInstance()->GetCurrentSourceID() );
                      PPTrackVector->push_back( aTrack );
                      if ( m_verboseLevel > 3  )
                        {
                          const G4Event* currentEvent = G4EventManager::GetEventManager()->GetConstCurrentEvent();
                          std::ofstream outFile;
                          G4String outF = "GoingOutParticles.txt";
                          outFile.open (outF ,std::ios::app);
                          outFile << " Event ID : " << currentEvent->GetEventID() << "         A " << particleName << " Primary Particle is detected";
                          outFile << " going out of the Phantom Parent ID " << ParentID <<"  track ID "<< TrackID << Gateendl;
                          ShowG4TrackInfos(outF, fTrack);
                        } // verbose end block
                    }

                  if ( Boundary == 0) // boundary == 0 so we kill particle after they passed phantom boundary
                    // so we WAIT one more step after phantom boundary !!!!!
                    {
                      if (   fKillNextIsSet == true  ) // the step number at which we just got out the phantom IS the current step number !!
                        {
                          if ( m_verboseLevel > 3 )
                            {
                              const G4Event* currentEvent = G4EventManager::GetEventManager()->GetConstCurrentEvent();
                              std::ofstream outFile;
                              G4String outF = "KilledAfterBoundaryParticles.txt";
                              outFile.open (outF ,std::ios::app);
                              outFile << " Event ID : " << currentEvent->GetEventID() << "         A " << particleName << " Primary Particle is detected";
                              outFile << " going out of the Phantom Parent ID " << ParentID <<"  track ID "<< TrackID << Gateendl;
                              ShowG4TrackInfos(outF, fTrack);
                            } // verbose end block
                          ///// We check that the next volume is not "OutOfWorld". If so NO NEED to save it !
                          G4String NxtVol = G4String("OutOfWorld");
                          if( fTrack->GetNextVolume() != 0 )
                            { NxtVol = (G4String)  (fTrack->GetNextVolume()->GetName()) ;}
                          if (   NxtVol != G4String( "OutOfWorld" )    )
                            {
                              fTrack->SetTrackStatus(fStpAKill);
                              GateTrack* aTrack = new GateTrack();
                              aTrack->Fill_Track( fTrack  );
                              aTrack->SetTime( GateSourceMgr::GetInstance()->GetTime() );
                              aTrack->SetEventID(EventID);
                              aTrack->SetRunID(RunID);
                              aTrack->SetWasKilled(0);
                              aTrack->SetSourceID(  GateSourceMgr::GetInstance()->GetCurrentSourceID() );
                              PPTrackVector->push_back( aTrack );
                            }
                        }
                      else {
                        fKillNextIsSet = true;  // we go for one more step and kill particle !
                      }
                    }  //  boundary == 0 end if  block
                  if (m_verboseLevel>0 )
                    {
                      std::ofstream outFile;
                      G4String outF = "GoingOutParticles.txt";
                      outFile.open ( outF ,std::ios::app);
                      const G4Event* currentEvent = G4EventManager::GetEventManager()->GetConstCurrentEvent();
                      outFile << " Event ID : " << currentEvent->GetEventID() << "         A " << particleName << " Particle is detected going out of the Phantom Parent ID " << ParentID <<"  track ID "<< TrackID << Gateendl;
                      if ( m_verboseLevel>3 )
                        {
                          ShowG4TrackInfos(outF, fTrack);
                        }
                    } // verbose end block
                } // test on particle IDs
              else
                { // particle goes out of the phantom but with wrong IDs so kill it to avoid it hit the detectors !
                  fTrack->SetTrackStatus( fStpAKill );
                }

            } // test next volume name is different end if block
          if ( NxtVol  == m_StartingVolName )   // here we are still in the same volume for next step
            {
              if ( test_ID == 1 ) // test if current particle has required ID datas
                {                 //// ok  check if it is killed in the phantom !!!!!
                  if ( fTrack->GetTrackStatus() == fStopAndKill )
                    {
                      //// in that case we also keep it because it is needed in GateTrajectoryNavigator methods
                      GateTrack* aTrack = new GateTrack();
                      aTrack->Fill_Track( fTrack  );
                      aTrack->SetTime( GateSourceMgr::GetInstance()->GetTime() );
                      aTrack->SetEventID(EventID);
                      aTrack->SetRunID(RunID);
                      aTrack->SetWasKilled(1);  /// the primary particle was killed during stepping inside the phantom !
                      aTrack->SetSourceID(  GateSourceMgr::GetInstance()->GetCurrentSourceID() );
                      PPTrackVector->push_back( aTrack );
                      if ( m_verboseLevel > 0 )
                        {
                          const G4Event* currentEvent = G4EventManager::GetEventManager()->GetConstCurrentEvent();
                          std::ofstream outFile;
                          G4String outF = "KilledInsideParticles.txt";
                          outFile.open ( outF ,std::ios::app);
                          outFile << " Event ID : " << currentEvent->GetEventID() << "         A " << particleName << " Primary Particle is killed inside the Phantom Parent ID "                                 << ParentID <<"  track ID "<< TrackID << Gateendl;
                          if ( m_verboseLevel > 3 )
                            {
                              ShowG4TrackInfos(outF, fTrack);
                            }
                        } // verbose end block
                    }
                } // test the particle IDs
            } //  case where we are still in the phantom for next step
        } // test on the Sensitive Detector Type
      else  // the particle was not created in a phantom so kill it to avoid it to hit detectors !
        {
          fTrack->SetTrackStatus( fStopAndKill );
        }
    } // Tracker Mode


  // GateDebugMessage("Actor", 1, "GateSteppingAction::UserSteppingAction à la fin\n");


  pCallbackMan->UserSteppingAction(theStep);

}

GateSteppingAction::~GateSteppingAction()
{
  for (std::vector<GateTrack*>::iterator it = PPTrackVector->begin(); it != PPTrackVector->end(); )
    {
      delete (*it);
      it = PPTrackVector->erase(it);
    }
  delete PPTrackVector;
  delete m_steppingMessenger;
}


void GateSteppingAction::ShowG4TrackInfos( G4String outF, G4Track* fTrack)
{
  std::ofstream outFile;
  outFile.open ( outF ,std::ios::app);
  outFile << "      -----------------------------------------------\n";
  outFile << "        G4Track Information  " << std::setw(20) << Gateendl;
  outFile << "      -----------------------------------------------\n";
  outFile << "     Particle Name          : " << fTrack->GetDefinition()->GetParticleName()<< Gateendl;
  outFile << "        Current Volume         : "
          << std::setw(20);
  if( fTrack->GetVolume() != 0 ) {
    outFile << fTrack->GetVolume()->GetName() << " ";
  } else {
    outFile << "OutOfWorld" << " ";
  }
  outFile << Gateendl;
  outFile << "        Step number         : " << std::setw(20) << fTrack->GetCurrentStepNumber()<< Gateendl;
#ifdef G4_USE_G4BESTUNIT_FOR_VERBOSE
  outFile << "        Position - x        : "
          << std::setw(20) << G4BestUnit(fTrack->GetPosition().x(), "Length")
          << Gateendl;
  outFile << "        Position - y        : "
          << std::setw(20) << G4BestUnit(fTrack->GetPosition().y(), "Length")
          << Gateendl;
  outFile << "        Position - z        : "
          << std::setw(20) << G4BestUnit(fTrack->GetPosition().z(), "Length")
          << Gateendl;
  outFile << "        Global Time         : "
          << std::setw(20) << G4BestUnit(fTrack->GetGlobalTime(), "Time")
          << Gateendl;
  outFile << "        Local Time          : "
          << std::setw(20) << G4BestUnit(fTrack->GetLocalTime(), "Time")
          << Gateendl;
#else
  outFile << "        Position - x (mm)   : "
          << std::setw(20) << fTrack->GetPosition().x() /mm
          << Gateendl;
  outFile << "        Position - y (mm)   : "
          << std::setw(20) << fTrack->GetPosition().y() /mm
          << Gateendl;
  outFile << "        Position - z (mm)   : "
          << std::setw(20) << fTrack->GetPosition().z() /mm
          << Gateendl;
  outFile << "        Global Time (ns)    : "
          << std::setw(20) << fTrack->GetGlobalTime() /ns
          << Gateendl;
  outFile << "        Local Time (ns)     : "
          << std::setw(20) << fTrack->GetLocalTime() /ns
          << Gateendl;
#endif
  outFile << "        Momentum Direct - x : "
          << std::setw(20) << fTrack->GetMomentumDirection().x()
          << Gateendl;
  outFile << "        Momentum Direct - y : "
          << std::setw(20) << fTrack->GetMomentumDirection().y()
          << Gateendl;
  outFile << "        Momentum Direct - z : "
          << std::setw(20) << fTrack->GetMomentumDirection().z()
          << Gateendl;
#ifdef G4_USE_G4BESTUNIT_FOR_VERBOSE
  outFile << "        Kinetic Energy      : "
#else
    outFile << "        Kinetic Energy (MeV): "
#endif
          << std::setw(20) << G4BestUnit(fTrack->GetKineticEnergy(), "Energy")
          << Gateendl;

  outFile << "        Polarization - x    : "
          << std::setw(20) << fTrack->GetPolarization().x()
          << Gateendl;
  outFile << "        Polarization - y    : "
          << std::setw(20) << fTrack->GetPolarization().y()
          << Gateendl;
  outFile << "        Polarization - z    : "
          << std::setw(20) << fTrack->GetPolarization().z()
          << Gateendl;
  outFile << "        Track Length        : "
          << std::setw(20) << G4BestUnit(fTrack->GetTrackLength(), "Length")
          << Gateendl;
  outFile << "        Track ID #          : "
          << std::setw(20) << fTrack->GetTrackID()
          << Gateendl;
  outFile << "        Parent Track ID #   : "
          << std::setw(20) << fTrack->GetParentID()
          << Gateendl;
  outFile << "        Next Volume         : "
          << std::setw(20);
  if( fTrack->GetNextVolume() != 0 ) {
    outFile << fTrack->GetNextVolume()->GetName() << " ";
  } else {
    outFile << "OutOfWorld" << " ";
  }
  outFile << Gateendl;
  outFile << "        Track Status        : "
          << std::setw(20);
  if( fTrack->GetTrackStatus() == fAlive ){
    outFile << " Alive";
  } else if( fTrack->GetTrackStatus() == fStopButAlive ){
    outFile << " StopButAlive";
  } else if( fTrack->GetTrackStatus() == fStopAndKill ){
    outFile << " StopAndKill";
  } else if( fTrack->GetTrackStatus() == fKillTrackAndSecondaries ){
    outFile << " KillTrackAndSecondaries";
  } else if( fTrack->GetTrackStatus() == fSuspend ){
    outFile << " Suspend";
  } else if( fTrack->GetTrackStatus() == fPostponeToNextEvent ){
    outFile << " PostponeToNextEvent";
  }
  outFile << Gateendl;
#ifdef G4_USE_G4BESTUNIT_FOR_VERBOSE
  outFile << "        Vertex - x          : "
          << std::setw(20) << G4BestUnit(fTrack->GetVertexPosition().x(),"Length")
          << Gateendl;
  outFile << "        Vertex - y          : "
          << std::setw(20) << G4BestUnit(fTrack->GetVertexPosition().y(),"Length")
          << Gateendl;
  outFile << "        Vertex - z          : "
          << std::setw(20) << G4BestUnit(fTrack->GetVertexPosition().z(),"Length")
          << Gateendl;
#else
  outFile << "        Vertex - x (mm)     : "
          << std::setw(20) << fTrack->GetVertexPosition().x()/mm
          << Gateendl;
  outFile << "        Vertex - y (mm)     : "
          << std::setw(20) << fTrack->GetVertexPosition().y()/mm
          << Gateendl;
  outFile << "        Vertex - z (mm)     : "
          << std::setw(20) << fTrack->GetVertexPosition().z()/mm
          << Gateendl;
#endif
  outFile << "        Vertex - Px (MomDir): "
          << std::setw(20) << fTrack->GetVertexMomentumDirection().x()
          << Gateendl;
  outFile << "        Vertex - Py (MomDir): "
          << std::setw(20) << fTrack->GetVertexMomentumDirection().y()
          << Gateendl;
  outFile << "        Vertex - Pz (MomDir): "
          << std::setw(20) << fTrack->GetVertexMomentumDirection().z()
          << Gateendl;
#ifdef G4_USE_G4BESTUNIT_FOR_VERBOSE
  outFile << "        Vertex - KineE      : "
#else
    outFile << "        Vertex - KineE (MeV): "
#endif
          << std::setw(20) << G4BestUnit(fTrack->GetVertexKineticEnergy(),"Energy")
          << Gateendl;

  outFile << "        Creator Process     : "
          << std::setw(20);
  if( fTrack->GetCreatorProcess() == NULL){
    outFile << " Event Generator\n";
  } else {
    outFile << fTrack->GetCreatorProcess()->GetProcessName() << Gateendl;
  }

  outFile << "      -----------------------------------------------\n";
}

//-----------------------------------------------------------------------------
#endif
