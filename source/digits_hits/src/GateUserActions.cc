/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATECALLBACKMANAGER_CC
#define GATECALLBACKMANAGER_CC

#include "GateUserActions.hh"
#include "GateActions.hh"

#include "G4UImanager.hh"
#include "G4VVisManager.hh"
//#include "G4Run.hh"


#include "GateSteppingVerbose.hh"
#include "G4SteppingManager.hh"

#include "G4SliceTimer.hh"

//class GateRecorderBase;
GateUserActions* GateUserActions::pUserActions=0;

//-----------------------------------------------------------------------------
GateUserActions::GateUserActions(GateRunManager* m, GateRecorderBase* r)
  : recorder(r)
{
  GateMessage("Core", 4,"GateUserActions Constructor start."<<G4endl);

  pUserActions = this;

  SetRunManager(m);

  // Initialisation
  mCurrentRun = 0;
  mCurrentEvent = 0;
  mCurrentTrack = 0;
  mCurrentStep = 0;

  mRunNumber = 0;
  mEventNumber = 0;
  mTrackNumber = 0;
  mStepNumber = 0;

  mIsTimeStudyActivated = false;


  // Set fGate' user action classes to the GateRunmanager :
  // Run/Event/Tracking/Stepping in order to get the callbacks
  GateRunAction* RunAction = new GateRunAction(this, recorder);
  GateEventAction* EventAction = new GateEventAction(this, recorder);
  GateTrackingAction* TrackingAction = new GateTrackingAction(this, recorder);
  GateSteppingAction* SteppingAction = new GateSteppingAction(this, recorder);

  pRunManager->SetUserAction(RunAction);
  pRunManager->SetUserAction(EventAction);
  pRunManager->SetUserAction(TrackingAction);
  pRunManager->SetUserAction(SteppingAction);

  //pRunManager->SetUserAction(dynamic_cast<G4UserRunAction *>(this)); //Don't know why this don't work
  //pRunManager->SetUserAction(dynamic_cast<G4UserEventAction *>(this));
  //pRunManager->SetUserAction(dynamic_cast<G4UserTrackingAction *>(this));
  //pRunManager->SetUserAction(dynamic_cast<G4UserSteppingAction *>(this));

  mTimer = new G4SliceTimer();

  GateMessage("Core", 4,"GateUserActions Constructor end."<<G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateUserActions::~GateUserActions()
{

  delete pUserActions;
  GateDebugMessageInc("Core", 4, "GateUserActions Destructor."<<G4endl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateUserActions::BeginOfRunAction(const G4Run* run)
{
  mRunNumber++;

  mTimer->Clear();
  mTimer->Start();

  mCurrentRun = run;
  GateActorManager::GetInstance()->BeginOfRunAction(run);

  // Prepare the visualization
  if (G4VVisManager::GetConcreteInstance()) {
    G4UImanager* UI = G4UImanager::GetUIpointer();
    UI->ApplyCommand("/vis/scene/notifyHandlers");
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateUserActions::EndOfRunAction(const G4Run* run)
{
  GateActorManager::GetInstance()->EndOfRunAction(run);

  if(mIsTimeStudyActivated){
     GateSteppingVerbose * steppingVerbose = (GateSteppingVerbose*)(G4VSteppingVerbose::GetInstance());
     steppingVerbose->EndOfRun();
  }

  // Run ended, update the visualization
  if (G4VVisManager::GetConcreteInstance()) {
    G4UImanager::GetUIpointer()->ApplyCommand("/vis/viewer/update");
  }
  mTimer->Stop();
  GateMessage("Core",1,"Run "<<mRunNumber - 1<<"  ---  Elapsed time = "<<mTimer->GetUserElapsed()<<G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateUserActions::BeginOfEventAction(const G4Event* evt)
{
  mCurrentEvent = evt;
  mEventNumber++;
  theListOfTrackIDInfo.clear();
  GateActorManager::GetInstance()->BeginOfEventAction(evt);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateUserActions::EndOfEventAction(const G4Event* evt)
{
  GateActorManager::GetInstance()->EndOfEventAction(evt);
//sizeof(v) + sizeof(T) * v.capacity();
// G4cout<<G4endl;
 // GateTrackIDInfo trInfo;
 //G4cout<<"Taille vecteur = "<<sizeof(theListOfTrackIDInfo)<<G4endl;
 //G4cout<<"Taille élément vecteur = "<<sizeof(trInfo)<<G4endl;
 //G4cout<<"Taille totale = "<<sizeof(theListOfTrackIDInfo)+sizeof(trInfo)*theListOfTrackIDInfo.size()<<G4endl;
 for(std::map<G4int,GateTrackIDInfo>::iterator i = theListOfTrackIDInfo.begin(); i != theListOfTrackIDInfo.begin(); /*EMPTY*/)
 {
   theListOfTrackIDInfo.erase(i++);
 }
 theListOfTrackIDInfo.clear();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateUserActions::PreUserTrackingAction(const G4Track* track)
{
  GateDebugMessage("Core", 3, "Pre Track " << track->GetTrackID() << "\n");

  GateTrackIDInfo trInfo;
  trInfo.SetParticleName(track->GetDefinition()->GetParticleName());
  trInfo.SetID(track->GetTrackID());
  trInfo.SetParentID(track->GetParentID());
  theListOfTrackIDInfo[track->GetTrackID()] = trInfo;


  //  theListOfTrackIDInfo[track->GetTrackID()] = new GateTrackIDInfo(track->GetDefinition()->GetParticleName(),track->GetTrackID(),track->GetParentID() );

  GateActorManager::GetInstance()->PreUserTrackingAction(track);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateUserActions::PostUserTrackingAction(const G4Track* track)
{
  GateActorManager::GetInstance()->PostUserTrackingAction(track);
  if(mIsTimeStudyActivated){
    GateSteppingVerbose * steppingVerbose = (GateSteppingVerbose*)(G4VSteppingVerbose::GetInstance());
    steppingVerbose->EndOfTrack();
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateUserActions::UserSteppingAction(const G4Step* step)
{
  GateActorManager::GetInstance()->UserSteppingAction(step);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateTrackIDInfo *GateUserActions::GetTrackIDInfo(G4int id)
{
  if(id>0) return (&theListOfTrackIDInfo[id]);
  else return 0;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateUserActions::EnableTimeStudy(G4String filename)
{
  GateSteppingVerbose * steppingVerbose = (GateSteppingVerbose*)(G4VSteppingVerbose::GetInstance());
  steppingVerbose->Initialize(filename,false);
  mIsTimeStudyActivated = true;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateUserActions::EnableTimeStudyForSteps(G4String filename)
{
  GateSteppingVerbose * steppingVerbose = (GateSteppingVerbose*)(G4VSteppingVerbose::GetInstance());
  steppingVerbose->Initialize(filename,true);
  mIsTimeStudyActivated = true;
}
//-----------------------------------------------------------------------------


#endif /* end #define GATECALLBACKMANAGER_CC */
