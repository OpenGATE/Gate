/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#ifndef GATEACTORMANAGER_CC
#define GATEACTORMANAGER_CC

#include "GateActorManager.hh"
#include "GateVActor.hh"
#include "GateMultiSensitiveDetector.hh"

//-----------------------------------------------------------------------------
GateActorManager::GateActorManager()
{
  GateDebugMessageInc("Actor",4,"GateActorManager() -- begin\n");

  pActorManagerMessenger = new GateActorManagerMessenger(this);
  IsInitialized =0;
  resetAfterSaving = false;
  GateDebugMessageDec("Actor",4,"GateActormanager() -- end\n");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateActorManager::~GateActorManager()
{
  GateDebugMessageInc("Actor",4,"~GateActorManager() -- begin\n");

  theListOfActorsEnabledForBeginOfRun.clear();
  theListOfActorsEnabledForEndOfRun.clear();
  theListOfActorsEnabledForBeginOfEvent.clear();
  theListOfActorsEnabledForEndOfEvent.clear();
  theListOfActorsEnabledForPreUserTrackingAction.clear();
  theListOfActorsEnabledForPostUserTrackingAction.clear();
  theListOfActorsEnabledForUserSteppingAction.clear();

    GateVActor* actor;
    while(!theListOfOwnedActors.empty())
    {
        // get first 'element'
        actor = theListOfOwnedActors.front();

        // remove it from the list
        theListOfOwnedActors.erase(theListOfOwnedActors.begin());

        GateMessage("Actor", 4, "~GateActorManager -- delete actor: " << actor->GetObjectName() << Gateendl );
        // delete the pointer
        delete actor;
    }


  delete pActorManagerMessenger;

  GateDebugMessageDec("Actor",4,"~GateActormanager() -- end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateActorManager::SetResetAfterSaving(bool reset)
{
  resetAfterSaving = reset;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
bool GateActorManager::GetResetAfterSaving() const
{
  return resetAfterSaving;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateActorManager::AddActor(G4String actorType, G4String actorName, int depth)
{
  GateDebugMessageInc("Actor",5,"Actor Manager -- AddActor(): "<<actorName<<" -- begin\n");
  if (GateActorManager::theListOfActorPrototypes[actorType])
  {
      auto a = GateActorManager::theListOfActorPrototypes[actorType](actorName,depth);
      theListOfActors.push_back(a);
      theListOfOwnedActors.push_back(a);
  }

  else GateWarning("Actor type: "<<actorType<<" does not exist!");
  GateDebugMessageDec("Actor",5,"Actor Manager -- AddActor(): "<<actorName<<" -- end\n\n");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateVActor*  GateActorManager::GetActor(const G4String &actorType, const G4String &actorName)
{
  for (std::vector<GateVActor*>::const_iterator iter=theListOfActors.begin(); iter!=theListOfActors.end(); iter++) {
    GateVActor *actor = *iter;
    if (actor->GetName()==actorName && actor->GetTypeName()==actorType) return actor;
  }
  return NULL;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateActorManager::CreateListsOfEnabledActors()
{
  std::vector<GateVActor*>::iterator sit;
  for (sit= theListOfActors.begin(); sit!=theListOfActors.end(); ++sit) {
    //if ((*sit)->GetObjectName() == "output") (*sit) = GateOutputMgr::GetInstance();
    //GateMessage("Core", 0, "Actor = " << (*sit)->GetObjectName() << Gateendl);

    (*sit)->Construct();
    if ((*sit)->IsBeginOfRunActionEnabled()       && IsInitialized<2) theListOfActorsEnabledForBeginOfRun.push_back( (*sit) );
    if ((*sit)->IsEndOfRunActionEnabled()         && IsInitialized<2) theListOfActorsEnabledForEndOfRun.push_back( (*sit) );
    if ((*sit)->IsBeginOfEventActionEnabled()     && IsInitialized<2) theListOfActorsEnabledForBeginOfEvent.push_back( (*sit) );
    if ((*sit)->IsEndOfEventActionEnabled()       && IsInitialized<2) theListOfActorsEnabledForEndOfEvent.push_back( (*sit) );
    if ((*sit)->IsPreUserTrackingActionEnabled()  && IsInitialized<2) theListOfActorsEnabledForPreUserTrackingAction.push_back( (*sit) );
    if ((*sit)->IsPostUserTrackingActionEnabled() && IsInitialized<2) theListOfActorsEnabledForPostUserTrackingAction.push_back( (*sit) );
    if ((*sit)->IsRecordEndOfAcquisitionEnabled() && IsInitialized<2) theListOfActorsEnabledForRecordEndOfAcquisition.push_back( (*sit) );

    //GateMessage("Core", 0, "IsUserSteppingActionEnabled = " << (*sit)->IsUserSteppingActionEnabled() << Gateendl);

    if ((*sit)->IsUserSteppingActionEnabled()) {
      if ( (*sit)->GetVolumeName()=="" ) {
        if (IsInitialized<2) theListOfActorsEnabledForUserSteppingAction.push_back( (*sit) );
      }
      else {
        SetMultiFunctionalDetector((*sit),(*sit)->GetVolume());
      }
    }
  }

  IsInitialized++;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateActorManager::PrintListOfActorTypes() const
{
  G4cout << "***********************\n";
  for (std::map<G4String,maker_actor>::const_iterator iter=theListOfActorPrototypes.begin(); iter!=theListOfActorPrototypes.end(); iter++) {
    G4cout << iter->first << Gateendl;
  }
  G4cout << "***********************\n";
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateActorManager::PrintListOfActors() const
{
  std::vector<GateVActor*>::const_iterator sit;
  for (sit= theListOfActors.begin(); sit!=theListOfActors.end(); ++sit)
    {
      GateMessage("Actor", 1,"Name = "<<  (*sit)->GetObjectName() <<"  Volume name = " << (*sit)->GetVolumeName() << Gateendl);
    }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
std::vector<GateVActor*> GateActorManager::ReturnListOfActors()
{
  return theListOfActors;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateActorManager::BeginOfRunAction(const G4Run* run)
{
  std::vector<GateVActor*>::iterator sit;

  //GateMessage("Core", 0, "Run " << run->GetRunID() << " is starting.\n");
  for (sit = theListOfActorsEnabledForBeginOfRun.begin(); sit!=theListOfActorsEnabledForBeginOfRun.end(); ++sit)
    (*sit)->BeginOfRunAction(run);

}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateActorManager::EndOfRunAction(const G4Run* run)
{
  std::vector<GateVActor*>::iterator sit;
  for (sit = theListOfActorsEnabledForEndOfRun.begin(); sit!=theListOfActorsEnabledForEndOfRun.end(); ++sit)
    (*sit)->EndOfRunAction(run);
  //GateMessage("Core", 0, "Run " << run->GetRunID() << " is ending.\n");
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateActorManager::BeginOfEventAction(const G4Event* evt)
{
  if (evt) mCurrentEventId = evt->GetEventID();
  std::vector<GateVActor*>::iterator sit;
  for (sit = theListOfActorsEnabledForBeginOfEvent.begin(); sit!=theListOfActorsEnabledForBeginOfEvent.end(); ++sit)
    (*sit)->BeginOfEventAction(evt);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateActorManager::EndOfEventAction(const G4Event* evt)
{
  std::vector<GateVActor*>::iterator sit;
  for (sit = theListOfActorsEnabledForEndOfEvent.begin(); sit!=theListOfActorsEnabledForEndOfEvent.end(); ++sit)
    (*sit)->EndOfEventAction(evt);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateActorManager::PreUserTrackingAction(const G4Track* track)
{
  // GateDebugMessage("Actor", 1, "listtrack= " << theListOfActorsEnabledForPreUserTrackingAction.size()
  //                    << Gateendl);
  std::vector<GateVActor*>::iterator sit;
  for (sit = theListOfActorsEnabledForPreUserTrackingAction.begin(); sit!=theListOfActorsEnabledForPreUserTrackingAction.end(); ++sit)
    {
      if ((*sit)->GetNumberOfFilters()!=0)
        if (!(*sit)->GetFilterManager()->Accept(track) ) continue;
      (*sit)->PreUserTrackingAction(0,track);
    }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateActorManager::PostUserTrackingAction(const G4Track* track)
{
  std::vector<GateVActor*>::iterator sit;
  for (sit = theListOfActorsEnabledForPostUserTrackingAction.begin(); sit!=theListOfActorsEnabledForPostUserTrackingAction.end(); ++sit)
    {
      if ((*sit)->GetNumberOfFilters()!=0)
        if (!(*sit)->GetFilterManager()->Accept(track) ) continue;
      (*sit)->PostUserTrackingAction(0,track);
    }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateActorManager::UserSteppingAction(const G4Step* step)
{
  std::vector<GateVActor*>::iterator sit;
  // GateDebugMessage("Actor", 1, "list = " << theListOfActorsEnabledForUserSteppingAction.size() << Gateendl);
  for (sit = theListOfActorsEnabledForUserSteppingAction.begin(); sit!=theListOfActorsEnabledForUserSteppingAction.end(); ++sit)
    {
      // GateDebugMessage("Actor", 1, "Step for " << (*sit)->GetObjectName());
      if ((*sit)->GetNumberOfFilters()!=0){
        if (!(*sit)->GetFilterManager()->Accept(step) ) continue;
      }
      (*sit)->UserSteppingAction(0, step);
    }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateActorManager::RecordEndOfAcquisition()
{
  std::vector<GateVActor*>::iterator sit;
  // GateDebugMessage("Actor", 1, "list = " << theListOfActorsEnabledForUserSteppingAction.size() << Gateendl);
  for (sit = theListOfActorsEnabledForRecordEndOfAcquisition.begin(); sit!=theListOfActorsEnabledForRecordEndOfAcquisition.end(); ++sit)
    {
      // GateDebugMessage("Actor", 1, "Step for " << (*sit)->GetObjectName());
      (*sit)->RecordEndOfAcquisition();
    }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateActorManager::SetMultiFunctionalDetector(GateVActor * actor, GateVVolume * volume)
{
  GateDebugMessageInc("Actor",4,"Actor Manager -- SetMFD -- begin "<<volume->GetLogicalVolume()<< Gateendl);

  if (!volume->GetLogicalVolume()->GetSensitiveDetector())
    {
      GateDebugMessage("Actor",5,"SetMFD -- Add new MSD - Attach to: "<<volume->GetLogicalVolume()->GetName()<< Gateendl);

      G4int nActor = theListOfMultiSensitiveDetector.size();
      std::ostringstream num;
      num << nActor;
      G4String detectorName = "MFD_"+ num.str();
      G4String detectorName2 = "MSD_"+ num.str();

      GateMultiSensitiveDetector * msd = new GateMultiSensitiveDetector(detectorName2);

      G4SDManager::GetSDMpointer()->AddNewDetector(msd);

      volume->GetLogicalVolume()->SetSensitiveDetector(msd);
      msd->SetMultiFunctionalDetector(detectorName);

      // We give "ownership of actor to msd, when msd is deleted, it will delete 'actor'
      msd->SetActor(actor);
      // remove the actor from the list of owned actors
      theListOfOwnedActors.erase(std::remove(theListOfOwnedActors.begin(), theListOfOwnedActors.end(), actor), theListOfOwnedActors.end());

      theListOfMultiSensitiveDetector.push_back(msd );
    }
  else if (G4StrUtil::contains(volume->GetLogicalVolume()->GetSensitiveDetector()->GetName(), "MSD") )
    {
      GateDebugMessage("Actor",5,"SetMFD -- MSD already attached to: "<<volume->GetLogicalVolume()->GetName()<< Gateendl);
      dynamic_cast<GateMultiSensitiveDetector*>(volume->GetLogicalVolume()->GetSensitiveDetector())->SetActor(actor);
    }
  else
    {
      GateDebugMessage("Actor",5,"SetMFD -- SD already attached to: "<<volume->GetLogicalVolume()->GetName()
                       <<" - Reaplace it by a MSD\n");

      G4int nActor = theListOfMultiSensitiveDetector.size();
      std::ostringstream num;
      num << nActor;
      G4String detectorName = "MFD_"+ num.str();
      G4String detectorName2 = "MSD_"+ num.str();

      // Remove attached SD by replacing with a deactivated clone SD
      G4VSensitiveDetector* replacementSD = new GateEmptySD(volume->GetLogicalVolume()->GetSensitiveDetector()->GetName());
      G4SDManager::GetSDMpointer()->AddNewDetector(replacementSD);
      replacementSD->Activate(false);

      theListOfMultiSensitiveDetector.push_back(new GateMultiSensitiveDetector(detectorName2));

      theListOfMultiSensitiveDetector[nActor]->SetSensitiveDetector(volume->GetLogicalVolume()->GetSensitiveDetector());
      volume->GetLogicalVolume()->SetSensitiveDetector(theListOfMultiSensitiveDetector[nActor]);
      G4SDManager::GetSDMpointer()->AddNewDetector(theListOfMultiSensitiveDetector[nActor]);

      theListOfMultiSensitiveDetector[nActor]->SetMultiFunctionalDetector(detectorName);
      theListOfMultiSensitiveDetector[nActor]->SetActor(actor);
    }

  GateObjectChildList * listOfChild = volume->GetTheChildList();
  for (size_t i =0;i<listOfChild->size();i++)
    {
      GateVVolume*  vol =  listOfChild->GetVolume(i) ;
      SetMultiFunctionalDetector(actor, vol);
    }

  // here indicate to the volume that he has to add the SD to his child ...
  // should cast the GetSensitiveDetector
  volume->PropagateSensitiveDetectorToChild(dynamic_cast<GateMultiSensitiveDetector*>(volume->GetLogicalVolume()->GetSensitiveDetector()));

  GateDebugMessageDec("Actor",4,"Actor Manager -- SetMFD -- end\n");

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
G4bool GateActorManager::AddFilter(G4String filterType, G4String actorName )
{
  GateDebugMessageDec("Actor",4,"AddFilter() -- begin\n");
  int nActor = -1;
  for (unsigned int i = 0;i<theListOfActors.size();i++)
    if (theListOfActors[i]->GetObjectName() == actorName) nActor = i;

  if (nActor==-1) GateError("Actor "<<actorName<<" not found!");

  if (GateActorManager::theListOfFilterPrototypes[filterType])
    {
      theListOfActors[nActor]->GetFilterManager()->AddFilter(GateActorManager::theListOfFilterPrototypes[filterType]("/gate/actor/"+theListOfActors[nActor]->GetObjectName()+"/"+filterType));
      theListOfActors[nActor]->IncNumberOfFilters();
    }
  else
    {
      GateWarning("Filter type: "<<filterType<<" does not exist!");
      return false;
    }

  GateDebugMessageDec("Actor",4,"AddFilter() -- end\n");
  return true;
}
//-----------------------------------------------------------------------------

GateActorManager *GateActorManager::singleton_ActorManager = 0;

#endif /* end #define GATEACTORMANAGER_CC */
