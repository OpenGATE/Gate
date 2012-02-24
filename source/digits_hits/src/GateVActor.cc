/*----------------------
  GATE version name: gate_v6

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/


#include "GateVActor.hh"
#include "G4Event.hh"
#include "GateActorMessenger.hh"
#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>

//-----------------------------------------------------------------------------
GateVActor::GateVActor(G4String name, G4int depth)
  :GateNamedObject(name), G4VPrimitiveScorer(name, depth)
{
  GateDebugMessageInc("Actor",4,"GateVActor() -- begin"<<G4endl);
  EnableBeginOfRunAction(false);
  EnableEndOfRunAction(true); // for save
  EnableBeginOfEventAction(false);
  EnableEndOfEventAction(true); // for save every n
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(false);
  mSaveFilename = "FilnameNotGivenForThisActor";
  mVolumeName = "";
  mVolume = 0;  
  EnableSaveEveryNEvents(0);
  EnableSaveEveryNSeconds(0);
  mNumOfFilters = 0;
  pFilterManager = new GateFilterManager(GetObjectName()+"_filter");
  GateDebugMessageDec("Actor",4,"GateVActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
GateVActor::~GateVActor()
{
  GateDebugMessageInc("Actor",4,"~GateVActor() -- begin"<<G4endl);
  delete pFilterManager;
  GateDebugMessageDec("Actor",4,"~GateVActor() -- end"<<G4endl);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// default callback for BeginOfRunAction 
void GateVActor::BeginOfRunAction(const G4Run*) 
{
  gettimeofday(&mTimeOfLastSaveEvent, NULL);
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// default callback for EndOfRunAction allowing to call Save
void GateVActor::EndOfRunAction(const G4Run*) 
{
  Save();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// default callback for EndOfEventAction allowing to call
// EndOfNEventAction (if it is enabled)
void GateVActor::EndOfEventAction(const G4Event*e) 
{
  int ne = e->GetEventID()+1;    
 
  // Save every n events
  if ((ne != 0) && (mSaveEveryNEvents != 0)) 
    if (ne % mSaveEveryNEvents == 0)  Save();

  // Save every n seconds
  if (mSaveEveryNSeconds != 0) { // need to check time
    struct timeval end;
    gettimeofday(&end, NULL);
    long seconds  = end.tv_sec  - mTimeOfLastSaveEvent.tv_sec;
    if (seconds > mSaveEveryNSeconds) {
      //GateMessage("Core", 0, "Actor " << GetName() << " : " << mSaveEveryNSeconds << " seconds." << G4endl);
      Save();
      mTimeOfLastSaveEvent = end;
    }
  }
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVActor::SetSaveFilename(G4String  f) 
{
  mSaveFilename = f;
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVActor::Save() 
{
  SaveData();
}
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void GateVActor::AttachToVolume(G4String /*volumeName*/)
{
  if (mVolumeName != "") {
    GateDebugMessageInc("Actor",4,"Attach "<<GetObjectName()<<" to volume -- begin"<<G4endl);
    mVolume =   GateObjectStore::GetInstance()->FindVolumeCreator(mVolumeName);
    // DD(mVolume);
    // DD(mVolume->GetLogicalVolume());
    GateDebugMessage("Actor",5,"actor attached to: "<<mVolume->GetObjectName()<<G4endl);
    GateDebugMessageDec("Actor",4,"Attach "<<GetObjectName()<<" to volume -- end\n"<<G4endl);
  }
}
//-----------------------------------------------------------------------------


