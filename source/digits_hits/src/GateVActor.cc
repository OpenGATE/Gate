/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#include "G4Event.hh"

#include "GateVActor.hh"
#include "GateActorMessenger.hh"
#include "GateActorManager.hh"
#include "GateMiscFunctions.hh"

#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>

//-----------------------------------------------------------------------------
GateVActor::GateVActor(G4String name, G4int depth)
  :GateNamedObject(name), G4VPrimitiveScorer(name, depth)
{
  GateDebugMessageInc("Actor",4,"GateVActor() -- begin\n");
  EnableBeginOfRunAction(false);
  EnableEndOfRunAction(true); // for save
  EnableBeginOfEventAction(false);
  EnableEndOfEventAction(true); // for save every n
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(false);
  EnableResetDataAtEachRun(false);
  EnableRecordEndOfAcquisition(false);
  mSaveFilename = "FilnameNotGivenForThisActor";
  mSaveInitialFilename = mSaveFilename;
  mVolumeName = "";
  mVolume = 0;
  EnableSaveEveryNEvents(0);
  EnableSaveEveryNSeconds(0);
  mNumOfFilters = 0;
  mOverWriteFilesFlag = true;
  pFilterManager = new GateFilterManager(GetObjectName()+"_filter");
  GateDebugMessageDec("Actor",4,"GateVActor() -- end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateVActor::~GateVActor()
{
  GateDebugMessageInc("Actor",4,"~GateVActor() -- begin\n");
  delete pFilterManager;
  GateDebugMessageDec("Actor",4,"~GateVActor() -- end\n");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// default callback for BeginOfRunAction
void GateVActor::BeginOfRunAction(const G4Run*)
{
  gettimeofday(&mTimeOfLastSaveEvent, NULL);
  if (mResetDataAtEachRun) {
    ResetData();
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// default callback for EndOfRunAction allowing to call Save
void GateVActor::EndOfRunAction(const G4Run*)
{
  SaveData();
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
    if (ne % mSaveEveryNEvents == 0)  SaveData();

  // Save every n seconds
  if (mSaveEveryNSeconds != 0) { // need to check time
    struct timeval end;
    gettimeofday(&end, NULL);
    long seconds  = end.tv_sec  - mTimeOfLastSaveEvent.tv_sec;
    if (seconds > mSaveEveryNSeconds) {
      //GateMessage("Core", 0, "Actor " << GetName() << " : " << mSaveEveryNSeconds << " seconds.\n");
      SaveData();
      mTimeOfLastSaveEvent = end;
    }
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVActor::SetSaveFilename(G4String  f)
{
  mSaveFilename = f;
  mSaveInitialFilename = f;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVActor::AttachToVolume(G4String /*volumeName*/)
{
  if (mVolumeName != "") {
    GateDebugMessageInc("Actor",4,"Attach "<<GetObjectName()<<" to volume -- begin\n");
    mVolume =   GateObjectStore::GetInstance()->FindVolumeCreator(mVolumeName);
    GateDebugMessage("Actor",5,"actor attached to: "<<mVolume->GetObjectName()<< Gateendl);
    GateDebugMessageDec("Actor",4,"Attach "<<GetObjectName()<<" to volume -- end\n");
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateVActor::SaveData()
{
  if (!this->mOverWriteFilesFlag) {
    mSaveFilename = GetSaveCurrentFilename(mSaveInitialFilename);
  }
}
//-----------------------------------------------------------------------------
