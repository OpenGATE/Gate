/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*
  \class  GateVActor
  \author thibault.frisson@creatis.insa-lyon.fr
  laurent.guigues@creatis.insa-lyon.fr
  david.sarrut@creatis.insa-lyon.fr
*/


//-----------------------------------------------------------------------------
/// \class GateVActor
//-----------------------------------------------------------------------------

#ifndef GATEVACTOR_HH
#define GATEVACTOR_HH

#include "globals.hh"
#include "G4String.hh"
#include <iomanip>
#include <vector>

#include "GateActorManager.hh"
#include "GateNamedObject.hh"
#include "GateMessageManager.hh"
#include "GateFilterManager.hh"
#include "GateObjectStore.hh"
#include "GateVVolume.hh"
#include "G4VPrimitiveScorer.hh"
#include "G4THitsMap.hh"
#include "G4TouchableHistory.hh"

class GateVActor :
  public GateNamedObject,
  public G4VPrimitiveScorer
{
public:
  GateVActor(G4String name, G4int depth=0);
  virtual ~GateVActor();

public:
  virtual void Initialize(G4HCofThisEvent*){}
  virtual void EndOfEvent(G4HCofThisEvent*){}
  virtual void clear(){}

public:
  virtual void DrawAll(){}
  virtual void PrintAll(){}

  //-----------------------------------------------------------------------------
  G4String GetTypeName() {return mTypeName;}
  void SetTypeName(G4String type) {mTypeName = type;}
  //-----------------------------------------------------------------------------
  /// Constructs the sensor (set world as parent by default)
  virtual void Construct(){}
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  // User's callbacks (do nothing here, could be overloaded). WARNING:
  // all are disabled by default. You must use for example
  // EnableBeginOfRunAction(true) and overload the callback to make it
  // happens.
  virtual void BeginOfRunAction(const G4Run*); // default action (get time)
  virtual void EndOfRunAction(const G4Run*); // default action (save)
  virtual void BeginOfEventAction(const G4Event*) {}
  virtual void EndOfEventAction(const G4Event*); // default action (save every n)
  virtual void PreUserTrackingAction(const GateVVolume *, const G4Track*) {}
  virtual void PostUserTrackingAction(const GateVVolume *, const G4Track*) {}
  virtual void UserSteppingAction(const GateVVolume *, const G4Step*) {}
  virtual void RecordEndOfAcquisition() {};
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// Attaches the sensor to a volume
  void AttachToVolume(G4String volumeName);
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  void EnableBeginOfRunAction(bool b)       { mIsBeginOfRunActionEnabled = b; }
  void EnableEndOfRunAction(bool b)         { mIsEndOfRunActionEnabled = b; }
  void EnableBeginOfEventAction(bool b)     { mIsBeginOfEventActionEnabled = b; }
  void EnableEndOfEventAction(bool b)       { mIsEndOfEventActionEnabled = b; }
  void EnablePreUserTrackingAction(bool b)  { mIsPreUserTrackingActionEnabled = b; }
  void EnablePostUserTrackingAction(bool b) { mIsPostUserTrackingActionEnabled = b; }
  void EnableUserSteppingAction(bool b)     { mIsUserSteppingActionEnabled = b; }
  void EnableRecordEndOfAcquisition(bool b) { mIsRecordEndOfAcquisitionEnabled = b; }
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  bool IsBeginOfRunActionEnabled() const       { return mIsBeginOfRunActionEnabled; }
  bool IsEndOfRunActionEnabled() const         { return mIsEndOfRunActionEnabled; }
  bool IsBeginOfEventActionEnabled() const     { return mIsBeginOfEventActionEnabled; }
  bool IsEndOfEventActionEnabled() const       { return mIsEndOfEventActionEnabled; }
  bool IsPreUserTrackingActionEnabled() const  { return mIsPreUserTrackingActionEnabled; }
  bool IsPostUserTrackingActionEnabled() const { return mIsPostUserTrackingActionEnabled; }
  bool IsUserSteppingActionEnabled() const     { return mIsUserSteppingActionEnabled; }
  bool IsRecordEndOfAcquisitionEnabled() const { return mIsRecordEndOfAcquisitionEnabled; }
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  void SetSaveFilename(G4String  f);
  G4String GetSaveFilename() { return mSaveFilename; }
  virtual void SaveData();
  virtual void ResetData() = 0;
  void EnableSaveEveryNEvents(int n) { mSaveEveryNEvents = n; }
  void EnableSaveEveryNSeconds(int n) { mSaveEveryNSeconds = n; }
  void SetOverWriteFilesFlag(bool b) { mOverWriteFilesFlag = b; }
  void EnableResetDataAtEachRun(bool b) { mResetDataAtEachRun = b; }
  //-----------------------------------------------------------------------------

  G4String GetVolumeName(){return mVolumeName;}
  GateVVolume * GetVolume(){return mVolume;}
  void SetVolumeName(G4String name){mVolumeName = name;}

  GateFilterManager * GetFilterManager(){return pFilterManager;}
  G4int GetNumberOfFilters() {return mNumOfFilters;}
  void IncNumberOfFilters() {mNumOfFilters++;}

protected:
  G4String mTypeName;
  G4String mVolumeName;
  GateVVolume * mVolume;

  GateFilterManager * pFilterManager;

  virtual G4bool ProcessHits(G4Step * step, G4TouchableHistory *) { UserSteppingAction(0, step); return true; }

  G4int mNumOfFilters;

  //-----------------------------------------------------------------------------
  bool mIsBeginOfRunActionEnabled;
  bool mIsEndOfRunActionEnabled;
  bool mIsBeginOfEventActionEnabled;
  bool mIsEndOfEventActionEnabled;
  bool mIsPreUserTrackingActionEnabled;
  bool mIsPostUserTrackingActionEnabled;
  bool mIsUserSteppingActionEnabled;
  bool mIsRecordEndOfAcquisitionEnabled;
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  int  mSaveEveryNEvents;
  int  mSaveEveryNSeconds;
  bool mOverWriteFilesFlag;
  bool mResetDataAtEachRun;
  G4String mSaveInitialFilename;
  G4String mSaveFilename;
  int mSaveFileDescriptor;
  struct timeval mTimeOfLastSaveEvent;
  //-----------------------------------------------------------------------------

};
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
#define MAKE_AUTO_CREATOR_ACTOR(NAME,CLASS)				\
  class NAME##Creator {							\
  public:								\
  NAME##Creator() {							\
    GateActorManager::GetInstance()->theListOfActorPrototypes[#NAME]= CLASS::make_sensor; } }; \
  static NAME##Creator ActorCreator##NAME;

#define FCT_FOR_AUTO_CREATOR_ACTOR(CLASS)				\
  static GateVActor *make_sensor(G4String name, G4int depth){ GateVActor *instance=new CLASS(name, depth); instance->SetTypeName(#CLASS); return instance; };
//-----------------------------------------------------------------------------

#endif /* end #define GATEVACTOR_HH */
