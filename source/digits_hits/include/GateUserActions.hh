/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  \class  GateUserActions
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/


//-----------------------------------------------------------------------------
/// \class GateUserActions
//-----------------------------------------------------------------------------

#ifndef GATECALLBACKMANAGER_HH
#define GATECALLBACKMANAGER_HH


#include "globals.hh"
#include "G4String.hh"
#include <iomanip>
#include <vector>

#include "GateRunManager.hh"

#include "GateMessageManager.hh"
#include "GateActorManager.hh"
#include "GateMessageManager.hh"
#include "GateTrajectory.hh"
#include "GateSteppingVerbose.hh"
#include "G4VSteppingVerbose.hh"


class GateRunAction;
class GateEventAction;
class G4SliceTimer;

class GateUserActions
{
public:
  GateUserActions(GateRunManager* m);
  ~GateUserActions();

  //-----------------------------------------------------------------------------
  /// \brief Sets the RunManager used
  /// *** MUST *** be called before simulation starts
  void SetRunManager(GateRunManager* m) { pRunManager = m; }
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// G4UserRunAction callback
  void BeginOfRunAction(const G4Run*);
  /// G4UserRunAction callback
  void EndOfRunAction(const G4Run*);
  /// G4UserEventAction callback
  void BeginOfEventAction(const G4Event*);
  /// G4UserEventAction callback
  void EndOfEventAction (const G4Event*);
  /// G4UserTrackingAction callback
  void PreUserTrackingAction(const G4Track*);
  /// G4UserTrackingAction callback
  void PostUserTrackingAction(const G4Track*);
  /// G4UserSteppingAction callback
  void UserSteppingAction(const G4Step*);
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// Returns the current Run
  const G4Run* GetCurrentRun() const { return mCurrentRun; }
  /// Returns the current Event
  const G4Event* GetCurrentEvent() const { return mCurrentEvent; }
  /// Returns the current Track
  const G4Track* GetCurrentTrack() const { return mCurrentTrack; }
  /// Returns the current Step
  const G4Step* GetCurrentStep() const { return mCurrentStep; }
  //-----------------------------------------------------------------------------

  G4int GetCurrentRunID(){return mCurrentRun->GetRunID();}

  static GateUserActions* GetUserActions() { return pUserActions; };

  GateTrackIDInfo *GetTrackIDInfo(G4int id);

  long int GetCurrentEventNumber() { return mEventNumber; }

  void EnableTimeStudy(G4String filename);
  void EnableTimeStudyForSteps(G4String filename);

protected:

  //-----------------------------------------------------------------------------
  /// Pointer on the GateRunmanager
  GateRunManager* pRunManager;
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// Current Run
  const G4Run* mCurrentRun;
  /// Current Event
  const G4Event* mCurrentEvent;
  /// Current Track
  const G4Track* mCurrentTrack;
  /// Current Step
  const G4Step* mCurrentStep;
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  long int mRunNumber;
  long int mEventNumber;
  long int mTrackNumber;
  long long int mStepNumber;
  long int mTrackNumberInCurrentEvent;
  long int mStepNumberInCurrentTrack;
  //-----------------------------------------------------------------------------

  static GateUserActions* pUserActions;


  GateRunAction* runAction;
  GateEventAction* eventAction;

  G4bool mIsTimeStudyActivated;
  G4SliceTimer* mTimer;

  std::map<G4int,GateTrackIDInfo> theListOfTrackIDInfo;


};

#endif /* end #define GATECALLBACKMANAGER_HH */
