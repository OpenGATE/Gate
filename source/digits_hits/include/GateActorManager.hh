/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


/*
  \class  GateActorManager
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
          david.sarrut@creatis.insa-lyon.fr
*/

//-----------------------------------------------------------------------------
/// \class GateActorManager
//-----------------------------------------------------------------------------

#ifndef GATEACTORMANAGER_HH
#define GATEACTORMANAGER_HH

#include "globals.hh"
#include <iomanip>
#include <vector>
#include <map>

#include <G4String.hh>
#include <G4SDManager.hh>
#include <G4MultiFunctionalDetector.hh>
#include <G4Run.hh>
#include <G4Event.hh>

#include "GateMessageManager.hh"
#include "GateVFilter.hh"
#include "GateActorManagerMessenger.hh"
#include "GateObjectChildList.hh"

class GateVActor;
class GateMultiSensitiveDetector;

class GateActorManager
{
public:
  ~GateActorManager();

  typedef std::vector<GateVActor*> GateActorVec;

  static GateActorManager *GetInstance()
  {
    if (singleton_ActorManager == 0)
    {
      //std::cout << "creating GateActorManager...\n";
      singleton_ActorManager = new GateActorManager;
    }
    //else std::cout << "GateActorManager already created!\n";
    return singleton_ActorManager;
  };

  void SetResetAfterSaving(bool reset);
  bool GetResetAfterSaving() const;

  void AddActor(G4String actorType, G4String actorName, int depth=0);
  void CreateListsOfEnabledActors();
  void PrintListOfActors() const;
  void PrintListOfActorTypes() const;
  GateVActor*  GetActor(const G4String &actorType, const G4String &actorName);

  GateActorVec ReturnListOfActors();

  //-----------------------------------------------------------------------------
  ///
  G4bool AddFilter(G4String filterType,G4String actorName );
  //-----------------------------------------------------------------------------

  //-----------------------------------------------------------------------------
  /// G4UserRunAction callback
  void BeginOfRunAction(const G4Run*);
  /// G4UserRunAction callback
  void EndOfRunAction(const G4Run*);
  /// G4UserEventAction callback
  void BeginOfEventAction(const G4Event*);
  /// G4UserEventAction callback
  void EndOfEventAction(const G4Event*);
  /// G4UserTrackingAction callback
  void PreUserTrackingAction(const G4Track*);
  /// G4UserTrackingAction callback
  void PostUserTrackingAction(const G4Track*);
  /// G4UserSteppingAction callback
  void UserSteppingAction(const G4Step*);
  //-----------------------------------------------------------------------------

  typedef GateVActor *(*maker_actor)(G4String name, G4int depth);
  std::map<G4String,maker_actor> theListOfActorPrototypes;

  typedef GateVFilter *(*maker_filter)(G4String name);
  std::map<G4String,maker_filter> theListOfFilterPrototypes;

  void SetMultiFunctionalDetector(GateVActor * actor, GateVVolume * volume);
  std::vector<GateMultiSensitiveDetector*> theListOfMultiSensitiveDetector;

  GateActorVec & GetTheListOfActors() { return theListOfActors; }

  G4int GetCurrentEventId() const { return mCurrentEventId; }

protected:
  //std::vector<GateMultiSensitiveDetector*> theListOfMultiSensitiveDetector;
  GateActorVec theListOfActors;
  GateActorVec theListOfActorsEnabledForBeginOfRun;
  GateActorVec theListOfActorsEnabledForEndOfRun;
  GateActorVec theListOfActorsEnabledForBeginOfEvent;
  GateActorVec theListOfActorsEnabledForEndOfEvent;
  GateActorVec theListOfActorsEnabledForPreUserTrackingAction;
  GateActorVec theListOfActorsEnabledForPostUserTrackingAction;
  GateActorVec theListOfActorsEnabledForUserSteppingAction;

  GateActorManagerMessenger* pActorManagerMessenger;  //pointer to the Messenger
  G4int mCurrentEventId;

private:
  int IsInitialized;
  bool resetAfterSaving;

  GateActorManager();
  static GateActorManager *singleton_ActorManager;
};

#endif /* end #define GATEACTORMANAGER_HH */
