/*----------------------
   GATE version name: gate_v6

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
#include "G4String.hh"
#include <iomanip>   
#include <vector>
#include <map>

#include "G4SDManager.hh"

#include "GateMessageManager.hh"

#include "GateVActor.hh"
#include "GateVFilter.hh"

#include "GateMultiSensitiveDetector.hh"
#include "G4MultiFunctionalDetector.hh"

#include "GateActorManagerMessenger.hh"
#include "GateObjectChildList.hh"

class GateActorManager
{
public:
  ~GateActorManager();

  static GateActorManager *GetInstance()
  {   
    if (singleton_ActorManager == 0)
    {
      //std::cout << "creating GateActorManager..." << std::endl;
      singleton_ActorManager = new GateActorManager;
    }
    //else std::cout << "GateActorManager already created!" << std::endl;
    return singleton_ActorManager;
  };

  void AddActor(G4String actorType, G4String actorName, int depth=0);
  void CreateListsOfEnabledActors();
  void GetListOfActors();
  //added by Emilia
  GateVActor*  GetActor(G4String actorType, G4String actorName, int depth=0);
  inline virtual void SetActor(GateVActor* anActor) { m_actor = anActor; }; 
  inline GateVActor*  GetActor() {return m_actor;};
  
  std::vector<GateVActor*> ReturnListOfActors();
  
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

  std::vector<GateVActor*> & GetTheListOfActors() { return theListOfActors; }

protected:
  //std::vector<GateMultiSensitiveDetector*> theListOfMultiSensitiveDetector;
  std::vector<GateVActor*> theListOfActors;
  std::vector<GateVActor*> theListOfActorsEnabledForBeginOfRun;
  std::vector<GateVActor*> theListOfActorsEnabledForEndOfRun;
  std::vector<GateVActor*> theListOfActorsEnabledForBeginOfEvent;
  std::vector<GateVActor*> theListOfActorsEnabledForEndOfEvent;
  std::vector<GateVActor*> theListOfActorsEnabledForPreUserTrackingAction;
  std::vector<GateVActor*> theListOfActorsEnabledForPostUserTrackingAction;
  std::vector<GateVActor*> theListOfActorsEnabledForUserSteppingAction;

  GateActorManagerMessenger* pActorManagerMessenger;  //pointer to the Messenger

private:
  int IsInitialized;

  GateVActor* m_actor;
  GateActorManager();
  static GateActorManager *singleton_ActorManager;
};

#endif /* end #define GATEACTORMANAGER_HH */
