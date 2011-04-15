/*!
  \class  GateDetectionProfileActor
  \author pierre.gueth@creatis.insa-lyon.fr
 */

#ifndef GATEDETECTIONPROFILEACTOR_HH
#define GATEDETECTIONPROFILEACTOR_HH

#include "GateVActor.hh"
#include "GateActorManager.hh"
#include "GateDetectionProfileActorMessenger.hh"

class GateDetectionProfileActor : public GateVActor
{
  public: 
    virtual ~GateDetectionProfileActor();

    FCT_FOR_AUTO_CREATOR_ACTOR(GateDetectionProfileActor)
    virtual void Construct();

    virtual void BeginOfRunAction(const G4Run * r);
    virtual void BeginOfEventAction(const G4Event *) ;
    virtual void UserSteppingAction(const GateVVolume *, const G4Step*);
    virtual void PreUserTrackingAction(const GateVVolume *, const G4Track*) ;
    virtual void PostUserTrackingAction(const GateVVolume *, const G4Track*) ;
    virtual void EndOfEventAction(const G4Event*);

    virtual void SaveData();
    virtual void ResetData();

  protected:
    GateDetectionProfileActor(G4String name, G4int depth=0);

    GateDetectionProfileActorMessenger * pMessenger;
};

MAKE_AUTO_CREATOR_ACTOR(DetectionProfileActor,GateDetectionProfileActor)

class GateDetectionProfilePrimaryTimerActor : public GateVActor
{
  public: 
    virtual ~GateDetectionProfilePrimaryTimerActor();

    FCT_FOR_AUTO_CREATOR_ACTOR(GateDetectionProfilePrimaryTimerActor)
    virtual void Construct();

    virtual void BeginOfRunAction(const G4Run * r);
    virtual void BeginOfEventAction(const G4Event *) ;
    virtual void UserSteppingAction(const GateVVolume *, const G4Step*);
    virtual void PreUserTrackingAction(const GateVVolume *, const G4Track*) ;
    virtual void PostUserTrackingAction(const GateVVolume *, const G4Track*) ;
    virtual void EndOfEventAction(const G4Event*);

    virtual void SaveData();
    virtual void ResetData();

  protected:
    GateDetectionProfilePrimaryTimerActor(G4String name, G4int depth=0);

    GateDetectionProfilePrimaryTimerActorMessenger * pMessenger;
};

MAKE_AUTO_CREATOR_ACTOR(DetectionProfilePrimaryTimerActor,GateDetectionProfilePrimaryTimerActor)

#endif
