/*!
  \class  GateDetectionProfileActor
  \author pierre.gueth@creatis.insa-lyon.fr
 */

#ifndef GATEDETECTIONPROFILEACTOR_HH
#define GATEDETECTIONPROFILEACTOR_HH

#ifdef G4ANALYSIS_USE_ROOT
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>

#include "GateVActor.hh"
#include "GateActorManager.hh"
#include "GateDetectionProfileActorMessenger.hh"

class GateDetectionProfileActor : public GateVActor
{
  public: 
    enum DetectionPosition { Beam, Detected, Middle };
    void SetTimer(const G4String &timerName);
    void SetDistanceThreshold(double distance);
    void SetDetectionPosition(DetectionPosition type);

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

    GateDetectionProfileActorMessenger *messenger;
    GateDetectionProfilePrimaryTimerActor *timerActor;
    bool firstStepForTrack;
    G4double distanceThreshold;
    DetectionPosition detectionPosition;
};

MAKE_AUTO_CREATOR_ACTOR(DetectionProfileActor,GateDetectionProfileActor)

class GateDetectionProfilePrimaryTimerActor : public GateVActor
{
  public: 
    struct TriggerData {
      G4double time;
      G4String name;
      G4ThreeVector position;
      G4ThreeVector direction;
    };

    bool IsTriggered() const;
    const TriggerData &GetTriggerData() const;

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
    TFile *rootFile;
    GateDetectionProfilePrimaryTimerActorMessenger *messenger;
    bool triggered;
    TriggerData data;
    TH1D *histoTime;
    TH1D *histoDirz;
    TH2D *histoPosition;
};

MAKE_AUTO_CREATOR_ACTOR(DetectionProfilePrimaryTimerActor,GateDetectionProfilePrimaryTimerActor)

#endif
#endif
