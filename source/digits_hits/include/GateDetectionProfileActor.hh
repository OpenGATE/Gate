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
#include <map>

#include "GateVImageActor.hh"
#include "GateActorManager.hh"
#include "GateDetectionProfileActorMessenger.hh"

class GateDetectionProfileActor : public GateVImageActor
{
  public: 
    enum DetectionPosition { Beam, Particle, Middle };
    void SetTimer(const G4String &timerName);
    void SetDistanceThreshold(double distance);
    void SetDetectionPosition(DetectionPosition type);

    virtual ~GateDetectionProfileActor();

    FCT_FOR_AUTO_CREATOR_ACTOR(GateDetectionProfileActor)
    virtual void Construct();

    virtual void UserSteppingActionInVoxel(const int index, const G4Step *step);
    virtual void UserPreTrackActionInVoxel(const int index, const G4Track *track);
    virtual void UserPostTrackActionInVoxel(const int index, const G4Track *track);

    virtual void SaveData();
    virtual void ResetData();

  protected:
    GateDetectionProfileActor(G4String name, G4int depth=0);

    GateDetectionProfileActorMessenger *messenger;
    GateDetectionProfilePrimaryTimerActor *timerActor;
    bool firstStepForTrack;
    bool detectedSomething;
    G4double detectedEnergy;
    G4double detectedWeight;
    G4double detectedTime;
    int detectedIndex;
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
    void AddReportForDetector(const G4String &detectorName);
    void ReportDetectedParticle(const G4String &detectorName, double time, double energy, double deltaEnergy, double weight);

    virtual ~GateDetectionProfilePrimaryTimerActor();

    FCT_FOR_AUTO_CREATOR_ACTOR(GateDetectionProfilePrimaryTimerActor)
    virtual void Construct();

    virtual void BeginOfEventAction(const G4Event *) ;
    virtual void UserSteppingAction(const GateVVolume *, const G4Step *);

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
    
    typedef std::map<G4String,TH2D*> HistosTimeEnergy;
    HistosTimeEnergy histosTimeEnergy;
    HistosTimeEnergy histosTimeDeltaEnergy;
    HistosTimeEnergy histosEnergyDeltaEnergy;
};

MAKE_AUTO_CREATOR_ACTOR(DetectionProfilePrimaryTimerActor,GateDetectionProfilePrimaryTimerActor)

#endif
#endif
