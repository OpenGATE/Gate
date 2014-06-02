/*!
*/

#include <GateConfiguration.h>

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
  void SetDeltaEnergyThreshold(double energy);
  void SetDetectionPosition(DetectionPosition type);
  void SetUseCristalNormal(bool state);
  void SetUseCristalPosition(bool state);

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
  bool useCristalNormal;
  bool useCristalPosition;
  G4double detectedEnergy;
  G4double detectedDeltaEnergy;
  G4double detectedWeight;
  G4double detectedTime;
  int detectedIndex;
  G4double distanceThreshold;
  G4double deltaEnergyThreshold;
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
  void SetDetectionSize(double size);
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
  double detectionSize;
  GateDetectionProfilePrimaryTimerActorMessenger *messenger;
  bool triggered;
  TriggerData data;
  TH1D *histoTime;
  TH1D *histoDirz;
  TH2D *histoPosition;

  typedef std::map<G4String,TH2D*> ActorHistos;
  ActorHistos histosTimeEnergy;
  ActorHistos histosTimeDeltaEnergy;
  ActorHistos histosEnergyDeltaEnergy;
  ActorHistos histosEnergyDeltaEnergyPercent;
};

MAKE_AUTO_CREATOR_ACTOR(DetectionProfilePrimaryTimerActor,GateDetectionProfilePrimaryTimerActor)

#endif
