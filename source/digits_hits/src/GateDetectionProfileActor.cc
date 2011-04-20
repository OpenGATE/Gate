/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GATEDETECTIONPROFILEACTOR_CC
#define GATEDETECTIONPROFILEACTOR_CC

#ifdef G4ANALYSIS_USE_ROOT

#include "GateDetectionProfileActor.hh"
#include "GateMiscFunctions.hh"

GateDetectionProfileActor::GateDetectionProfileActor(G4String name, G4int depth):
  GateVImageActor(name,depth)
{
  messenger  = new GateDetectionProfileActorMessenger(this);
  timerActor = NULL;
  firstStepForTrack = true;
  distanceThreshold = 0;
  detectionPosition = Beam;
}

void GateDetectionProfileActor::SetTimer(const G4String &timerName)
{
  GateVActor *abstractActor = GateActorManager::GetInstance()->GetActor("GateDetectionProfilePrimaryTimerActor",timerName);
  if (!abstractActor) {
    GateWarning("can't find timer actor named " << timerName << G4endl);
    return;
  }
  timerActor = dynamic_cast<GateDetectionProfilePrimaryTimerActor*>(abstractActor);
}

GateDetectionProfileActor::~GateDetectionProfileActor()
{
  delete messenger;
}

void GateDetectionProfileActor::SetDistanceThreshold(double distance)
{
  distanceThreshold = distance;
}

void GateDetectionProfileActor::SetDetectionPosition(GateDetectionProfileActor::DetectionPosition type)
{
  detectionPosition = type;
}

void GateDetectionProfileActor::Construct()
{
  GateVImageActor::Construct();

  if (!timerActor) GateError("set timer actor");

  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(false);
  EnablePreUserTrackingAction(true);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);
  EnableEndOfEventAction(false);

  mImage.Allocate();
  //G4cout << "resolution=" << mImage.GetResolution() << " halfSize=" << mImage.GetHalfSize() << G4endl;

  ResetData();
}

void GateDetectionProfileActor::SaveData()
{
  mImage.Write(mSaveFilename);
}

void GateDetectionProfileActor::ResetData() 
{
  mImage.Fill(0);
}

void GateDetectionProfileActor::UserPreTrackActionInVoxel(const int, const G4Track *)
{
  firstStepForTrack = true;
}

void GateDetectionProfileActor::UserPostTrackActionInVoxel(const int, const G4Track *)
{
}

void GateDetectionProfileActor::UserSteppingActionInVoxel(const int, const G4Step *step)
{
  if (!firstStepForTrack) return;
  firstStepForTrack = false;

  const bool isSecondary = (step->GetTrack()->GetLogicalVolumeAtVertex()==mVolume->GetLogicalVolume()); //FIXME dirty hacked to know is the particle was created inside the volume
  if (isSecondary) return;

  if (!timerActor->IsTriggered()) return;

  const G4StepPoint *point = step->GetPreStepPoint();
  const GateDetectionProfilePrimaryTimerActor::TriggerData &triggerData = timerActor->GetTriggerData();

  const G4double deltaTime = point->GetGlobalTime()-triggerData.time;
  const G4double weight = point->GetWeight();

  // find minimum distance between two lines according to 
  // http://softsurfer.com/Archive/algorithm_0106/algorithm_0106.htm
  G4double minDistance;
  G4ThreeVector minPosition;
  {
    const G4double a = triggerData.direction.mag2();
    const G4double b = triggerData.direction.dot(point->GetMomentumDirection());
    const G4double c = point->GetMomentumDirection().mag2();
    const G4ThreeVector w0 = triggerData.position - point->GetPosition();
    const G4double d = w0.dot(triggerData.direction);
    const G4double e = w0.dot(point->GetMomentumDirection());
    //G4cout << "a=" << a << " b=" << b << " c=" << c << " d=" << d << " e=" << e << G4endl;

    const G4double det = a*c-b*b;
    if (det==0) return; // lines are parallel so there is no precise detection point that can be precised
    const G4double sc = (b*e - c*d)/det;
    const G4double tc = (a*e - b*d)/det;
    //G4cout << "det=" << det << " sc=" << sc << " tc=" << tc << G4endl;

    const G4ThreeVector minBeam = triggerData.position + triggerData.direction*sc;
    const G4ThreeVector minDetected = point->GetPosition() + point->GetMomentumDirection()*tc;
    minDistance = (minBeam-minDetected).mag();
    //G4cout << "minDistance=" << minDistance << " minBeam=" << minBeam << " minDetected=" << minDetected << G4endl;

    switch (detectionPosition) {
      case Beam:
	minPosition = minBeam;
	break;
      case Detected:
	minPosition = minDetected;
	break;
      case Middle:
	minPosition = (minDetected + minBeam)/2.;
	break;
    }
  }

  if (distanceThreshold>0 && minDistance>distanceThreshold) return;

  int index = mImage.GetIndexFromPosition(minPosition);
  if (index>=0) mImage.AddValue(index,weight);

  GateMessage("Actor",0,
    "detector hittted" <<
    " name=" << step->GetTrack()->GetParticleDefinition()->GetParticleName() << 
    " flytime=" << deltaTime/ns << 
    " position=" << minPosition/mm << 
    " distance=" << minDistance << 
    " index=" << index << G4endl);
}

GateDetectionProfilePrimaryTimerActor::GateDetectionProfilePrimaryTimerActor(G4String name, G4int depth):
  GateVActor(name,depth)
{
  messenger = new GateDetectionProfilePrimaryTimerActorMessenger(this);
  rootFile  = NULL;
  triggered = false;
}

GateDetectionProfilePrimaryTimerActor::~GateDetectionProfilePrimaryTimerActor()
{
  delete messenger;
}

void GateDetectionProfilePrimaryTimerActor::Construct()
{
  GateVActor::Construct();

  // Enable callbacks
  EnableBeginOfRunAction(false);
  EnableBeginOfEventAction(true);
  EnablePreUserTrackingAction(false);
  EnablePostUserTrackingAction(false);
  EnableUserSteppingAction(true);
  EnableEndOfEventAction(false);

  rootFile = new TFile(mSaveFilename,"RECREATE");

  histoTime = new TH1D("triggerTime","Trigger Time",100,0.,1.);
  histoTime->SetXTitle("time [ns]");

  histoPosition = new TH2D("triggerPosition","Trigger Position",101,-5.,5.,101,-5.,5.);
  histoPosition->SetXTitle("x [mm]");
  histoPosition->SetYTitle("y [mm]");

  histoDirz = new TH1D("triggerDirection","Trigger Direction",100,.5,1.5);

  ResetData();
}

bool GateDetectionProfilePrimaryTimerActor::IsTriggered() const
{
  return triggered;
}

const GateDetectionProfilePrimaryTimerActor::TriggerData &GateDetectionProfilePrimaryTimerActor::GetTriggerData() const
{
  return data;
}

void GateDetectionProfilePrimaryTimerActor::SaveData()
{
  rootFile->Write();
  rootFile->Close();
}

void GateDetectionProfilePrimaryTimerActor::ResetData() 
{
  histoTime->Reset();
  histoPosition->Reset();
}

void GateDetectionProfilePrimaryTimerActor::BeginOfEventAction(const G4Event*)
{
  triggered = false;
}

void GateDetectionProfilePrimaryTimerActor::UserSteppingAction(const GateVVolume*, const G4Step *step)
{
  if (triggered) return;
  if (!step->IsLastStepInVolume()) return;
  triggered = true;

  const G4StepPoint *point = step->GetPostStepPoint();

  data.name = step->GetTrack()->GetDefinition()->GetParticleName();
  data.position = point->GetPosition();
  data.direction = point->GetMomentumDirection();
  data.time = point->GetGlobalTime();
  G4double weight = point->GetWeight();

  histoTime->Fill(data.time,weight);
  histoPosition->Fill(data.position[0],data.position[1],weight);
  histoDirz->Fill(data.direction[2],weight);

  GateMessage("Actor",4,"triggered by " << data.name << " at " << data.time/ns << "ns " << data.position/mm << "mm" << G4endl);
}

#endif 
#endif 
