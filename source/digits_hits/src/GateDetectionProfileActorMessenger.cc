/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateDetectionProfileActorMessenger.hh"
#include "GateDetectionProfileActor.hh"

GateDetectionProfileActorMessenger::GateDetectionProfileActorMessenger(GateDetectionProfileActor *v) :
  GateImageActorMessenger(v),
  actor(v)
{
  G4String base = baseName + actor->GetObjectName();
  {
    cmdSetTimer = new G4UIcmdWithAString((base+"/setTimerActor").c_str(),this);
    cmdSetTimer->SetGuidance("Set timer actor used for triggering. Should be GateDetectionProfilePrimaryTimerActor.");
    cmdSetTimer->SetParameterName("timerActor",false);
  }
  {
    cmdSetDistanceThreshold = new G4UIcmdWithADoubleAndUnit((base+"/setDistanceThreshold").c_str(),this);
    cmdSetDistanceThreshold->SetGuidance("Set distance threshold to reject bad reconstruction. value<=0 means no thresholding is performed. Default is value=0 (no thresholding).");
    cmdSetDistanceThreshold->SetUnitCategory("Length");
    cmdSetDistanceThreshold->SetParameterName("maxDistance",false);
  }
  {
    cmdSetDeltaEnergyThreshold = new G4UIcmdWithADoubleAndUnit((base+"/setDeltaEnergyThreshold").c_str(),this);
    cmdSetDeltaEnergyThreshold->SetGuidance("Set energy deposition threshold to detect event. value<0 means no thresholding is performed. Default is value=-1 (all event used).");
    cmdSetDeltaEnergyThreshold->SetUnitCategory("Energy");
    cmdSetDeltaEnergyThreshold->SetParameterName("minDeltaEnergy",false);
  }
  {
    cmdSetDetectionPosition = new G4UIcmdWithAString((base+"/setDetectionPosition").c_str(),this);
    cmdSetDetectionPosition->SetGuidance("Set which position is used for fill profiles. Default is middle.");
    cmdSetDetectionPosition->SetCandidates("beam particle middle");
    cmdSetDetectionPosition->SetParameterName("detectedPosition",false);
  }
  {
    cmdSetUseCristalNormal = new G4UIcmdWithABool((base+"/useCristalNormal").c_str(),this);
    cmdSetUseCristalNormal->SetGuidance("Use of cristal direction instead of momentum direction in reconstruction algorithm.");
    cmdSetUseCristalNormal->SetParameterName("useCristalNormal",false);
  }
  {
    cmdSetUseCristalPosition = new G4UIcmdWithABool((base+"/useCristalPosition").c_str(),this);
    cmdSetUseCristalPosition->SetGuidance("Use of cristal direction instead of momentum direction in reconstruction algorithm.");
    cmdSetUseCristalPosition->SetParameterName("useCristalPosition",false);
  }
}

GateDetectionProfileActorMessenger::~GateDetectionProfileActorMessenger()
{
  delete cmdSetTimer;
  delete cmdSetDistanceThreshold;
  delete cmdSetDeltaEnergyThreshold;
  delete cmdSetDetectionPosition;
}

void GateDetectionProfileActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd==cmdSetTimer) actor->SetTimer(newValue);
  if (cmd==cmdSetDistanceThreshold) actor->SetDistanceThreshold(G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue));
  if (cmd==cmdSetDeltaEnergyThreshold) actor->SetDeltaEnergyThreshold(G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue));
  if (cmd==cmdSetDetectionPosition) {
    if (newValue=="beam") actor->SetDetectionPosition(GateDetectionProfileActor::Beam);
    else if (newValue=="particle") actor->SetDetectionPosition(GateDetectionProfileActor::Particle);
    else if (newValue=="middle") actor->SetDetectionPosition(GateDetectionProfileActor::Middle);
    else assert(false);
  }
  if (cmd==cmdSetUseCristalNormal) actor->SetUseCristalNormal(G4UIcmdWithABool::GetNewBoolValue(newValue));
  if (cmd==cmdSetUseCristalPosition) actor->SetUseCristalPosition(G4UIcmdWithABool::GetNewBoolValue(newValue));

  GateImageActorMessenger::SetNewValue(cmd,newValue);
}


GateDetectionProfilePrimaryTimerActorMessenger::GateDetectionProfilePrimaryTimerActorMessenger(GateDetectionProfilePrimaryTimerActor * v) :
  GateActorMessenger(v),
  actor(v)
{
  G4String base = baseName + actor->GetObjectName();
  {
    cmdAddReportForDetector = new G4UIcmdWithAString((base+"/addReportForDetector").c_str(),this);
    cmdAddReportForDetector->SetGuidance("Will add flytime / energy report for detector to output file.");
    cmdAddReportForDetector->SetParameterName("detectorActor",false);
  }
  {
    cmdSetDetectionSize = new G4UIcmdWithADoubleAndUnit((base+"/setDetectionSize").c_str(),this);
    cmdSetDetectionSize->SetGuidance("Set the size of detected particles histogram.");
    cmdSetDetectionSize->SetUnitCategory("Length");
    cmdSetDetectionSize->SetParameterName("Size",false);
  }
}

GateDetectionProfilePrimaryTimerActorMessenger::~GateDetectionProfilePrimaryTimerActorMessenger()
{
  delete cmdAddReportForDetector;
  delete cmdSetDetectionSize;
}


void GateDetectionProfilePrimaryTimerActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd==cmdAddReportForDetector) actor->AddReportForDetector(newValue);
  if (cmd==cmdSetDetectionSize) actor->SetDetectionSize(G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue));
  GateActorMessenger::SetNewValue(cmd,newValue);
}
