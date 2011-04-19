/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GATEDETECTIONPROFILEACTORMESSENGER_CC
#define GATEDETECTIONPROFILEACTORMESSENGER_CC

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
    cmdSetThreshold = new G4UIcmdWithADoubleAndUnit((base+"/setDistanceThreshold").c_str(),this);
    cmdSetThreshold->SetGuidance("Set distance threshold to reject bad reconstruction. value<=0 means no thresholding is performed.");
    cmdSetThreshold->SetUnitCategory("Length");
    cmdSetThreshold->SetParameterName("maxDistance",false);
  }
  {
    cmdSetDetectionPosition = new G4UIcmdWithAString((base+"/setDetectionPosition").c_str(),this);
    cmdSetDetectionPosition->SetGuidance("Set which position is used for fill profiles.");
    cmdSetDetectionPosition->SetCandidates("beam detected middle");
    cmdSetDetectionPosition->SetParameterName("detectedPosition",false);
  }
}

GateDetectionProfileActorMessenger::~GateDetectionProfileActorMessenger()
{
  delete cmdSetTimer;
  delete cmdSetThreshold;
  delete cmdSetDetectionPosition;
}

void GateDetectionProfileActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd==cmdSetTimer) actor->SetTimer(newValue);
  if (cmd==cmdSetThreshold) actor->SetDistanceThreshold(G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue));
  if (cmd==cmdSetDetectionPosition) {
    if (newValue=="beam") actor->SetDetectionPosition(GateDetectionProfileActor::Beam);
    else if (newValue=="detected") actor->SetDetectionPosition(GateDetectionProfileActor::Detected);
    else if (newValue=="middle") actor->SetDetectionPosition(GateDetectionProfileActor::Middle);
    else assert(false);
  }

  GateImageActorMessenger::SetNewValue(cmd,newValue);
}


GateDetectionProfilePrimaryTimerActorMessenger::GateDetectionProfilePrimaryTimerActorMessenger(GateDetectionProfilePrimaryTimerActor * v) :
  GateActorMessenger(v),
  pActor(v)
{
  //BuildCommands(baseName+pActor->GetObjectName());
  //bb = base+"/energySpectrum/setEmin";
  //pEminCmd = new G4UIcmdWithADoubleAndUnit(bb, this); 
  //guidance = G4String("Set minimum energy of the energy spectrum");
  //pEminCmd->SetGuidance(guidance);
  //pEminCmd->SetParameterName("Emin", false);
  //pEminCmd->SetDefaultUnit("MeV");

  //bb = base+"/energyLossHisto/setNumberOfBins";
  //pEdepNBinsCmd = new G4UIcmdWithAnInteger(bb, this); 
  //guidance = G4String("Set number of bins of the energy loss histogram");
  //pEdepNBinsCmd->SetGuidance(guidance);
  //pEdepNBinsCmd->SetParameterName("Nbins", false);
}

GateDetectionProfilePrimaryTimerActorMessenger::~GateDetectionProfilePrimaryTimerActorMessenger()
{
  //delete pEmaxCmd;
  //delete pEminCmd;
}


void GateDetectionProfilePrimaryTimerActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  //if(cmd == pEminCmd) pActor->SetEmin(  pEminCmd->GetNewDoubleValue(newValue)  ) ;
  GateActorMessenger::SetNewValue(cmd,newValue);
}

#endif
