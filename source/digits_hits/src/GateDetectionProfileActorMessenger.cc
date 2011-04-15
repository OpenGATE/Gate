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

GateDetectionProfileActorMessenger::GateDetectionProfileActorMessenger(GateDetectionProfileActor * v)
: GateActorMessenger(v),
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

GateDetectionProfileActorMessenger::~GateDetectionProfileActorMessenger()
{
  //delete pEmaxCmd;
  //delete pEminCmd;
}

void GateDetectionProfileActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  //if(cmd == pEminCmd) pActor->SetEmin(  pEminCmd->GetNewDoubleValue(newValue)  ) ;
  GateActorMessenger::SetNewValue(cmd,newValue);
}


GateDetectionProfilePrimaryTimerActorMessenger::GateDetectionProfilePrimaryTimerActorMessenger(GateDetectionProfilePrimaryTimerActor * v)
: GateActorMessenger(v),
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
