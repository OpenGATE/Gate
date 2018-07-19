/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#ifndef GATENTLEDOSEACTORMESSENGER_CC
#define GATENTLEDOSEACTORMESSENGER_CC

#include "GateNTLEDoseActorMessenger.hh"
#include "GateNTLEDoseActor.hh"

//-----------------------------------------------------------------------------
GateNTLEDoseActorMessenger::GateNTLEDoseActorMessenger(GateNTLEDoseActor* sensor)
  :GateImageActorMessenger(sensor),
   pDoseActor(sensor)
{
  pEnableEdepCmd                        = 0;
  pEnableEdepSquaredCmd                 = 0;
  pEnableEdepUncertaintyCmd             = 0;

  pEnableDoseCmd                        = 0;
  pEnableDoseSquaredCmd                 = 0;
  pEnableDoseUncertaintyCmd             = 0;

  pEnableFluxCmd                        = 0;
  pEnableFluxSquaredCmd                 = 0;
  pEnableFluxUncertaintyCmd             = 0;

  pEnableDoseCorrectionCmd              = 0;
  pEnableDoseCorrectionTLECmd           = 0;

  pEnableKFExtrapolationCmd             = 0;
  pEnableKFDACmd                        = 0;
  pEnableKermaFactorDumpCmd             = 0;
  pEnableKillSecondaryCmd               = 0;

  pEnableKermaEquivalentFactorCmd       = 0;
  pEnablePhotonKermaEquivalentFactorCmd = 0;

  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateNTLEDoseActorMessenger::~GateNTLEDoseActorMessenger()
{
  if(pEnableEdepCmd)                  delete pEnableEdepCmd;
  if(pEnableEdepSquaredCmd)           delete pEnableEdepSquaredCmd;
  if(pEnableEdepUncertaintyCmd)       delete pEnableEdepUncertaintyCmd;

  if(pEnableDoseCmd)                  delete pEnableDoseCmd;
  if(pEnableDoseSquaredCmd)           delete pEnableDoseSquaredCmd;
  if(pEnableDoseUncertaintyCmd)       delete pEnableDoseUncertaintyCmd;

  if(pEnableFluxCmd)                  delete pEnableFluxCmd;
  if(pEnableFluxSquaredCmd)           delete pEnableFluxSquaredCmd;
  if(pEnableFluxUncertaintyCmd)       delete pEnableFluxUncertaintyCmd;

  if(pEnableDoseCorrectionCmd)        delete pEnableDoseCorrectionCmd;
  if(pEnableDoseCorrectionTLECmd)     delete pEnableDoseCorrectionTLECmd;

  if(pEnableKFExtrapolationCmd)       delete pEnableKFExtrapolationCmd;
  if(pEnableKFDACmd)                  delete pEnableKFDACmd;
  if(pEnableKermaFactorDumpCmd)       delete pEnableKermaFactorDumpCmd;
  if(pEnableKillSecondaryCmd)         delete pEnableKillSecondaryCmd;

  if(pEnableKermaEquivalentFactorCmd)       delete pEnableKermaEquivalentFactorCmd;
  if(pEnablePhotonKermaEquivalentFactorCmd) delete pEnablePhotonKermaEquivalentFactorCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNTLEDoseActorMessenger::BuildCommands(G4String base)
{
  G4String n = base+"/enableEdep";
  pEnableEdepCmd = new G4UIcmdWithABool(n, this);
  G4String guid = G4String("Enable Edep computation");
  pEnableEdepCmd->SetGuidance(guid);

  n = base+"/enableSquaredEdep";
  pEnableEdepSquaredCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable squared Edep computation");
  pEnableEdepSquaredCmd->SetGuidance(guid);

  n = base+"/enableUncertaintyEdep";
  pEnableEdepUncertaintyCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable uncertainty Edep computation");
  pEnableEdepUncertaintyCmd->SetGuidance(guid);


  n = base+"/enableDose";
  pEnableDoseCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable dose computation");
  pEnableDoseCmd->SetGuidance(guid);

  n = base+"/enableSquaredDose";
  pEnableDoseSquaredCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable squared dose computation");
  pEnableDoseSquaredCmd->SetGuidance(guid);

  n = base+"/enableUncertaintyDose";
  pEnableDoseUncertaintyCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable uncertainty dose computation");
  pEnableDoseUncertaintyCmd->SetGuidance(guid);


  n = base+"/enableFlux";
  pEnableFluxCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable neutron flux computation");
  pEnableFluxCmd->SetGuidance(guid);

  n = base+"/enableSquaredFlux";
  pEnableFluxSquaredCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable squared neutron flux computation");
  pEnableFluxSquaredCmd->SetGuidance(guid);

  n = base+"/enableUncertaintyFlux";
  pEnableFluxUncertaintyCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable neuton flux uncertainty computation");
  pEnableFluxUncertaintyCmd->SetGuidance(guid);


  n = base+"/enableDoseCorrection";
  pEnableDoseCorrectionCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable dose correction computation");
  pEnableDoseCorrectionCmd->SetGuidance(guid);

  n = base+"/enableDoseCorrectionTLE";
  pEnableDoseCorrectionTLECmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable dose correction (with TLE) computation");
  pEnableDoseCorrectionTLECmd->SetGuidance(guid);


  n = base+"/enableKermaFactorExtrapolation";
  pEnableKFExtrapolationCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable kerma factor extrapolation under 0.025 eV");
  pEnableKFExtrapolationCmd->SetGuidance(guid);

  n = base+"/enableKermaFactorDA";
  pEnableKFDACmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable kerma factor generated from DoseActor");
  pEnableKFDACmd->SetGuidance(guid);

  n = base+"/enableKermaFactorDump";
  pEnableKermaFactorDumpCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable kerma factor graph dump");
  pEnableKermaFactorDumpCmd->SetGuidance(guid);

  n = base+"/enableKillSecondary";
  pEnableKillSecondaryCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable the killing of secondary particles");
  pEnableKillSecondaryCmd->SetGuidance(guid);


  pEnableKermaEquivalentFactorCmd = new G4UIcmdWithABool(G4String(base+"/enableKermaEquivalentFactor"), this);
  pEnableKermaEquivalentFactorCmd->SetGuidance(G4String("Enable usage of neutron kerma equivalent factors instead of kerma factors"));

  pEnablePhotonKermaEquivalentFactorCmd = new G4UIcmdWithABool(G4String(base+"/enablePhotonKermaEquivalentFactor"), this);
  pEnablePhotonKermaEquivalentFactorCmd->SetGuidance(G4String("Enable usage of photon kerma equivalent factors instead of MuEn"));
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNTLEDoseActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pEnableEdepCmd)                  pDoseActor->EnableEdepImage(pEnableEdepCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableEdepSquaredCmd)           pDoseActor->EnableEdepSquaredImage(pEnableEdepSquaredCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableEdepUncertaintyCmd)       pDoseActor->EnableEdepUncertaintyImage(pEnableEdepUncertaintyCmd->GetNewBoolValue(newValue));

  if (cmd == pEnableDoseCmd)                  pDoseActor->EnableDoseImage(pEnableDoseCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseSquaredCmd)           pDoseActor->EnableDoseSquaredImage(pEnableDoseSquaredCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseUncertaintyCmd)       pDoseActor->EnableDoseUncertaintyImage(pEnableDoseUncertaintyCmd->GetNewBoolValue(newValue));

  if (cmd == pEnableFluxCmd)                  pDoseActor->EnableFluxImage(pEnableFluxCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableFluxSquaredCmd)           pDoseActor->EnableFluxSquaredImage(pEnableFluxSquaredCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableFluxUncertaintyCmd)       pDoseActor->EnableFluxUncertaintyImage(pEnableFluxUncertaintyCmd->GetNewBoolValue(newValue));

  if (cmd == pEnableDoseCorrectionCmd)        pDoseActor->EnableDoseCorrection(pEnableDoseCorrectionCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableDoseCorrectionTLECmd)     pDoseActor->EnableDoseCorrectionTLE(pEnableDoseCorrectionTLECmd->GetNewBoolValue(newValue));

  if (cmd == pEnableKFExtrapolationCmd)       pDoseActor->EnableKFExtrapolation(pEnableKFExtrapolationCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableKFDACmd)                  pDoseActor->EnableKFDA(pEnableKFDACmd->GetNewBoolValue(newValue));
  if (cmd == pEnableKermaFactorDumpCmd)       pDoseActor->EnableKermaFactorDump(pEnableKermaFactorDumpCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableKillSecondaryCmd)         pDoseActor->EnableKillSecondary(pEnableKillSecondaryCmd->GetNewBoolValue(newValue));

  if (cmd == pEnableKermaEquivalentFactorCmd)       pDoseActor->EnableKermaEquivalentFactor(pEnableKermaEquivalentFactorCmd->GetNewBoolValue(newValue));
  if (cmd == pEnablePhotonKermaEquivalentFactorCmd) pDoseActor->EnablePhotonKermaEquivalentFactor(pEnablePhotonKermaEquivalentFactorCmd->GetNewBoolValue(newValue));

  GateImageActorMessenger::SetNewValue(cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif
