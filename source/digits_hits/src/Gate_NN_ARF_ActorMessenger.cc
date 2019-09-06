/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "Gate_NN_ARF_ActorMessenger.hh"
#include "Gate_NN_ARF_Actor.hh"

//-----------------------------------------------------------------------------
Gate_NN_ARF_ActorMessenger::Gate_NN_ARF_ActorMessenger(Gate_NN_ARF_Actor* sensor):
  GateActorMessenger(sensor), pDIOActor(sensor)
{
  BuildCommands(baseName + sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
Gate_NN_ARF_ActorMessenger::~Gate_NN_ARF_ActorMessenger()
{
  delete pSetEnergyWindowNamesCmd;
  delete pSetModeFlagCmd;
  delete pSetMaxAngleCmd;
  delete pSetRRFactorCmd;
  delete pSetNNModelCmd;
  delete pSetNNDictCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_ActorMessenger::BuildCommands(G4String base)
{
  G4String n = base + "/setEnergyWindowNames";
  pSetEnergyWindowNamesCmd = new G4UIcmdWithAString(n, this);
  auto guid = G4String("Set the names of the energy windows to output (e.g. scatter, peak1 etc)");
  pSetEnergyWindowNamesCmd->SetGuidance(guid);

  n = base + "/setMode";
  pSetModeFlagCmd = new G4UIcmdWithAString(n, this);
  guid = G4String("If 'train': store [theta phi E w]. If 'test': store [x y theta phi E]");
  // if 'trainE', store [theta phi E Ew]
  pSetModeFlagCmd->SetGuidance(guid);

  n = base + "/setMaxAngle";
  pSetMaxAngleCmd = new G4UIcmdWithADoubleAndUnit(n, this);
  guid = G4String("Do not store data if angle larger than the given treshold (in degree, only for 'train' mode)");
  pSetMaxAngleCmd->SetGuidance(guid);

  n = base + "/setRussianRoulette";
  pSetRRFactorCmd = new G4UIcmdWithAnInteger(n, this);
  guid = G4String("Apply Russian Roulette VRT to data outside all energy windows. The given nb is the factor (1/w w=weight), integer (e.g. 50)");
  pSetRRFactorCmd->SetGuidance(guid);

  n = base + "/setNNModel";
  pSetNNModelCmd = new G4UIcmdWithAString(n, this);
  guid = G4String("Path to the neural network model in .pt");
  pSetNNModelCmd->SetGuidance(guid);

  n = base + "/setNNDict";
  pSetNNDictCmd = new G4UIcmdWithAString(n, this);
  guid = G4String("Path to the neural network dictionary in .json");
  pSetNNDictCmd->SetGuidance(guid);

  n = base + "/setImage";
  pSetImageCmd = new G4UIcmdWithAString(n, this);
  guid = G4String("Path to the output image of the NN");
  pSetImageCmd->SetGuidance(guid);

  n = base + "/setSpacingX";
  pSetSpacingXCmd = new G4UIcmdWithADoubleAndUnit(n, this);
  guid = G4String("Spacing of the output image along X");
  pSetSpacingXCmd->SetGuidance(guid);
  pSetSpacingXCmd->SetDefaultUnit("mm");

  n = base + "/setSpacingY";
  pSetSpacingYCmd = new G4UIcmdWithADoubleAndUnit(n, this);
  guid = G4String("Spacing of the output image along Y");
  pSetSpacingYCmd->SetGuidance(guid);
  pSetSpacingYCmd->SetDefaultUnit("mm");

  n = base + "/setSizeX";
  pSetSizeXCmd = new G4UIcmdWithAnInteger(n, this);
  guid = G4String("Size of the output image in pixel along X");
  pSetSizeXCmd->SetGuidance(guid);

  n = base + "/setSizeY";
  pSetSizeYCmd = new G4UIcmdWithAnInteger(n, this);
  guid = G4String("Size of the output image in pixel along Y");
  pSetSizeYCmd->SetGuidance(guid);

  n = base + "/setCollimatorLength";
  pSetCollimatorLengthCmd = new G4UIcmdWithADoubleAndUnit(n, this);
  guid = G4String("collimator+ half crystal length");
  pSetCollimatorLengthCmd->SetGuidance(guid);
  pSetCollimatorLengthCmd->SetDefaultUnit("mm");

  n = base + "/setBatchSize";
  pSetBatchSizeCmd = new G4UIcmdWithADouble(n, this);
  guid = G4String("Batch size for GPU. Large value is faster, but may require too much GPU memory.");
  pSetBatchSizeCmd->SetGuidance(guid);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_ActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pSetEnergyWindowNamesCmd)    pDIOActor->SetEnergyWindowNames(newValue);
  if (cmd == pSetModeFlagCmd)             pDIOActor->SetMode(newValue);
  if (cmd == pSetMaxAngleCmd)             pDIOActor->SetMaxAngle(pSetMaxAngleCmd->GetNewDoubleValue(newValue));
  if (cmd == pSetRRFactorCmd)             pDIOActor->SetRRFactor(pSetRRFactorCmd->GetNewIntValue(newValue));
  if (cmd == pSetNNModelCmd)              pDIOActor->SetNNModel(newValue);
  if (cmd == pSetNNDictCmd)               pDIOActor->SetNNDict(newValue);
  if (cmd == pSetImageCmd)                pDIOActor->SetImage(newValue);
  if (cmd == pSetSpacingXCmd)             pDIOActor->SetSpacing(pSetSpacingXCmd->GetNewDoubleValue(newValue), 0);
  if (cmd == pSetSpacingYCmd)             pDIOActor->SetSpacing(pSetSpacingYCmd->GetNewDoubleValue(newValue), 1);
  if (cmd == pSetSizeXCmd)                pDIOActor->SetSize(pSetSizeXCmd->GetNewIntValue(newValue), 0);
  if (cmd == pSetSizeYCmd)                pDIOActor->SetSize(pSetSizeYCmd->GetNewIntValue(newValue), 1);
  if (cmd == pSetCollimatorLengthCmd)     pDIOActor->SetCollimatorLength(pSetCollimatorLengthCmd->GetNewDoubleValue(newValue));
  if (cmd == pSetBatchSizeCmd)            pDIOActor->SetBatchSize(pSetBatchSizeCmd->GetNewDoubleValue(newValue));
  GateActorMessenger::SetNewValue(cmd, newValue);
}
//-----------------------------------------------------------------------------
