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
  pSetModeFlagCmd->SetGuidance(guid);

  n = base + "/setMaxAngle";
  pSetMaxAngleCmd = new G4UIcmdWithADoubleAndUnit(n, this);
  guid = G4String("Do not store data if angle larger than the given treshold (in degree, only for 'train' mode)");
  pSetMaxAngleCmd->SetGuidance(guid);

  n = base + "/setRussianRoulette";
  pSetRRFactorCmd = new G4UIcmdWithAnInteger(n, this);
  guid = G4String("Apply Russian Roulette VRT to data outside all energy windows. The given nb is the factor (1/w w=weight), integer (e.g. 50)");
  pSetRRFactorCmd->SetGuidance(guid);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void Gate_NN_ARF_ActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pSetEnergyWindowNamesCmd)    pDIOActor->SetEnergyWindowNames(newValue);
  if (cmd == pSetModeFlagCmd)             pDIOActor->SetMode(newValue);
  if (cmd == pSetMaxAngleCmd)             pDIOActor->SetMaxAngle(pSetMaxAngleCmd->GetNewDoubleValue(newValue));
  if (cmd == pSetRRFactorCmd)             pDIOActor->SetRRFactor(pSetRRFactorCmd->GetNewIntValue(newValue));
  GateActorMessenger::SetNewValue(cmd, newValue);
}
//-----------------------------------------------------------------------------
