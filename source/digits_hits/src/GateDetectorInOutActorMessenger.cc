/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateDetectorInOutActorMessenger.hh"
#include "GateDetectorInOutActor.hh"

//-----------------------------------------------------------------------------
GateDetectorInOutActorMessenger::GateDetectorInOutActorMessenger(GateDetectorInOutActor* sensor):
  GateActorMessenger(sensor), pDIOActor(sensor)
{
  BuildCommands(baseName + sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateDetectorInOutActorMessenger::~GateDetectorInOutActorMessenger()
{
  delete pSetOutputWindowNamesCmd;
  delete pSetOutputInDataOnlyFlagCmd;
  delete pSetMaxAngleCmd;
  delete pSetRRFactorCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActorMessenger::BuildCommands(G4String base)
{
  G4String n = base + "/setOutputWindowNames";
  pSetOutputWindowNamesCmd = new G4UIcmdWithAString(n, this);
  auto guid = G4String("Set the names of the energy windows to output (e.g. scatter, peak1 etc)");
  pSetOutputWindowNamesCmd->SetGuidance(guid);

  n = base + "/setOutputInDataOnly";
  pSetOutputInDataOnlyFlagCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("If 'true', onlye output the In Data (and not the Out)");
  pSetOutputInDataOnlyFlagCmd->SetGuidance(guid);

  n = base + "/setMaxAngle";
  pSetMaxAngleCmd = new G4UIcmdWithADoubleAndUnit(n, this);
  guid = G4String("Do not store data if angle larger than the given treshold ()in degree)");
  pSetMaxAngleCmd->SetGuidance(guid);

  n = base + "/setOutsideRussianRouletteFactor";
  pSetRRFactorCmd = new G4UIcmdWithAnInteger(n, this);
  guid = G4String("Apply Russian Roulette VRT to data outside all energy windows. The given nb is the factor (1/w w=weight), integer (e.g. 50)");
  pSetRRFactorCmd->SetGuidance(guid);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDetectorInOutActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pSetOutputWindowNamesCmd)    pDIOActor->SetOutputWindowNames(newValue);
  if (cmd == pSetOutputInDataOnlyFlagCmd) pDIOActor->SetOutputInDataOnlyFlag(pSetOutputInDataOnlyFlagCmd->GetNewBoolValue(newValue));
  if (cmd == pSetMaxAngleCmd)             pDIOActor->SetMaxAngle(pSetMaxAngleCmd->GetNewDoubleValue(newValue));
  if (cmd == pSetRRFactorCmd)             pDIOActor->SetRRFactor(pSetRRFactorCmd->GetNewIntValue(newValue));
  GateActorMessenger::SetNewValue(cmd, newValue);
}
//-----------------------------------------------------------------------------
