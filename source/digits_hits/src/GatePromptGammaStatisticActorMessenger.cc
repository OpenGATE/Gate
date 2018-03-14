/*----------------------
  GATE version name: gate_v7

  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/


#include "GatePromptGammaStatisticActorMessenger.hh"
#include "GatePromptGammaStatisticActor.hh"

//-----------------------------------------------------------------------------
GatePromptGammaStatisticActorMessenger::GatePromptGammaStatisticActorMessenger(GatePromptGammaStatisticActor* v)
  : GateActorMessenger(v), pActor(v)
{
  BuildCommands(baseName+pActor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePromptGammaStatisticActorMessenger::~GatePromptGammaStatisticActorMessenger()
{
  delete pProtonEMinCmd;
  delete pProtonEMaxCmd;
  delete pGammaEMinCmd;
  delete pGammaEMaxCmd;
  delete pProtonNbBinsCmd;
  delete pGammaNbBinsCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaStatisticActorMessenger::BuildCommands(G4String base)
{
  G4String bb = base+"/setProtonMinEnergy";
  pProtonEMinCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  G4String guidance = G4String("Set minimum energy of the proton energy spectrum");
  pProtonEMinCmd->SetGuidance(guidance);
  pProtonEMinCmd->SetDefaultUnit("MeV");

  bb = base+"/setProtonMaxEnergy";
  pProtonEMaxCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set maximum energy of the proton energy spectrum");
  pProtonEMaxCmd->SetGuidance(guidance);
  pProtonEMaxCmd->SetDefaultUnit("MeV");

  bb = base+"/setGammaMinEnergy";
  pGammaEMinCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set minimum energy of the gamma energy spectrum");
  pGammaEMinCmd->SetGuidance(guidance);
  pGammaEMinCmd->SetDefaultUnit("MeV");

  bb = base+"/setGammaMaxEnergy";
  pGammaEMaxCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set maximum energy of the gamma energy spectrum");
  pGammaEMaxCmd->SetGuidance(guidance);
  pGammaEMaxCmd->SetDefaultUnit("MeV");

  bb = base+"/setProtonNbBins";
  pProtonNbBinsCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = G4String("Set number of bins of the energy proton histograms");
  pProtonNbBinsCmd->SetGuidance(guidance);
  pProtonNbBinsCmd->SetParameterName("Nbins", false);

  bb = base+"/setGammaNbBins";
  pGammaNbBinsCmd = new G4UIcmdWithAnInteger(bb, this);
  guidance = G4String("Set number of bins of the energy proton histograms");
  pGammaNbBinsCmd->SetGuidance(guidance);
  pGammaNbBinsCmd->SetParameterName("Nbins", false);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePromptGammaStatisticActorMessenger::SetNewValue(G4UIcommand* cmd,
                                                                    G4String newValue)
{

  if (cmd == pProtonEMinCmd) pActor->SetProtonEMin(pProtonEMinCmd->GetNewDoubleValue(newValue));
  if (cmd == pProtonEMaxCmd) pActor->SetProtonEMax(pProtonEMaxCmd->GetNewDoubleValue(newValue));
  if (cmd == pGammaEMinCmd) pActor->SetGammaEMin(pGammaEMinCmd->GetNewDoubleValue(newValue));
  if (cmd == pGammaEMaxCmd) pActor->SetGammaEMax(pGammaEMaxCmd->GetNewDoubleValue(newValue));

  if (cmd == pProtonNbBinsCmd) pActor->SetProtonNbBins(pProtonNbBinsCmd->GetNewIntValue(newValue));
  if (cmd == pGammaNbBinsCmd) pActor->SetGammaNbBins(pGammaNbBinsCmd->GetNewIntValue(newValue));

  GateActorMessenger::SetNewValue(cmd,newValue);
}
//-----------------------------------------------------------------------------
