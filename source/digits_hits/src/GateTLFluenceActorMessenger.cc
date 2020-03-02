/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATETLFLUENCEACTORMESSENGER_CC
#define GATETLFLUENCEACTORMESSENGER_CC

#include "GateTLFluenceActorMessenger.hh"

#include "GateTLFluenceActor.hh"

//-----------------------------------------------------------------------------
GateTLFluenceActorMessenger::GateTLFluenceActorMessenger(GateTLFluenceActor* sensor)
  :GateImageActorMessenger(sensor),
   pFluenceActor(sensor)
{
  pEnableFluenceCmd= 0;
  pEnableEnergyFluenceCmd = 0;
  pEnableFluenceSquaredCmd= 0;
  pEnableFluenceUncertaintyCmd= 0;
  pEnableEnergyFluenceSquaredCmd= 0;
  pEnableEnergyFluenceUncertaintyCmd= 0;

  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateTLFluenceActorMessenger::~GateTLFluenceActorMessenger()
{
  if(pEnableFluenceCmd) delete pEnableFluenceCmd;
  if(pEnableEnergyFluenceCmd) delete pEnableEnergyFluenceCmd;
  
  if(pEnableFluenceSquaredCmd) delete pEnableFluenceSquaredCmd;
  if(pEnableEnergyFluenceSquaredCmd) delete pEnableEnergyFluenceSquaredCmd;
  
  if(pEnableFluenceUncertaintyCmd) delete pEnableFluenceUncertaintyCmd;
  if(pEnableEnergyFluenceUncertaintyCmd) delete pEnableEnergyFluenceUncertaintyCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTLFluenceActorMessenger::BuildCommands(G4String base)
{

  G4String  n = base+"/enableFluence";
  pEnableFluenceCmd = new G4UIcmdWithABool(n, this);
  G4String guid = G4String("Enable fluence computation");
  pEnableFluenceCmd->SetGuidance(guid);

  n = base+"/enableEnergyFluence";
  pEnableEnergyFluenceCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable energy fluence computation");
  pEnableEnergyFluenceCmd->SetGuidance(guid);

  n = base+"/enableSquaredFluence";
  pEnableFluenceSquaredCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable squared fluence computation");
  pEnableFluenceSquaredCmd->SetGuidance(guid);

  n = base+"/enableFluenceUncertainty";
  pEnableFluenceUncertaintyCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable fluence uncertainty computation");
  pEnableFluenceUncertaintyCmd->SetGuidance(guid);

  n = base+"/enableSquaredEnergyFluence";
  pEnableEnergyFluenceSquaredCmd = new G4UIcmdWithABool(n, this);
  guid = G4String("Enable squared energy fluence computation");
  pEnableEnergyFluenceSquaredCmd->SetGuidance(guid);

  n = base+"/enableEnergyFluenceUncertainty";
  pEnableEnergyFluenceUncertaintyCmd= new G4UIcmdWithABool(n, this);
  guid = G4String("Enable energy fluence uncertainty computation");
  pEnableEnergyFluenceUncertaintyCmd->SetGuidance(guid);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTLFluenceActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pEnableFluenceCmd) pFluenceActor->EnableFluenceImage(pEnableFluenceCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableEnergyFluenceCmd) pFluenceActor->EnableEnergyFluenceImage(pEnableEnergyFluenceCmd->GetNewBoolValue(newValue));
  if (cmd == pEnableFluenceSquaredCmd) pFluenceActor->EnableFluenceSquaredImage(pEnableFluenceSquaredCmd->GetNewBoolValue(newValue));
 if (cmd == pEnableFluenceUncertaintyCmd) pFluenceActor->EnableFluenceUncertaintyImage(pEnableFluenceUncertaintyCmd->GetNewBoolValue(newValue));
 if (cmd == pEnableEnergyFluenceSquaredCmd) pFluenceActor->EnableEnergyFluenceSquaredImage(pEnableEnergyFluenceSquaredCmd->GetNewBoolValue(newValue));
 if (cmd == pEnableEnergyFluenceUncertaintyCmd) pFluenceActor->EnableEnergyFluenceUncertaintyImage(pEnableEnergyFluenceUncertaintyCmd->GetNewBoolValue(newValue));

 GateImageActorMessenger::SetNewValue(cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATETLEDOSEACTORMESSENGER_CC */
