/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/


#include "GateMuMapActorMessenger.hh"
#include "GateMuMapActor.hh"
#include "GateImageActorMessenger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"

//-----------------------------------------------------------------------------
GateMuMapActorMessenger::GateMuMapActorMessenger(GateMuMapActor* sensor)
  :GateImageActorMessenger(sensor),
  pMuMapActor(sensor)
{
  pSetEnergyCmd= 0;

  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateMuMapActorMessenger::~GateMuMapActorMessenger()
{
  if(pSetEnergyCmd) delete pSetEnergyCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMuMapActorMessenger::BuildCommands(G4String base)
{
  G4String  n = base+"/setEnergy";
  pSetEnergyCmd= new G4UIcmdWithADoubleAndUnit(n, this);
  pSetEnergyCmd->SetGuidance("Set energy value");
  pSetEnergyCmd->SetParameterName("energy",false);
  pSetEnergyCmd->SetDefaultUnit("MeV");

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMuMapActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pSetEnergyCmd)  pMuMapActor->SetEnergy(G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue));

  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

