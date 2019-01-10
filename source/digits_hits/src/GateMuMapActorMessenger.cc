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
  //add Mu Unit
  new G4UnitDefinition("1/cm","1/cm","Mu",1.0/centimeter);
  new G4UnitDefinition("1/m","1/m","Mu",1.0/meter);
  new G4UnitDefinition("1/mm","1/mm","Mu",1.0/mm);

  pSetEnergyCmd= 0;
  pSetMuUnitCmd= 0;

  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateMuMapActorMessenger::~GateMuMapActorMessenger()
{
  if(pSetEnergyCmd) delete pSetEnergyCmd;
  if(pSetMuUnitCmd) delete pSetMuUnitCmd;
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

  n = base+"/setMuUnit";
  pSetMuUnitCmd= new G4UIcmdWithADoubleAndUnit(n, this);
  pSetMuUnitCmd->SetGuidance("Set Mu Unit");
  pSetMuUnitCmd->SetParameterName("MuUnit",false);
  pSetMuUnitCmd->SetDefaultUnit("1/cm");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMuMapActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pSetEnergyCmd)  pMuMapActor->SetEnergy(G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue));
  if (cmd == pSetMuUnitCmd)  pMuMapActor->SetMuUnit(G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue));

  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

