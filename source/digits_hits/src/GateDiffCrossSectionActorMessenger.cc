/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#include "GateDiffCrossSectionActorMessenger.hh"
#include "GateDiffCrossSectionActor.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"


//-----------------------------------------------------------------------------
GateDiffCrossSectionActorMessenger::GateDiffCrossSectionActorMessenger( GateDiffCrossSectionActor* sensor):GateActorMessenger( sensor), pDiffCrossSectionActor(sensor)
{
  BuildCommands( baseName + sensor->GetObjectName( ));
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateDiffCrossSectionActorMessenger::~GateDiffCrossSectionActorMessenger()
{
  delete pSetEnergyCmd;
  delete pSetAngleCmd;
  delete pReadListEnergyCmd;
  delete pReadListAngleCmd;
  delete pSetMaterialCmd;
  delete pReadMaterialListCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDiffCrossSectionActorMessenger::BuildCommands( G4String base)
{
  G4String n;
  G4String guid;
  n = base+"/setEnergy";
  pSetEnergyCmd = new G4UIcmdWithADoubleAndUnit(n, this);
  guid = G4String( "Set the energy");
  pSetEnergyCmd->SetGuidance( guid);

  n = base+"/readEnergyList";
  pReadListEnergyCmd = new G4UIcmdWithAString(n, this);
  guid = G4String( "List of energies");
  pReadListEnergyCmd->SetGuidance( guid);

  n = base+"/setAngle";
  pSetAngleCmd = new G4UIcmdWithADoubleAndUnit(n, this);
  guid = G4String( "Set the angle");
  pSetAngleCmd->SetGuidance( guid);

  n = base+"/readAngleList";
  pReadListAngleCmd = new G4UIcmdWithAString(n, this);
  guid = G4String( "List of angles");
  pReadListAngleCmd->SetGuidance( guid);

  n = base+"/setMaterial";
  pSetMaterialCmd = new G4UIcmdWithAString(n, this);
  guid = G4String( "Material to Mu calcul");
  pSetMaterialCmd->SetGuidance( guid);

  n = base+"/readMaterialList";
  pReadMaterialListCmd = new G4UIcmdWithAString(n, this);
  guid = G4String( "List of materials");
  pReadMaterialListCmd->SetGuidance( guid);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateDiffCrossSectionActorMessenger::SetNewValue( G4UIcommand* cmd, G4String newValue)
{
  if( cmd == pSetEnergyCmd)
    {
      pDiffCrossSectionActor->SetEnergy(pSetEnergyCmd->GetNewDoubleValue(newValue));
    }
  if( cmd == pReadListEnergyCmd)
    {
      pDiffCrossSectionActor->ReadListEnergy(newValue);
    }

  if( cmd == pSetAngleCmd)
    {
      pDiffCrossSectionActor->SetAngle(pSetAngleCmd->GetNewDoubleValue(newValue));
    }
  if( cmd == pReadListAngleCmd)
    {
      pDiffCrossSectionActor->ReadListAngle(newValue);
    }
  if( cmd == pSetMaterialCmd)
    {
      pDiffCrossSectionActor->SetMaterial(newValue);
    }
  if( cmd == pReadMaterialListCmd)
    {
      pDiffCrossSectionActor->ReadMaterialList(newValue);
      //pMuCalculatorActor->ReadListEnergy(pReadListEnergyCmd->GetNewStringValue(newValue));
    }
  GateActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------
