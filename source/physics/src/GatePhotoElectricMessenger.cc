/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/



#include "GatePhotoElectricMessenger.hh"

#include "G4UIcmdWithADoubleAndUnit.hh"

//-----------------------------------------------------------------------------
GatePhotoElectricMessenger::GatePhotoElectricMessenger(GateVProcess *pb):GateEMStandardProcessMessenger(pb)
{
  mAuger = false;
  mLowEnergyElectron = -1.;
  mLowEnergyGamma = -1.;
  BuildCommands(pb->GetG4ProcessName() );
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePhotoElectricMessenger::~GatePhotoElectricMessenger()
{
  delete pActiveAugerCmd;
  delete pSetLowEElectron;
  delete pSetLowEGamma;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePhotoElectricMessenger::BuildCommands(G4String base)
{

  G4String baseModel ="";
  baseModel +=  mPrefix;
  baseModel += base;

  G4String bb;
  G4String guidance;

  bb = baseModel+"/setAugerElectron";
  pActiveAugerCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Activation of Auger electron production";
  pActiveAugerCmd->SetGuidance(guidance);
 
  bb = baseModel+"/setDeltaRayCut";
  pSetLowEElectron = new G4UIcmdWithADoubleAndUnit(bb,this);
  guidance = "Set the production cut (in energy) for delta-rays by low-energy photo-electric";
  pSetLowEElectron->SetGuidance(guidance);

  bb = baseModel+"/setXRayCut";
  pSetLowEGamma = new G4UIcmdWithADoubleAndUnit(bb,this);
  guidance = "Set the production cut (in energy) for X-rays by low-energy photo-electric";
  pSetLowEGamma->SetGuidance(guidance);

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePhotoElectricMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if(command == pActiveAugerCmd){
    mAuger =  pActiveAugerCmd->GetNewBoolValue(param);
  }
  if(command == pSetLowEElectron){
    mLowEnergyElectron = pSetLowEElectron->GetNewDoubleValue(param);
  }
  if(command == pSetLowEGamma){
    mLowEnergyGamma = pSetLowEGamma->GetNewDoubleValue(param);
  }

  GateEMStandardProcessMessenger::SetNewValue( command,  param);

}
//-----------------------------------------------------------------------------

