/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  \class GateThermalActorMessenger
  \brief This class is the GateThermalActor messenger. 
  \author vesna.cuplov@gmail.com
*/

#ifndef GATETHERMALACTORMESSENGER_CC
#define GATETHERMALACTORMESSENGER_CC

// This actor is only compiled if ITK is available
#include "GateConfiguration.h"
#ifdef  GATE_USE_ITK

#include "GateThermalActorMessenger.hh"
#include "GateThermalActor.hh"

//-----------------------------------------------------------------------------
GateThermalActorMessenger::GateThermalActorMessenger(GateThermalActor* sensor)
  :GateImageActorMessenger(sensor),
  pThermalActor(sensor)
{
  pTimeCmd = 0;
  pDiffusivityCmd = 0;
  pBloodPerfusionRateCmd = 0;
  pBloodDensityCmd = 0;
  pBloodHeatCapacityCmd = 0;
  pTissueDensityCmd = 0;
  pTissueHeatCapacityCmd = 0;
  pScaleCmd = 0;
  pNumTimeFramesCmd = 0;


  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateThermalActorMessenger::~GateThermalActorMessenger()
{

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateThermalActorMessenger::BuildCommands(G4String base)
{

  pTimeCmd = new G4UIcmdWithADouble((base+"/setDiffusionTime").c_str(),this);
  pTimeCmd->SetGuidance("Set the diffusion time");

  pDiffusivityCmd = new G4UIcmdWithADouble((base+"/setThermalDiffusivity").c_str(),this);
  pDiffusivityCmd->SetGuidance("Set the tissue thermal diffusivity");

  pBloodPerfusionRateCmd = new G4UIcmdWithADouble((base+"/setBloodPerfusionRate").c_str(),this);
  pBloodPerfusionRateCmd->SetGuidance("Set the blood perfusion rate");

  pBloodDensityCmd = new G4UIcmdWithADouble((base+"/setBloodDensity").c_str(),this);
  pBloodDensityCmd->SetGuidance("Set the blood density");

  pBloodHeatCapacityCmd = new G4UIcmdWithADouble((base+"/setBloodHeatCapacity").c_str(),this);
  pBloodHeatCapacityCmd->SetGuidance("Set the blood heat capacity");

  pTissueDensityCmd = new G4UIcmdWithADouble((base+"/setTissueDensity").c_str(),this);
  pTissueDensityCmd->SetGuidance("Set the tissue density");

  pTissueHeatCapacityCmd = new G4UIcmdWithADouble((base+"/setTissueHeatCapacity").c_str(),this);
  pTissueHeatCapacityCmd->SetGuidance("Set the tissue heat capacity");

  pScaleCmd = new G4UIcmdWithADouble((base+"/setSimulationScale").c_str(),this);
  pScaleCmd->SetGuidance("Set simulation scale");

  pNumTimeFramesCmd = new G4UIcmdWithAnInteger((base+"/setNumberOfTimeFrames").c_str(),this);
  pNumTimeFramesCmd->SetGuidance("Set number of time frames");

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateThermalActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if(cmd == pTimeCmd) pThermalActor->setTime(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pDiffusivityCmd) pThermalActor->setDiffusivity(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pBloodPerfusionRateCmd) pThermalActor->setBloodPerfusionRate(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pBloodDensityCmd) pThermalActor->setBloodDensity(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pBloodHeatCapacityCmd) pThermalActor->setBloodHeatCapacity(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pTissueDensityCmd) pThermalActor->setTissueDensity(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pTissueHeatCapacityCmd) pThermalActor->setTissueHeatCapacity(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pScaleCmd) pThermalActor->setScale(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pNumTimeFramesCmd) pThermalActor->setNumberOfTimeFrames(  G4UIcmdWithAnInteger::GetNewIntValue(newValue)  );

  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEThermalActorMESSENGER_CC */

#endif // end define USE_ITK
