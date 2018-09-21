/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class GateThermalActorMessenger
  \brief This class is the GateThermalActor messenger. 
  \author fsmekens@gmail.com
*/

#ifndef GATEMULTIMATERIALTHERMALACTORMESSENGER_CC
#define GATEMULTIMATERIALTHERMALACTORMESSENGER_CC

// This actor is only compiled if ITK is available
#include "GateConfiguration.h"
#ifdef  GATE_USE_ITK

#include "GateMultiMaterialThermalActorMessenger.hh"
#include "GateMultiMaterialThermalActor.hh"

//-----------------------------------------------------------------------------
GateMultiMaterialThermalActorMessenger::GateMultiMaterialThermalActorMessenger(GateMultiMaterialThermalActor* sensor)
  :GateImageActorMessenger(sensor),
  pThermalActor(sensor)
{
  pRelaxationTimeCmd = 0;
  pDiffusivityCmd = 0;
  pSetPerfusionRateByMaterialCmd = 0;
  pSetPerfusionRateByConstantCmd = 0;
  pSetPerfusionRateByImageCmd = 0;
  pBloodDensityCmd = 0;
  pBloodHeatCapacityCmd = 0;
  pTissueHeatCapacityCmd = 0;
  pEnableStepDiffusionCmd = 0;
  pSetMeasurementFilenameCmd = 0;

  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateMultiMaterialThermalActorMessenger::~GateMultiMaterialThermalActorMessenger()
{
  if(pRelaxationTimeCmd) delete pRelaxationTimeCmd;
  if(pDiffusivityCmd) delete pDiffusivityCmd;
  if(pSetPerfusionRateByMaterialCmd) delete pSetPerfusionRateByMaterialCmd;
  if(pSetPerfusionRateByConstantCmd) delete pSetPerfusionRateByConstantCmd;
  if(pSetPerfusionRateByImageCmd) delete pSetPerfusionRateByImageCmd;
  if(pBloodDensityCmd) delete pBloodDensityCmd;
  if(pBloodHeatCapacityCmd) delete pBloodHeatCapacityCmd;
  if(pTissueHeatCapacityCmd) delete pTissueHeatCapacityCmd;
  if(pEnableStepDiffusionCmd) delete pEnableStepDiffusionCmd;
  if(pSetMeasurementFilenameCmd) delete pSetMeasurementFilenameCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActorMessenger::BuildCommands(G4String base)
{
  pRelaxationTimeCmd = new G4UIcmdWithADoubleAndUnit((base+"/setRelaxationTime").c_str(),this);
  pRelaxationTimeCmd->SetGuidance("Set the relaxation time applied after the simulation");

  pDiffusivityCmd = new G4UIcmdWithADouble((base+"/setThermalDiffusivity").c_str(),this);
  pDiffusivityCmd->SetGuidance("Set the tissue thermal diffusivity");  

  pSetPerfusionRateByMaterialCmd = new G4UIcmdWithABool((base+"/setBloodPerfusionRateByMaterial").c_str(),this);
  pSetPerfusionRateByMaterialCmd->SetGuidance("Activate blood perfusion by material ('PERFUSIONRATE' in Material.xml)");

  pSetPerfusionRateByConstantCmd = new G4UIcmdWithADouble((base+"/setBloodPerfusionRateByConstant").c_str(),this);
  pSetPerfusionRateByConstantCmd->SetGuidance("Activate a global blood perfusion with the specified perfusion rate [s-1]");

  pSetPerfusionRateByImageCmd = new G4UIcmdWithAString((base+"/setBloodPerfusionRateByImage").c_str(),this);
  pSetPerfusionRateByImageCmd->SetGuidance("Activate blood perfusion using the specified image");

  pBloodDensityCmd = new G4UIcmdWithADoubleAndUnit((base+"/setBloodDensity").c_str(),this);
  pBloodDensityCmd->SetGuidance("Set the blood density");

  pBloodHeatCapacityCmd = new G4UIcmdWithADouble((base+"/setBloodHeatCapacity").c_str(),this);
  pBloodHeatCapacityCmd->SetGuidance("Set the blood heat capacity");

  pTissueHeatCapacityCmd = new G4UIcmdWithADouble((base+"/setTissueHeatCapacity").c_str(),this);
  pTissueHeatCapacityCmd->SetGuidance("Set the tissue heat capacity");
  
  pEnableStepDiffusionCmd = new G4UIcmdWithABool((base+"/enableStepDiffusion").c_str(),this);
  pEnableStepDiffusionCmd->SetGuidance("Enable time-dependent diffusion");
  
  pSetMeasurementFilenameCmd = new G4UIcmdWithAString((base+"/setMeasurementFilename").c_str(),this);
  pSetMeasurementFilenameCmd->SetGuidance("Give a list of ROI for which the deposited energy over time will be stored");
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateMultiMaterialThermalActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if(cmd == pRelaxationTimeCmd) pThermalActor->setRelaxationTime(  G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue)  );
  if(cmd == pDiffusivityCmd) pThermalActor->setDiffusivity(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pSetPerfusionRateByMaterialCmd) pThermalActor->SetBloodPerfusionByMaterial(  G4UIcmdWithABool::GetNewBoolValue(newValue)  );
  if(cmd == pSetPerfusionRateByConstantCmd) pThermalActor->SetBloodPerfusionByConstant(  G4UIcmdWithADouble::GetNewDoubleValue(newValue) / s  );
  if(cmd == pSetPerfusionRateByImageCmd) pThermalActor->SetBloodPerfusionByImage(  newValue  );
  if(cmd == pBloodDensityCmd) pThermalActor->setBloodDensity(  G4UIcmdWithADoubleAndUnit::GetNewDoubleValue(newValue)  );
  if(cmd == pBloodHeatCapacityCmd) pThermalActor->setBloodHeatCapacity(  G4UIcmdWithADouble::GetNewDoubleValue(newValue) * joule/(kg*kelvin) );
  if(cmd == pTissueHeatCapacityCmd) pThermalActor->setTissueHeatCapacity(  G4UIcmdWithADouble::GetNewDoubleValue(newValue) * joule/(kg*kelvin) );
  if(cmd == pEnableStepDiffusionCmd) pThermalActor->enableStepDiffusion(  G4UIcmdWithABool::GetNewBoolValue(newValue)  );
  if(cmd == pSetMeasurementFilenameCmd) pThermalActor->SetMeasurementFilename(  newValue  );

  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEMULTIMATERIALTHERMALACTORMESSENGER_CC */

#endif /* end #define USE_ITK */
