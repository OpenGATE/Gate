/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class GateNanoActorMessenger
  \brief This class is the GateNanoActor messenger. 
  \author vesna.cuplov@gmail.com
*/

#ifndef GATENANOACTORMESSENGER_CC
#define GATENANOACTORMESSENGER_CC

#include "GateNanoActorMessenger.hh"
#include "GateNanoActor.hh"

//-----------------------------------------------------------------------------
GateNanoActorMessenger::GateNanoActorMessenger(GateNanoActor* sensor)
  :GateImageActorMessenger(sensor),
  pNanoActor(sensor)
{
  pTimeCmd = 0;
  pDiffusivityCmd = 0;
  pBodyTempCmd = 0;
  pBloodTempCmd = 0;
  pNanoTempCmd = 0;
  pBloodPerfusionRateCmd = 0;
  pBloodDensityCmd = 0;
  pBloodHeatCapacityCmd = 0;
  pTissueDensityCmd = 0;
  pTissueHeatCapacityCmd = 0;
  pTissueThermalConductivityCmd = 0;
  pNanoAbsorptionCrossSectionCmd = 0;
  pNanoDensityCmd = 0;
  pScaleCmd = 0;

  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateNanoActorMessenger::~GateNanoActorMessenger()
{

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNanoActorMessenger::BuildCommands(G4String base)
{

  pTimeCmd = new G4UIcmdWithADouble((base+"/setTime").c_str(),this);
  pTimeCmd->SetGuidance("Set the diffusion time");

  pDiffusivityCmd = new G4UIcmdWithADouble((base+"/setDiffusivity").c_str(),this);
  pDiffusivityCmd->SetGuidance("Set the thermal diffusivity");

  pBodyTempCmd = new G4UIcmdWithADouble((base+"/setBodyTemperature").c_str(),this);
  pBodyTempCmd->SetGuidance("Set the body temperature");

  pBloodTempCmd = new G4UIcmdWithADouble((base+"/setBloodTemperature").c_str(),this);
  pBloodTempCmd->SetGuidance("Set the blood temperature");

  pNanoTempCmd = new G4UIcmdWithADouble((base+"/setNanoMaximumTemperature").c_str(),this);
  pNanoTempCmd->SetGuidance("Set the nano object maxiumum temperature");

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

  pTissueThermalConductivityCmd = new G4UIcmdWithADouble((base+"/setTissueThermalConductivity").c_str(),this);
  pTissueThermalConductivityCmd->SetGuidance("Set the tissue thermal conductivity");

  pNanoAbsorptionCrossSectionCmd = new G4UIcmdWithADouble((base+"/setNanoAbsorptionCrossSection").c_str(),this);
  pNanoAbsorptionCrossSectionCmd->SetGuidance("Set the nano particle absorption cross section");

  pNanoDensityCmd = new G4UIcmdWithADouble((base+"/setNanoDensity").c_str(),this);
  pNanoDensityCmd->SetGuidance("Set the nano object density");

  pScaleCmd = new G4UIcmdWithADouble((base+"/setScale").c_str(),this);
  pScaleCmd->SetGuidance("Set simulation scale");

}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateNanoActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if(cmd == pTimeCmd) pNanoActor->setTime(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pDiffusivityCmd) pNanoActor->setDiffusivity(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pBodyTempCmd) pNanoActor->setBodyTemperature(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pBloodTempCmd) pNanoActor->setBloodTemperature(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pNanoTempCmd) pNanoActor->setNanoMaximumTemperature(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pBloodPerfusionRateCmd) pNanoActor->setBloodPerfusionRate(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pBloodDensityCmd) pNanoActor->setBloodDensity(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pBloodHeatCapacityCmd) pNanoActor->setBloodHeatCapacity(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pTissueDensityCmd) pNanoActor->setTissueDensity(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pTissueHeatCapacityCmd) pNanoActor->setTissueHeatCapacity(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pTissueThermalConductivityCmd) pNanoActor->setTissueThermalConductivity(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pNanoAbsorptionCrossSectionCmd) pNanoActor->setNanoAbsorptionCrossSection(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pNanoDensityCmd) pNanoActor->setNanoDensity(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );
  if(cmd == pScaleCmd) pNanoActor->setScale(  G4UIcmdWithADouble::GetNewDoubleValue(newValue)  );

  GateImageActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATENANOACTORMESSENGER_CC */
