/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#include "GatePhaseSpaceActorMessenger.hh"
#ifdef G4ANALYSIS_USE_ROOT

#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithoutParameter.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"

#include "GatePhaseSpaceActor.hh"

//-----------------------------------------------------------------------------
GatePhaseSpaceActorMessenger::GatePhaseSpaceActorMessenger(GatePhaseSpaceActor* sensor)
  :GateActorMessenger(sensor),pActor(sensor)
{
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GatePhaseSpaceActorMessenger::~GatePhaseSpaceActorMessenger()
{
  delete pEnableMassCmd;
  delete pEnableEkineCmd;
  delete pEnablePositionXCmd;
  delete pEnablePositionYCmd;
  delete pEnablePositionZCmd;
  delete pEnableDirectionXCmd;
  delete pEnableDirectionYCmd;
  delete pEnableDirectionZCmd;
  delete pEnableProdVolumeCmd;
  delete pEnableProdProcessCmd;
  delete pEnableParticleNameCmd;
  delete pCoordinateInVolumeFrameCmd;
  delete pEnableWeightCmd;
  delete pEnableTimeCmd;
  delete pMaxSizeCmd;
  delete pInOrOutGoingParticlesCmd;
  delete pEnableSecCmd;
  delete pEnableStoreAllStepCmd;
  delete bEnablePrimaryEnergyCmd;
  delete bCoordinateFrameCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePhaseSpaceActorMessenger::BuildCommands(G4String base)
{
  G4String guidance;
  G4String bb;

  bb = base+"/enableEkine";
  pEnableEkineCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save kinetic energy of particles in the phase space file.";
  pEnableEkineCmd->SetGuidance(guidance);
  pEnableEkineCmd->SetParameterName("State",false);

  bb = base+"/enableXPosition";
  pEnablePositionXCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save position of particles along X axis in the phase space file.";
  pEnablePositionXCmd->SetGuidance(guidance);
  pEnablePositionXCmd->SetParameterName("State",false);

  bb = base+"/enableXDirection";
  pEnableDirectionXCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save direction of particles along X axis in the phase space file.";
  pEnableDirectionXCmd->SetGuidance(guidance);
  pEnableDirectionXCmd->SetParameterName("State",false);

  bb = base+"/enableYPosition";
  pEnablePositionYCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save position of particles along Y axis in the phase space file.";
  pEnablePositionYCmd->SetGuidance(guidance);
  pEnablePositionYCmd->SetParameterName("State",false);

  bb = base+"/enableYDirection";
  pEnableDirectionYCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save direction of particles along Y axis in the phase space file.";
  pEnableDirectionYCmd->SetGuidance(guidance);
  pEnableDirectionYCmd->SetParameterName("State",false);

  bb = base+"/enableZPosition";
  pEnablePositionZCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save position of particles along Z axis in the phase space file.";
  pEnablePositionZCmd->SetGuidance(guidance);
  pEnablePositionZCmd->SetParameterName("State",false);

  bb = base+"/enableZDirection";
  pEnableDirectionZCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save direction of particles along Z axis in the phase space file.";
  pEnableDirectionZCmd->SetGuidance(guidance);
  pEnableDirectionZCmd->SetParameterName("State",false);

  bb = base+"/enableProductionVolume";
  pEnableProdVolumeCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save the name of the volume, where particle is created, in the phase space file.";
  pEnableProdVolumeCmd->SetGuidance(guidance);
  pEnableProdVolumeCmd->SetParameterName("State",false);

  bb = base+"/enableProductionProcess";
  pEnableProdProcessCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save the name of the process, which produced the particle, in the phase space file.";
  pEnableProdProcessCmd->SetGuidance(guidance);
  pEnableProdProcessCmd->SetParameterName("State",false);

  bb = base+"/enableParticleName";
  pEnableParticleNameCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save the name of particles in the phase space file.";
  pEnableParticleNameCmd->SetGuidance(guidance);
  pEnableParticleNameCmd->SetParameterName("State",false);

  bb = base+"/enableWeight";
  pEnableWeightCmd =  new G4UIcmdWithABool(bb,this);
  guidance = "Save the weight of particles in the phase space file.";
  pEnableWeightCmd->SetGuidance(guidance);
  pEnableWeightCmd->SetParameterName("State",false);

  bb = base+"/enableTime";
  pEnableTimeCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save the time of particles in the phase space file.";
  pEnableTimeCmd->SetGuidance(guidance);
  pEnableTimeCmd->SetParameterName("State",false);

  bb = base+"/enableMass";
  pEnableMassCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Save the mass of particles in the phase space file.";
  pEnableMassCmd->SetGuidance(guidance);
  pEnableMassCmd->SetParameterName("State",false);

  bb = base+"/storeSecondaries";
  pEnableSecCmd =  new G4UIcmdWithABool(bb,this);
  guidance = "Store the secondary particles created in the attached volume.";
  pEnableSecCmd->SetGuidance(guidance);
  pEnableSecCmd->SetParameterName("State",false);

  bb = base+"/storeOutgoingParticles";
  pInOrOutGoingParticlesCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Store the outgoing particles instead of incoming particles.";
  pInOrOutGoingParticlesCmd->SetGuidance(guidance);

  bb = base+"/storeAllStep";
  pEnableStoreAllStepCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Store all step inside the attached volume";
  pEnableStoreAllStepCmd->SetGuidance(guidance);
  pEnableStoreAllStepCmd->SetParameterName("State",false);

  bb = base+"/useVolumeFrame";
  pCoordinateInVolumeFrameCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Record coordinate in the actor volume frame.";
  pCoordinateInVolumeFrameCmd->SetGuidance(guidance);

  bb = base+"/setMaxFileSize";
  pMaxSizeCmd = new G4UIcmdWithADoubleAndUnit(bb, this);
  guidance = G4String("Set maximum size of the phase space file. When the file reaches the maximum size, a new file is automatically created.");
  pMaxSizeCmd->SetGuidance(guidance);
  pMaxSizeCmd->SetParameterName("Size", false);
  pMaxSizeCmd->SetUnitCategory("Memory size");

  bb = base+"/enablePrimaryEnergy";
  bEnablePrimaryEnergyCmd = new G4UIcmdWithABool(bb,this);
  guidance = "Store the energy of the primary particle for every hit.";
  bEnablePrimaryEnergyCmd->SetGuidance(guidance);

  bb = base+"/setCoordinateFrame";
  bCoordinateFrameCmd = new G4UIcmdWithAString(bb, this);
  guidance = "Store the hit coordinates in the frame of the frame passed as an argument.";
  bCoordinateFrameCmd->SetGuidance(guidance);
  bCoordinateFrameCmd->SetParameterName("Coordinate Frame",false);


}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GatePhaseSpaceActorMessenger::SetNewValue(G4UIcommand* command, G4String param)
{
  if(command == pEnableEkineCmd) pActor->SetIsEkineEnabled(pEnableEkineCmd->GetNewBoolValue(param));
  if(command == pEnablePositionXCmd) pActor->SetIsXPositionEnabled(pEnablePositionXCmd->GetNewBoolValue(param));
  if(command == pEnableDirectionXCmd) pActor->SetIsXDirectionEnabled(pEnableDirectionXCmd->GetNewBoolValue(param));
  if(command == pEnablePositionYCmd) pActor->SetIsYPositionEnabled(pEnablePositionYCmd->GetNewBoolValue(param));
  if(command == pEnableDirectionYCmd) pActor->SetIsYDirectionEnabled(pEnableDirectionYCmd->GetNewBoolValue(param));
  if(command == pEnablePositionZCmd) pActor->SetIsZPositionEnabled(pEnablePositionZCmd->GetNewBoolValue(param));
  if(command == pEnableDirectionZCmd) pActor->SetIsZDirectionEnabled(pEnableDirectionZCmd->GetNewBoolValue(param));
  if(command == pEnableProdProcessCmd) pActor->SetIsProdProcessEnabled(pEnableProdProcessCmd->GetNewBoolValue(param));
  if(command == pEnableProdVolumeCmd) pActor->SetIsProdVolumeEnabled(pEnableProdVolumeCmd->GetNewBoolValue(param));
  if(command == pEnableParticleNameCmd) pActor->SetIsParticleNameEnabled(pEnableParticleNameCmd->GetNewBoolValue(param));
  if(command == pEnableWeightCmd) pActor->SetIsWeightEnabled(pEnableWeightCmd->GetNewBoolValue(param));
  if(command == pEnableTimeCmd) pActor->SetIsTimeEnabled(pEnableTimeCmd->GetNewBoolValue(param));
  if(command == pEnableMassCmd) pActor->SetIsMassEnabled(pEnableMassCmd->GetNewBoolValue(param));
  if(command == pCoordinateInVolumeFrameCmd) pActor->SetUseVolumeFrame(pCoordinateInVolumeFrameCmd->GetNewBoolValue(param));
  if(command == pInOrOutGoingParticlesCmd) pActor->SetStoreOutgoingParticles(pInOrOutGoingParticlesCmd->GetNewBoolValue(param));
  if(command == pEnableStoreAllStepCmd) pActor->SetIsAllStep(pEnableStoreAllStepCmd->GetNewBoolValue(param));
  if(command == pEnableSecCmd) pActor->SetIsSecStored(pEnableSecCmd->GetNewBoolValue(param));
  if(command == pSaveEveryNEventsCmd || command == pSaveEveryNSecondsCmd)  GateError("saveEveryNEvents and saveEveryNSeconds commands are not available with phase space actor. But you can use the setMaxFileSize command.");
  if(command == pMaxSizeCmd) pActor->SetMaxFileSize(pMaxSizeCmd->GetNewDoubleValue(param));
  if(command == bEnablePrimaryEnergyCmd) pActor->SetIsPrimaryEnergyEnabled(bEnablePrimaryEnergyCmd->GetNewBoolValue(param));
  if(command == bCoordinateFrameCmd) {pActor->SetCoordFrame(param);pActor->SetEnableCoordFrame();};

  GateActorMessenger::SetNewValue(command ,param );
}
//-----------------------------------------------------------------------------

#endif
