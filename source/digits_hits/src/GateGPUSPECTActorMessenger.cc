/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEGPUSPECTACTORMESSENGER_CC
#define GATEGPUSPECTACTORMESSENGER_CC

#include "GateGPUSPECTActorMessenger.hh"
#include "GateGPUSPECTActor.hh"

#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4UIcmdWith3Vector.hh"


//-----------------------------------------------------------------------------
GateGPUSPECTActorMessenger::GateGPUSPECTActorMessenger(GateGPUSPECTActor* sensor)
  :GateActorMessenger(sensor),
  pSPECTActor(sensor)
{ 
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateGPUSPECTActorMessenger::~GateGPUSPECTActorMessenger()
{
  delete pSetGPUDeviceIDCmd;
  delete pSetGPUBufferCmd;
  delete pSetHoleHexaHeightCmd;
  delete pSetHoleHexaRadiusCmd;
  delete pSetHoleHexaRotAxisCmd;
  delete pSetHoleHexaRotAngleCmd;
  delete pSetHoleHexaMaterialCmd;
  delete pSetCubArrayRepNumXCmd;
  delete pSetCubArrayRepNumYCmd;
  delete pSetCubArrayRepNumZCmd;
  delete pSetCubArrayRepVecCmd;
  delete pSetLinearRepNumCmd;
  delete pSetLinearRepVecCmd; 
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateGPUSPECTActorMessenger::BuildCommands(G4String base)
{
  G4String n = base+"/setGPUDeviceID"; 
  pSetGPUDeviceIDCmd = new G4UIcmdWithAnInteger(n, this); 
  G4String guid = G4String("Set the CUDA Device ID");
  pSetGPUDeviceIDCmd->SetGuidance(guid);
  pSetGPUDeviceIDCmd->SetParameterName("DeviceID", false);

  n = base+"/setGPUBufferSize"; 
  pSetGPUBufferCmd = new G4UIcmdWithAnInteger(n, this); 
  guid = G4String("Set the buffer size for the gpu (nb of particles)");
  pSetGPUBufferCmd->SetGuidance(guid);
  pSetGPUBufferCmd->SetParameterName("Size", false);

  n = base+"/setHoleHexaHeight"; 
  pSetHoleHexaHeightCmd = new G4UIcmdWithADoubleAndUnit(n, this); 
  guid = G4String("Set the height of an hexagonal hole");
  pSetHoleHexaHeightCmd->SetGuidance(guid);
  pSetHoleHexaHeightCmd->SetParameterName("Height", false);
  pSetHoleHexaHeightCmd->SetDefaultUnit("cm");

  n = base+"/setHoleHexaRadius"; 
  pSetHoleHexaRadiusCmd = new G4UIcmdWithADoubleAndUnit(n, this); 
  guid = G4String("Set the radius of an hexagonal hole");
  pSetHoleHexaRadiusCmd->SetGuidance(guid);
  pSetHoleHexaRadiusCmd->SetParameterName("Radius", false);
  pSetHoleHexaRadiusCmd->SetDefaultUnit("mm");

  n = base+"/setHoleHexaRotAxis"; 
  pSetHoleHexaRotAxisCmd = new G4UIcmdWith3Vector(n, this); 
  guid = G4String("Set the rotation axis of the hexagonal hole");
  pSetHoleHexaRotAxisCmd->SetGuidance(guid);
  pSetHoleHexaRotAxisCmd->SetParameterName("RotX", "RotY", "RotZ", false);

  n = base+"/setHoleHexaRotAngle"; 
  pSetHoleHexaRotAngleCmd = new G4UIcmdWithADoubleAndUnit(n, this); 
  guid = G4String("Set the rotation angle of the hexagonal hole");
  pSetHoleHexaRotAngleCmd->SetGuidance(guid);
  pSetHoleHexaRotAngleCmd->SetParameterName("Angle", false);
  pSetHoleHexaRotAngleCmd->SetDefaultUnit("deg");

  n = base+"/setHoleHexaMaterial"; 
  pSetHoleHexaMaterialCmd = new G4UIcmdWithAString(n, this); 
  guid = G4String("Set the material of the hexagonal hole");
  pSetHoleHexaMaterialCmd->SetGuidance(guid);
  pSetHoleHexaMaterialCmd->SetParameterName("Material", false);

  n = base+"/setCubArrayRepNumX"; 
  pSetCubArrayRepNumXCmd = new G4UIcmdWithAnInteger(n, this); 
  guid = G4String("Set the number of repetitions in X");
  pSetCubArrayRepNumXCmd->SetGuidance(guid);
  pSetCubArrayRepNumXCmd->SetParameterName("CubArrayRepNumX", false);

  n = base+"/setCubArrayRepNumY"; 
  pSetCubArrayRepNumYCmd = new G4UIcmdWithAnInteger(n, this); 
  guid = G4String("Set the number of repetitions in Y");
  pSetCubArrayRepNumYCmd->SetGuidance(guid);
  pSetCubArrayRepNumYCmd->SetParameterName("CubArrayRepNumY", false);

  n = base+"/setCubArrayRepNumZ"; 
  pSetCubArrayRepNumZCmd = new G4UIcmdWithAnInteger(n, this); 
  guid = G4String("Set the number of repetitions in Z");
  pSetCubArrayRepNumZCmd->SetGuidance(guid);
  pSetCubArrayRepNumZCmd->SetParameterName("CubArrayRepNumZ", false);

  n = base+"/setCubArrayRepVec"; 
  pSetCubArrayRepVecCmd = new G4UIcmdWith3VectorAndUnit(n, this); 
  guid = G4String("Set the vector to translate the object");
  pSetCubArrayRepVecCmd->SetGuidance(guid);
  pSetCubArrayRepVecCmd->SetParameterName("CubArrayRepVecX", "CubArrayRepVecY", "CubArrayRepVecZ", false);
  pSetCubArrayRepVecCmd->SetDefaultUnit("mm");  

  n = base+"/setLinearRepNum"; 
  pSetLinearRepNumCmd = new G4UIcmdWithAnInteger(n, this); 
  guid = G4String("Set the number of linear repetitions");
  pSetLinearRepNumCmd->SetGuidance(guid);
  pSetLinearRepNumCmd->SetParameterName("LinearRepNum", false);

  n = base+"/setLinearRepVec"; 
  pSetLinearRepVecCmd = new G4UIcmdWith3VectorAndUnit(n, this); 
  guid = G4String("Set the vector for the linear repetition");
  pSetLinearRepVecCmd->SetGuidance(guid);
  pSetLinearRepVecCmd->SetParameterName("LinearRepVecX", "LinearRepVecY", "LinearRepVecZ", false);
  pSetLinearRepVecCmd->SetDefaultUnit("mm"); 
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateGPUSPECTActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pSetGPUDeviceIDCmd) 
    pSPECTActor->SetGPUDeviceID(pSetGPUDeviceIDCmd->GetNewIntValue(newValue));
  if (cmd == pSetGPUBufferCmd) 
    pSPECTActor->SetGPUBufferSize(pSetGPUBufferCmd->GetNewIntValue(newValue));
  if (cmd == pSetHoleHexaHeightCmd) 
    pSPECTActor->SetHoleHexaHeight(pSetHoleHexaHeightCmd->GetNewDoubleValue(newValue));
  if (cmd == pSetHoleHexaRadiusCmd) 
    pSPECTActor->SetHoleHexaRadius(pSetHoleHexaRadiusCmd->GetNewDoubleValue(newValue));
  if (cmd == pSetHoleHexaRotAxisCmd) 
    pSPECTActor->SetHoleHexaRotAxis(pSetHoleHexaRotAxisCmd->GetNew3VectorValue(newValue));
  if (cmd == pSetHoleHexaRotAngleCmd) 
    pSPECTActor->SetHoleHexaRotAngle(pSetHoleHexaRotAngleCmd->GetNewDoubleValue(newValue));
  if (cmd == pSetHoleHexaMaterialCmd) 
    pSPECTActor->SetHoleHexaMaterial(newValue);
  if (cmd == pSetCubArrayRepNumXCmd) 
    pSPECTActor->SetCubArrayRepNumX(pSetCubArrayRepNumXCmd->GetNewIntValue(newValue));
  if (cmd == pSetCubArrayRepNumYCmd) 
    pSPECTActor->SetCubArrayRepNumY(pSetCubArrayRepNumYCmd->GetNewIntValue(newValue));
  if (cmd == pSetCubArrayRepNumZCmd) 
    pSPECTActor->SetCubArrayRepNumZ(pSetCubArrayRepNumZCmd->GetNewIntValue(newValue));
  if (cmd == pSetCubArrayRepVecCmd) 
    pSPECTActor->SetCubArrayRepVec(pSetCubArrayRepVecCmd->GetNew3VectorValue(newValue));
  if (cmd == pSetLinearRepNumCmd) 
    pSPECTActor->SetLinearRepNum(pSetLinearRepNumCmd->GetNewIntValue(newValue));
  if (cmd == pSetLinearRepVecCmd) 
    pSPECTActor->SetLinearRepVec(pSetLinearRepVecCmd->GetNew3VectorValue(newValue));
  GateActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEGPUSPECTACTORMESSENGER_CC */
