/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GateGPUPhotRadTheraActorMESSENGER_CC
#define GateGPUPhotRadTheraActorMESSENGER_CC

#include "GateGPUPhotRadTheraActorMessenger.hh"
#include "GateGPUPhotRadTheraActor.hh"

//-----------------------------------------------------------------------------
GateGPUPhotRadTheraActorMessenger::GateGPUPhotRadTheraActorMessenger(GateGPUPhotRadTheraActor* sensor)
  :GateActorMessenger(sensor),
  pPhotRadTheraActor(sensor)
{
  pSetGPUDeviceIDCmd = 0;
  pSetGPUBufferCmd = 0;
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateGPUPhotRadTheraActorMessenger::~GateGPUPhotRadTheraActorMessenger()
{
  if (pSetGPUDeviceIDCmd) delete pSetGPUDeviceIDCmd;
  if (pSetGPUBufferCmd) delete pSetGPUBufferCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateGPUPhotRadTheraActorMessenger::BuildCommands(G4String base)
{
  G4String n = base+"/setGPUDeviceID"; 
  pSetGPUDeviceIDCmd = new G4UIcmdWithAnInteger(n, this); 
  G4String guid = G4String("Set the CUDA Device ID");
  pSetGPUDeviceIDCmd->SetGuidance(guid);

  n = base+"/setGPUBufferSize"; 
  pSetGPUBufferCmd = new G4UIcmdWithAnInteger(n, this); 
  guid = G4String("Set the buffer size for the gpu (nb of particles)");
  pSetGPUBufferCmd->SetGuidance(guid);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateGPUPhotRadTheraActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pSetGPUDeviceIDCmd) 
    pPhotRadTheraActor->SetGPUDeviceID(pSetGPUDeviceIDCmd->GetNewIntValue(newValue));
  if (cmd == pSetGPUBufferCmd) 
    pPhotRadTheraActor->SetGPUBufferSize(pSetGPUBufferCmd->GetNewIntValue(newValue));
  GateActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GateGPUPhotRadTheraActorMESSENGER_CC */
