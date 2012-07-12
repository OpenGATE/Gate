/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATETRACKINGGPUACTORMESSENGER_CC
#define GATETRACKINGGPUACTORMESSENGER_CC

#include "GateTrackingGPUActorMessenger.hh"
#include "GateTrackingGPUActor.hh"

//-----------------------------------------------------------------------------
GateTrackingGPUActorMessenger::GateTrackingGPUActorMessenger(GateTrackingGPUActor* sensor)
  :GateActorMessenger(sensor),
  pTrackingGPUActor(sensor)
{
  pSetCudaDeviceIDCmd = 0;
  pSetGPUBufferCmd = 0;
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateTrackingGPUActorMessenger::~GateTrackingGPUActorMessenger()
{
  if (pSetCudaDeviceIDCmd) delete pSetCudaDeviceIDCmd;
  if (pSetGPUBufferCmd) delete pSetGPUBufferCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTrackingGPUActorMessenger::BuildCommands(G4String base)
{
  G4String n = base+"/setCudaDeviceID"; 
  pSetCudaDeviceIDCmd = new G4UIcmdWithAnInteger(n, this); 
  G4String guid = G4String("Set the CUDA Device ID");
  pSetCudaDeviceIDCmd->SetGuidance(guid);

  n = base+"/setGPUBufferSize"; 
  pSetGPUBufferCmd = new G4UIcmdWithAnInteger(n, this); 
  guid = G4String("Set the buffer size for the gpu (nb of particles)");
  pSetGPUBufferCmd->SetGuidance(guid);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateTrackingGPUActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pSetCudaDeviceIDCmd) 
    pTrackingGPUActor->SetCudaDeviceID(pSetCudaDeviceIDCmd->GetNewIntValue(newValue));
  if (cmd == pSetGPUBufferCmd) 
    pTrackingGPUActor->SetGPUBufferSize(pSetGPUBufferCmd->GetNewIntValue(newValue));
  GateActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATETRACKINGGPUACTORMESSENGER_CC */
