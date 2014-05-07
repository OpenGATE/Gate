/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/


#ifndef GATEGPUTRANSTOMOACTORMESSENGER_CC
#define GATEGPUTRANSTOMOACTORMESSENGER_CC

#include "GateGPUTransTomoActorMessenger.hh"
#include "GateGPUTransTomoActor.hh"

//-----------------------------------------------------------------------------
GateGPUTransTomoActorMessenger::GateGPUTransTomoActorMessenger(GateGPUTransTomoActor* sensor)
  :GateActorMessenger(sensor),
  pTransTomoActor(sensor)
{
  pSetGPUDeviceIDCmd = 0;
  pSetGPUBufferCmd = 0;
  BuildCommands(baseName+sensor->GetObjectName());
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
GateGPUTransTomoActorMessenger::~GateGPUTransTomoActorMessenger()
{
  if (pSetGPUDeviceIDCmd) delete pSetGPUDeviceIDCmd;
  if (pSetGPUBufferCmd) delete pSetGPUBufferCmd;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateGPUTransTomoActorMessenger::BuildCommands(G4String base)
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
void GateGPUTransTomoActorMessenger::SetNewValue(G4UIcommand* cmd, G4String newValue)
{
  if (cmd == pSetGPUDeviceIDCmd) 
    pTransTomoActor->SetGPUDeviceID(pSetGPUDeviceIDCmd->GetNewIntValue(newValue));
  if (cmd == pSetGPUBufferCmd) 
    pTransTomoActor->SetGPUBufferSize(pSetGPUBufferCmd->GetNewIntValue(newValue));
  GateActorMessenger::SetNewValue( cmd, newValue);
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEGPUTRANSTOMOACTORMESSENGER_CC */
