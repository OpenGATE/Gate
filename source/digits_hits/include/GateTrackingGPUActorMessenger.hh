/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  GateTrackingGPUActorMessenger
*/

#ifndef GATETRACKINGGPUACTORMESSENGER_HH
#define GATETRACKINGGPUACTORMESSENGER_HH

#include "G4UIcmdWithAnInteger.hh"
#include "GateImageActorMessenger.hh"

class GateTrackingGPUActor;
class GateTrackingGPUActorMessenger : public GateActorMessenger
{
public:
  GateTrackingGPUActorMessenger(GateTrackingGPUActor* sensor);
  virtual ~GateTrackingGPUActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateTrackingGPUActor * pTrackingGPUActor;
  G4UIcmdWithAnInteger * pSetGPUDeviceIDCmd;
  G4UIcmdWithAnInteger * pSetGPUBufferCmd;
};

#endif /* end #define GATEDOSEACTORMESSENGER_HH*/
