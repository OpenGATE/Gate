/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  GateGPUPhotRadTheraActorMessenger
*/

#ifndef GateGPUPhotRadTheraActorMESSENGER_HH
#define GateGPUPhotRadTheraActorMESSENGER_HH

#include "G4UIcmdWithAnInteger.hh"
#include "GateImageActorMessenger.hh"

class GateGPUPhotRadTheraActor;
class GateGPUPhotRadTheraActorMessenger : public GateActorMessenger
{
public:
  GateGPUPhotRadTheraActorMessenger(GateGPUPhotRadTheraActor* sensor);
  virtual ~GateGPUPhotRadTheraActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateGPUPhotRadTheraActor * pPhotRadTheraActor;
  G4UIcmdWithAnInteger * pSetGPUDeviceIDCmd;
  G4UIcmdWithAnInteger * pSetGPUBufferCmd;
};

#endif /* end #define GateGPUPhotRadTheraActor_HH*/
