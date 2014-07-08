/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  GATEGPUTransTomoActorMessenger
*/

#ifndef GATEGPUTRANSTOMOACTORMESSENGER_HH
#define GATEGPUTRANSTOMOACTORMESSENGER_HH

#include "G4UIcmdWithAnInteger.hh"
#include "GateImageActorMessenger.hh"

class GateGPUTransTomoActor;
class GateGPUTransTomoActorMessenger : public GateActorMessenger
{
public:
  GateGPUTransTomoActorMessenger(GateGPUTransTomoActor* sensor);
  virtual ~GateGPUTransTomoActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateGPUTransTomoActor * pTransTomoActor;
  G4UIcmdWithAnInteger * pSetGPUDeviceIDCmd;
  G4UIcmdWithAnInteger * pSetGPUBufferCmd;
};

#endif /* end #define GATEGPUTRANSTOMOACTOR_HH*/
