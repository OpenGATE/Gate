/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*!
  \class  GateWashOutActorMessenger
  \author I. Martinez-Rovira (immamartinez@gmail.com)
          S. Jan (sebastien.jan@cea.fr)
*/

#ifndef GATEWASHOUTACTORMESSENGER_HH
#define GATEWASHOUTACTORMESSENGER_HH

#include "GateUIcmdWithAStringADoubleAndADoubleWithUnit.hh"
#include "GateActorMessenger.hh"
#include "GateMessenger.hh"

class GateWashOutActor;
class GateWashOutActorMessenger : public GateActorMessenger
{
public:
  GateWashOutActorMessenger(GateWashOutActor* sensor);
  virtual ~GateWashOutActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand* command, G4String newValue);

  G4double ScaleValue(G4double value,G4String unit);

protected:

  GateWashOutActor * pWashOutActor;

  G4UIcmdWithAString* ReadWashOutTableCmd;

};

#endif /* end #define GATEWASHOUTACTORMESSENGER_HH*/
