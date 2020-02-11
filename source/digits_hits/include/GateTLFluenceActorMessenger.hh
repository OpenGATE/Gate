/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateTLFluenceActorMessenger
  \author anders.garpebring@umu.se
*/

#ifndef GATETLFLUENCEACTORMESSENGER_HH
#define GATETLFLUENCEACTORMESSENGER_HH

#include "G4UIcmdWithABool.hh"
#include "GateImageActorMessenger.hh"

class GateTLFluenceActor;
class GateTLFluenceActorMessenger : public GateImageActorMessenger
{
public:
  GateTLFluenceActorMessenger(GateTLFluenceActor* sensor);
  virtual ~GateTLFluenceActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateTLFluenceActor * pFluenceActor;

  G4UIcmdWithABool * pEnableFluenceCmd;
  G4UIcmdWithABool * pEnableEnergyFluenceCmd;
  G4UIcmdWithABool * pEnableFluenceUncertaintyCmd;
  G4UIcmdWithABool * pEnableFluenceSquaredCmd;
  G4UIcmdWithABool * pEnableEnergyFluenceSquaredCmd;
  G4UIcmdWithABool * pEnableEnergyFluenceUncertaintyCmd;
};

#endif /* end #define GATETLFLUENCEACTORMESSENGER_HH*/
