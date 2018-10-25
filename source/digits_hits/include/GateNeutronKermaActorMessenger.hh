/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateNeutronKermaActorMessenger
  \author Thomas DESCHLER (thomas@deschler.fr)
*/

#ifndef GATENEUTRONKERMAACTORMESSENGER_HH
#define GATENEUTRONKERMAACTORMESSENGER_HH

#include "G4UIcmdWithABool.hh"
#include "GateImageActorMessenger.hh"

class GateNeutronKermaActor;
class GateNeutronKermaActorMessenger : public GateImageActorMessenger
{
public:
  GateNeutronKermaActorMessenger(GateNeutronKermaActor* sensor);
  virtual ~GateNeutronKermaActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateNeutronKermaActor * pKermaActor;

  G4UIcmdWithABool * pEnableDoseCmd;
  G4UIcmdWithABool * pEnableDoseSquaredCmd;
  G4UIcmdWithABool * pEnableDoseUncertaintyCmd;

  G4UIcmdWithABool * pEnableEdepCmd;
  G4UIcmdWithABool * pEnableEdepSquaredCmd;
  G4UIcmdWithABool * pEnableEdepUncertaintyCmd;

  G4UIcmdWithABool * pEnableNumberOfHitsCmd;
};

#endif
