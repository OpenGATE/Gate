/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateKermaActorMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATEKERMAACTORMESSENGER_HH
#define GATEKERMAACTORMESSENGER_HH

#include "G4UIcmdWithABool.hh"
#include "GateImageActorMessenger.hh"

class GateKermaActor;
class GateKermaActorMessenger : public GateImageActorMessenger
{
public:
  GateKermaActorMessenger(GateKermaActor* sensor);
  virtual ~GateKermaActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateKermaActor * pKermaActor;

  G4UIcmdWithABool * pEnableDoseCmd;
  G4UIcmdWithABool * pEnableDoseSquaredCmd;
  G4UIcmdWithABool * pEnableDoseUncertaintyCmd;
  G4UIcmdWithABool * pEnableDoseToWaterCmd;
  G4UIcmdWithABool * pEnableDoseToWaterSquaredCmd;
  G4UIcmdWithABool * pEnableDoseToWaterUncertaintyCmd;
  G4UIcmdWithABool * pEnableEdepCmd;
  G4UIcmdWithABool * pEnableEdepSquaredCmd;
  G4UIcmdWithABool * pEnableEdepUncertaintyCmd;
  G4UIcmdWithABool * pEnableNumberOfHitsCmd;
  G4UIcmdWithABool * pEnableDoseNormCmd;
  G4UIcmdWithABool * pEnableDoseToWaterNormCmd;
};

#endif /* end #define GATEDOSEACTORMESSENGER_HH*/
