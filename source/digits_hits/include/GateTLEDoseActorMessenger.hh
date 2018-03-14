/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  \class  GateTLEDoseActorMessenger
  \author fabien.baldacci@creatis.insa-lyon.fr
*/

#ifndef GATETLEDOSEACTORMESSENGER_HH
#define GATETLEDOSEACTORMESSENGER_HH

#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAString.hh"
#include "GateImageActorMessenger.hh"

class GateTLEDoseActor;
class GateTLEDoseActorMessenger : public GateImageActorMessenger
{
public:
  GateTLEDoseActorMessenger(GateTLEDoseActor* sensor);
  virtual ~GateTLEDoseActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateTLEDoseActor * pDoseActor;

  G4UIcmdWithABool * pEnableDoseCmd;
  G4UIcmdWithABool * pEnableDoseSquaredCmd;
  G4UIcmdWithABool * pEnableDoseUncertaintyCmd;
  G4UIcmdWithABool * pEnableEdepCmd;
  G4UIcmdWithABool * pEnableEdepSquaredCmd;
  G4UIcmdWithABool * pEnableEdepUncertaintyCmd;
  G4UIcmdWithABool * pEnableDoseNormToMaxCmd;
  G4UIcmdWithABool * pEnableDoseNormToIntegralCmd;

  G4UIcmdWithAString * pSetDoseAlgorithmCmd;
  G4UIcmdWithAString * pImportMassImageCmd;
  G4UIcmdWithAString * pVolumeFilterCmd;
  G4UIcmdWithAString * pMaterialFilterCmd;
};

#endif /* end #define GATETLEDOSEACTORMESSENGER_HH*/
