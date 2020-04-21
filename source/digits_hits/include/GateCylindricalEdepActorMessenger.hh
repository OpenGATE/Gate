/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateCylindircalEdepActorMessenger
  * based on GateDoseActorMessenger
  \author A.Resch
*/

#ifndef GATECYLINDRICALEDEPACTORMESSENGER_HH
#define GATECYLINDRICALEDEPACTORMESSENGER_HH

#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAString.hh"
#include "GateImageActorMessenger.hh"

class GateCylindricalEdepActor;
class GateCylindricalEdepActorMessenger : public GateImageActorMessenger
{
public:
  GateCylindricalEdepActorMessenger(GateCylindricalEdepActor* sensor);
  virtual ~GateCylindricalEdepActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateCylindricalEdepActor * pCylindicalEdepActor;

  G4UIcmdWithABool * pEnableDoseCmd;
  G4UIcmdWithABool * pEnableDoseSquaredCmd;
  G4UIcmdWithABool * pEnableDoseUncertaintyCmd;
  G4UIcmdWithABool * pEnableDoseToWaterCmd;
  G4UIcmdWithABool * pEnableDoseToWaterSquaredCmd;
  G4UIcmdWithABool * pEnableDoseToWaterUncertaintyCmd;
  G4UIcmdWithABool * pEnableEdepCmd;
  
  G4UIcmdWithABool * pEnableEdepHadElasticCmd;
  G4UIcmdWithABool * pEnableEdepInelasticCmd;
  G4UIcmdWithABool * pEnableEdepRestCmd;
  
  G4UIcmdWithABool * pEnableEdepSquaredCmd;
  G4UIcmdWithABool * pEnableEdepUncertaintyCmd;
  G4UIcmdWithABool * pEnableNumberOfHitsCmd;
  G4UIcmdWithABool * pEnableDoseNormToMaxCmd;
  G4UIcmdWithABool * pEnableDoseNormToIntegralCmd;
  G4UIcmdWithABool * pEnableDoseToWaterNormCmd;
  G4UIcmdWithAString * pSetDoseAlgorithmCmd;
  G4UIcmdWithAString * pImportMassImageCmd;
  G4UIcmdWithAString * pExportMassImageCmd;
};

#endif /* end #define GateCylindricalEdepActorMESSENGER_HH*/
