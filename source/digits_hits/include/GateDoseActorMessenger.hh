/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateDoseActorMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
          david.sarrut@creatis.insa-lyon.fr
*/

#ifndef GATEDOSEACTORMESSENGER_HH
#define GATEDOSEACTORMESSENGER_HH

#include "G4UIcmdWithABool.hh"
#include "G4UIcmdWithAString.hh"
#include "GateImageActorMessenger.hh"

class GateDoseActor;
class GateDoseActorMessenger : public GateImageActorMessenger
{
public:
  GateDoseActorMessenger(GateDoseActor* sensor);
  virtual ~GateDoseActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateDoseActor * pDoseActor;

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
  G4UIcmdWithABool * pEnableDoseNormToMaxCmd;
  G4UIcmdWithABool * pEnableDoseNormToIntegralCmd;
  G4UIcmdWithABool * pEnableDoseToWaterNormCmd;
  G4UIcmdWithABool * pEnablePeakFinderDoseCalculationCmd;
  G4UIcmdWithAString * pSetDoseAlgorithmCmd;
  G4UIcmdWithAString * pImportMassImageCmd;
  G4UIcmdWithAString * pExportMassImageCmd;

};

#endif /* end #define GATEDOSEACTORMESSENGER_HH*/
