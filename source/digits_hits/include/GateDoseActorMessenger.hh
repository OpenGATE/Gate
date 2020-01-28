/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
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
  //Edep
  G4UIcmdWithABool * pEnableEdepCmd;
  G4UIcmdWithABool * pEnableEdepSquaredCmd;
  G4UIcmdWithABool * pEnableEdepUncertaintyCmd;
  //Dose
  G4UIcmdWithABool * pEnableDoseCmd;
  G4UIcmdWithABool * pEnableDoseSquaredCmd;
  G4UIcmdWithABool * pEnableDoseUncertaintyCmd;
  G4UIcmdWithABool * pEnableDoseNormToMaxCmd;
  G4UIcmdWithABool * pEnableDoseNormToIntegralCmd;
    //Efficiency option
  G4UIcmdWithAString * pSetDoseEfficiencyCmd;
    //Efficiency option by Z (by ion atomic number)
  G4UIcmdWithAString * pSetDoseEfficiencyByZCmd;
  //DoseToWater
  G4UIcmdWithABool * pEnableDoseToWaterCmd;
  G4UIcmdWithABool * pEnableDoseToWaterSquaredCmd;
  G4UIcmdWithABool * pEnableDoseToWaterUncertaintyCmd;
  G4UIcmdWithABool * pEnableDoseToWaterNormToMaxCmd;
  G4UIcmdWithABool * pEnableDoseToWaterNormToIntegralCmd;
  //DoseToOtherMaterial
  G4UIcmdWithABool * pEnableDoseToOtherMaterialCmd;
  G4UIcmdWithABool * pEnableDoseToOtherMaterialSquaredCmd;
  G4UIcmdWithABool * pEnableDoseToOtherMaterialUncertaintyCmd;
  G4UIcmdWithABool * pEnableDoseToOtherMaterialNormToMaxCmd;
  G4UIcmdWithABool * pEnableDoseToOtherMaterialNormToIntegralCmd;
  G4UIcmdWithAString * pSetOtherMaterialCmd;
  //Others
  G4UIcmdWithABool * pEnableNumberOfHitsCmd;
  G4UIcmdWithAString * pSetDoseAlgorithmCmd;
  G4UIcmdWithAString * pImportMassImageCmd;
  G4UIcmdWithAString * pExportMassImageCmd;
  G4UIcmdWithAString * pVolumeFilterCmd;
  G4UIcmdWithAString * pMaterialFilterCmd;
  G4UIcmdWithABool * pTestFlagCmd;
  //Regions
  G4UIcmdWithAString * pDoseRegionInputCmd;
  G4UIcmdWithAString * pDoseRegionOutputCmd;
  G4UIcmdWithAString * pDoseRegionAddRegionCmd;
};

#endif /* end #define GATEDOSEACTORMESSENGER_HH*/
