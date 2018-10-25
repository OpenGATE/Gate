/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

/*
  \class GateMultiMaterialThermalActorMessenger
  \brief This class is the GateMultiMaterialThermalActor messenger. 
  \author fsmekens@gmail.com
*/

#ifndef GATEMULTIMATERIALTHERMALACTORMESSENGER_HH
#define GATEMULTIMATERIALTHERMALACTORMESSENGER_HH

#include "G4UIcmdWithABool.hh"
#include "GateImageActorMessenger.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAString.hh"

class GateMultiMaterialThermalActor;
class GateMultiMaterialThermalActorMessenger : public GateImageActorMessenger
{
public:
  GateMultiMaterialThermalActorMessenger(GateMultiMaterialThermalActor* sensor);
  virtual ~GateMultiMaterialThermalActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateMultiMaterialThermalActor * pThermalActor;
  
  G4UIcmdWithADoubleAndUnit* pRelaxationTimeCmd;
  G4UIcmdWithADouble* pDiffusivityCmd;
  G4UIcmdWithABool* pSetPerfusionRateByMaterialCmd;
  G4UIcmdWithADouble* pSetPerfusionRateByConstantCmd;
  G4UIcmdWithAString* pSetPerfusionRateByImageCmd;
  G4UIcmdWithADoubleAndUnit* pBloodDensityCmd;
  G4UIcmdWithADouble* pBloodHeatCapacityCmd;
  G4UIcmdWithADouble* pTissueHeatCapacityCmd;
  G4UIcmdWithABool* pEnableStepDiffusionCmd;
  G4UIcmdWithAString* pSetMeasurementFilenameCmd;
};

#endif /* end #define GATEMULTIMATERIALTHERMALACTORMESSENGER_HH*/

