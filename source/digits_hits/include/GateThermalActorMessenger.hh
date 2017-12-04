/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*
  \class GateThermalActorMessenger
  \brief This class is the GateThermalActor messenger. 
  \author vesna.cuplov@gmail.com
*/

#ifndef GATETHERMALACTORMESSENGER_HH
#define GATETHERMALACTORMESSENGER_HH

#include "G4UIcmdWithABool.hh"
#include "GateImageActorMessenger.hh"
#include "G4UIcmdWithADouble.hh"

class GateThermalActor;
class GateThermalActorMessenger : public GateImageActorMessenger
{
public:
  GateThermalActorMessenger(GateThermalActor* sensor);
  virtual ~GateThermalActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateThermalActor * pThermalActor;
  G4UIcmdWithADouble* pTimeCmd;
  G4UIcmdWithADouble* pDiffusivityCmd;
  G4UIcmdWithADouble* pBloodPerfusionRateCmd;
  G4UIcmdWithADouble* pBloodDensityCmd;
  G4UIcmdWithADouble* pBloodHeatCapacityCmd;
  G4UIcmdWithADouble* pTissueDensityCmd;
  G4UIcmdWithADouble* pTissueHeatCapacityCmd;
  G4UIcmdWithADouble* pScaleCmd;
  G4UIcmdWithAnInteger* pNumTimeFramesCmd;

};

#endif /* end #define GATETHERMALACTORMESSENGER_HH*/

