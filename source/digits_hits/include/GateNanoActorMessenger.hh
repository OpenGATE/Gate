/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class GateNanoActorMessenger
  \brief This class is the GateNanoActor messenger. 
  \author vesna.cuplov@gmail.com
*/

#ifndef GATENANOACTORMESSENGER_HH
#define GATENANOACTORMESSENGER_HH

#include "G4UIcmdWithABool.hh"
#include "GateImageActorMessenger.hh"
#include "G4UIcmdWithADouble.hh"

class GateNanoActor;
class GateNanoActorMessenger : public GateImageActorMessenger
{
public:
  GateNanoActorMessenger(GateNanoActor* sensor);
  virtual ~GateNanoActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  GateNanoActor * pNanoActor;
  G4UIcmdWithADouble* pTimeCmd;
  G4UIcmdWithADouble* pDiffusivityCmd;
  G4UIcmdWithADouble* pBodyTempCmd;
  G4UIcmdWithADouble* pBloodTempCmd;
  G4UIcmdWithADouble* pNanoTempCmd;
  G4UIcmdWithADouble* pBloodPerfusionRateCmd;
  G4UIcmdWithADouble* pBloodDensityCmd;
  G4UIcmdWithADouble* pBloodHeatCapacityCmd;
  G4UIcmdWithADouble* pTissueDensityCmd;
  G4UIcmdWithADouble* pTissueHeatCapacityCmd;
  G4UIcmdWithADouble* pTissueThermalConductivityCmd;
  G4UIcmdWithADouble* pNanoAbsorptionCrossSectionCmd;
  G4UIcmdWithADouble* pNanoDensityCmd;
  G4UIcmdWithADouble* pScaleCmd;

};

#endif /* end #define GATENANOACTORMESSENGER_HH*/
