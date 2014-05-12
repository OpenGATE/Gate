/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

#include "GateConfiguration.h"
#ifdef G4ANALYSIS_USE_ROOT
/*
  \class  GatePhaseSpaceActorMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
	  pierre.gueth@creatis.insa-lyon.fr
*/

#ifndef GATEPHASESPACEACTORMESSENGER_HH
#define GATEPHASESPACEACTORMESSENGER_HH

#include "globals.hh"
#include "GateActorMessenger.hh"

class G4UIcmdWithABool;
class G4UIcmdWithoutParameter;
class G4UIcmdWithADoubleAndUnit;
class G4UIcmdWithAString;
class GatePhaseSpaceActor;

class GatePhaseSpaceActorMessenger : public  GateActorMessenger
{
public:
  GatePhaseSpaceActorMessenger(GatePhaseSpaceActor* sensor);
  virtual ~GatePhaseSpaceActorMessenger();

  void BuildCommands(G4String base);
  virtual void SetNewValue(G4UIcommand*, G4String);

protected:
  GatePhaseSpaceActor * pActor;

  G4UIcmdWithABool* pEnableEkineCmd;
  G4UIcmdWithABool* pEnablePositionXCmd;
  G4UIcmdWithABool* pEnablePositionYCmd;
  G4UIcmdWithABool* pEnablePositionZCmd;
  G4UIcmdWithABool* pEnableDirectionXCmd;
  G4UIcmdWithABool* pEnableDirectionYCmd;
  G4UIcmdWithABool* pEnableDirectionZCmd;
  G4UIcmdWithABool* pEnableProdVolumeCmd;
  G4UIcmdWithABool* pEnableProdProcessCmd;
  G4UIcmdWithABool* pEnableParticleNameCmd;
  G4UIcmdWithABool* pEnableWeightCmd;
  G4UIcmdWithABool* pEnableTimeCmd;
  G4UIcmdWithABool* pEnableMassCmd;
  G4UIcmdWithABool* pEnableSecCmd;
  G4UIcmdWithABool* pEnableStoreAllStepCmd;
  G4UIcmdWithABool* pCoordinateInVolumeFrameCmd;
  G4UIcmdWithADoubleAndUnit* pMaxSizeCmd;
  G4UIcmdWithABool * pInOrOutGoingParticlesCmd;

};

#endif /* end #define GATESOURCEACTORMESSENGER_HH*/
#endif
