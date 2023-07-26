/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*
  \class  GatePhaseSpaceActorMessenger
  \author thibault.frisson@creatis.insa-lyon.fr
  laurent.guigues@creatis.insa-lyon.fr
  david.sarrut@creatis.insa-lyon.fr
  pierre.gueth@creatis.insa-lyon.fr
  brent.huisman@creatis.insa-lyon.fr
*/

#ifndef GATEPHASESPACEACTORMESSENGER_HH
#define GATEPHASESPACEACTORMESSENGER_HH

#include "globals.hh"
#include "GateActorMessenger.hh"

class G4UIcmdWithABool;

class G4UIcmdWithoutParameter;

class G4UIcmdWithADoubleAndUnit;

class G4UIcmdWithAString;

class G4UIcmdWith3VectorAndUnit;

class GatePhaseSpaceActor;

class GatePhaseSpaceActorMessenger : public GateActorMessenger
{
public:
  GatePhaseSpaceActorMessenger(GatePhaseSpaceActor *sensor);

  virtual ~GatePhaseSpaceActorMessenger();

  void BuildCommands(G4String base);

  virtual void SetNewValue(G4UIcommand *, G4String);

protected:
  GatePhaseSpaceActor *pActor;

  G4UIcmdWithABool *pEnableChargeCmd;
  G4UIcmdWithABool *pEnableAtomicNumberCmd;
  G4UIcmdWithABool *pEnableElectronicDEDXCmd;
  G4UIcmdWithABool *pEnableTotalDEDXCmd;
  G4UIcmdWithABool *pEnableEkineCmd;
  G4UIcmdWithABool *pEnablePositionXCmd;
  G4UIcmdWithABool *pEnablePositionYCmd;
  G4UIcmdWithABool *pEnablePositionZCmd;
  G4UIcmdWithABool *pEnableDirectionXCmd;
  G4UIcmdWithABool *pEnableDirectionYCmd;
  G4UIcmdWithABool *pEnableDirectionZCmd;
  G4UIcmdWithABool *pEnableProdVolumeCmd;
  G4UIcmdWithABool *pEnableProdProcessCmd;
  G4UIcmdWithABool *pEnableParticleNameCmd;
  G4UIcmdWithABool *pEnableWeightCmd;
  G4UIcmdWithABool *pEnableTimeCmd;
  G4UIcmdWithABool* pEnableIonTimeCmd;
  G4UIcmdWithABool *pEnableTimeFromBeginOfEventCmd;
  G4UIcmdWithABool *pEnableTrackLengthCmd;
  G4UIcmdWithABool *pEnableMassCmd;
  G4UIcmdWithABool *pEnableSecCmd;
  G4UIcmdWithABool *pEnableStoreAllStepCmd;
  G4UIcmdWithABool *pCoordinateInVolumeFrameCmd;
  G4UIcmdWithADoubleAndUnit *pMaxSizeCmd;
  G4UIcmdWithABool *pInOrOutGoingParticlesCmd;
  G4UIcmdWithABool *bEnablePrimaryEnergyCmd;
  G4UIcmdWithABool *bEnableEmissionPointCmd;
  G4UIcmdWithAString *bCoordinateFrameCmd;
  G4UIcmdWithABool *bEnableLocalTimeCmd;
  G4UIcmdWithAString *bSpotIDFromSourceCmd;
  G4UIcmdWithABool *bEnablePDGCodeCmd;
  G4UIcmdWithABool *bEnableCompactCmd;
  G4UIcmdWithABool *pEnableNuclearFlagCmd;
  G4UIcmdWithABool *bEnableSphereProjection;
  G4UIcmdWith3VectorAndUnit *bSetSphereProjectionCenter;
  G4UIcmdWithADoubleAndUnit *bSetSphereProjectionRadius;
  G4UIcmdWithABool *bEnableTranslationAlongDirection;
  G4UIcmdWithADoubleAndUnit *bSetTranslationAlongDirectionLength;

  G4UIcmdWithABool *pEnableTOutCmd;
  G4UIcmdWithABool *pEnableTProdCmd;
  G4UIcmdWithAString *pUseMaskCmd;
  G4UIcmdWithABool *pEnableKillCmd;
};

#endif /* end #define GATESOURCEACTORMESSENGER_HH*/
