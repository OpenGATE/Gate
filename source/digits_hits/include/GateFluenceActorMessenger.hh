/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  \class  GateFluenceActorMessenger
  \author simon.rit@creatis.insa-lyon.fr
*/

#ifndef GATEFLUENCEACTORMESSENGER_HH
#define GATEFLUENCEACTORMESSENGER_HH

#include "GateImageActorMessenger.hh"

class GateFluenceActor;
class GateFluenceActorMessenger : public GateImageActorMessenger
{
public:
  GateFluenceActorMessenger(GateFluenceActor* sensor);
  virtual ~GateFluenceActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:

  G4UIcmdWithABool * pEnableSquaredCmd;
  G4UIcmdWithABool * pEnableStepLengthCmd;
  G4UIcmdWithABool * pEnableUncertaintyCmd;
  G4UIcmdWithABool * pEnableNormCmd;
  G4UIcmdWithABool * pEnableNumberOfHitsCmd;
  G4UIcmdWithABool * pSetIgnoreWeightCmd;
  GateFluenceActor * pFluenceActor;
  G4UIcmdWithAString * pSetResponseDetectorFileCmd;
  G4UIcmdWithABool * pEnableScatterCmd;
  G4UIcmdWithAString * pSetScatterOrderFilenameCmd;
  G4UIcmdWithAString * pSetSeparateProcessFilenameCmd;
};

#endif /* end #define GATEFLUENCEACTORMESSENGER_HH*/
