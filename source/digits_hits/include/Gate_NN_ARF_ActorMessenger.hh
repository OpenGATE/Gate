/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

/*
  \class  Gate_NN_ARF_ActorMessenger
*/

#ifndef GATE_NN_ARF_ACTORMESSENGER_HH
#define GATE_NN_ARF_ACTORMESSENGER_HH

#include "GateActorMessenger.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithAnInteger.hh"
#include "G4UIcmdWithADouble.hh"

class Gate_NN_ARF_Actor;

//-----------------------------------------------------------------------------
class Gate_NN_ARF_ActorMessenger : public GateActorMessenger
{
public:
  Gate_NN_ARF_ActorMessenger(Gate_NN_ARF_Actor* sensor);
  virtual ~Gate_NN_ARF_ActorMessenger();

  void BuildCommands(G4String base);
  void SetNewValue(G4UIcommand*, G4String);

protected:
  Gate_NN_ARF_Actor         * pDIOActor;
  G4UIcmdWithAString        * pSetEnergyWindowNamesCmd;
  G4UIcmdWithAString        * pSetModeFlagCmd;
  G4UIcmdWithADoubleAndUnit * pSetMaxAngleCmd;
  G4UIcmdWithAnInteger      * pSetRRFactorCmd;
  G4UIcmdWithAString        * pSetNNModelCmd;
  G4UIcmdWithAString        * pSetNNDictCmd;
  G4UIcmdWithAString        * pSetImageCmd;
  G4UIcmdWithADoubleAndUnit * pSetSpacingXCmd;
  G4UIcmdWithADoubleAndUnit * pSetSpacingYCmd;
  G4UIcmdWithAnInteger      * pSetSizeXCmd;
  G4UIcmdWithAnInteger      * pSetSizeYCmd;
  G4UIcmdWithADoubleAndUnit * pSetCollimatorLengthCmd;
  G4UIcmdWithADouble        * pSetBatchSizeCmd;
};
//-----------------------------------------------------------------------------

#endif /* end #define GATE_NN_ARF_ACTORMESSENGER_HH */
