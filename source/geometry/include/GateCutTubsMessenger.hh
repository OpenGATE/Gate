/*
 * GateCutTubsMessenger.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATECUTTUBSMESSENGER_HH_
#define GATECUTTUBSMESSENGER_HH_

#include "globals.hh"
#include "GateVolumeMessenger.hh"

class GateCutTubs;

class GateCutTubsMessenger: public GateVolumeMessenger
{
public:
  GateCutTubsMessenger(GateCutTubs* itsCreator);
  ~GateCutTubsMessenger();

  void SetNewValue(G4UIcommand*, G4String);
  virtual inline GateCutTubs* GetCutTubsCreator()
    { return (GateCutTubs*)GetVolumeCreator(); }

private:
  G4UIcmdWithADoubleAndUnit* CutTubsInnerRCmd;
  G4UIcmdWithADoubleAndUnit* CutTubsOuterRCmd;
  G4UIcmdWithADoubleAndUnit* CutTubsStartPhiCmd;
  G4UIcmdWithADoubleAndUnit* CutTubsDeltaPhiCmd;
  G4UIcmdWithADoubleAndUnit* CutTubsZLengthCmd;
  G4UIcmdWith3Vector* CutTubsNegNormCmd;
  G4UIcmdWith3Vector* CutTubsPosNormCmd;
};

#endif /* GATECUTTUBSMESSENGER_HH_ */
