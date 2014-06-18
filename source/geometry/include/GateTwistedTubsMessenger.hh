/*
 * GateTwistedTubsMessenger.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATETWISTEDTUBSMESSENGER_HH_
#define GATETWISTEDTUBSMESSENGER_HH_

#include "globals.hh"
#include "GateVolumeMessenger.hh"

class GateTwistedTubs;

class GateTwistedTubsMessenger: public GateVolumeMessenger
{
public:
  GateTwistedTubsMessenger(GateTwistedTubs* itsCreator);
  ~GateTwistedTubsMessenger();

  void SetNewValue(G4UIcommand*, G4String);
  virtual inline GateTwistedTubs* GetTwistedTubsCreator()
    { return (GateTwistedTubs*)GetVolumeCreator(); }

private:
  G4UIcmdWithADoubleAndUnit* TwistedTubsInnerRCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTubsOuterRCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTubsPosZCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTubsNegZCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTubsTotalPhiCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTubsTwistAngleCmd;
  G4UIcmdWithAnInteger* TwistedTubsNSegmentCmd;
};



#endif /* GATETWISTEDTUBSMESSENGER_HH_ */
