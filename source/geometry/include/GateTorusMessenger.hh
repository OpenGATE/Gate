/*
 * GateTorusMessenger.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATETORUSMESSENGER_HH_
#define GATETORUSMESSENGER_HH_

#include "globals.hh"
#include "GateVolumeMessenger.hh"

class GateTorus;

class GateTorusMessenger: public GateVolumeMessenger
{
public:
  GateTorusMessenger(GateTorus* itsCreator);
  ~GateTorusMessenger();

  void SetNewValue(G4UIcommand*, G4String);
  virtual inline GateTorus* GetTorusCreator()
    { return (GateTorus*)GetVolumeCreator(); }

private:
  G4UIcmdWithADoubleAndUnit* TorusInnerRCmd;
  G4UIcmdWithADoubleAndUnit* TorusOuterRCmd;
  G4UIcmdWithADoubleAndUnit* TorusStartPhiCmd;
  G4UIcmdWithADoubleAndUnit* TorusDeltaPhiCmd;
  G4UIcmdWithADoubleAndUnit* TorusTorusRCmd;
};

#endif /* GATETORUSMESSENGER_HH_ */
