/*
 * GateTwistedTrapMessenger.hh
 *
 *  Created on: 2014.06.11.
 *      Author: patayg
 */

#ifndef GATETWISTEDTRAPMESSENGER_HH_
#define GATETWISTEDTRAPMESSENGER_HH_

#include "globals.hh"
#include "GateVolumeMessenger.hh"

class GateTwistedTrap;

class GateTwistedTrapMessenger: public GateVolumeMessenger
{
public:
  GateTwistedTrapMessenger(GateTwistedTrap* itsCreator);
  ~GateTwistedTrapMessenger();

  void SetNewValue(G4UIcommand*, G4String);
  virtual inline GateTwistedTrap* GetTwistedTrapCreator()
    { return (GateTwistedTrap*)GetVolumeCreator(); }

private:
  G4UIcmdWithADoubleAndUnit* TwistedTrapYMinusLengthCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTrapYPlusLengthCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTrapX1MinusLengthCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTrapX2MinusLengthCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTrapX1PlusLengthCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTrapX2PlusLengthCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTrapZLengthCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTrapTwistAngleCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTrapPolarAngleCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTrapAzimuthalAngleCmd;
  G4UIcmdWithADoubleAndUnit* TwistedTrapTiltAngleCmd;
};

#endif /* GATETWISTEDTRAPMESSENGER_HH_ */
